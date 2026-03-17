# -*- coding: utf-8 -*-
"""
wcy_eval.py — WCY Evaluation Framework v1.0
=============================================
Addresses the E1-C finding: keyword matching underestimates LLM accuracy.
This module provides three-axis scoring:

  Axis S: Structural Score  — format compliance via wcy_parser
  Axis M: Meaning Score     — semantic equivalence via LLM-as-judge
  Axis P: Provenance Score  — from= chain validity

Usage (Colab):
  !pip install anthropic tiktoken -q
  # requires wcy_parser.py in the same directory

Execution: Colab Pro, set ANTHROPIC_API_KEY directly
"""

import subprocess, sys
for pkg in ["anthropic", "tiktoken", "numpy"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import json, time, os, re, math
from dataclasses import dataclass, field
from typing import Any
import numpy as np

# wcy_parser must be in the same directory.
# In Colab: from google.colab import files; files.upload() then import
try:
    from wcy_parser import (
        parse_wcy, flatten, extract_voids, validate,
        resolve_chain, WCYLine, ValidationResult
    )
    print("✓ wcy_parser imported")
except ImportError:
    print("⚠ wcy_parser.py not found — upload it first")
    print("  In Colab: from google.colab import files; files.upload()")

# ↓↓↓ Insert your API key here ↓↓↓
API_KEY = "sk-ant-api03-YOUR_KEY_HERE"
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

import anthropic
client = anthropic.Anthropic(api_key=API_KEY)
SONNET = "claude-sonnet-4-20250514"
HAIKU  = "claude-haiku-4-5-20251001"
print("✓ API ready")


###############################################################
# AXIS S: Structural Score
# Precise format compliance measurement via wcy_parser
###############################################################

@dataclass
class StructuralScore:
    """Axis S result."""
    parse_rate: float         # fraction of lines parseable (0.0–1.0)
    phase_valid: float        # fraction of lines with valid phase markers
    depth_valid: float        # fraction of lines respecting depth ≤ 2
    from_valid: float         # fraction of valid from= references (if any)
    void_has_hint: float      # fraction of void-B markers that include hint=
    overall: float            # weighted composite score
    details: dict = field(default_factory=dict)


def score_structural(response_text: str) -> StructuralScore:
    """
    Compute the structural score for LLM-generated WCY text.
    Scores format compliance only — no keyword matching.
    """
    if not response_text.strip():
        return StructuralScore(0, 0, 0, 0, 0, 0)

    lines_raw = [l for l in response_text.split('\n') if l.strip()]
    total_lines = len(lines_raw)
    if total_lines == 0:
        return StructuralScore(0, 0, 0, 0, 0, 0)

    # attempt to parse
    try:
        parsed_lines = flatten(parse_wcy(response_text))
    except Exception:
        parsed_lines = []

    parse_rate = len(parsed_lines) / total_lines

    # phase marker validity
    valid_phase = sum(
        1 for l in lines_raw
        if l.strip() and l.strip()[0] in '.!:>~'
        and len(l.strip()) > 1 and l.strip()[1] == ' '
    )
    phase_valid = valid_phase / total_lines

    # depth compliance
    depth_valid = sum(1 for l in parsed_lines if l.depth <= 2) / max(len(parsed_lines), 1)

    # from= reference validity
    lines_with_from = [l for l in parsed_lines if l.from_refs]
    if lines_with_from:
        all_nums = {l.line_num for l in parsed_lines}
        valid_refs = sum(
            1 for l in lines_with_from
            for ref in l.from_refs
            if ref in all_nums and ref < l.line_num
        )
        total_refs = sum(len(l.from_refs) for l in lines_with_from)
        from_valid = valid_refs / max(total_refs, 1)
    else:
        from_valid = 1.0  # no from= references = no penalty

    # void-B hint inclusion rate
    void_lines = [l for l in parsed_lines if l.is_void]
    if void_lines:
        void_has_hint = sum(
            1 for l in void_lines if 'hint' in l.tags
        ) / len(void_lines)
    else:
        void_has_hint = 1.0

    # weighted composite
    overall = (
        parse_rate     * 0.30 +
        phase_valid    * 0.30 +
        depth_valid    * 0.15 +
        from_valid     * 0.15 +
        void_has_hint  * 0.10
    )

    return StructuralScore(
        parse_rate=parse_rate,
        phase_valid=phase_valid,
        depth_valid=depth_valid,
        from_valid=from_valid,
        void_has_hint=void_has_hint,
        overall=overall,
        details={
            'total_raw_lines': total_lines,
            'parsed_lines': len(parsed_lines),
            'lines_with_from': len(lines_with_from),
            'void_lines':       len(void_lines),
        }
    )


###############################################################
# AXIS M: Meaning Score
# Semantic preservation measured via embedding similarity
# Resolves the E1-C false negative (Sonnet paraphrases scored as 62% with keywords)
###############################################################

# NOTE: get_embedding() / cosine_sim() are not used internally.
# score_meaning() uses LLM-as-judge instead (more reliable than embedding
# similarity for reasoning quality). Kept as stubs for external use.
def get_embedding(text: str) -> list[float]:
    """Stub — Anthropic API has no dedicated embedding endpoint.
    Replace with OpenAI text-embedding-3-small or Cohere embed-v3 if needed."""
    raise NotImplementedError("Use score_meaning() for semantic comparison")


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity helper for external embedding use."""
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def score_meaning(
    response: str,
    reference: str,
    question: str,
    model: str = SONNET,
) -> float:
    """
    Axis M: LLM-as-judge semantic equivalence score (0.0–1.0).

    Strategy: a separate judge LLM scores whether two answers convey
    the same meaning on a 0.0–1.0 scale. Compares semantics, not keywords.

    This resolves the E1-C false negative:
    'elevated BP (165/100)' == 'hypertension' → judge returns 1.0
    """
    JUDGE_SYS = """You are a semantic equivalence judge.
Given a question and two answers (A and B), score how semantically equivalent they are on a scale of 0.0 to 1.0.
- 1.0: identical meaning, possibly different wording
- 0.8: same core claim, minor details differ
- 0.5: partially overlapping, some key points differ
- 0.2: same topic but different claims
- 0.0: contradictory or completely different

Output ONLY a JSON object: {"score": <float>, "reasoning": "<1 sentence>"}
No other text."""

    user = f"""Question: {question}

Answer A (reference): {reference}

Answer B (to evaluate): {response}

Score how semantically equivalent Answer B is to Answer A."""

    try:
        msg = client.messages.create(
            model=SONNET,
            max_tokens=100,
            system=JUDGE_SYS,
            messages=[{"role": "user", "content": user}]
        )
        raw = msg.content[0].text.strip()
        # primary: JSON parse
        try:
            data = json.loads(raw)
            return float(data.get("score", 0.0))
        except json.JSONDecodeError:
            pass
        # fallback: regex-extract first float in [0.0, 1.0]
        m = re.search(r'(?:score["\s:]+)([\d.]+)', raw)
        if m:
            return min(max(float(m.group(1)), 0.0), 1.0)
        # last resort: scan for any float-shaped token
        floats = re.findall(r'(0\.\d+|1\.0)', raw)
        return float(floats[0]) if floats else 0.5  # 0.5 (ambiguous, not a hard fail)
    except Exception as e:
        print(f"  [score_meaning error] {e}")
        return 0.0


###############################################################
# AXIS P: Provenance Score
# from= chain completeness and root traceability accuracy
###############################################################

@dataclass
class ProvenanceScore:
    """Axis P result."""
    ref_validity: float      # fraction of valid from= references
    chain_depth: int         # maximum observed chain depth
    root_reachable: float    # fraction of chains that reach a root
    overall: float


def score_provenance(response: str, ground_truth_root: str | None = None) -> ProvenanceScore:
    """
    Axis P: score from= chain completeness.

    ground_truth_root: expected root value (if known)
    """
    try:
        parsed = flatten(parse_wcy(response))
    except Exception:
        return ProvenanceScore(0, 0, 0, 0)

    if not parsed:
        return ProvenanceScore(1.0, 0, 1.0, 1.0)

    all_nums = {l.line_num for l in parsed}
    total_refs = sum(len(l.from_refs) for l in parsed)

    if total_refs == 0:
        return ProvenanceScore(1.0, 0, 1.0, 1.0)

    # reference validity ratio
    valid_refs = sum(
        1 for l in parsed
        for ref in l.from_refs
        if ref in all_nums and ref < l.line_num
    )
    ref_validity = valid_refs / total_refs

    # maximum chain depth (traced from the last lines with from=)
    lines_with_from = [l for l in parsed if l.from_refs]
    max_depth = 0
    root_successes = 0

    for line in lines_with_from[-3:]:  # sample last 3 lines only
        chain = resolve_chain(parsed, line.line_num)
        depth = len(chain)
        max_depth = max(max_depth, depth)

        if chain:
            root_line = chain[0]
            if ground_truth_root:
                # check if root line text contains the expected root value
                root_text = root_line.raw.lower()
                if ground_truth_root.lower() in root_text:
                    root_successes += 1
            else:
                root_successes += 1  # reaching any root counts as success

    root_reachable = root_successes / max(len(lines_with_from[-3:]), 1)

    overall = ref_validity * 0.5 + root_reachable * 0.5

    return ProvenanceScore(
        ref_validity=ref_validity,
        chain_depth=max_depth,
        root_reachable=root_reachable,
        overall=overall,
    )


###############################################################
# ── Composite Scoring: WCYScore ─────────────────────────────────────────────
###############################################################

@dataclass
class WCYScore:
    """Composite score across all three axes."""
    structural: StructuralScore
    meaning: float            # Axis M (0.0–1.0)
    provenance: ProvenanceScore
    composite: float          # weighted composite score
    model: str = ""
    task: str = ""
    format: str = ""          # 'wcy' or 'json'

    def __str__(self):
        return (
            f"[{self.model} / {self.task} / {self.format}]\n"
            f"  S(struct)={self.structural.overall:.2f}  "
            f"M(meaning)={self.meaning:.2f}  "
            f"P(prov)={self.provenance.overall:.2f}  "
            f"→ composite={self.composite:.2f}"
        )


def score_full(
    response: str,
    question: str,
    reference_answer: str,
    model: str = "",
    task: str = "",
    fmt: str = "wcy",
    ground_truth_root: str | None = None,
    skip_meaning: bool = False,
) -> WCYScore:
    """
    Full three-axis scoring.

    skip_meaning=True: skip API call, score structure and provenance only.
    """
    s = score_structural(response) if fmt == "wcy" else StructuralScore(1,1,1,1,1,1)
    m = score_meaning(response, reference_answer, question) if not skip_meaning else 0.0
    p = score_provenance(response, ground_truth_root) if fmt == "wcy" else ProvenanceScore(1,0,1,1)

    # per-format weighting
    if fmt == "wcy":
        composite = s.overall * 0.30 + m * 0.50 + p.overall * 0.20
    else:
        composite = m  # JSON: meaning only

    return WCYScore(
        structural=s, meaning=m, provenance=p,
        composite=composite, model=model, task=task, format=fmt
    )



# ── Runs only when executed directly (skipped on `import wcy_eval`) ──────────


if __name__ == '__main__':
    # E1-C re-run — investigating the Sonnet WCY 62% anomaly from Sprint 3

    print("\n" + "=" * 70)
    print("  E1-C RERUN — Parser-based + Meaning Score Evaluation")
    print("=" * 70)

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    count = lambda t: len(enc.encode(str(t)))

    WCY_SPEC = """WCY FORMAT:
    PHASE MARKERS: . observe  : infer  > act  ~ meta  ! exception
    SLOTS: tag=value | bare_value | ?tag (void) | from=N | conf=0.xx
    One marker per line, space after marker. Max depth 2."""

    # ── Test cases (same data as E1-C) ──────────────────────────────────────

    TEST_CASES = [
        {
            "name": "Read -- cardiovascular risk",
            "data_json": json.dumps({
                "patient": {"name": "Choi", "age": 55, "bp": "165/100", "bmi": 31.2,
                            "smoking": "yes", "family_htn": True},
                "labs": {"fasting_glucose": 128, "HbA1c": 6.4, "cholesterol": 245},
                "lifestyle": {"exercise": "none", "diet": "high_sodium"}
            }, indent=2),
            "data_wcy": """. patient=Choi  age=55  bp=165/100  bmi=31.2
    . history  smoker  family_htn=yes
    . labs  glucose=128  HbA1c=6.4  cholesterol=245
    . lifestyle  exercise=none  diet=high_sodium""",
            "question": "What are the top 2 cardiovascular risk factors? What single intervention would have the greatest impact?",
            "reference_answer": "The top two cardiovascular risk factors are hypertension (BP 165/100) and smoking. The single intervention with greatest impact would be blood pressure control, as hypertension is the most modifiable and dangerous risk factor in this profile.",
            "ground_truth_root": "choi",
        },
        {
            "name": "Read -- patient identification",
            "data_json": json.dumps({
                "patients": [
                    {"id": "P-001", "name": "Kim",  "age": 45, "temp": 38.5, "dx": "influenza"},
                    {"id": "P-002", "name": "Lee",  "age": 62, "temp": 37.1, "dx": "hypertension"},
                    {"id": "P-003", "name": "Park", "age": 38, "temp": 39.2, "dx": "viral_infection"},
                ]
            }, indent=2),
            "data_wcy": """~ patient:id,name,age,temp,dx
    . patient  P-001  Kim   45  38.5  influenza
    . patient  P-002  Lee   62  37.1  hypertension
    . patient  P-003  Park  38  39.2  viral_infection""",
            "question": "Which patient has the highest temperature? What is their diagnosis?",
            "reference_answer": "Park (P-003) has the highest temperature at 39.2°C, with a diagnosis of viral infection.",
            "ground_truth_root": "park",
        },
    ]

    SYS_READ_JSON = "You are a medical data analyst. Answer questions about patient data provided as JSON. Be concise and direct."
    SYS_READ_WCY  = f"You are a medical data analyst. Answer questions about patient data in WCY format. Be concise.\n{WCY_SPEC}"

    results = []

    for model_name, model_id in [("Sonnet", SONNET), ("Haiku", HAIKU)]:
        print(f"\n  ── {model_name} ──")

        for tc in TEST_CASES:
            print(f"\n  Task: {tc['name']}")

            for fmt in ["json", "wcy"]:
                data = tc["data_json"] if fmt == "json" else tc["data_wcy"]
                sys  = SYS_READ_JSON  if fmt == "json" else SYS_READ_WCY
                user = f"Data:\n{data}\n\nQuestion: {tc['question']}"

                msg = client.messages.create(
                    model=model_id, max_tokens=300, system=sys,
                    messages=[{"role": "user", "content": user}]
                )
                resp = msg.content[0].text
                raw_in  = msg.usage.input_tokens
                raw_out = msg.usage.output_tokens
                sys_tok = count(sys)
                pay_in  = raw_in - sys_tok

                # ── three-axis scoring ───────────────────────────────────────
                ws = score_full(
                    response=resp,
                    question=tc["question"],
                    reference_answer=tc["reference_answer"],
                    model=model_name, task=tc["name"], fmt=fmt,
                    ground_truth_root=tc["ground_truth_root"],
                    skip_meaning=False,
                )

                # legacy keyword score (for comparison)
                kw_keys = [tc["ground_truth_root"], "hypertension", "bp", "blood pressure",
                           "smoking", "park", "viral", "39.2"]
                kw_hits = sum(1 for k in kw_keys if k.lower() in resp.lower())
                kw_score = min(kw_hits / 3, 1.0)  # 3+ hits = full score

                row = {
                    "model": model_name, "task": tc["name"], "fmt": fmt,
                    "pay_in": pay_in, "out": raw_out,
                    "S": ws.structural.overall, "M": ws.meaning,
                    "P": ws.provenance.overall, "composite": ws.composite,
                    "kw_legacy": kw_score, "response": resp,
                }
                results.append(row)

                # print row
                print(f"    [{fmt.upper():>4}] S={ws.structural.overall:.2f}  "
                      f"M={ws.meaning:.2f}  P={ws.provenance.overall:.2f}  "
                      f"→ {ws.composite:.2f}  (legacy_kw={kw_score:.2f})")
                print(f"           pay_in={pay_in}  out={raw_out}")
                print(f"           resp: {resp[:80].replace(chr(10),' ')}...")

                time.sleep(0.8)

    # ── Comparison summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY: New Score vs Legacy Keyword Score")
    print(f"  {'Model':<8} {'Fmt':<5} {'S(struct)':>10} {'M(meaning)':>11} {'P(prov)':>8} {'Composite':>10} {'Legacy KW':>10}")
    print(f"  {'─'*68}")

    for model_name in ["Sonnet", "Haiku"]:
        for fmt in ["json", "wcy"]:
            rows = [r for r in results if r["model"]==model_name and r["fmt"]==fmt]
            if not rows:
                continue
            avg_S   = sum(r["S"] for r in rows) / len(rows)
            avg_M   = sum(r["M"] for r in rows) / len(rows)
            avg_P   = sum(r["P"] for r in rows) / len(rows)
            avg_C   = sum(r["composite"] for r in rows) / len(rows)
            avg_kw  = sum(r["kw_legacy"] for r in rows) / len(rows)
            print(f"  {model_name:<8} {fmt:<5} {avg_S:>10.2f} {avg_M:>11.2f} {avg_P:>8.2f} {avg_C:>10.2f} {avg_kw:>10.2f}")

    print(f"\n  Key check: Sonnet WCY composite vs legacy kw -- does this reframe E1-C?")
    sonnet_wcy = [r for r in results if r["model"]=="Sonnet" and r["fmt"]=="wcy"]
    if sonnet_wcy:
        new_avg  = sum(r["composite"] for r in sonnet_wcy) / len(sonnet_wcy)
        old_avg  = sum(r["kw_legacy"]  for r in sonnet_wcy) / len(sonnet_wcy)
        print(f"  Sonnet WCY: new={new_avg:.2f}  legacy={old_avg:.2f}  diff={new_avg-old_avg:+.2f}")

    print(f"\n  Parser-based detail (WCY condition):")
    print(f"  {'Model':<8} {'Task':<35} {'phase_v':>8} {'from_v':>7} {'parse_r':>8}")
    for r in results:
        if r["fmt"] == "wcy":
            s = score_structural(r["response"])
            print(f"  {r['model']:<8} {r['task']:<35} {s.phase_valid:>8.2f} {s.from_valid:>7.2f} {s.parse_rate:>8.2f}")

