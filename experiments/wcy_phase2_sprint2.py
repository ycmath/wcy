# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  WCY Phase 2 — Sprint 2
  WCY-Unique Features Validation
═══════════════════════════════════════════════════════════════

Experiments:
  E3-A: Void-B (?) Utilization — WCY 고유 미탐 후보 처리 검증
  E2-C: Shared State Accumulation — 3에이전트 순차 침전 시뮬레이션
  E1-B: Reference Chain Depth — from= N-hop 추적 정확도

성공 기준:
  E3-A: 에이전트가 ? 마커를 인식하고 처리 ≥80%
  E2-C: 에이전트가 선행 침전을 올바르게 참조 ≥90%
  E1-B: from= 체인 추적 정확도, 깊이 5까지 유지

실행 환경: Colab Pro (Python 3.10+)
필요: ANTHROPIC_API_KEY (Colab 시크릿)
"""

###############################################################
# CELL 1: Setup
###############################################################

import subprocess, sys
for pkg in ["anthropic", "tiktoken"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import json, time, os, re
from textwrap import dedent
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

# ↓↓↓ 여기에 API 키를 직접 입력하세요 ↓↓↓
API_KEY = "sk-ant-api03-YOUR_KEY_HERE"
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

import anthropic
client = anthropic.Anthropic(api_key=API_KEY)
USE_API = True
print(f"✓ Setup: API=ON")

MODEL = "claude-sonnet-4-20250514"
count = lambda t: len(enc.encode(str(t)))

def call_api(system, user, max_tokens=1500, label=""):
    """Single API call with token tracking."""
    if not USE_API:
        print(f"  [API OFF] Would call: {label}")
        return None, 0, 0
    msg = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}]
    )
    in_tok  = msg.usage.input_tokens
    out_tok = msg.usage.output_tokens
    text    = msg.content[0].text
    return text, in_tok, out_tok


# ── WCY Spec (시스템 프롬프트용, ~140 tokens) ─────────────────
WCY_SPEC = dedent("""
WCY FORMAT SPEC (brief):
PHASE MARKERS (line-start):
  .  observe  — confirmed data/fact
  :  infer    — derived conclusion (may include conf=, conf_range=)
  >  act      — output, call, side effect
  ~  meta     — schema / context declaration
  !  exception — error, missing data

SLOTS (space-separated within line):
  tag=value    named slot
  bare_value   positional slot
  ?tag         void-B: unexplored candidate requiring investigation
               (hint= points to relevant evidence; conf_range= expected range)
  from=N,M     evidence trail (this line derives from lines N and M)
  conf=0.xx    confidence (0.0–1.0)

RULES:
  - One phase marker per line, followed by space
  - Line numbers matter: from=N references line N (1-indexed, counting non-empty lines)
  - Blank line = semantic block boundary
  - Max nesting depth: 2 (2-space indent)
""").strip()

print(f"WCY spec tokens: {count(WCY_SPEC)}")


###############################################################
# CELL 2: E3-A — Void-B (?) Utilization
#
# 목표: ?tag 마커가 있는 WCY 문서를 주고 "미해결 항목을 처리하라"
#      에이전트가 ? 마커를 인식 → hint= 추적 → 처리 여부 측정
#      비교: 동일 정보를 JSON으로 (명시적 gap 표시 없음)
###############################################################

print("\n" + "=" * 70)
print("  E3-A: VOID-B (?) UTILIZATION")
print("=" * 70)

# ── 3가지 케이스 정의 ──────────────────────────────────────────

VOID_CASES = [
    {
        "name": "Case A — Simple (1 void)",
        "wcy": dedent("""
            ~ context  case=E-2026-001
            . patient=Kim  age=45  presented=clinic
            . symptoms  fatigue  weight_loss  night_sweats  duration=3weeks
            . vitals  temp=37.8  hr=95  bp=128/82
            : ?primary_diagnosis  hint=fatigue,weight_loss,night_sweats  conf_range=0.3..0.8
        """).strip(),
        "json": json.dumps({
            "case": "E-2026-001",
            "patient": {"name": "Kim", "age": 45, "presented": "clinic"},
            "symptoms": ["fatigue", "weight_loss", "night_sweats"],
            "duration": "3 weeks",
            "vitals": {"temp": 37.8, "hr": 95, "bp": "128/82"},
            "pending": ["primary_diagnosis"]
        }, indent=2),
        "expected_voids": 1
    },
    {
        "name": "Case B — Medium (3 voids)",
        "wcy": dedent("""
            ~ context  case=E-2026-002
            . patient=Lee  age=62  presented=ER
            . chief_complaint  sudden_onset_weakness  left_side  duration=2h
            . history  hypertension  diabetes  smoker=yes
            . vitals  temp=36.9  hr=88  bp=178/96
            . meds  lisinopril  metformin
            : ?stroke_type  hint=sudden_onset,left_weakness,bp  conf_range=0.4..0.8
            : ?imaging_priority  hint=symptom_duration,stroke_type  conf_range=0.5..0.9
            : ?contraindications  hint=meds,diabetes,stroke_type  conf_range=0.2..0.6
        """).strip(),
        "json": json.dumps({
            "case": "E-2026-002",
            "patient": {"name": "Lee", "age": 62, "presented": "ER"},
            "chief_complaint": "sudden onset left-sided weakness, 2h duration",
            "history": ["hypertension", "diabetes", "smoker"],
            "vitals": {"temp": 36.9, "hr": 88, "bp": "178/96"},
            "medications": ["lisinopril", "metformin"],
            "unresolved": [
                {"item": "stroke_type", "relevant": ["sudden_onset", "left_weakness", "bp"]},
                {"item": "imaging_priority", "relevant": ["symptom_duration", "stroke_type"]},
                {"item": "contraindications", "relevant": ["medications", "diabetes", "stroke_type"]}
            ]
        }, indent=2),
        "expected_voids": 3
    },
    {
        "name": "Case C — Complex (5 voids, chained)",
        "wcy": dedent("""
            ~ context  case=E-2026-003
            . patient=Park  age=38  sex=F  presented=OB
            . complaint  severe_headache  visual_changes  edema  ga=34weeks
            . vitals  bp=165/108  hr=94  proteinuria=+3
            . labs  plt=88k  ast=180  alt=145  ldh=elevated
            : ?diagnosis  hint=bp,proteinuria,ga  conf_range=0.5..0.9
            : ?severity  hint=labs,bp,symptoms  conf_range=0.4..0.8
            : ?delivery_timing  hint=diagnosis,ga,severity  conf_range=0.3..0.7
            : ?magnesium_protocol  hint=diagnosis,severity  conf_range=0.5..0.9
            : ?fetal_lung_maturity  hint=delivery_timing,ga  conf_range=0.4..0.8
        """).strip(),
        "json": json.dumps({
            "case": "E-2026-003",
            "patient": {"name": "Park", "age": 38, "sex": "F", "presented": "OB"},
            "complaint": ["severe headache", "visual changes", "edema"],
            "ga": "34 weeks",
            "vitals": {"bp": "165/108", "hr": 94, "proteinuria": "+3"},
            "labs": {"platelets": "88k", "AST": 180, "ALT": 145, "LDH": "elevated"},
            "unresolved": [
                "diagnosis",
                "severity_assessment",
                "delivery_timing",
                "magnesium_protocol",
                "fetal_lung_maturity"
            ]
        }, indent=2),
        "expected_voids": 5
    }
]

# ── 평가 함수 ─────────────────────────────────────────────────

def count_void_markers(wcy_text):
    """WCY 입력에서 ?tag 수 세기."""
    return sum(1 for line in wcy_text.split('\n')
               if line.strip().startswith(': ?') or line.strip().startswith(': ?'))

def score_void_resolution(response, void_tags):
    """
    LLM 응답에서 void 처리 여부 평가.
    void_tags: ['primary_diagnosis', 'stroke_type', ...]
    반환: (resolved_count, total, details)
    """
    resp_lower = response.lower()
    resolved = []
    for tag in void_tags:
        tag_clean = tag.replace('_', ' ')
        # 1) WCY >나 : 줄에 태그가 등장하는지
        in_wcy = bool(re.search(rf'[>:] .*{re.escape(tag)}', response))
        # 2) 자연어로 다룬 경우
        in_nl  = tag_clean in resp_lower or tag.replace('_','') in resp_lower
        resolved.append({
            "tag": tag,
            "addressed": in_wcy or in_nl,
            "in_wcy_line": in_wcy,
            "in_natural_language": in_nl
        })
    n_resolved = sum(1 for r in resolved if r["addressed"])
    return n_resolved, len(void_tags), resolved

VOID_SYSTEM_WCY = f"""{WCY_SPEC}

You are a clinical AI assistant. Review the WCY case document and address ALL unresolved items marked with ? (void-B markers). For each ?, provide your assessment using WCY syntax. Follow the hint= guidance and respect the conf_range= bounds."""

VOID_SYSTEM_JSON = """You are a clinical AI assistant. Review the clinical case JSON and address ALL unresolved/pending items listed. Provide your assessment for each item."""

print("\n[Token counts — no API needed]")
print(f"{'Case':<30} {'WCY input':>10} {'JSON input':>11} {'Diff':>7}")
print("─" * 62)

e3a_token_results = []
for case in VOID_CASES:
    t_wcy  = count(VOID_SYSTEM_WCY + case["wcy"])
    t_json = count(VOID_SYSTEM_JSON + case["json"])
    diff   = t_wcy - t_json
    print(f"  {case['name']:<28} {t_wcy:>10} {t_json:>11} {diff:>+7}")
    e3a_token_results.append({"case": case["name"], "wcy": t_wcy, "json": t_json})

# ── API 실험 ──────────────────────────────────────────────────
if USE_API:
    print("\n[LLM Experiment — WCY vs JSON void resolution]")

    e3a_results = []
    for case in VOID_CASES:
        print(f"\n  ── {case['name']} ──")

        # void 태그 추출
        void_tags = re.findall(r':\s+\?(\w+)', case["wcy"])

        # WCY 조건
        wcy_resp, wcy_in, wcy_out = call_api(
            VOID_SYSTEM_WCY, case["wcy"], max_tokens=1200,
            label=f"{case['name']} WCY"
        )
        if wcy_resp:
            n_res_wcy, total, wcy_details = score_void_resolution(wcy_resp, void_tags)
            print(f"    WCY: {n_res_wcy}/{total} voids addressed  |  in={wcy_in} out={wcy_out}")
            for d in wcy_details:
                icon = "✓" if d["addressed"] else "✗"
                mode = "WCY-line" if d["in_wcy_line"] else ("NL" if d["in_natural_language"] else "MISS")
                print(f"      {icon} ?{d['tag']:<25} [{mode}]")

        # JSON 조건
        json_resp, json_in, json_out = call_api(
            VOID_SYSTEM_JSON, case["json"], max_tokens=1200,
            label=f"{case['name']} JSON"
        )
        if json_resp:
            n_res_json, total, json_details = score_void_resolution(json_resp, void_tags)
            print(f"    JSON: {n_res_json}/{total} items addressed  |  in={json_in} out={json_out}")

        if wcy_resp and json_resp:
            e3a_results.append({
                "case": case["name"],
                "n_voids": total,
                "wcy_resolved": n_res_wcy,
                "json_resolved": n_res_json,
                "wcy_in_tok": wcy_in, "wcy_out_tok": wcy_out,
                "json_in_tok": json_in, "json_out_tok": json_out,
            })
        time.sleep(1)

    # 요약
    if e3a_results:
        print(f"\n{'='*70}")
        print(f"  E3-A SUMMARY")
        print(f"  {'Case':<30} {'Voids':>6} {'WCY%':>7} {'JSON%':>7} {'Δ':>6}")
        print(f"  {'─'*55}")
        total_wcy = total_json = total_voids = 0
        for r in e3a_results:
            wcy_pct  = r["wcy_resolved"]  / r["n_voids"] * 100
            json_pct = r["json_resolved"] / r["n_voids"] * 100
            delta    = wcy_pct - json_pct
            total_wcy   += r["wcy_resolved"]
            total_json  += r["json_resolved"]
            total_voids += r["n_voids"]
            print(f"  {r['case']:<30} {r['n_voids']:>6} {wcy_pct:>6.0f}% {json_pct:>6.0f}% {delta:>+5.0f}%")
        avg_wcy  = total_wcy  / total_voids * 100
        avg_json = total_json / total_voids * 100
        print(f"  {'─'*55}")
        print(f"  {'OVERALL':<30} {total_voids:>6} {avg_wcy:>6.0f}% {avg_json:>6.0f}% {avg_wcy-avg_json:>+5.0f}%")
        print(f"\n  성공 기준: WCY ≥80% → {'✓ PASS' if avg_wcy >= 80 else '✗ FAIL'}")


###############################################################
# CELL 3: E2-C — Shared State Accumulation
#
# 3 에이전트(Intake → Diagnostician → Treatment)가
# 같은 환자 케이스에 순차적으로 침전을 쌓음.
# 각 라운드는 이전 라운드의 출력을 입력으로 받음.
# 측정: 누적 컨텍스트 크기, 선행 침전 참조 정확도
###############################################################

print("\n\n" + "=" * 70)
print("  E2-C: SHARED STATE ACCUMULATION")
print("=" * 70)

# ── 초기 케이스 정의 ─────────────────────────────────────────

CASE_BASE = {
    "id": "C-2026-007",
    "patient": "Choi",
    "age": 58,
    "sex": "M",
    "presented": "ER",
    "chief_complaint": "chest_pain",
    "onset_hours": 3,
    "vitals": {"temp": 37.1, "hr": 112, "bp": "92/64"},
    "history": ["hypertension", "hyperlipidemia", "ex_smoker"],
    "meds": [{"name": "amlodipine", "dose": 5}, {"name": "rosuvastatin", "dose": 20}]
}

# ── WCY 초기 입력 ────────────────────────────────────────────

ROUND1_WCY_INPUT = dedent(f"""
~ context  case={CASE_BASE['id']}
. patient={CASE_BASE['patient']}  age={CASE_BASE['age']}  sex={CASE_BASE['sex']}  presented={CASE_BASE['presented']}
. chief_complaint  {CASE_BASE['chief_complaint']}  onset={CASE_BASE['onset_hours']}h
. vitals  temp={CASE_BASE['vitals']['temp']}  hr={CASE_BASE['vitals']['hr']}  bp={CASE_BASE['vitals']['bp']}
. history  {' '.join(CASE_BASE['history'])}
. meds  {' '.join(f"{m['name']}_{m['dose']}mg" for m in CASE_BASE['meds'])}
""").strip()

ROUND1_JSON_INPUT = json.dumps({
    "case": CASE_BASE["id"],
    "patient": {"name": CASE_BASE["patient"], "age": CASE_BASE["age"],
                "sex": CASE_BASE["sex"], "presented": CASE_BASE["presented"]},
    "chief_complaint": CASE_BASE["chief_complaint"],
    "onset_hours": CASE_BASE["onset_hours"],
    "vitals": CASE_BASE["vitals"],
    "history": CASE_BASE["history"],
    "medications": CASE_BASE["meds"]
}, indent=2)

# ── 에이전트 시스템 프롬프트 ──────────────────────────────────

INTAKE_SYS_WCY = f"""{WCY_SPEC}

You are the Intake agent. Review the initial patient data. Output:
1. Confirm received data (. lines)
2. Identify immediate concerns (: lines with conf=)
3. Order priority workup (> lines)
4. Flag any unknowns with ?tag void markers
Keep output concise. Use WCY only."""

INTAKE_SYS_JSON = """You are the Intake agent for an ER patient. Review the patient data.
Provide your assessment as JSON with keys: confirmed_data, immediate_concerns (with confidence), priority_workup, unknowns."""

DIAG_SYS_WCY = f"""{WCY_SPEC}

You are the Diagnostician agent. You receive the full case context (Intake output + initial data).
Add your diagnostic reasoning using WCY. Reference prior observations using from=N (line numbers from the full context you receive).
Output:
1. Lab/imaging results integration (. lines)
2. Differential diagnoses (: lines with conf=, from=)
3. Further orders (> lines)
4. Remaining unknowns (?tag)"""

DIAG_SYS_JSON = """You are the Diagnostician agent. You receive the full case context including Intake assessment.
Add your diagnostic reasoning as JSON with keys: findings, differentials (with confidence and evidence references), further_orders, unknowns."""

TREATMENT_SYS_WCY = f"""{WCY_SPEC}

You are the Treatment agent. You receive the complete case context (Intake + Diagnostician outputs).
Synthesize all prior evidence and provide treatment plan using WCY.
You MUST use from=N to reference the specific prior lines that support each treatment decision.
Output:
1. Final diagnosis with confidence (: diagnosis= conf= from=)
2. Treatment actions (> lines with rationale from=)
3. Monitoring plan (> monitor lines)"""

TREATMENT_SYS_JSON = """You are the Treatment agent. You receive the complete case context.
Synthesize all prior evidence and provide:
1. Final diagnosis with confidence and evidence references
2. Treatment plan with rationale
3. Monitoring plan
Format as JSON."""

def check_from_references(response_text, context_lines):
    """from=N 참조가 유효한 라인 번호를 가리키는지 확인."""
    refs = re.findall(r'from=([\d,]+)', response_text)
    valid = invalid = 0
    details = []
    for ref_str in refs:
        for n_str in ref_str.split(','):
            n_str = n_str.strip()
            if n_str.isdigit():
                n = int(n_str)
                if 1 <= n <= context_lines:
                    valid += 1
                    details.append((n, "valid"))
                else:
                    invalid += 1
                    details.append((n, f"invalid (max={context_lines})"))
    return valid, invalid, details

def count_non_empty_lines(text):
    return sum(1 for l in text.split('\n') if l.strip())

print(f"\n[Token counts — initial input]")
print(f"  WCY Round 1 input:  {count(ROUND1_WCY_INPUT):>6} tokens")
print(f"  JSON Round 1 input: {count(ROUND1_JSON_INPUT):>6} tokens")

# ── API 실험 ─────────────────────────────────────────────────
if USE_API:
    print("\n[3-Agent Pipeline Simulation]")

    def run_pipeline(mode):
        """mode: 'wcy' or 'json'"""
        print(f"\n  ── {mode.upper()} Pipeline ──")

        if mode == 'wcy':
            r1_sys = INTAKE_SYS_WCY
            r2_sys = DIAG_SYS_WCY
            r3_sys = TREATMENT_SYS_WCY
            r1_input = ROUND1_WCY_INPUT
        else:
            r1_sys = INTAKE_SYS_JSON
            r2_sys = DIAG_SYS_JSON
            r3_sys = TREATMENT_SYS_JSON
            r1_input = ROUND1_JSON_INPUT

        history = {"rounds": [], "cumulative_context": ""}

        # Round 1: Intake
        r1_resp, r1_in, r1_out = call_api(r1_sys, r1_input, max_tokens=600, label="Intake")
        history["rounds"].append({"agent": "Intake", "in": r1_in, "out": r1_out, "resp": r1_resp})
        print(f"    Round 1 Intake:       in={r1_in:>5}  out={r1_out:>4}")

        # Round 2: Diagnostician (receives initial + Round 1)
        r2_context = f"{r1_input}\n\n--- Intake Agent Output ---\n{r1_resp}"
        # Simulate: ECG and troponin results added (realistic scenario)
        lab_addition = (
            "\n\n--- Lab Results ---\n. ECG  ST_depression  leads=V4-V6  hr=112\n. troponin=0.18  ref=<0.04  critical=true\n. BNP=340  ref=<100"
            if mode == 'wcy' else
            '\n\n--- Lab Results ---\n' + json.dumps({"ECG": "ST depression V4-V6", "troponin": 0.18, "troponin_ref": "<0.04", "BNP": 340}, indent=2)
        )
        r2_context += lab_addition
        r2_resp, r2_in, r2_out = call_api(r2_sys, r2_context, max_tokens=700, label="Diagnostician")
        history["rounds"].append({"agent": "Diagnostician", "in": r2_in, "out": r2_out, "resp": r2_resp})
        print(f"    Round 2 Diagnostician: in={r2_in:>5}  out={r2_out:>4}")

        # Round 3: Treatment (receives everything)
        r3_context = f"{r2_context}\n\n--- Diagnostician Agent Output ---\n{r2_resp}"
        r3_resp, r3_in, r3_out = call_api(r3_sys, r3_context, max_tokens=800, label="Treatment")
        history["rounds"].append({"agent": "Treatment", "in": r3_in, "out": r3_out, "resp": r3_resp})
        print(f"    Round 3 Treatment:     in={r3_in:>5}  out={r3_out:>4}")

        # from= 검증 (WCY만)
        if mode == 'wcy' and r3_resp:
            ctx_lines = count_non_empty_lines(r3_context)
            v_ok, v_bad, v_det = check_from_references(r3_resp, ctx_lines)
            print(f"    from= references: {v_ok} valid, {v_bad} invalid (context={ctx_lines} lines)")
            history["from_valid"] = v_ok
            history["from_invalid"] = v_bad

        total_in  = sum(r["in"]  for r in history["rounds"])
        total_out = sum(r["out"] for r in history["rounds"])
        print(f"    TOTAL: in={total_in}  out={total_out}  grand={total_in+total_out}")

        history["total_in"] = total_in
        history["total_out"] = total_out
        return history

    wcy_pipeline  = run_pipeline('wcy')
    time.sleep(2)
    json_pipeline = run_pipeline('json')

    # 비교 요약
    print(f"\n{'='*70}")
    print(f"  E2-C SUMMARY: Token Accumulation")
    print(f"  {'Metric':<30} {'JSON':>8} {'WCY':>8} {'Savings':>9}")
    print(f"  {'─'*58}")

    for i, r_label in enumerate(["Round 1 (Intake)", "Round 2 (Diagnos)", "Round 3 (Treat)"]):
        j_in  = json_pipeline["rounds"][i]["in"]
        w_in  = wcy_pipeline["rounds"][i]["in"]
        sav   = (j_in - w_in) / j_in * 100 if j_in else 0
        print(f"  {r_label+' input':<30} {j_in:>8} {w_in:>8} {sav:>+8.1f}%")

    j_tot = json_pipeline["total_in"] + json_pipeline["total_out"]
    w_tot = wcy_pipeline["total_in"]  + wcy_pipeline["total_out"]
    sav_tot = (j_tot - w_tot) / j_tot * 100
    print(f"  {'─'*58}")
    print(f"  {'GRAND TOTAL':<30} {j_tot:>8} {w_tot:>8} {sav_tot:>+8.1f}%")

    if "from_valid" in wcy_pipeline:
        total_refs = wcy_pipeline["from_valid"] + wcy_pipeline["from_invalid"]
        pct = wcy_pipeline["from_valid"] / max(total_refs, 1) * 100
        print(f"\n  from= validity: {wcy_pipeline['from_valid']}/{total_refs} ({pct:.0f}%)")
        print(f"  성공 기준: ≥90% 참조 정확도 → {'✓ PASS' if pct >= 90 else '△ CHECK'}")


###############################################################
# CELL 4: E1-B — Reference Chain Depth (from= N-hop)
#
# from= 체인을 1-hop에서 12-hop까지 쌓아가며
# LLM이 "최초 근거 사실이 무엇인가"를 올바르게 역추적하는지 측정.
# 비교: JSON 중첩 참조 vs WCY flat from= 체인
###############################################################

print("\n\n" + "=" * 70)
print("  E1-B: REFERENCE CHAIN DEPTH (from= N-hop)")
print("=" * 70)

def make_wcy_chain(depth, domain="medical"):
    """
    depth-hop WCY 추론 체인 생성.
    각 줄은 이전 줄에서 from= 으로 파생됨.
    반환: (wcy_text, answer_fact, line_count)
    """
    if domain == "medical":
        seed_fact = "patient=Yoon  age=71  presented=ER  chief_complaint=syncope"
        chain_steps = [
            ("bp=88/56  hr=42  alert=reduced",          "vitals_critical",    "hypotension_bradycardia"),
            ("ECG=complete_AV_block  p_rate=80  v_rate=42", "ECG_finding",   "complete_heart_block"),
            ("cardiac_output=reduced  from=prev",         "hemodynamics",    "cardiogenic_shock_risk"),
            ("temporary_pacing_indicated",                "intervention",    "pacemaker_needed"),
            ("pacing_threshold=0.7mA  rate_set=70",       "pacing_param",    "effective_pacing"),
            ("bp_improved=112/74  hr=70  alert=improving","response",        "hemodynamic_stabilization"),
            ("permanent_pacemaker_indicated",             "long_term",       "permanent_pacing"),
            ("DDD_mode  AV_delay=160ms",                  "pacemaker_spec",  "optimal_programming"),
            ("follow_up=6weeks  remote_monitoring=yes",   "care_plan",       "outpatient_followup"),
            ("prognosis=good  ejection_fraction=55",      "outcome",         "favorable_prognosis"),
            ("ICD_not_indicated  LVEF_above_35",          "ICD_decision",    "no_ICD_needed"),
            ("final_dx=sick_sinus_syndrome  conf=0.88",   "final_diag",      "sick_sinus_syndrome"),
        ]
    else:
        seed_fact = "project=Alpha  started=2024-01  budget=500k"
        chain_steps = [
            ("team_size=8  lead=Kim",                     "team_info",       "8_person_team"),
            ("phase1_complete  deliverable=prototype",    "milestone1",      "prototype_done"),
            ("user_testing=positive  nps=72",             "feedback",        "high_satisfaction"),
            ("phase2_approved  budget_increase=200k",     "phase2_start",    "expanded_budget"),
            ("beta_launch=2024-09  users=1200",           "launch",          "beta_launched"),
            ("revenue_month1=45k  growth=28pct",          "revenue",         "strong_growth"),
            ("series_A_raised=2M",                        "funding",         "series_A"),
            ("team_expanded=25  new_hires=17",            "scaling",         "team_growth"),
            ("product_v2_shipped  features=12",           "v2_release",      "v2_done"),
            ("enterprise_contract=signed  arr=800k",      "enterprise",      "enterprise_deal"),
            ("series_B_target=10M",                       "series_B",        "series_B_planned"),
            ("final_valuation=45M  conf=0.75",            "valuation",       "45M_valuation"),
        ]

    lines = [f". {seed_fact}"]  # line 1
    for i in range(min(depth, len(chain_steps))):
        data, tag, value = chain_steps[i]
        prev_line = i + 1  # 1-indexed, previous line
        lines.append(f": {tag}={value}  from={prev_line}")
    wcy_text = "\n".join(lines)

    # 정답: seed fact의 핵심 값
    if domain == "medical":
        answer = "Yoon"  # patient name from seed
    else:
        answer = "Alpha"  # project name from seed
    return wcy_text, answer, len(lines)

def make_json_chain(depth, domain="medical"):
    """동등한 JSON 중첩 참조 구조."""
    if domain == "medical":
        base = {
            "observation": {
                "patient": "Yoon", "age": 71, "presented": "ER", "complaint": "syncope"
            }
        }
        steps = [
            ("vitals", {"bp": "88/56", "hr": 42, "alert": "reduced"}),
            ("ECG", {"finding": "complete_AV_block", "p_rate": 80, "v_rate": 42}),
            ("hemodynamics", {"status": "cardiogenic_shock_risk", "based_on": "vitals"}),
            ("intervention", {"required": "temporary_pacing", "based_on": "hemodynamics"}),
            ("pacing", {"threshold": "0.7mA", "rate": 70, "based_on": "intervention"}),
            ("response", {"bp": "112/74", "hr": 70, "based_on": "pacing"}),
            ("long_term", {"plan": "permanent_pacemaker", "based_on": "response"}),
            ("pacemaker_spec", {"mode": "DDD", "av_delay": 160, "based_on": "long_term"}),
            ("care_plan", {"follow_up": "6 weeks", "monitoring": "remote", "based_on": "pacemaker_spec"}),
            ("outcome", {"prognosis": "good", "ef": "55%", "based_on": "care_plan"}),
            ("ICD_decision", {"ICD": "not indicated", "reason": "LVEF>35%", "based_on": "outcome"}),
            ("final", {"dx": "sick_sinus_syndrome", "conf": 0.88, "based_on": "ICD_decision"}),
        ]
        answer = "Yoon"
    else:
        answer = "Alpha"
        steps = []  # simplified for non-medical

    obj = base.copy()
    current = obj
    for i in range(min(depth, len(steps))):
        key, val = steps[i]
        obj[key] = val
    return json.dumps(obj, indent=2), answer

# ── 토큰 카운트 (API 불필요) ───────────────────────────────────
depths = [1, 2, 3, 5, 8, 12]
print(f"\n[Token counts by chain depth — no API needed]")
print(f"  {'Depth':>6} {'WCY tok':>9} {'JSON tok':>10} {'WCY/JSON':>10}")
print(f"  {'─'*40}")

chain_token_results = []
for d in depths:
    wcy_text, _, _ = make_wcy_chain(d)
    json_text, _   = make_json_chain(d)
    t_wcy  = count(wcy_text)
    t_json = count(json_text)
    ratio  = t_wcy / t_json * 100
    print(f"  {d:>6} {t_wcy:>9} {t_json:>10} {ratio:>9.1f}%")
    chain_token_results.append({"depth": d, "wcy": t_wcy, "json": t_json})

# ── API 실험 ─────────────────────────────────────────────────
if USE_API:
    print("\n[LLM Tracking Accuracy by Depth]")

    CHAIN_SYS_WCY = f"""{WCY_SPEC}

You are given a WCY inference chain where each line builds on previous lines via from= references.
Answer the question about the original source fact. Be brief — give only the answer value."""

    CHAIN_SYS_JSON = """You are given a JSON object representing an inference chain where each step references prior steps.
Answer the question about the original source observation. Be brief — give only the answer value."""

    QUESTION_MEDICAL = "What is the patient's name in the original observation?"
    ANSWER_MEDICAL   = "yoon"

    e1b_results = []
    for d in depths:
        wcy_text, _, chain_len = make_wcy_chain(d)
        json_text, _           = make_json_chain(d)

        q_wcy  = f"{wcy_text}\n\nQuestion: {QUESTION_MEDICAL}"
        q_json = f"{json_text}\n\nQuestion: {QUESTION_MEDICAL}"

        wcy_resp, wcy_in, wcy_out = call_api(CHAIN_SYS_WCY, q_wcy, max_tokens=50, label=f"chain-d{d}-WCY")
        json_resp, json_in, json_out = call_api(CHAIN_SYS_JSON, q_json, max_tokens=50, label=f"chain-d{d}-JSON")

        wcy_correct  = ANSWER_MEDICAL in wcy_resp.lower()  if wcy_resp  else False
        json_correct = ANSWER_MEDICAL in json_resp.lower() if json_resp else False

        print(f"  Depth {d:>2}: WCY={'✓' if wcy_correct else '✗'} ({wcy_in+wcy_out:>4}tok)  "
              f"JSON={'✓' if json_correct else '✗'} ({json_in+json_out:>4}tok)"
              f"  WCY-ans='{(wcy_resp or '').strip()[:30]}'")

        e1b_results.append({
            "depth": d,
            "wcy_correct": wcy_correct, "json_correct": json_correct,
            "wcy_tok": wcy_in+wcy_out, "json_tok": json_in+json_out
        })
        time.sleep(0.5)

    # 요약
    print(f"\n{'='*70}")
    print(f"  E1-B SUMMARY")
    print(f"  {'Depth':>6} {'WCY':>6} {'JSON':>6}")
    print(f"  {'─'*22}")
    for r in e1b_results:
        print(f"  {r['depth']:>6} {'✓' if r['wcy_correct'] else '✗':>6} {'✓' if r['json_correct'] else '✗':>6}")

    # 최대 정확 깊이 찾기
    wcy_max_depth  = max((r["depth"] for r in e1b_results if r["wcy_correct"]),  default=0)
    json_max_depth = max((r["depth"] for r in e1b_results if r["json_correct"]), default=0)
    print(f"\n  WCY 최대 정확 깊이: {wcy_max_depth}")
    print(f"  JSON 최대 정확 깊이: {json_max_depth}")
    print(f"  성공 기준: WCY ≥ 깊이 5 → {'✓ PASS' if wcy_max_depth >= 5 else '✗ FAIL'}")


###############################################################
# CELL 5: Final Summary
###############################################################

print("\n\n" + "═" * 70)
print("  SPRINT 2 — FINAL SUMMARY")
print("═" * 70)

print("""
실험 구성:
  E3-A: Void-B (?) 처리 — 3케이스, 1/3/5 void markers, WCY vs JSON
  E2-C: 공유 상태 축적 — 3에이전트 파이프라인, 침전 누적, from= 검증
  E1-B: 참조 체인 깊이 — 1/2/3/5/8/12 hop, 역추적 정확도

가설 (Sprint 2 성공 기준):
  E3-A: LLM이 ?마커를 ≥80% 처리 → void-B는 WCY 고유 실용 기능
  E2-C: 3라운드 from= 참조 ≥90% 유효 → 침전 체인 신뢰성
  E1-B: WCY가 JSON보다 깊은 체인에서 더 정확 → flat from=의 우위

다음 단계:
  Sprint 2 결과 확보 후:
  → 모든 성공 기준 충족 시: Position Paper v0.1 → v1.0 업그레이드
  → void-B 검증되면: ASMT 통합 경로 (경로 B) 개시 가능
  → Sprint 3: E2-D (검증 오버헤드) + E1-C (모델 래더)
""")
