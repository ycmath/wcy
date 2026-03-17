# -*- coding: utf-8 -*-
"""
wcy_phase5_pipeline.py — WCY Trace Generation Pipeline v1.0
=============================================================

Phase 5: large-scale training data generation

Goal: 12 seed traces -> 500+ usable traces
Method: systematic generation over domain x difficulty x void_depth matrix

Quality gates (3 levels):
  Gate 1 -- Structure:   parse_r >= 0.70  (parser can read the output)
  Gate 2 -- Void:        void_generated >= 1  (at least one boundary marked)
  Gate 3 -- Resolution:  resolution_rate >= 0.50  (exploration completed)

Outputs:
  wcy_traces_v1.jsonl        <- all generated traces
  wcy_traces_v1_clean.jsonl  <- gate-passing traces only
  wcy_pipeline_report.txt    <- generation statistics

Execution: Colab Pro, requires wcy_parser.py
Note: many API calls (50 tasks x ~2 calls = ~100 calls)
      estimated cost ~$0.50-1.00
"""

import subprocess, sys
for pkg in ["anthropic", "tiktoken"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import json, time, random, re, os
from dataclasses import dataclass, field
from textwrap import dedent
import tiktoken

try:
    from wcy_parser import parse_wcy, flatten, extract_voids
    print("✓ wcy_parser v1.1 imported")
except ImportError:
    print("⚠ Upload wcy_parser.py to the same directory first")

# ↓↓↓ Insert your API key here ↓↓↓
API_KEY = "sk-ant-api03-YOUR_KEY_HERE"
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

import anthropic
client = anthropic.Anthropic(api_key=API_KEY)
SONNET = "claude-sonnet-4-20250514"

enc = tiktoken.get_encoding("cl100k_base")
count = lambda t: len(enc.encode(str(t)))
print("✓ Setup complete\n")


###############################################################
# Matrix definition
# domain(8) x difficulty(3) x void_depth(2) = 48 combinations
# RUNS_PER_COMBO traces per combination
# Set RUNS_PER_COMBO=10 for ~480 additional traces per run
###############################################################

DOMAINS = {
    "medical":       "medical diagnosis and treatment decisions",
    "code":          "software debugging and system design",
    "scientific":    "scientific hypothesis testing",
    "legal":         "legal reasoning and judgment",
    "mathematical":  "mathematical proofs and computation",
    "strategic":     "strategic decision-making",
    "philosophical": "philosophical and epistemological analysis",
    "engineering":   "system design and trade-offs",
}

DIFFICULTIES = {
    "simple":  {"void_target": 2, "max_tokens": 400, "description": "single causal chain, clear resolution"},
    "medium":  {"void_target": 4, "max_tokens": 700, "description": "multiple hypotheses, partial resolution"},
    "complex": {"void_target": 6, "max_tokens": 1200, "description": "cascade with unresolvable !warning"},
}

VOID_DEPTHS = {
    "single":  "each ?tag resolved independently",
    "cascade": "resolution of one ?tag generates new ?tags",
}


###############################################################
# Task template generator
# Dynamically generates tasks per domain
###############################################################

TASK_SEEDS = {
    "medical": [
        ("Patient aged {age} ({sex}), {symptom} for {duration} days, {lab_finding} found. Diagnosis and treatment plan.",
         {"age": [28,45,62,71,38,55], "sex": ["male","female"],
          "symptom": ["acute chest pain","high fever and chills","dyspnea","abdominal pain","headache"],
          "duration": [1,2,3,7,14], "lab_finding": ["troponin elevated","WBC 18000","CRP 45","D-dimer positive"]}),
        ("Patient co-administering {drug_a} and {drug_b}. Evaluate interaction and alternatives.",
         {"drug_a": ["warfarin","metformin","lisinopril","atorvastatin"],
          "drug_b": ["ciprofloxacin","ibuprofen","amiodarone","fluconazole"]}),
    ],
    "code": [
        ("{service} service showing {symptom}, started after {change}. Diagnose cause and fix.",
         {"service": ["API server","batch pipeline","ML inference server","database"],
          "symptom": ["memory leak","response latency spike","error rate increase","CPU overload"],
          "change": ["deployment v3.1","infra upgrade","2x traffic increase","library update"]}),
        ("Optimise {algorithm} from {current} to {target}. Analyse trade-offs.",
         {"algorithm": ["graph traversal","string matching","sorting","cache design"],
          "current": ["O(n²)","O(n³)","O(2ⁿ)"],
          "target": ["O(n log n)","O(n)","O(1) amortized"]}),
    ],
    "scientific": [
        ("Unexpected {observation} in experiment {experiment}. Hypothesise cause and propose verification.",
         {"experiment": ["protein crystallisation","cell culture","chemical reaction","physical measurement"],
          "observation": ["yield drop","unexpected byproduct","rate change","measurement anomaly"]}),
        ("Correlation r={r} found between {phenomenon} and {variable}. Evaluate causal inference.",
         {"phenomenon": ["income level","exercise frequency","sleep duration","stress index"],
          "variable": ["longevity","cognitive function","cardiovascular health","immune markers"],
          "r": ["0.72","0.85","0.61","0.43"]}),
    ],
    "legal": [
        ("Dispute over interpretation of {clause} clause. Analyse positions of {party_a} vs {party_b}.",
         {"clause": ["liability cap","force majeure","IP ownership","confidentiality"],
          "party_a": ["client (Party A)","plaintiff","buyer"],
          "party_b": ["vendor (Party B)","defendant","seller"]}),
        ("Determine whether {action} violates {law}. Analyse element satisfaction.",
         {"action": ["sharing information with a third party","contract non-performance","hiring competitor employees","price-fixing"],
          "law": ["Personal Data Protection Act","Fair Trade Act","Labour Law","Civil Code breach of obligation"]}),
    ],
    "mathematical": [
        ("Proposition: {claim}. Prove or provide a counterexample.",
         {"claim": ["there are infinitely many primes","a continuous function is integrable",
                    "if P≠NP then RSA is secure","every rational number is a real number"]}),
        ("{problem_type} problem: {setup}. Step-by-step solution.",
         {"problem_type": ["probability","optimisation","statistical estimation","differential equation"],
          "setup": ["optimal strategy in incomplete-information game","cost minimisation under constraints",
                    "population mean estimation from n=30 sample","infected count prediction using SIR model"]}),
    ],
    "strategic": [
        ("{company_type} considering entry into {market}. Analyse risks and opportunities.",
         {"company_type": ["startup","large corporation","foreign company"],
          "market": ["Southeast Asian market","B2B SaaS","healthcare AI","retail fintech"]}),
        ("Achieve {goal} with limited {resource}. Priority decision framework.",
         {"resource": ["6-month runway","team of 5","budget of 100M KRW"],
          "goal": ["achieve PMF","prepare Series A","monetisation","secure technology patents"]}),
    ],
    "philosophical": [
        ("Claim: \"{claim}\". Validate premises and analyse counterarguments.",
         {"claim": ["free will exists","AI cannot be conscious",
                    "morality is culturally relative","knowledge is justified true belief"]}),
        ("Judgment and limits from a {ethical_framework} perspective on {scenario}.",
         {"scenario": ["autonomous vehicle trolley problem","AI decision-making fairness","privacy vs public safety"],
          "ethical_framework": ["utilitarianism","deontology","virtue ethics","contractarianism"]}),
    ],
    "engineering": [
        ("Design a {system_type} system for {requirements}. Architecture decisions and trade-offs.",
         {"system_type": ["distributed cache","message queue","API gateway","ML serving"],
          "requirements": ["1M rps reads","99.99% availability","p99 <10ms","global distribution"]}),
        ("{tech_a} vs {tech_b} technology selection. Evaluation criteria and recommendation.",
         {"tech_a": ["PostgreSQL","Kafka","Kubernetes","REST"],
          "tech_b": ["MongoDB","RabbitMQ","Nomad","GraphQL"]}),
    ],
}

def generate_task(domain: str, difficulty: str, void_depth: str) -> dict:
    """Generate a task for the given domain x difficulty x void_depth combination."""
    templates = TASK_SEEDS.get(domain, TASK_SEEDS["medical"])
    template_text, params = random.choice(templates)

    # fill in template parameters
    filled = template_text
    for key, values in params.items():
        if f"{{{key}}}" in filled:
            filled = filled.replace(f"{{{key}}}", str(random.choice(values)))

    diff_cfg = DIFFICULTIES[difficulty]
    void_instruction = {
        "single":  f"Generate {diff_cfg['void_target']} ?tags and resolve each independently.",
        "cascade": f"Generate {diff_cfg['void_target']}+ ?tags and track any new ?tags that emerge during resolution. Mark unresolvable ones with ! warning.",
    }[void_depth]

    return {
        "domain": domain,
        "difficulty": difficulty,
        "void_depth": void_depth,
        "task_text": filled,
        "void_instruction": void_instruction,
        "max_tokens": diff_cfg["max_tokens"],
    }


###############################################################
# WCY reasoning quality measurement
###############################################################

def measure_quality(response: str) -> dict:
    lines_raw = [l for l in response.split('\n') if l.strip()]
    total = len(lines_raw)
    if total == 0:
        return {"parse_rate": 0, "void_generated": 0, "void_resolved": 0,
                "resolution_rate": 0, "infer_count": 0, "act_count": 0,
                "warning_count": 0, "usable": False}

    valid = sum(1 for l in lines_raw
                if l.strip() and l.strip()[0] in '.!:>~'
                and len(l.strip()) > 1 and l.strip()[1] in (' ', '|'))
    parse_rate = valid / total

    try:
        parsed = flatten(parse_wcy(response))
    except:
        parsed = []

    void_lines    = [l for l in parsed if l.is_void]
    void_tags     = [t for l in void_lines for t in l.void_tags]
    infer_lines   = [l for l in parsed if l.phase == ':' and not l.is_void]
    act_lines     = [l for l in parsed if l.phase == '>']
    warning_lines = [l for l in parsed if l.phase == '!']

    # resolution rate: check if ?tag name appears in later : lines (loose match)
    resolved = 0
    for tag in void_tags:
        core = tag.split('_')[0].lower()
        for il in infer_lines:
            tags_str = str(il.tags).lower()
            if core in tags_str or tag.lower() in tags_str:
                resolved += 1
                break

    resolution_rate = resolved / max(len(void_tags), 1)

    usable = (parse_rate >= 0.70 and
              len(void_tags) >= 1 and
              resolution_rate >= 0.50)

    return {
        "parse_rate": round(parse_rate, 3),
        "void_generated": len(void_tags),
        "void_resolved": resolved,
        "resolution_rate": round(resolution_rate, 3),
        "infer_count": len(infer_lines),
        "act_count": len(act_lines),
        "warning_count": len(warning_lines),
        "usable": usable,
    }


###############################################################
# System prompt (includes Phase 4 few-shot examples)
###############################################################

FEW_SHOT_COMPACT = dedent("""
Example A (code/medium):
~ task  memory_leak  domain=software_engineering
. service=web_api  symptom=50MB_per_hour  started_after=v2.3_release
: ?leak_source  hint=v2.3_changes,async_tasks  conf_range=0.6..0.9
> profile  memory_by_object  tool=pympler  reason=from=3
. profiling  dict_objects=+180MB  coroutine=+12MB
: leak_source=background_task_accumulation  conf=0.89  from=3,5
: ?cleanup_pattern  hint=fastapi_background_tasks  conf_range=0.7..0.9
> review  task_lifecycle  reason=from=7
. code_review  no_done_callback  accumulate_in_queue
: cleanup_pattern=missing_disposal  conf=0.94  from=7,9
> implement  done_callback  from=10
! note  monitor_deployment_async_changes_risky  from=5,9

Example B (philosophical/complex cascade):
~ task  claim_verification  domain=epistemology
. claim  "correlation implies causation"
. data  r=0.87  p<0.001
: ?causal_direction  hint=temporal_sequence  conf_range=0.1..0.4
: ?confounding  hint=third_variable  conf_range=0.6..0.9
> investigate  temporal_sequence  reason=from=3
. finding  simultaneous_not_sequential
: causal_direction=simultaneous  conf=0.85  from=3,5
: ?mechanism  hint=biological_pathway  conf_range=0.05..0.15  from=5
> search  mechanism  reason=from=7
. result  no_mechanism_found
: mechanism=implausible  conf=0.98  from=7,9
: claim_validity=invalid  conf=0.97  from=3,5,7,9
! warning  classic_confounding_fallacy  from=10
""").strip()

SYS_PIPELINE = f"""WCY reasoning format:
. observe  -- confirmed fact
: infer    -- derived conclusion (conf= recommended, from= for provenance)
> act      -- investigation action (reason=from=N)
~ meta     -- task declaration
! exception -- warning or unresolvable marker

?tag rules:
  : ?unknown  hint=direction      conf_range=L..H  <- mark unknown
  > investigate  reason=from=?_line               <- investigate
  . new_finding  result=value                      <- observe
  : resolved=value  conf=0.xx  from=?,obs          <- resolved

{FEW_SHOT_COMPACT}

rule: if you generate a ?tag, complete the resolution cycle. unresolvable -> ! warning.
no markdown, no natural language. WCY output only."""


###############################################################
# Pipeline execution
###############################################################

# Execution plan (adjustable)
# Default: 48 combos x 1 = 48 tasks (fast)
# Large-scale: set RUNS_PER_COMBO=10 for ~480 additional traces per run
RUNS_PER_COMBO = 1   # fast run; set to 10 for ~480 additional traces

# domain order (diversity first)
DOMAIN_ORDER  = list(DOMAINS.keys())
DIFF_ORDER    = ["simple", "medium", "complex"]
DEPTH_ORDER   = ["single", "cascade"]

# build combination list
combos = [
    (d, diff, depth)
    for d in DOMAIN_ORDER
    for diff in DIFF_ORDER
    for depth in DEPTH_ORDER
]
random.seed(42)
random.shuffle(combos)  # shuffle for variety

print(f"Plan: {len(combos)} combos x {RUNS_PER_COMBO} = {len(combos)*RUNS_PER_COMBO} tasks")
print(f"Estimated time: ~{len(combos)*RUNS_PER_COMBO*3//60} min\n")

all_traces = []
stats = {"total": 0, "usable": 0, "gate1_fail": 0, "gate2_fail": 0, "gate3_fail": 0}

OUTPUT_FILE = "wcy_traces_v1.jsonl"
CLEAN_FILE  = "wcy_traces_v1_clean.jsonl"

# initialise output files
open(OUTPUT_FILE, 'w').close()
open(CLEAN_FILE, 'w').close()

print(f"{'#':>4} {'Domain':<14} {'Diff':<8} {'Depth':<8} {'parse':>6} {'?gen':>5} {'?res%':>6} {'gate':>5}")
print("─" * 62)

for run_i in range(RUNS_PER_COMBO):
    for ci, (domain, difficulty, void_depth) in enumerate(combos):
        task = generate_task(domain, difficulty, void_depth)
        trace_id = f"v1_{ci:03d}_{run_i}"
        stats["total"] += 1

        user_msg = f"Task: {task['task_text']}\n\n{task['void_instruction']}"

        try:
            msg = client.messages.create(
                model=SONNET,
                max_tokens=task["max_tokens"],
                system=SYS_PIPELINE,
                messages=[{"role": "user", "content": user_msg}]
            )
            resp = msg.content[0].text
        except Exception as e:
            print(f"  ERROR [{trace_id}]: {e}")
            time.sleep(5)
            continue

        q = measure_quality(resp)

        # quality gates
        gate1 = q["parse_rate"] >= 0.70
        gate2 = q["void_generated"] >= 1
        gate3 = q["resolution_rate"] >= 0.50

        if not gate1: stats["gate1_fail"] += 1
        if not gate2: stats["gate2_fail"] += 1
        if not gate3: stats["gate3_fail"] += 1

        gate_str = "✓" if (gate1 and gate2 and gate3) else (
            "G1" if not gate1 else ("G2" if not gate2 else "G3"))

        if gate1 and gate2 and gate3:
            stats["usable"] += 1

        trace = {
            "id": trace_id,
            "domain": domain,
            "difficulty": difficulty,
            "void_depth": void_depth,
            "task": task["task_text"],
            "void_instruction": task["void_instruction"],
            "wcy_reasoning": resp,
            "quality": q,
            "model": SONNET,
            "spec_version": "1.1",
            "usable": gate1 and gate2 and gate3,
        }

        # save in real time
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(trace, ensure_ascii=False) + '\n')
        if trace["usable"]:
            with open(CLEAN_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(trace, ensure_ascii=False) + '\n')

        print(f"{stats['total']:>4} {domain:<14} {difficulty:<8} {void_depth:<8} "
              f"{q['parse_rate']:>6.0%} {q['void_generated']:>5} "
              f"{q['resolution_rate']:>5.0%} {gate_str:>5}")

        time.sleep(1.2)  # rate limit

# ── Final report ────────────────────────────────────────────────
print(f"\n{'═'*62}")
print(f"  PIPELINE COMPLETE")
print(f"{'═'*62}")
print(f"  Total:     {stats['total']}")
print(f"  Usable:    {stats['usable']} ({stats['usable']/max(stats['total'],1)*100:.0f}%)")
print(f"  Gate 1 fail (parse_r): {stats['gate1_fail']}")
print(f"  Gate 2 fail (void>=1): {stats['gate2_fail']}")
print(f"  Gate 3 fail (res>=50%): {stats['gate3_fail']}")

# per-domain statistics
from collections import defaultdict
domain_stats = defaultdict(lambda: {"total":0,"usable":0,"avg_void":0,"avg_res":0})
with open(OUTPUT_FILE) as f:
    for line in f:
        t = json.loads(line)
        d = t["domain"]
        domain_stats[d]["total"] += 1
        domain_stats[d]["usable"] += int(t["usable"])
        domain_stats[d]["avg_void"] += t["quality"]["void_generated"]
        domain_stats[d]["avg_res"]  += t["quality"]["resolution_rate"]

print(f"\n  By domain:")
print(f"  {'Domain':<14} {'total':>6} {'usable':>7} {'?avg':>6} {'res%':>6}")
print(f"  {'─'*44}")
for dom, s in sorted(domain_stats.items()):
    n = max(s["total"], 1)
    print(f"  {dom:<14} {s['total']:>6} {s['usable']:>7} "
          f"{s['avg_void']/n:>6.1f} {s['avg_res']/n*100:>5.0f}%")

# save report
report = {
    "stats": stats,
    "domain_stats": dict(domain_stats),
    "output_file": OUTPUT_FILE,
    "clean_file": CLEAN_FILE,
}
with open("wcy_pipeline_report.json", "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n  Saved:")
print(f"  {OUTPUT_FILE}    -- all traces")
print(f"  {CLEAN_FILE} -- usable traces only")
print(f"  wcy_pipeline_report.json -- statistics")
print(f"\n  Download the three files from the Colab file panel")
