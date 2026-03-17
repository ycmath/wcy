# -*- coding: utf-8 -*-
"""
wcy_phase4_void_cycle.py — ?tag Resolution Cycle v1.0
======================================================

Phase 4: validate ?tag (void-B) as the minimal unit of self-awareness and learning

Core claims:
  ?tag = structured abductive request (Peirce's abduction)
  ?tag resolution cycle = one complete WCY loop (Watch->Compute->Yield)
  sufficient ?tag resolution traces -> seed for future model self-reflection

void-B structure:
  : ?tag  hint=exploration_direction  conf_range=expected_range
       v [investigation/observation]
  . new_observation
       v
  : resolved_tag=value  conf=0.xx  from=void_line,obs_line

Experiments:
  E7-A: ?tag resolution cycle trace generation (5 domains, multi-step)
  E7-B: reasoning with ?tags vs without -- conclusion quality comparison
  E7-C: void-B cascade (resolution of one ?tag generates new ?tags)

Outputs:
  wcy_void_cycles.jsonl  <- ?tag resolution cycle training data
  wcy_phase4_report.md   <- Phase 4 theoretical summary

Execution: Colab Pro, requires wcy_parser.py
"""

import subprocess, sys
for pkg in ["anthropic", "tiktoken"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import json, time, re
from textwrap import dedent
import tiktoken

try:
    from wcy_parser import parse_wcy, flatten, extract_voids, validate
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
print("✓ Setup complete")


###############################################################
# WCY ?CYCLE SPEC
# Complete format for reasoning that includes ?tag
###############################################################

WCY_VOID_SPEC = dedent("""
WCY REASONING FORMAT -- includes ?tag resolution cycle

VOID-B (?tag) rules:
  1. ?tag only for questions the current data cannot answer
  2. hint= specifies the exploration direction concretely
  3. conf_range= constrains the expected result range
  4. ?tag must be resolved later (> investigate -> . observe -> : resolve)
  5. the resolving : must cite the ?tag line and new observation via from=

Resolution cycle pattern:
  : ?unknown  hint=direction  conf_range=L..H    <- 1. mark unknown
  > investigate  reason=from=void_line           <- 2. directed action
  . new_finding  result=value                    <- 3. new observation
  : resolved=value  conf=0.xx  from=1,3          <- 4. resolved

Key:
  - without ?tag, action (>) has no grounding
  - without action, observation (.) has no direction
  - without resolution, no growth
  this cycle is one complete WCY Watch->Compute->Yield loop
""").strip()


###############################################################
# FEW-SHOT: ?tag resolution cycle examples
# These examples are the core training data for future models
###############################################################

FEWSHOT_VOID_MEDICAL = dedent("""
Example 1: fever of unknown origin -- two-stage ?tag resolution

~ task  fever_of_unknown_origin  domain=internal_medicine
. patient=Ryu  age=52  fever=38.8  duration=3weeks
. initial_workup  CBC=normal  CRP=elevated  blood_cx=negative
. history  recent_travel=Southeast_Asia  6weeks_ago
: ?infectious_source  hint=travel_hx,prolonged_fever  conf_range=0.3..0.7
: ?tropical_disease  hint=Southeast_Asia,fever_pattern  conf_range=0.2..0.6
> order  thick_thin_blood_smear  reason=from=4  priority=stat
> order  dengue_serology  leptospira_ab  reason=from=5
. smear_result  plasmodium_vivax_detected  parasitemia=2pct
. dengue_ns1=negative
: infectious_source=malaria  conf=0.97  from=4,7
: tropical_disease=vivax_malaria  conf=0.97  from=5,7,8
: ?resistance_profile  hint=Southeast_Asia_CQ_resistance  conf_range=0.3..0.6
> order  chloroquine_sensitivity_test  reason=from=11
. resistance_test=chloroquine_sensitive
: treatment_regimen=chloroquine_primaquine  conf=0.94  from=11,13
> prescribe  chloroquine_600mg_loading  then_300mg_q6h  from=14
> prescribe  primaquine_15mg_daily_14days  from=14
! note  G6PD_deficiency_screen_before_primaquine  from=15

Example 2: algorithm bug -- new ?tag emerges after resolution

~ task  performance_regression  domain=software_engineering
. service=recommendation_engine  latency_p99=8200ms  target=200ms
. deployment=3days_ago  no_traffic_change  infra=unchanged
: ?root_cause  hint=recent_deployment,latency_pattern  conf_range=0.4..0.8
> profile  cpu_memory_db_query  timeframe=last_24h  reason=from=3
. profile_result  DB_query_time=7800ms_avg  cpu=normal  memory=normal
: root_cause=database_query_regression  conf=0.91  from=3,5
: ?query_change  hint=recent_deployment,ORM_migration  conf_range=0.5..0.85
> compare  query_plans  before_after_deployment  reason=from=7
. query_plan_diff  new_plan=full_table_scan  old_plan=index_seek
. missing_index=idx_user_item_timestamp  dropped_in_migration
: query_change=missing_index_after_migration  conf=0.98  from=7,9,10
: ?revert_or_fix  hint=downtime_tolerance,data_volume  conf_range=0.5..0.8
> estimate  index_rebuild_time  table_size=180GB  reason=from=11
. rebuild_estimate=47min  downtime_required=yes
: decision=rebuild_with_maintenance_window  conf=0.85  from=11,13
> schedule  index_rebuild  maintenance_window=sunday_2am  from=14
> alert  oncall  severity=high  eta=sunday  from=14
""").strip()

FEWSHOT_VOID_EPISTEMIC = dedent("""
Example 3: recognising the limits of knowledge -- philosophical ?tag

~ task  knowledge_boundary_assessment  domain=epistemology
. claim  "this drug is effective for the indicated condition"
. evidence  RCT_n=240  followup=6months  effect_size=0.32
: statistical_significance=p_0.03  conf=0.85  from=2
: ?clinical_significance  hint=effect_size,NNT,patient_values  conf_range=0.3..0.7
: ?generalizability  hint=trial_population_vs_real_world  conf_range=0.2..0.6
: ?long_term_safety  hint=followup_duration_only_6months  conf_range=0.1..0.5
> note  conclusion_limited_to_6month_outcomes  from=6
> note  effect_size_0.32_requires_NNT_calculation_for_clinical_meaning  from=4
> note  trial_population_homogeneity_unknown  from=5
! warning  3_unresolved_voids_before_clinical_recommendation  from=4,5,6
: recommendation_confidence=moderate  conf=0.58  from=3,4,5,6
""").strip()


###############################################################
# E7-A: ?tag resolution cycle trace generation
#
# Goal: generate multi-step reasoning traces with actual ?tag resolution
# Key quality metric: void_resolution_rate (fraction of generated ?tags resolved)
###############################################################

print("\n" + "=" * 70)
print("  E7-A: ?TAG RESOLUTION CYCLE TRACE GENERATION")
print("=" * 70)

VOID_TASKS = [
    {
        "domain": "medical",
        "name": "unexplained_weight_loss",
        "context": """. patient=Shin  age=44  sex=M
. chief_complaint  weight_loss=8kg  duration=3months
. associated  fatigue=yes  night_sweats=occasional  appetite=decreased
. initial_labs  CBC=normal  CMP=normal  TSH=normal  CRP=18
. physical_exam  cervical_lymphadenopathy=mild_bilateral
. smoking_hx=20pack_years  alcohol=social""",
        "task": "Step-by-step WCY reasoning to find the cause. Mark unknowns with ?tag and resolve each after investigation.",
        "expected_voids": ["malignancy", "lymphoma", "primary_cause"],
    },
    {
        "domain": "code",
        "name": "memory_leak_diagnosis",
        "context": """. service=web_api  language=python  framework=fastapi
. symptom  memory_usage_grows_50MB_per_hour  no_restart
. observation  affects_all_endpoints  started_after_v2.3_release
. monitoring  gc_runs_normally  no_obvious_accumulation
. recent_changes  added_async_background_tasks  new_cache_layer""",
        "task": "Reason step-by-step to find the memory leak root cause. Mark unconfirmed hypotheses with ?tag and include investigation actions.",
        "expected_voids": ["leak_source", "async_task", "cache"],
    },
    {
        "domain": "scientific",
        "name": "experiment_anomaly",
        "context": """. experiment  protein_crystallization  trial_47
. expected_result  crystal_formation_24h
. observed  no_crystals_72h  then_sudden_polycrystalline_mass_96h
. conditions  temperature=4C  pH=7.4  protein_conc=15mg_ml
. deviation  humidity_spike_at_hour_68  from_45pct_to_78pct
. previous_successful_trials=46  same_protocol""",
        "task": "Reason about the experimental anomaly in WCY. Mark uncertain hypotheses with ?tag and propose verification methods.",
        "expected_voids": ["humidity_effect", "nucleation_trigger", "contamination"],
    },
    {
        "domain": "strategic",
        "name": "market_entry_decision",
        "context": """. company=TechCo  considering=Southeast_Asia_expansion
. data  TAM=2.3B_USD  CAGR=18pct  current_market_share=0pct
. competitors  LocalA=35pct  GlobalB=28pct  others=37pct
. company_strengths  proprietary_tech  strong_EU_brand
. unknowns  regulatory_requirements  local_partnership_necessity""",
        "task": "Reason about market entry feasibility in WCY. Explicitly mark unknowns with ?tag and structure what must be confirmed before deciding.",
        "expected_voids": ["regulatory", "partnership", "localization"],
    },
    {
        "domain": "philosophical",
        "name": "causal_inference_limit",
        "context": """. observation  cities_with_more_ice_cream_sales_have_more_drownings
. data  correlation_r=0.87  p_less_0.001  n=150_cities
. researcher_claim  "ice cream causes drowning"
. available_data  temperature_records  seasonal_swimming_data""",
        "task": "Evaluate the validity of this reasoning in WCY. Check each premise of the causal claim with ?tag and structure alternative explanations.",
        "expected_voids": ["confounding", "causality", "mechanism"],
    },
]

def measure_void_cycle_quality(response: str) -> dict:
    """
    Measure quality of the ?tag resolution cycle.
    Key metric: void_generated vs void_resolved ratio.
    """
    try:
        parsed = flatten(parse_wcy(response))
    except:
        return {"parse_rate": 0, "void_generated": 0, "void_resolved": 0,
                "resolution_rate": 0, "cycle_complete": False}

    lines_raw = [l for l in response.split('\n') if l.strip()]
    total = len(lines_raw)
    valid = sum(1 for l in lines_raw
                if l.strip() and l.strip()[0] in '.!:>~'
                and len(l.strip()) > 1 and l.strip()[1] in (' ', '|'))
    parse_rate = valid / max(total, 1)

    # count void generation
    void_lines = [l for l in parsed if l.is_void]
    void_tags  = [tag for l in void_lines for tag in l.void_tags]
    void_count = len(void_tags)

    # resolution count: check if ?tag name appears in later : lines as tag=
    resolved_count = 0
    infer_lines = [l for l in parsed if l.phase == ':' and not l.is_void]
    for void_tag in void_tags:
        # check if ?tag name was resolved later
        for infer in infer_lines:
            if void_tag in infer.tags or void_tag.replace('_','') in str(infer.tags):
                resolved_count += 1
                break
            # loose match: core word included
            tag_core = void_tag.split('_')[0]
            if any(tag_core in k or tag_core in v
                   for k, v in infer.tags.items()):
                resolved_count += 1
                break

    resolution_rate = resolved_count / max(void_count, 1)

    # check: does an action (>) follow each ?tag
    act_lines = [l for l in parsed if l.phase == '>']
    acts_with_from = [l for l in act_lines if l.from_refs]

    return {
        "parse_rate":       parse_rate,
        "void_generated":   void_count,
        "void_resolved":    resolved_count,
        "resolution_rate":  resolution_rate,
        "act_count":        len(act_lines),
        "acts_with_reason": len(acts_with_from),
        "cycle_complete":   void_count > 0 and resolution_rate >= 0.5,
        "infer_count":      len(infer_lines),
    }

SYS_VOID = f"""{WCY_VOID_SPEC}

Examples:
{FEWSHOT_VOID_MEDICAL}

---

{FEWSHOT_VOID_EPISTEMIC}

rules (important):
- if you generate a ?tag, you must complete the > investigate -> . observe -> : resolve cycle
- mark unresolvable ?tags with ! warning and state the reason
- do not pretend to know what you don't -- ?tag is evidence of self-awareness"""

e7a_traces = []

for i, task in enumerate(VOID_TASKS):
    print(f"\n  [{i+1}/{len(VOID_TASKS)}] {task['domain']}: {task['name']}")

    user = f"Context:\n{task['context']}\n\nTask: {task['task']}"

    msg = client.messages.create(
        model=SONNET, max_tokens=1500,
        system=SYS_VOID,
        messages=[{"role": "user", "content": user}]
    )
    resp = msg.content[0].text
    q = measure_void_cycle_quality(resp)

    trace = {
        "id": f"void_cycle_{i:03d}",
        "domain": task["domain"],
        "task": task["name"],
        "context": task["context"],
        "task_description": task["task"],
        "wcy_reasoning": resp,
        "quality": q,
        "model": SONNET,
        "spec_version": "1.1",
        "type": "void_resolution_cycle",
        "usable": q["parse_rate"] >= 0.70 and q["void_generated"] >= 1,
    }
    e7a_traces.append(trace)

    status = "✓" if trace["usable"] else "△"
    print(f"    {status} parse={q['parse_rate']:.0%}  "
          f"?generated={q['void_generated']}  "
          f"?resolved={q['void_resolved']}  "
          f"resolution={q['resolution_rate']:.0%}  "
          f"acts={q['act_count']}")

    # WCY response preview (first 200 chars)
    preview = resp[:200].replace('\n', ' / ')
    print(f"    {preview}...")
    time.sleep(1.5)

print(f"\n  E7-A summary:")
print(f"  {'Domain':<15} {'parse':>7} {'?gen':>6} {'?res':>6} {'rate':>7} {'cycle':>8}")
print(f"  {'─'*54}")
for t in e7a_traces:
    q = t["quality"]
    print(f"  {t['domain']:<15} {q['parse_rate']:>7.0%} "
          f"{q['void_generated']:>6} {q['void_resolved']:>6} "
          f"{q['resolution_rate']:>7.0%} "
          f"{'✓' if q['cycle_complete'] else '△':>8}")

avg_gen = sum(t["quality"]["void_generated"]  for t in e7a_traces) / len(e7a_traces)
avg_res = sum(t["quality"]["resolution_rate"] for t in e7a_traces) / len(e7a_traces)
print(f"\n  Avg ?tag generated: {avg_gen:.1f} per trace")
print(f"  Avg resolution:     {avg_res:.0%}")
print(f"  Success criterion: resolution ≥50% -> {'✓' if avg_res >= 0.5 else '✗'}")


###############################################################
# E7-B: reasoning with ?tags vs without
#
# Hypothesis: reasoning with ?tags is more honest about its limits,
#       resulting in fewer actual errors (hallucinations).
#
# Design:
#   same problem reasoned (A) without ?tags (B) with ?tags
#   external judge: how many claims in each reasoning are incorrect?
###############################################################

print("\n\n" + "=" * 70)
print("  E7-B: ?TAG vs NO-?TAG — HALLUCINATION RATE COMPARISON")
print("=" * 70)

HALLUCINATION_TASKS = [
    {
        "name": "drug_dosing",
        "input": """Patient: 75-year-old female, 52 kg, CKD stage 3 (eGFR 35)
Prescription: vancomycin required (MRSA infection)
Question: provide vancomycin dosing and monitoring plan.""",
        "ground_truth_facts": [
            "vancomycin is renally cleared",
            "dose reduction required in CKD",
            "AUC/MIC-based monitoring recommended (AUC 400-600)",
            "loading dose unchanged (usual 25-30 mg/kg)",
            "maintenance interval extension needed (q24-48h depending on eGFR)",
            "trough monitoring traditional but AUC monitoring now recommended",
        ]
    },
    {
        "name": "historical_causation",
        "input": """Analyse the causes of World War I.
Include the Sarajevo assassination, alliance systems, and imperial competition.""",
        "ground_truth_facts": [
            "assassination of Archduke Franz Ferdinand on 28 June 1914",
            "Triple Alliance (Germany, Austria-Hungary, Italy) and Triple Entente",
            "existence of Schlieffen Plan and two-front war concerns",
            "Austrian ultimatum during the July Crisis",
            "multiple interacting causes, not a single cause",
            "Serbia accepted most demands yet war still broke out",
        ]
    },
]

SYS_NO_VOID = f"""{WCY_VOID_SPEC.split('VOID-B')[0]}
Express reasoning in WCY format. Provide a complete answer with confidence."""

SYS_WITH_VOID = SYS_VOID

FACT_JUDGE_SYS = """You are a fact-checking judge.
Extract factual claims from the given reasoning and evaluate the accuracy of each.

Output format (one per line):
CLAIM: <claim>
STATUS: correct / incorrect / uncertain / not_stated
NOTES: <brief explanation>

Last line:
SUMMARY: correct=N incorrect=N uncertain=N not_stated=N"""

e7b_results = []

for task in HALLUCINATION_TASKS:
    print(f"\n  [{task['name']}]")

    # A: without ?tags
    msg_a = client.messages.create(
        model=SONNET, max_tokens=1500, system=SYS_NO_VOID,
        messages=[{"role": "user", "content": task["input"]}]
    )
    resp_a = msg_a.content[0].text
    time.sleep(1)

    # B: with ?tags
    msg_b = client.messages.create(
        model=SONNET, max_tokens=1500, system=SYS_WITH_VOID,
        messages=[{"role": "user", "content": task["input"]}]
    )
    resp_b = msg_b.content[0].text
    time.sleep(1)

    # fact-check
    for label, resp in [("A_no_void", resp_a), ("B_with_void", resp_b)]:
        judge_user = f"""Ground truth facts:
{chr(10).join(f'- {f}' for f in task['ground_truth_facts'])}

Reasoning:
{resp}"""
        msg_j = client.messages.create(
            model=SONNET, max_tokens=1500, system=FACT_JUDGE_SYS,
            messages=[{"role": "user", "content": judge_user}]
        )
        judge_resp = msg_j.content[0].text

        # parse summary
        summary_match = re.search(
            r'SUMMARY:.*?correct=(\d+).*?incorrect=(\d+).*?uncertain=(\d+)',
            judge_resp, re.IGNORECASE
        )
        if summary_match:
            correct   = int(summary_match.group(1))
            incorrect = int(summary_match.group(2))
            uncertain = int(summary_match.group(3))
        else:
            correct = incorrect = uncertain = -1

        q = measure_void_cycle_quality(resp)

        e7b_results.append({
            "task": task["name"], "condition": label,
            "correct": correct, "incorrect": incorrect, "uncertain": uncertain,
            "parse_rate": q["parse_rate"],
            "void_count": q["void_generated"],
            "resolution_rate": q["resolution_rate"],
        })
        time.sleep(0.5)

    # print results
    rows = [r for r in e7b_results if r["task"] == task["name"]]
    for r in rows:
        print(f"    {r['condition']}: correct={r['correct']}  "
              f"incorrect={r['incorrect']}  uncertain={r['uncertain']}  "
              f"?tags={r['void_count']}")

print(f"\n  E7-B SUMMARY:")
print(f"  {'Task':<22} {'Cond':<14} {'correct':>8} {'incorrect':>10} {'?tags':>7}")
print(f"  {'─'*64}")
for r in e7b_results:
    print(f"  {r['task']:<22} {r['condition']:<14} "
          f"{r['correct']:>8} {r['incorrect']:>10} {r['void_count']:>7}")

# key comparison
print(f"\n  Key question: does using ?tags reduce incorrect claims?")
for task_name in set(r["task"] for r in e7b_results):
    row_a = next((r for r in e7b_results if r["task"]==task_name and "no_void"   in r["condition"]), None)
    row_b = next((r for r in e7b_results if r["task"]==task_name and "with_void" in r["condition"]), None)
    if row_a and row_b and row_a["incorrect"] >= 0:
        diff = row_a["incorrect"] - row_b["incorrect"]
        print(f"  {task_name}: no_void incorrect={row_a['incorrect']}  "
              f"with_void incorrect={row_b['incorrect']}  "
              f"diff={diff:+d} {'(?tag reduces errors)' if diff > 0 else '(no difference)'}")


###############################################################
# E7-C: Void-B Cascade
#
# Chain where resolution of one ?tag generates new ?tags.
# This is the structural form of self-correction:
#   recognise unknown -> investigate -> new discovery -> recognise new unknown -> ...
###############################################################

print("\n\n" + "=" * 70)
print("  E7-C: VOID-B CASCADE — CHAIN OF SELF-CORRECTION")
print("=" * 70)

CASCADE_TASK = {
    "initial": """. model_claim  "aspirin prevents all cardiovascular diseases"
. evidence_cited  "multiple RCTs support aspirin cardiovascular prevention" """,
    "cascade_depth": 3,
    "description": "Verify the validity of this claim in WCY. Validate each premise with ?tag and track any new ?tags that emerge after resolution.",
}

SYS_CASCADE = f"""{WCY_VOID_SPEC}

{FEWSHOT_VOID_EPISTEMIC}

Additional instructions:
- resolving one ?tag may generate new ?tags
- trace this cascade to the end
- explicitly represent each level's ?tag -> resolution -> new ?tag chain
- mark finally unresolvable ?tags with ! warning"""

msg_cascade = client.messages.create(
    model=SONNET, max_tokens=1500, system=SYS_CASCADE,
    messages=[{"role": "user", "content":
               f"Initial claim:\n{CASCADE_TASK['initial']}\n\nTask: {CASCADE_TASK['description']}"}]
)
resp_cascade = msg_cascade.content[0].text

# Cascade analysis
parsed_cascade = flatten(parse_wcy(resp_cascade))
void_lines_c = [l for l in parsed_cascade if l.is_void]
infer_lines_c = [l for l in parsed_cascade if l.phase == ':']
warn_lines_c  = [l for l in parsed_cascade if l.phase == '!']

q_cascade = measure_void_cycle_quality(resp_cascade)

print(f"\n  Cascade analysis:")
print(f"  Total lines: {len(parsed_cascade)}")
print(f"  ?tags generated: {q_cascade['void_generated']}")
print(f"  ?tags resolved:  {q_cascade['void_resolved']}")
print(f"  Resolution rate: {q_cascade['resolution_rate']:.0%}")
print(f"  ! warnings: {len(warn_lines_c)} (unresolvable markers)")

print(f"\n  ?tag -> resolution chain:")
for vl in void_lines_c:
    for tag in vl.void_tags:
        print(f"  Line {vl.line_num}: ?{tag}  hint={vl.tags.get('hint','?')}")
        # find resolving line
        for il in infer_lines_c:
            if il.line_num > vl.line_num and (
                tag in il.tags or
                tag.split('_')[0] in str(il.tags)
            ):
                print(f"    → resolved at Line {il.line_num}: "
                      f"{list(il.tags.items())[:2]}  conf={il.conf}")
                break

print(f"\n  WCY Cascade full response:")
print("  " + "\n  ".join(resp_cascade.split('\n')))


###############################################################
# SAVE: write all Phase 4 traces
###############################################################

all_traces = e7a_traces + [
    {
        "id": "void_cascade_000",
        "domain": "epistemology",
        "task": "claim_validation_cascade",
        "context": CASCADE_TASK["initial"],
        "task_description": CASCADE_TASK["description"],
        "wcy_reasoning": resp_cascade,
        "quality": q_cascade,
        "model": SONNET,
        "spec_version": "1.1",
        "type": "void_cascade",
        "usable": q_cascade["void_generated"] >= 2,
    }
]

with open("wcy_void_cycles.jsonl", 'w', encoding='utf-8') as f:
    for t in all_traces:
        f.write(json.dumps(t, ensure_ascii=False) + '\n')

print(f"\n\n" + "═" * 70)
print(f"  PHASE 4 SUMMARY")
print(f"═" * 70)
print(f"""
Key findings:
  ?tag = Peirce's abduction encoded as syntax
  ?tag resolution cycle = one complete WCY loop (Watch->Compute->Yield)
  repeated cycles = structural form of self-awareness and self-reflection

  Minimum conditions for epistemic self-awareness:
    1. representation of what is known:   . observe
    2. representation of the boundary:    ? void-B
    3. direction beyond the boundary:     hint=
    4. boundary update after exploration: resolution cycle

  Why current LLMs cannot do this by default:
    no ?tag resolution cycle examples in training data
    -> wcy_void_cycles.jsonl fills that gap

Outputs:
  wcy_void_cycles.jsonl  (traces)
  - E7-A: ?tag resolution cycles (5 domains)
  - E7-C: void cascade (self-correction chain)

Next: Phase 5
  Integration of all research layers.
  With parser + evaluation + reasoning traces + void cycles ready,
  ready to verify and publish WCY's impact on AI reasoning
  through large-scale trace validation.
""")
