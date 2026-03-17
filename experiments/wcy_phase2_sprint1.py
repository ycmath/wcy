# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  WCY Phase 2 — Sprint 1
  Scale Test + Multi-Agent Delegation + Tool Call Format
═══════════════════════════════════════════════════════════════

Experiments:
  E1-A: Scale Ladder (10/50/200/500 rows)
  E2-A: 3-Agent Delegation Chain (JSON vs WCY token cost)
  E2-B: Tool Call Format (function calling schema comparison)
"""

###############################################################
# CELL 1: Setup
###############################################################

import subprocess, sys
for pkg in ["anthropic", "tiktoken"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import json, time, random, os
from collections import defaultdict
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

# API
API_KEY = "sk-ant-api03-YOUR_KEY_HERE"
try:
    from google.colab import userdata
    API_KEY = userdata.get('ANTHROPIC_API_KEY')
except:
    API_KEY = os.environ.get('ANTHROPIC_API_KEY', API_KEY)

client = None
USE_API = False
if API_KEY:
    import anthropic
    client = anthropic.Anthropic(api_key=API_KEY)
    USE_API = True

MODEL = "claude-sonnet-4-20250514"
count = lambda t: len(enc.encode(t))

print(f"✓ Setup: API={'ON' if USE_API else 'OFF'}")


###############################################################
# CELL 2: Data Generators
###############################################################

NAMES = ["Kim", "Lee", "Park", "Choi", "Jung", "Kang", "Cho", "Yoon",
         "Jang", "Lim", "Han", "Oh", "Seo", "Shin", "Kwon", "Hwang",
         "Ahn", "Song", "Yoo", "Hong", "Moon", "Bae", "Noh", "Ryu"]
SYMPTOMS = ["fever", "cough", "headache", "fatigue", "nausea", "dizziness",
            "rash", "joint_pain", "chest_pain", "shortness_of_breath",
            "sore_throat", "runny_nose", "back_pain", "abdominal_pain"]
MEDS = ["acetaminophen", "ibuprofen", "amoxicillin", "loratadine",
        "omeprazole", "metformin", "lisinopril", "atorvastatin"]
DIAGS = ["influenza", "migraine", "gastritis", "hypertension",
         "bronchitis", "allergic_rhinitis", "UTI", "diabetes_type2"]

random.seed(42)

def gen_patients(n):
    """Generate n patient records as list of dicts."""
    patients = []
    for i in range(n):
        name = NAMES[i % len(NAMES)] + str(i // len(NAMES) + 1 if i >= len(NAMES) else "")
        syms = random.sample(SYMPTOMS, random.randint(1, 4))
        meds = random.sample(MEDS, random.randint(0, 3))
        patients.append({
            "id": f"P-{1000+i}",
            "name": name,
            "age": random.randint(20, 80),
            "temperature": round(random.uniform(36.0, 40.0), 1),
            "symptoms": syms,
            "medications": [{"name": m, "dose_mg": random.choice([100,200,250,500])}
                            for m in meds],
            "diagnosis": {
                "primary": random.choice(DIAGS),
                "confidence": round(random.uniform(0.3, 0.95), 2)
            }
        })
    return patients


def patients_to_json(patients):
    return json.dumps({"patients": patients}, indent=2, ensure_ascii=False)

def patients_to_json_compact(patients):
    return json.dumps({"patients": patients}, ensure_ascii=False, separators=(',',':'))

def patients_to_wcy_hybrid(patients):
    lines = ["~ patient:id,name,age,temp  symptom:list  med:name,dose  diagnosis:primary,conf"]
    for p in patients:
        lines.append(f'~ context  {p["id"]}')
        lines.append(f'. patient  {p["id"]}  {p["name"]}  {p["age"]}  {p["temperature"]}')
        lines.append(f'. symptoms  {" ".join(p["symptoms"])}')
        if p["medications"]:
            for m in p["medications"]:
                lines.append(f'. med  {m["name"]}  {m["dose_mg"]}')
        else:
            lines.append('. med  none')
        d = p["diagnosis"]
        lines.append(f': diagnosis  {d["primary"]}  {d["confidence"]}')
        lines.append('')
    return '\n'.join(lines)

def patients_to_wcy_positional(patients):
    lines = []
    for p in patients:
        lines.append(f'. {p["id"]}  {p["name"]}  {p["age"]}  {p["temperature"]}')
        lines.append(f'. symptoms  {" ".join(p["symptoms"])}')
        if p["medications"]:
            for m in p["medications"]:
                lines.append(f'. med  {m["name"]}  {m["dose_mg"]}')
        else:
            lines.append('. med  none')
        d = p["diagnosis"]
        lines.append(f': {d["primary"]}  {d["confidence"]}')
    return '\n'.join(lines)

print("✓ Generators ready")


###############################################################
# CELL 3: E1-A — Scale Ladder (Token Density at Scale)
###############################################################

print("=" * 70)
print("  E1-A: SCALE LADDER — Token Density at Scale")
print("=" * 70)

scale_results = []
for n in [10, 50, 200, 500]:
    patients = gen_patients(n)
    jp = patients_to_json(patients)
    jc = patients_to_json_compact(patients)
    wh = patients_to_wcy_hybrid(patients)
    wp = patients_to_wcy_positional(patients)

    t_jp, t_jc, t_wh, t_wp = count(jp), count(jc), count(wh), count(wp)

    print(f"\n--- {n} patients ---")
    print(f"  {'Format':<20} {'Tokens':>7} {'vs Pretty':>10} {'vs Compact':>11}")
    print(f"  {'─'*50}")
    print(f"  {'JSON pretty':<20} {t_jp:>7} {'base':>10} {(t_jp-t_jc)/t_jp*100-100:>+10.1f}%")
    print(f"  {'JSON compact':<20} {t_jc:>7} {(t_jp-t_jc)/t_jp*100:>+9.1f}% {'base':>11}")
    print(f"  {'WCY hybrid':<20} {t_wh:>7} {(t_jp-t_wh)/t_jp*100:>+9.1f}% {(t_jc-t_wh)/t_jc*100:>+10.1f}%")
    print(f"  {'WCY positional':<20} {t_wp:>7} {(t_jp-t_wp)/t_jp*100:>+9.1f}% {(t_jc-t_wp)/t_jc*100:>+10.1f}%")

    scale_results.append({
        "n": n, "json_pretty": t_jp, "json_compact": t_jc,
        "wcy_hybrid": t_wh, "wcy_positional": t_wp,
        "sav_hyb_vs_pretty": (t_jp-t_wh)/t_jp*100,
        "sav_pos_vs_pretty": (t_jp-t_wp)/t_jp*100,
        "sav_hyb_vs_compact": (t_jc-t_wh)/t_jc*100,
        "sav_pos_vs_compact": (t_jc-t_wp)/t_jc*100,
    })

# Summary trend
print(f"\n{'='*70}")
print(f"  SCALE TREND SUMMARY")
print(f"  {'N':>5} {'Hybrid vs Pretty':>18} {'Pos vs Pretty':>15} {'Hybrid vs Compact':>19} {'Pos vs Compact':>16}")
for r in scale_results:
    print(f"  {r['n']:>5} {r['sav_hyb_vs_pretty']:>+17.1f}% {r['sav_pos_vs_pretty']:>+14.1f}% {r['sav_hyb_vs_compact']:>+18.1f}% {r['sav_pos_vs_compact']:>+15.1f}%")


###############################################################
# CELL 4: E2-B — Tool Call Format Comparison
###############################################################

print("\n" + "=" * 70)
print("  E2-B: TOOL CALL FORMAT — Function Calling Overhead")
print("=" * 70)

# 5 tool definitions
TOOLS_JSON = [
    {
        "name": "get_patient_record",
        "description": "Retrieve patient medical record",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "Patient ID"},
                "fields": {"type": "array", "items": {"type": "string"},
                           "description": "Fields to retrieve"}
            },
            "required": ["patient_id"]
        }
    },
    {
        "name": "order_lab_test",
        "description": "Order a laboratory test",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "test_name": {"type": "string"},
                "priority": {"type": "string", "enum": ["routine", "urgent", "stat"]}
            },
            "required": ["patient_id", "test_name"]
        }
    },
    {
        "name": "prescribe_medication",
        "description": "Prescribe medication to patient",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "medication": {"type": "string"},
                "dose_mg": {"type": "number"},
                "frequency": {"type": "string"},
                "duration_days": {"type": "integer"}
            },
            "required": ["patient_id", "medication", "dose_mg", "frequency"]
        }
    },
    {
        "name": "send_alert",
        "description": "Send alert to medical staff",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "message": {"type": "string"}
            },
            "required": ["recipient", "priority", "message"]
        }
    },
    {
        "name": "schedule_followup",
        "description": "Schedule follow-up appointment",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "days_from_now": {"type": "integer"},
                "department": {"type": "string"},
                "reason": {"type": "string"}
            },
            "required": ["patient_id", "days_from_now"]
        }
    }
]

TOOLS_WCY = """\
~ tool  get_patient_record  "Retrieve patient medical record"
  . patient_id  string  required
  . fields  array[string]

~ tool  order_lab_test  "Order a laboratory test"
  . patient_id  string  required
  . test_name  string  required
  . priority  string  routine|urgent|stat

~ tool  prescribe_medication  "Prescribe medication to patient"
  . patient_id  string  required
  . medication  string  required
  . dose_mg  number  required
  . frequency  string  required
  . duration_days  integer

~ tool  send_alert  "Send alert to medical staff"
  . recipient  string  required
  . priority  string  low|medium|high|critical  required
  . message  string  required

~ tool  schedule_followup  "Schedule follow-up appointment"
  . patient_id  string  required
  . days_from_now  integer  required
  . department  string
  . reason  string"""

# 10 tool calls
CALLS_JSON = [
    {"name": "get_patient_record", "arguments": {"patient_id": "P-1001", "fields": ["vitals", "medications"]}},
    {"name": "order_lab_test", "arguments": {"patient_id": "P-1001", "test_name": "CBC", "priority": "stat"}},
    {"name": "prescribe_medication", "arguments": {"patient_id": "P-1001", "medication": "tamiflu", "dose_mg": 75, "frequency": "q12h", "duration_days": 5}},
    {"name": "send_alert", "arguments": {"recipient": "Dr. Lee", "priority": "high", "message": "Patient P-1001 fever >39C"}},
    {"name": "get_patient_record", "arguments": {"patient_id": "P-1002", "fields": ["diagnosis", "labs"]}},
    {"name": "order_lab_test", "arguments": {"patient_id": "P-1002", "test_name": "CT_head", "priority": "urgent"}},
    {"name": "schedule_followup", "arguments": {"patient_id": "P-1001", "days_from_now": 7, "department": "pulmonology", "reason": "persistent cough"}},
    {"name": "prescribe_medication", "arguments": {"patient_id": "P-1002", "medication": "sumatriptan", "dose_mg": 50, "frequency": "prn", "duration_days": 30}},
    {"name": "send_alert", "arguments": {"recipient": "Nurse Station B", "priority": "medium", "message": "P-1002 needs monitoring q4h"}},
    {"name": "schedule_followup", "arguments": {"patient_id": "P-1002", "days_from_now": 14, "department": "neurology", "reason": "recurring migraines"}},
]

CALLS_WCY = """\
> get_patient_record  P-1001  fields=vitals,medications
> order_lab_test  P-1001  CBC  stat
> prescribe_medication  P-1001  tamiflu  75  q12h  days=5
> send_alert  Dr.Lee  high  "Patient P-1001 fever >39C"
> get_patient_record  P-1002  fields=diagnosis,labs
> order_lab_test  P-1002  CT_head  urgent
> schedule_followup  P-1001  days=7  pulmonology  "persistent cough"
> prescribe_medication  P-1002  sumatriptan  50  prn  days=30
> send_alert  Nurse_Station_B  medium  "P-1002 needs monitoring q4h"
> schedule_followup  P-1002  days=14  neurology  "recurring migraines\""""

# Measure
schema_json = json.dumps(TOOLS_JSON, indent=2)
calls_json = json.dumps(CALLS_JSON, indent=2)
schema_wcy = TOOLS_WCY
calls_wcy = CALLS_WCY

t_schema_j = count(schema_json)
t_schema_w = count(schema_wcy)
t_calls_j = count(calls_json)
t_calls_w = count(calls_wcy)

print(f"\n  {'Component':<25} {'JSON':>7} {'WCY':>7} {'Savings':>9}")
print(f"  {'─'*50}")
print(f"  {'Schema (5 tools)':<25} {t_schema_j:>7} {t_schema_w:>7} {(t_schema_j-t_schema_w)/t_schema_j*100:>+8.1f}%")
print(f"  {'Calls (10 invocations)':<25} {t_calls_j:>7} {t_calls_w:>7} {(t_calls_j-t_calls_w)/t_calls_j*100:>+8.1f}%")
total_j = t_schema_j + t_calls_j
total_w = t_schema_w + t_calls_w
print(f"  {'Total':<25} {total_j:>7} {total_w:>7} {(total_j-total_w)/total_j*100:>+8.1f}%")

# Per-call amortized (schema sent once per session, calls sent each time)
# Simulate: schema + N calls
print(f"\n  Amortized cost (schema once + N calls):")
print(f"  {'N calls':<10} {'JSON total':>12} {'WCY total':>12} {'Savings':>9}")
for n_calls in [1, 5, 10, 20, 50]:
    j_total = t_schema_j + t_calls_j * n_calls / 10  # scale proportionally
    w_total = t_schema_w + t_calls_w * n_calls / 10
    print(f"  {n_calls:<10} {j_total:>12.0f} {w_total:>12.0f} {(j_total-w_total)/j_total*100:>+8.1f}%")


###############################################################
# CELL 5: E2-A — 3-Agent Delegation Chain (API 필요)
###############################################################

TASK = """Analyze patient P-1001 (Kim, age 42, temp 38.5, symptoms: fever + cough + fatigue for 7 days).
Currently taking acetaminophen 500mg q6h.
Provide: risk assessment, recommended tests, treatment plan."""

WCY_SPEC_SHORT = """\
WCY format: lines starting with . (data), : (inference), > (action), ~ (meta), ! (error).
Slots are space-separated. tag=value for named, bare for positional. from=N for evidence trail."""

if not USE_API:
    print("\n⏭ E2-A 건너뜀 (API 필요)")
else:
    print("\n" + "=" * 70)
    print("  E2-A: 3-AGENT DELEGATION CHAIN")
    print("=" * 70)

    results_e2a = {"json": {}, "wcy": {}}

    for fmt in ["json", "wcy"]:
        print(f"\n{'─'*60}")
        print(f"  Pipeline: {fmt.upper()}")
        print(f"{'─'*60}")

        if fmt == "json":
            sys_planner = "You are a medical planning agent. Output your plan as JSON. Include: assessment, tests_needed, treatment_steps."
            sys_researcher = "You are a medical research agent. You receive a plan in JSON. Research each item and output detailed findings as JSON."
            sys_writer = "You are a medical report writer. You receive research findings in JSON. Write a concise clinical report. Output as JSON with sections: summary, findings, recommendations."
        else:
            sys_planner = f"You are a medical planning agent. Output your plan in WCY format.\n{WCY_SPEC_SHORT}"
            sys_researcher = f"You are a medical research agent. You receive a plan in WCY format. Research each item and output detailed findings in WCY format.\n{WCY_SPEC_SHORT}"
            sys_writer = f"You are a medical report writer. You receive research findings in WCY format. Write a concise clinical report in WCY format.\n{WCY_SPEC_SHORT}"

        # Agent 1: Planner
        print(f"\n  Agent 1 (Planner)...")
        r1 = client.messages.create(
            model=MODEL, max_tokens=800, temperature=0,
            system=sys_planner,
            messages=[{"role": "user", "content": f"Task:\n{TASK}"}]
        )
        out1 = r1.content[0].text
        t1_in, t1_out = r1.usage.input_tokens, r1.usage.output_tokens
        print(f"    Input: {t1_in} tok, Output: {t1_out} tok")

        time.sleep(2)

        # Agent 2: Researcher (receives Planner output)
        print(f"  Agent 2 (Researcher)...")
        r2 = client.messages.create(
            model=MODEL, max_tokens=1000, temperature=0,
            system=sys_researcher,
            messages=[{"role": "user", "content": f"Plan from Planner:\n{out1}\n\nOriginal task:\n{TASK}\n\nResearch each item in detail."}]
        )
        out2 = r2.content[0].text
        t2_in, t2_out = r2.usage.input_tokens, r2.usage.output_tokens
        print(f"    Input: {t2_in} tok, Output: {t2_out} tok")

        time.sleep(2)

        # Agent 3: Writer (receives Planner + Researcher output)
        print(f"  Agent 3 (Writer)...")
        r3 = client.messages.create(
            model=MODEL, max_tokens=1000, temperature=0,
            system=sys_writer,
            messages=[{"role": "user", "content": f"Research findings:\n{out2}\n\nOriginal plan:\n{out1}\n\nWrite the final clinical report."}]
        )
        out3 = r3.content[0].text
        t3_in, t3_out = r3.usage.input_tokens, r3.usage.output_tokens
        print(f"    Input: {t3_in} tok, Output: {t3_out} tok")

        total_in = t1_in + t2_in + t3_in
        total_out = t1_out + t2_out + t3_out
        total = total_in + total_out

        results_e2a[fmt] = {
            "agent1": {"in": t1_in, "out": t1_out},
            "agent2": {"in": t2_in, "out": t2_out},
            "agent3": {"in": t3_in, "out": t3_out},
            "total_in": total_in, "total_out": total_out, "total": total,
        }

        print(f"\n  Total: input={total_in}, output={total_out}, GRAND={total}")

    # Comparison
    j, w = results_e2a["json"], results_e2a["wcy"]
    print(f"\n{'='*60}")
    print(f"  E2-A COMPARISON: JSON vs WCY Pipeline")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'JSON':>8} {'WCY':>8} {'Savings':>9}")
    print(f"  {'─'*52}")
    for label, jv, wv in [
        ("Agent 1 input",  j["agent1"]["in"],  w["agent1"]["in"]),
        ("Agent 1 output", j["agent1"]["out"], w["agent1"]["out"]),
        ("Agent 2 input",  j["agent2"]["in"],  w["agent2"]["in"]),
        ("Agent 2 output", j["agent2"]["out"], w["agent2"]["out"]),
        ("Agent 3 input",  j["agent3"]["in"],  w["agent3"]["in"]),
        ("Agent 3 output", j["agent3"]["out"], w["agent3"]["out"]),
        ("─── Total input",  j["total_in"],  w["total_in"]),
        ("─── Total output", j["total_out"], w["total_out"]),
        ("═══ GRAND TOTAL",  j["total"],     w["total"]),
    ]:
        sav = (jv - wv) / jv * 100 if jv > 0 else 0
        print(f"  {label:<25} {jv:>8} {wv:>8} {sav:>+8.1f}%")

    # Show WCY agent outputs (samples)
    print(f"\n{'─'*60}")
    print(f"  WCY Agent 1 (Planner) sample output:")
    for line in out1.split('\n')[:12]:
        print(f"    {line}")
    if len(out1.split('\n')) > 12:
        print(f"    ...")


###############################################################
# CELL 6: E2-A Comprehension Check (API 필요)
###############################################################

if USE_API:
    print("\n" + "=" * 70)
    print("  E2-A QUALITY CHECK: Can Agent 3 outputs answer the same questions?")
    print("=" * 70)

    # Ask same questions about both final reports
    QUALITY_QS = [
        "What is the risk level for this patient?",
        "What tests are recommended?",
        "What medication changes are suggested?",
    ]

    for fmt in ["json", "wcy"]:
        # Get Agent 3 output from the pipeline we just ran
        pipeline_sys = f"Answer based on the report below. Be concise."
        # We need to re-get the output - store it during E2-A
        # For now, just note this check should be added
        pass

    print("  [Quality comparison requires storing agent outputs — see Sprint 2]")


###############################################################
# CELL 7: Comprehensive Summary
###############################################################

print("\n" + "=" * 70)
print("  ═══ SPRINT 1 RESULTS ═══")
print("=" * 70)

# E1-A summary
print("\n📊 E1-A: Scale Ladder (WCY hybrid vs JSON pretty)")
print(f"  {'N':>5} {'Savings':>10}")
for r in scale_results:
    print(f"  {r['n']:>5} {r['sav_hyb_vs_pretty']:>+9.1f}%")

print(f"\n  Trend: {'IMPROVING' if scale_results[-1]['sav_hyb_vs_pretty'] > scale_results[0]['sav_hyb_vs_pretty'] else 'STABLE'} with scale")

# E2-B summary
print(f"\n📊 E2-B: Tool Call Format")
print(f"  Schema: {(t_schema_j-t_schema_w)/t_schema_j*100:+.1f}%")
print(f"  Calls:  {(t_calls_j-t_calls_w)/t_calls_j*100:+.1f}%")
print(f"  Total:  {(total_j-total_w)/total_j*100:+.1f}%")

# E2-A summary (if available)
if USE_API and results_e2a["json"].get("total"):
    j_tot = results_e2a["json"]["total"]
    w_tot = results_e2a["wcy"]["total"]
    print(f"\n📊 E2-A: 3-Agent Pipeline")
    print(f"  JSON total: {j_tot} tokens")
    print(f"  WCY total:  {w_tot} tokens")
    print(f"  Savings:    {(j_tot-w_tot)/j_tot*100:+.1f}%")

print(f"\n{'='*70}")
print(f"  SUCCESS CRITERIA CHECK")
print(f"{'='*70}")
print(f"  [1] Token efficiency ≥30% in pipeline:  {'TBD' if not USE_API else ('✓' if (j_tot-w_tot)/j_tot*100 >= 30 else '✗')}")
print(f"  [2] Scale savings improve with N:        {'✓' if scale_results[-1]['sav_hyb_vs_pretty'] >= scale_results[0]['sav_hyb_vs_pretty'] else '✗'}")
print(f"  [3] Tool call savings ≥35%:              {'✓' if (total_j-total_w)/total_j*100 >= 35 else '✗'}")

print("\n✅ Sprint 1 complete")
