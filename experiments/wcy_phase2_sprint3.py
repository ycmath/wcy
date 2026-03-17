# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  WCY Phase 2 — Sprint 3
  Protocol Integration & Cross-Model Validation
═══════════════════════════════════════════════════════════════

Experiments:
  E2-D: Verification Overhead Reduction
        — 생성+검증 2단계 파이프라인의 reformatting 비용 측정
  E4-A: MCP Message Format Replacement
        — 실제 MCP 교환 사이클 JSON vs WCY 토큰 비교
  E1-C: Model Size Ladder
        — Claude Haiku vs Sonnet 동일 태스크 포맷 적응력 비교

측정 방법 (공정한 비교):
  - WCY 스펙 (~220 tokens)은 시스템 프롬프트에 포함되지만
    측정에서 제외함 (세션 고정 비용 = amortized overhead)
  - 측정 대상: payload tokens = input_tokens - count(system_prompt)
  - JSON 조건도 동일하게 system prompt 토큰 제외

실행 환경: Colab Pro (Python 3.10+)
필요: ANTHROPIC_API_KEY (코드 내 직접 입력)
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
count = lambda t: len(enc.encode(str(t)))

# ↓↓↓ 여기에 API 키를 직접 입력하세요 ↓↓↓
API_KEY = "sk-ant-api03-YOUR_KEY_HERE"
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

import anthropic
client = anthropic.Anthropic(api_key=API_KEY)
USE_API = True
print("✓ Setup: API=ON")

SONNET = "claude-sonnet-4-20250514"
HAIKU  = "claude-haiku-4-5-20251001"

# ── WCY Spec (시스템 프롬프트 고정 비용 — 측정 제외) ──────────────
WCY_SPEC = dedent("""
WCY FORMAT SPEC:
PHASE MARKERS (line-start, followed by space):
  .  observe  — confirmed data/fact
  :  infer    — derived conclusion (conf=, from=N)
  >  act      — output, call, side effect
  ~  meta     — schema / context declaration
  !  exception — error, missing data

SLOTS (space-separated):
  tag=value    named slot
  bare_value   positional slot
  ?tag         void-B: unexplored candidate (hint=, conf_range=)
  from=N,M     evidence trail (derives from lines N, M)
  conf=0.xx    confidence 0.0–1.0
""").strip()

WCY_SPEC_TOKENS = count(WCY_SPEC)
print(f"WCY spec (excluded from measurement): {WCY_SPEC_TOKENS} tokens")

def call_api(system, user, model=SONNET, max_tokens=1000, label=""):
    """
    API 호출 + payload 토큰만 반환.
    payload_in = input_tokens - system_prompt_tokens (스펙 주입 비용 제외)
    """
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}]
    )
    raw_in  = msg.usage.input_tokens
    raw_out = msg.usage.output_tokens
    text    = msg.content[0].text
    sys_tok = count(system)
    payload_in = raw_in - sys_tok   # 스펙 제외한 실제 페이로드
    return text, payload_in, raw_out, raw_in


###############################################################
# CELL 2: E2-D — Verification Overhead Reduction
#
# 가설: WCY의 phase marker가 이미 의도를 인코딩하므로
#       검증 에이전트가 reformatting 없이 바로 처리 가능.
#       → 검증 단계 입력 토큰 절감.
#
# 설계:
#   Generator → code + annotations
#   Verifier  → receives Generator output, produces verdict
#
#   JSON 경로: Generator → JSON metadata → Verifier parses JSON
#   WCY 경로:  Generator → WCY lines    → Verifier reads WCY directly
#
# 공정한 비교:
#   - Generator 시스템 프롬프트 토큰 제외
#   - Verifier 시스템 프롬프트 토큰 제외
#   - 측정: Generator output tokens + Verifier payload input tokens
###############################################################

print("\n" + "=" * 70)
print("  E2-D: VERIFICATION OVERHEAD REDUCTION")
print("=" * 70)

# ── 5개 코드 태스크 ───────────────────────────────────────────
CODE_TASKS = [
    {
        "name": "fibonacci",
        "prompt": "Write a Python function fibonacci(n) that returns the nth Fibonacci number.",
        "expected_issues": ["exponential complexity", "no memoization", "stack overflow risk"]
    },
    {
        "name": "binary_search",
        "prompt": "Write a Python function binary_search(arr, target) that returns the index or -1.",
        "expected_issues": ["off-by-one", "empty array", "duplicate handling"]
    },
    {
        "name": "rate_limiter",
        "prompt": "Write a Python class RateLimiter that allows max N requests per second.",
        "expected_issues": ["thread safety", "clock precision", "memory leak"]
    },
    {
        "name": "csv_parser",
        "prompt": "Write a Python function parse_csv(text) that returns a list of dicts.",
        "expected_issues": ["quoted fields", "newlines in fields", "encoding"]
    },
    {
        "name": "lru_cache",
        "prompt": "Write a Python class LRUCache(capacity) with get(key) and put(key, value).",
        "expected_issues": ["O(1) requirement", "eviction order", "negative capacity"]
    }
]

# ── 시스템 프롬프트 ───────────────────────────────────────────

GEN_SYS_JSON = """You are a code generation agent. Given a coding task:
1. Write the Python code
2. Output a JSON object with keys:
   - "code": the complete Python code as a string
   - "complexity": time complexity (e.g. "O(n)")
   - "known_issues": list of known limitations or edge cases
   - "confidence": float 0.0-1.0
Do not include markdown fences. Output valid JSON only."""

GEN_SYS_WCY = f"""You are a code generation agent. Given a coding task, output in WCY format:
{WCY_SPEC}
Example output structure:
~ task  <name>  lang=python
. code  <the complete function or class, on one line using semicolons if needed>
: complexity=<O(n)>  conf=<0.0-1.0>
: ?known_issue  hint=<specific_concern>  conf_range=0.5..0.9
> ready  from=1"""

VER_SYS_JSON = """You are a code verification agent. You receive a JSON object from a generator agent.
Analyze the code and metadata. Output a JSON verdict with keys:
- "correctness": true/false
- "issues_confirmed": list of issues you found (can overlap with known_issues)
- "issues_missed": issues the generator missed
- "verdict": "approve" | "revise" | "reject"
- "confidence": float 0.0-1.0"""

VER_SYS_WCY = f"""You are a code verification agent. You receive WCY output from a generator agent.
Analyze and output a WCY verdict.
{WCY_SPEC}
Use from=N to reference specific generator output lines you are verifying."""

# ── 실험 실행 ─────────────────────────────────────────────────
e2d_results = []

print(f"\n{'Task':<18} {'Gen out':>8} {'Ver payload_in':>15} {'Total':>8}  (JSON vs WCY)")
print("─" * 70)

for task in CODE_TASKS:
    row = {"name": task["name"]}

    # ── JSON 경로 ─────────────────────────────────────────────
    gen_json_resp, gen_json_pay_in, gen_json_out, _ = call_api(
        GEN_SYS_JSON, task["prompt"], label=f"{task['name']}_gen_json"
    )
    time.sleep(0.5)

    # Verifier receives Generator's JSON output as its user message
    ver_json_resp, ver_json_pay_in, ver_json_out, _ = call_api(
        VER_SYS_JSON,
        f"Generator output:\n{gen_json_resp}\n\nOriginal task: {task['prompt']}",
        label=f"{task['name']}_ver_json"
    )
    time.sleep(0.5)

    # ── WCY 경로 ─────────────────────────────────────────────
    gen_wcy_resp, gen_wcy_pay_in, gen_wcy_out, _ = call_api(
        GEN_SYS_WCY, task["prompt"], label=f"{task['name']}_gen_wcy"
    )
    time.sleep(0.5)

    # Verifier receives Generator's WCY output — no reformatting needed
    ver_wcy_resp, ver_wcy_pay_in, ver_wcy_out, _ = call_api(
        VER_SYS_WCY,
        f"Generator output:\n{gen_wcy_resp}\n\nOriginal task: {task['prompt']}",
        label=f"{task['name']}_ver_wcy"
    )
    time.sleep(0.5)

    # 총합: Generator output tokens + Verifier payload input tokens
    # (Generator의 payload_in은 비교 대상 아님 — 동일 프롬프트이므로)
    json_total = gen_json_out + ver_json_pay_in + ver_json_out
    wcy_total  = gen_wcy_out  + ver_wcy_pay_in  + ver_wcy_out
    savings    = (json_total - wcy_total) / json_total * 100

    row.update({
        "gen_json_out": gen_json_out, "gen_wcy_out": gen_wcy_out,
        "ver_json_pay_in": ver_json_pay_in, "ver_wcy_pay_in": ver_wcy_pay_in,
        "ver_json_out": ver_json_out, "ver_wcy_out": ver_wcy_out,
        "json_total": json_total, "wcy_total": wcy_total, "savings": savings,
        "gen_json_resp": gen_json_resp, "gen_wcy_resp": gen_wcy_resp,
        "ver_json_resp": ver_json_resp, "ver_wcy_resp": ver_wcy_resp,
    })
    e2d_results.append(row)

    print(f"  {task['name']:<16} "
          f"J:{gen_json_out:>4}/W:{gen_wcy_out:>4}  "
          f"J:{ver_json_pay_in:>4}/W:{ver_wcy_pay_in:>4}  "
          f"J:{json_total:>4}/W:{wcy_total:>4}  {savings:>+5.1f}%")

# 요약
print(f"\n{'='*70}")
print(f"  E2-D SUMMARY (payload tokens, spec excluded)")
print(f"  {'Metric':<35} {'JSON':>7} {'WCY':>7} {'Savings':>9}")
print(f"  {'─'*60}")

avg_gen_json = sum(r["gen_json_out"]     for r in e2d_results) / len(e2d_results)
avg_gen_wcy  = sum(r["gen_wcy_out"]      for r in e2d_results) / len(e2d_results)
avg_ver_jin  = sum(r["ver_json_pay_in"]  for r in e2d_results) / len(e2d_results)
avg_ver_win  = sum(r["ver_wcy_pay_in"]   for r in e2d_results) / len(e2d_results)
avg_ver_jout = sum(r["ver_json_out"]     for r in e2d_results) / len(e2d_results)
avg_ver_wout = sum(r["ver_wcy_out"]      for r in e2d_results) / len(e2d_results)
avg_jtot     = sum(r["json_total"]       for r in e2d_results) / len(e2d_results)
avg_wtot     = sum(r["wcy_total"]        for r in e2d_results) / len(e2d_results)

for label, jv, wv in [
    ("Generator output (avg)",          avg_gen_json,  avg_gen_wcy),
    ("Verifier payload input (avg)",     avg_ver_jin,   avg_ver_win),
    ("Verifier output (avg)",            avg_ver_jout,  avg_ver_wout),
    ("Total (gen_out+ver_in+ver_out)",   avg_jtot,      avg_wtot),
]:
    sav = (jv - wv) / jv * 100 if jv > 0 else 0
    print(f"  {label:<35} {jv:>7.0f} {wv:>7.0f} {sav:>+8.1f}%")

avg_sav = (avg_jtot - avg_wtot) / avg_jtot * 100
print(f"\n  성공 기준: Verifier input ≥20% 절감 → {'✓ PASS' if (avg_ver_jin-avg_ver_win)/avg_ver_jin*100 >= 20 else '✗ FAIL'}")
print(f"  핵심 질문: WCY에서 from= 역추적이 검증 구조화에 사용되는가?")

# WCY verifier 응답 샘플 출력
print(f"\n  WCY Verifier 응답 샘플 ({e2d_results[0]['name']}):")
for line in e2d_results[0]['ver_wcy_resp'].split('\n')[:10]:
    print(f"    {line}")

print(f"\n  from= 사용 횟수 (WCY verifier, 5 tasks):")
for r in e2d_results:
    n_from = len(re.findall(r'from=', r['ver_wcy_resp']))
    print(f"    {r['name']:<18}: {n_from} from= references")


###############################################################
# CELL 3: E4-A — MCP Message Format Replacement
#
# 실제 MCP 교환 사이클 10개를 JSON vs WCY로 표현.
# API 호출 불필요 — 순수 토큰 카운트 비교.
# 각 MCP 메시지 유형별로 구조화된 비교.
###############################################################

print("\n\n" + "=" * 70)
print("  E4-A: MCP MESSAGE FORMAT REPLACEMENT")
print("=" * 70)
print("  [토큰 카운트만 — API 불필요]")

# ── MCP 메시지 쌍 정의 (JSON vs WCY 등가 표현) ────────────────

MCP_EXCHANGES = [

    {
        "name": "1. tools/list request",
        "json": json.dumps({
            "jsonrpc": "2.0", "method": "tools/list", "id": 1
        }),
        "wcy": "> tools/list  id=1"
    },

    {
        "name": "2. tools/list response (5 tools)",
        "json": json.dumps({
            "jsonrpc": "2.0", "id": 1, "result": {
                "tools": [
                    {"name": "get_patient_record", "description": "Retrieve patient medical record",
                     "inputSchema": {"type": "object", "properties": {
                         "patient_id": {"type": "string"}, "fields": {"type": "array"}}, "required": ["patient_id"]}},
                    {"name": "order_lab_test", "description": "Order a laboratory test",
                     "inputSchema": {"type": "object", "properties": {
                         "patient_id": {"type": "string"}, "test_name": {"type": "string"},
                         "priority": {"type": "string", "enum": ["stat", "urgent", "routine"]}}, "required": ["patient_id", "test_name"]}},
                    {"name": "prescribe_medication", "description": "Prescribe medication to patient",
                     "inputSchema": {"type": "object", "properties": {
                         "patient_id": {"type": "string"}, "medication": {"type": "string"},
                         "dose_mg": {"type": "number"}, "frequency": {"type": "string"},
                         "duration_days": {"type": "integer"}}, "required": ["patient_id", "medication", "dose_mg"]}},
                    {"name": "send_alert", "description": "Send clinical alert",
                     "inputSchema": {"type": "object", "properties": {
                         "recipient": {"type": "string"}, "priority": {"type": "string"},
                         "message": {"type": "string"}}, "required": ["recipient", "priority", "message"]}},
                    {"name": "schedule_followup", "description": "Schedule patient follow-up",
                     "inputSchema": {"type": "object", "properties": {
                         "patient_id": {"type": "string"}, "days_from_now": {"type": "integer"},
                         "department": {"type": "string"}, "reason": {"type": "string"}},
                         "required": ["patient_id", "days_from_now", "department"]}},
                ]
            }
        }, indent=2),
        "wcy": dedent("""
            ~ tool  get_patient_record
              . patient_id  string  required
              . fields  array
            ~ tool  order_lab_test
              . patient_id  string  required
              . test_name  string  required
              . priority  string  stat|urgent|routine
            ~ tool  prescribe_medication
              . patient_id  string  required
              . medication  string  required
              . dose_mg  number  required
              . frequency  string
              . duration_days  integer
            ~ tool  send_alert
              . recipient  string  required
              . priority  string  required
              . message  string  required
            ~ tool  schedule_followup
              . patient_id  string  required
              . days_from_now  integer  required
              . department  string  required
              . reason  string
        """).strip()
    },

    {
        "name": "3. tools/call — get_patient_record",
        "json": json.dumps({
            "jsonrpc": "2.0", "method": "tools/call", "id": 2,
            "params": {"name": "get_patient_record",
                       "arguments": {"patient_id": "P-1001", "fields": ["vitals", "medications", "diagnosis"]}}
        }),
        "wcy": "> tools/call  get_patient_record  id=2\n  . patient_id=P-1001\n  . fields  vitals  medications  diagnosis"
    },

    {
        "name": "4. tools/call result — patient record",
        "json": json.dumps({
            "jsonrpc": "2.0", "id": 2, "result": {
                "content": [{"type": "text", "text": json.dumps({
                    "patient_id": "P-1001", "name": "Kim", "age": 45,
                    "vitals": {"temperature": 38.5, "heart_rate": 92, "blood_pressure": "128/84"},
                    "medications": [{"name": "acetaminophen", "dose_mg": 500, "frequency": "q6h"},
                                    {"name": "loratadine", "dose_mg": 10, "frequency": "qd"}],
                    "diagnosis": {"primary": "influenza", "confidence": 0.78}
                })}]
            }
        }, indent=2),
        "wcy": dedent("""
            . result  id=2  tool=get_patient_record
            . patient  P-1001  Kim  age=45
            . vitals  temp=38.5  hr=92  bp=128/84
            . med  acetaminophen  500  q6h
            . med  loratadine  10  qd
            : diagnosis=influenza  conf=0.78
        """).strip()
    },

    {
        "name": "5. tools/call — order_lab_test",
        "json": json.dumps({
            "jsonrpc": "2.0", "method": "tools/call", "id": 3,
            "params": {"name": "order_lab_test",
                       "arguments": {"patient_id": "P-1001", "test_name": "CBC", "priority": "stat"}}
        }),
        "wcy": "> tools/call  order_lab_test  id=3\n  . patient_id=P-1001\n  . test_name=CBC  priority=stat"
    },

    {
        "name": "6. tools/call — prescribe_medication",
        "json": json.dumps({
            "jsonrpc": "2.0", "method": "tools/call", "id": 4,
            "params": {"name": "prescribe_medication",
                       "arguments": {"patient_id": "P-1001", "medication": "oseltamivir",
                                     "dose_mg": 75, "frequency": "q12h", "duration_days": 5}}
        }),
        "wcy": "> tools/call  prescribe_medication  id=4\n  . patient_id=P-1001\n  . medication=oseltamivir  dose=75  freq=q12h  days=5"
    },

    {
        "name": "7. error response",
        "json": json.dumps({
            "jsonrpc": "2.0", "id": 5,
            "error": {"code": -32602, "message": "Invalid params",
                      "data": {"field": "patient_id", "reason": "Patient P-9999 not found"}}
        }),
        "wcy": "! error  id=5  code=-32602\n  . field=patient_id\n  . reason  patient_P-9999_not_found"
    },

    {
        "name": "8. prompts/get request",
        "json": json.dumps({
            "jsonrpc": "2.0", "method": "prompts/get", "id": 6,
            "params": {"name": "clinical_assessment",
                       "arguments": {"patient_id": "P-1001", "context": "fever_workup"}}
        }),
        "wcy": "> prompts/get  clinical_assessment  id=6\n  . patient_id=P-1001  context=fever_workup"
    },

    {
        "name": "9. resources/read request",
        "json": json.dumps({
            "jsonrpc": "2.0", "method": "resources/read", "id": 7,
            "params": {"uri": "clinical://protocols/fever-management/v2"}
        }),
        "wcy": "> resources/read  clinical://protocols/fever-management/v2  id=7"
    },

    {
        "name": "10. send_alert + schedule_followup (batch)",
        "json": json.dumps([
            {"jsonrpc": "2.0", "method": "tools/call", "id": 8,
             "params": {"name": "send_alert",
                        "arguments": {"recipient": "Dr. Lee", "priority": "medium",
                                      "message": "P-1001 prescribed oseltamivir, follow-up in 7d"}}},
            {"jsonrpc": "2.0", "method": "tools/call", "id": 9,
             "params": {"name": "schedule_followup",
                        "arguments": {"patient_id": "P-1001", "days_from_now": 7,
                                      "department": "pulmonology", "reason": "post-influenza follow-up"}}}
        ], indent=2),
        "wcy": dedent("""
            > tools/call  send_alert  id=8
              . recipient=Dr.Lee  priority=medium
              . message  P-1001_prescribed_oseltamivir_followup_7d
            > tools/call  schedule_followup  id=9
              . patient_id=P-1001  days=7
              . department=pulmonology  reason=post-influenza_followup
        """).strip()
    },
]

# ── 토큰 카운트 ───────────────────────────────────────────────
print(f"\n  {'Exchange':<45} {'JSON':>7} {'WCY':>7} {'Savings':>9}")
print(f"  {'─'*72}")

e4a_results = []
for ex in MCP_EXCHANGES:
    tj = count(ex["json"])
    tw = count(ex["wcy"])
    sav = (tj - tw) / tj * 100
    e4a_results.append({"name": ex["name"], "json": tj, "wcy": tw, "sav": sav})
    print(f"  {ex['name']:<45} {tj:>7} {tw:>7} {sav:>+8.1f}%")

# 합계 및 요약
total_j = sum(r["json"] for r in e4a_results)
total_w = sum(r["wcy"]  for r in e4a_results)
total_sav = (total_j - total_w) / total_j * 100
print(f"  {'─'*72}")
print(f"  {'TOTAL (10 exchanges)':<45} {total_j:>7} {total_w:>7} {total_sav:>+8.1f}%")

# 카테고리별 분류
request_idx  = [0, 2, 4, 5, 7, 8]  # requests
response_idx = [1, 3]               # responses
error_idx    = [6]
batch_idx    = [9]

for cat_name, idxs in [
    ("Requests (call/get/read)", request_idx),
    ("Responses (result data)", response_idx),
    ("Error messages",           error_idx),
    ("Batch calls",              batch_idx),
]:
    cj = sum(e4a_results[i]["json"] for i in idxs)
    cw = sum(e4a_results[i]["wcy"]  for i in idxs)
    print(f"  {cat_name:<40}: {(cj-cw)/cj*100:>+6.1f}%  (J={cj}, W={cw})")

print(f"\n  성공 기준: 전체 MCP 교환 ≥40% 절감 → {'✓ PASS' if total_sav >= 40 else '✗ FAIL'}")
print(f"  성공 기준: 스키마(tools/list) ≥50% 절감 → {'✓ PASS' if e4a_results[1]['sav'] >= 50 else '✗ FAIL'}")

# ── API로 파싱 정확도 검증 (선택) ─────────────────────────────
print(f"\n[WCY MCP 파싱 정확도 검증 — API]")

MCP_PARSE_SYS_JSON = """You are an MCP message parser. Given JSON-RPC messages, extract:
1. For each message: method/type, id, tool name (if applicable), parameters/results
Output as a simple list."""

MCP_PARSE_SYS_WCY = f"""You are an MCP message parser. Given WCY-encoded MCP messages, extract:
1. For each message: method/type, id, tool name (if applicable), parameters/results
Output as a simple list.
{WCY_SPEC}"""

# 전체 10개 교환을 하나의 입력으로 합치기
all_json = "\n---\n".join(ex["json"] for ex in MCP_EXCHANGES)
all_wcy  = "\n---\n".join(ex["wcy"]  for ex in MCP_EXCHANGES)

Q = "Extract: for each of the 10 MCP exchanges, state (1) the operation type, (2) the id, (3) tool name if any, (4) key parameter or result."

json_resp, json_pay_in, json_out, _ = call_api(
    MCP_PARSE_SYS_JSON, f"{all_json}\n\n{Q}", max_tokens=800, label="mcp_parse_json"
)
time.sleep(1)
wcy_resp, wcy_pay_in, wcy_out, _ = call_api(
    MCP_PARSE_SYS_WCY, f"{all_wcy}\n\n{Q}", max_tokens=800, label="mcp_parse_wcy"
)

print(f"  Parse task — payload input:  JSON={json_pay_in}  WCY={wcy_pay_in}  Δ={(json_pay_in-wcy_pay_in)/json_pay_in*100:+.1f}%")
print(f"  Parse task — output tokens:  JSON={json_out}  WCY={wcy_out}")

# 10개 항목 모두 추출했는지 대략 확인
def count_extractions(resp):
    nums = set(re.findall(r'\b([1-9]|10)\b', resp))
    ids  = re.findall(r'id[=: ]+(\d+)', resp.lower())
    return len(ids)

json_ex = count_extractions(json_resp)
wcy_ex  = count_extractions(wcy_resp)
print(f"  Extracted IDs found:         JSON≈{json_ex}/10  WCY≈{wcy_ex}/10")


###############################################################
# CELL 4: E1-C — Model Size Ladder
#
# 동일 태스크를 Haiku vs Sonnet으로 실행.
# 측정: (1) WCY 형식 준수율, (2) 포맷별 토큰 절감 패턴,
#       (3) 소형 모델에서 WCY 이득이 더 큰지 (용량 절약 가설)
#
# 공정한 비교:
#   - 동일 시스템 프롬프트, 동일 payload
#   - payload_in = input_tokens - system_tokens
###############################################################

print("\n\n" + "=" * 70)
print("  E1-C: MODEL SIZE LADDER — Haiku vs Sonnet")
print("=" * 70)

# ── 테스트 케이스 (Phase 1과 동일 난이도 중간) ─────────────────
TEST_CASES_E1C = [
    {
        "name": "Read — medium",
        "type": "read",
        "data_json": json.dumps({
            "patients": [
                {"id": "P-1001", "name": "Kim", "age": 45, "temperature": 38.5,
                 "symptoms": ["fever", "cough", "fatigue"],
                 "medications": [{"name": "acetaminophen", "dose_mg": 500}],
                 "diagnosis": {"primary": "influenza", "confidence": 0.78}},
                {"id": "P-1002", "name": "Lee", "age": 62, "temperature": 37.1,
                 "symptoms": ["headache", "dizziness"],
                 "medications": [{"name": "lisinopril", "dose_mg": 10}, {"name": "atorvastatin", "dose_mg": 20}],
                 "diagnosis": {"primary": "hypertension", "confidence": 0.91}},
                {"id": "P-1003", "name": "Park", "age": 38, "temperature": 39.2,
                 "symptoms": ["fever", "rash", "joint_pain"],
                 "medications": [],
                 "diagnosis": {"primary": "viral_infection", "confidence": 0.55}},
            ]
        }, indent=2),
        "data_wcy": dedent("""
            ~ patient:id,name,age,temp  symptom:list  med:name,dose  dx:primary,conf
            . patient  P-1001  Kim  45  38.5
            . symptoms  fever  cough  fatigue
            . med  acetaminophen  500
            : diagnosis=influenza  conf=0.78

            . patient  P-1002  Lee  62  37.1
            . symptoms  headache  dizziness
            . med  lisinopril  10
            . med  atorvastatin  20
            : diagnosis=hypertension  conf=0.91

            . patient  P-1003  Park  38  39.2
            . symptoms  fever  rash  joint_pain
            . med  none
            : diagnosis=viral_infection  conf=0.55
        """).strip(),
        "question": "Which patient has the highest temperature? What is their diagnosis and confidence?",
        "answer_key": ["park", "p-1003", "viral", "0.55"]
    },
    {
        "name": "Generate — medium",
        "type": "generate",
        "schema": "patient record: id, name, age, temperature, 2-3 symptoms, 1-2 medications, primary diagnosis with confidence",
        "answer_key": []
    },
    {
        "name": "Infer — multi-hop",
        "type": "infer",
        "data_json": json.dumps({
            "observation": {"patient": "Choi", "age": 55, "bp": "165/100", "bmi": 31.2,
                            "family_history": "hypertension", "smoking": "yes"},
            "lab_results": {"fasting_glucose": 128, "HbA1c": 6.4, "total_cholesterol": 245},
            "lifestyle": {"exercise": "none", "diet": "high_sodium"}
        }, indent=2),
        "data_wcy": dedent("""
            . patient=Choi  age=55  bp=165/100  bmi=31.2
            . history  family_htn  smoker
            . labs  glucose=128  HbA1c=6.4  cholesterol=245
            . lifestyle  exercise=none  diet=high_sodium
        """).strip(),
        "question": "What are the top 2 cardiovascular risk factors? What single intervention would have the greatest impact?",
        "answer_key": ["hypertension", "smoking", "blood pressure", "bp"]
    }
]

READ_SYS_JSON  = "You are a medical data analyst. Answer questions about patient data provided as JSON. Be concise."
READ_SYS_WCY   = f"You are a medical data analyst. Answer questions about patient data in WCY format. Be concise.\n{WCY_SPEC}"
GEN_SYS_JSON2  = "You are a medical record generator. Generate a patient record as JSON matching the given schema."
GEN_SYS_WCY2   = f"You are a medical record generator. Generate a patient record in WCY format matching the given schema.\n{WCY_SPEC}"
INFER_SYS_JSON = "You are a clinical decision support system. Given patient data as JSON, answer the clinical question. Be concise."
INFER_SYS_WCY  = f"You are a clinical decision support system. Given patient data in WCY format, answer the clinical question. Be concise.\n{WCY_SPEC}"

def check_answer(response, answer_keys):
    resp_lower = response.lower()
    return sum(1 for k in answer_keys if k.lower() in resp_lower), len(answer_keys)

e1c_results = []

for model_name, model_id in [("Sonnet", SONNET), ("Haiku", HAIKU)]:
    print(f"\n  ── {model_name} ({model_id}) ──")
    model_rows = []

    for tc in TEST_CASES_E1C:
        row = {"model": model_name, "task": tc["name"]}

        if tc["type"] == "read":
            sys_j, sys_w = READ_SYS_JSON, READ_SYS_WCY
            user_j = f"Data:\n{tc['data_json']}\n\nQuestion: {tc['question']}"
            user_w = f"Data:\n{tc['data_wcy']}\n\nQuestion: {tc['question']}"
        elif tc["type"] == "generate":
            sys_j, sys_w = GEN_SYS_JSON2, GEN_SYS_WCY2
            user_j = user_w = f"Generate a patient record: {tc['schema']}"
        else:  # infer
            sys_j, sys_w = INFER_SYS_JSON, INFER_SYS_WCY
            user_j = f"Data:\n{tc['data_json']}\n\nQuestion: {tc['question']}"
            user_w = f"Data:\n{tc['data_wcy']}\n\nQuestion: {tc['question']}"

        # JSON
        resp_j, pay_in_j, out_j, _ = call_api(sys_j, user_j, model=model_id, max_tokens=400, label=f"{model_name}_{tc['name']}_json")
        time.sleep(0.5)
        # WCY
        resp_w, pay_in_w, out_w, _ = call_api(sys_w, user_w, model=model_id, max_tokens=400, label=f"{model_name}_{tc['name']}_wcy")
        time.sleep(0.5)

        # 정확도 체크 (read/infer만)
        acc_j = acc_w = None
        if tc["answer_key"]:
            n_j, tot = check_answer(resp_j, tc["answer_key"])
            n_w, _   = check_answer(resp_w, tc["answer_key"])
            acc_j = n_j / tot * 100
            acc_w = n_w / tot * 100

        # WCY 형식 준수율 (generate만)
        wcy_compliance = None
        if tc["type"] == "generate":
            valid_lines = sum(1 for l in resp_w.split('\n')
                              if l.strip() and l.strip()[0] in '.!:>~' and len(l.strip()) > 1 and l.strip()[1] == ' ')
            total_lines = sum(1 for l in resp_w.split('\n') if l.strip())
            wcy_compliance = valid_lines / max(total_lines, 1) * 100

        row.update({
            "pay_in_j": pay_in_j, "pay_in_w": pay_in_w,
            "out_j": out_j, "out_w": out_w,
            "acc_j": acc_j, "acc_w": acc_w,
            "wcy_compliance": wcy_compliance,
            "resp_j": resp_j, "resp_w": resp_w
        })
        model_rows.append(row)

        sav_in  = (pay_in_j - pay_in_w) / pay_in_j * 100 if pay_in_j > 0 else 0
        sav_out = (out_j - out_w) / out_j * 100 if out_j > 0 else 0
        acc_str = f"J:{acc_j:.0f}%/W:{acc_w:.0f}%" if acc_j is not None else (f"compliance={wcy_compliance:.0f}%" if wcy_compliance is not None else "")
        print(f"    {tc['name']:<22} in:{sav_in:>+5.1f}%  out:{sav_out:>+5.1f}%  {acc_str}")

    e1c_results.extend(model_rows)

# ── 모델별 비교 요약 ──────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  E1-C SUMMARY: Haiku vs Sonnet")
print(f"  {'Task':<22} {'Sonnet in_sav':>14} {'Haiku in_sav':>13} {'Sonnet out_sav':>15} {'Haiku out_sav':>14}")
print(f"  {'─'*80}")

for tc in TEST_CASES_E1C:
    s_rows = [r for r in e1c_results if r["model"]=="Sonnet" and r["task"]==tc["name"]]
    h_rows = [r for r in e1c_results if r["model"]=="Haiku"  and r["task"]==tc["name"]]
    if s_rows and h_rows:
        s, h = s_rows[0], h_rows[0]
        s_sav_in  = (s["pay_in_j"]-s["pay_in_w"])/s["pay_in_j"]*100 if s["pay_in_j"]>0 else 0
        h_sav_in  = (h["pay_in_j"]-h["pay_in_w"])/h["pay_in_j"]*100 if h["pay_in_j"]>0 else 0
        s_sav_out = (s["out_j"]-s["out_w"])/s["out_j"]*100 if s["out_j"]>0 else 0
        h_sav_out = (h["out_j"]-h["out_w"])/h["out_j"]*100 if h["out_j"]>0 else 0
        print(f"  {tc['name']:<22} {s_sav_in:>+13.1f}% {h_sav_in:>+12.1f}% {s_sav_out:>+14.1f}% {h_sav_out:>+13.1f}%")

# WCY 형식 준수율 비교
print(f"\n  WCY format compliance (generate task):")
for model_name in ["Sonnet", "Haiku"]:
    rows = [r for r in e1c_results if r["model"]==model_name and r["wcy_compliance"] is not None]
    if rows:
        avg_c = sum(r["wcy_compliance"] for r in rows) / len(rows)
        print(f"    {model_name}: {avg_c:.1f}%")

# 정확도 비교
print(f"\n  Answer accuracy (read+infer tasks):")
for model_name in ["Sonnet", "Haiku"]:
    rows = [r for r in e1c_results if r["model"]==model_name and r["acc_j"] is not None]
    if rows:
        avg_j = sum(r["acc_j"] for r in rows) / len(rows)
        avg_w = sum(r["acc_w"] for r in rows) / len(rows)
        print(f"    {model_name}: JSON={avg_j:.0f}%  WCY={avg_w:.0f}%  Δ={avg_w-avg_j:+.0f}%")

print(f"\n  성공 기준: 두 모델 모두 WCY 정확도 ≥90%")
for model_name in ["Sonnet", "Haiku"]:
    rows = [r for r in e1c_results if r["model"]==model_name and r["acc_w"] is not None]
    if rows:
        avg_w = sum(r["acc_w"] for r in rows) / len(rows)
        print(f"    {model_name} WCY accuracy {avg_w:.0f}% → {'✓ PASS' if avg_w >= 90 else '✗ FAIL'}")


###############################################################
# CELL 5: Sprint 3 Final Summary
###############################################################

print("\n\n" + "═" * 70)
print("  SPRINT 3 — FINAL SUMMARY")
print("═" * 70)

print("""
측정 방법:
  - WCY 스펙 주입 비용(~220 tokens)은 모든 실험에서 측정 제외
  - payload_in = input_tokens - count(system_prompt)
  - 동일 시스템 프롬프트 제외 원칙을 JSON/WCY 양 조건에 동등 적용

실험 결과:
  E2-D: Generator → Verifier 파이프라인
        핵심 질문: WCY from= 역추적이 검증 단계를 구조화하는가?
        (결과 위 참조)

  E4-A: MCP 교환 10개 토큰 비교
        tools/list 스키마: 단일 최대 절감 항목
        (결과 위 참조)

  E1-C: Haiku vs Sonnet
        소형 모델에서 WCY 이득이 더 큰가? (용량 절약 가설)
        (결과 위 참조)

다음 단계:
  모든 Sprint 결과 통합 → Position Paper v1.1 (cross-model 검증 섹션 추가)
  또는: ASMT 통합 경로 (경로 B) 개시
""")
