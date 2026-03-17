# -*- coding: utf-8 -*-
"""
wcy_phase4_void_cycle.py — ?tag Resolution Cycle v1.0
======================================================

Phase 4: ?tag (void-B)를 자기인식과 학습의 최소 단위로 검증

핵심 주장:
  ?tag = 구조화된 가추 요청 (Peirce's abduction)
  ?tag 해소 사이클 = WCY의 완전한 한 바퀴 (Watch→Compute→Yield)
  충분한 ?tag 해소 trace → 미래 모델의 자기반성 능력 seed

void-B의 구조:
  : ?tag  hint=탐색방향  conf_range=기대범위
       ↓ [탐색/관찰]
  . new_observation
       ↓
  : resolved_tag=value  conf=0.xx  from=void_line,obs_line

실험:
  E7-A: ?tag 해소 사이클 trace 생성 (5개 도메인, 다단계)
  E7-B: ?tag가 있는 추론 vs 없는 추론 — 결론 품질 비교
  E7-C: void-B cascade (한 ?tag 해소가 새 ?tag를 낳는 체인)

산출물:
  wcy_void_cycles.jsonl  ← ?tag 해소 사이클 학습 데이터
  wcy_phase4_report.md   ← Phase 4 이론적 요약

실행: Colab Pro, wcy_parser.py 필요
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
    print("⚠ wcy_parser.py를 같은 디렉토리에 업로드하세요")

# ↓↓↓ API 키 직접 입력 ↓↓↓
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
# ?tag가 포함된 추론의 완전한 형식
###############################################################

WCY_VOID_SPEC = dedent("""
WCY REASONING FORMAT — ?tag 해소 사이클 포함

VOID-B (?tag) 규칙:
  1. ?tag는 현재 데이터로 답할 수 없는 질문에만 사용
  2. hint=는 탐색 방향을 구체적으로 지정
  3. conf_range=는 예상 결과의 범위를 제약
  4. ?tag는 반드시 이후에 해소되어야 함 (> 탐색 → . 관찰 → : 해소)
  5. 해소된 :는 from=으로 해당 ?tag 줄과 새 관찰 줄을 모두 참조

해소 사이클 패턴:
  : ?unknown  hint=탐색방향  conf_range=L..H    ← 1. 모름 표시
  > investigate  reason=from=void_line           ← 2. 탐색 행동
  . new_finding  result=value                    ← 3. 새 관찰
  : resolved=value  conf=0.xx  from=1,3          ← 4. 해소

핵심:
  - ?tag 없이는 탐색 행동(>)이 근거를 잃음
  - 탐색 없이는 관찰(.)이 방향을 잃음
  - 해소 없이는 성장이 없음
  이 사이클이 WCY의 Watch→Compute→Yield 한 바퀴
""").strip()


###############################################################
# FEW-SHOT: ?tag 해소 사이클 예시
# 이 예시들이 미래 모델의 핵심 학습 데이터
###############################################################

FEWSHOT_VOID_MEDICAL = dedent("""
예시 1: 원인불명 발열 — ?tag 2단계 해소

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

예시 2: 알고리즘 버그 — ?tag 해소 후 새 ?tag 발생

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
예시 3: 자기 지식의 한계 인식 — 철학적 ?tag

~ task  knowledge_boundary_assessment  domain=epistemology
. claim  "이 약물이 해당 질환에 효과적이다"
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
# E7-A: ?tag 해소 사이클 Trace 생성
#
# 목표: ?tag가 실제로 해소되는 다단계 추론 trace 생성
# 핵심 품질 기준: void_resolution_rate (생성된 ?tag가 해소된 비율)
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
        "task": "원인을 찾기 위한 단계적 WCY 추론. 모르는 것은 반드시 ?tag로 표시하고, 탐색 후 해소하라.",
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
        "task": "메모리 누수 원인을 단계적으로 추론하라. 확인되지 않은 것은 ?tag로 표시하고 탐색 행동을 포함하라.",
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
        "task": "실험 이상 현상의 원인을 WCY로 추론하라. 불확실한 가설은 ?tag로 표시하고 검증 방법을 제시하라.",
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
        "task": "시장 진입 가능성을 WCY로 추론하라. 알 수 없는 것은 ?tag로 명시하고, 의사결정 전 확인해야 할 것들을 구조화하라.",
        "expected_voids": ["regulatory", "partnership", "localization"],
    },
    {
        "domain": "philosophical",
        "name": "causal_inference_limit",
        "context": """. observation  cities_with_more_ice_cream_sales_have_more_drownings
. data  correlation_r=0.87  p_less_0.001  n=150_cities
. researcher_claim  "ice cream causes drowning"
. available_data  temperature_records  seasonal_swimming_data""",
        "task": "이 추론의 타당성을 WCY로 평가하라. 인과 주장의 각 전제를 ?tag로 검증하고 대안 설명을 구조화하라.",
        "expected_voids": ["confounding", "causality", "mechanism"],
    },
]

def measure_void_cycle_quality(response: str) -> dict:
    """
    ?tag 해소 사이클의 품질 측정.
    핵심: void_generated vs void_resolved 비율
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

    # void 생성 카운트
    void_lines = [l for l in parsed if l.is_void]
    void_tags  = [tag for l in void_lines for tag in l.void_tags]
    void_count = len(void_tags)

    # void 해소 카운트: 생성된 ?tag 이름이 이후 : 줄에 tag=로 등장하는지
    resolved_count = 0
    infer_lines = [l for l in parsed if l.phase == ':' and not l.is_void]
    for void_tag in void_tags:
        # ?tag 이름이 나중에 해소됐는지 확인
        for infer in infer_lines:
            if void_tag in infer.tags or void_tag.replace('_','') in str(infer.tags):
                resolved_count += 1
                break
            # 느슨한 매칭: 핵심 단어 포함
            tag_core = void_tag.split('_')[0]
            if any(tag_core in k or tag_core in v
                   for k, v in infer.tags.items()):
                resolved_count += 1
                break

    resolution_rate = resolved_count / max(void_count, 1)

    # 탐색 행동(>) 이 ?tag 이후에 있는가
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

예시:
{FEWSHOT_VOID_MEDICAL}

---

{FEWSHOT_VOID_EPISTEMIC}

규칙 (중요):
- ?tag를 생성하면 반드시 > 탐색 → . 관찰 → : 해소 순서를 완성하라
- 해소 불가능한 ?tag는 ! warning으로 표시하고 이유를 명시
- 모르는 것을 아는 척하지 마라 — ?tag가 자기인식의 증거다"""

e7a_traces = []

for i, task in enumerate(VOID_TASKS):
    print(f"\n  [{i+1}/{len(VOID_TASKS)}] {task['domain']}: {task['name']}")

    user = f"컨텍스트:\n{task['context']}\n\n태스크: {task['task']}"

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

    # WCY 응답 미리보기 (처음 200자)
    preview = resp[:200].replace('\n', ' / ')
    print(f"    {preview}...")
    time.sleep(1.5)

print(f"\n  E7-A 요약:")
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
print(f"\n  평균 ?tag 생성: {avg_gen:.1f}개/trace")
print(f"  평균 해소율:   {avg_res:.0%}")
print(f"  성공 기준: 해소율 ≥50% → {'✓' if avg_res >= 0.5 else '✗'}")


###############################################################
# E7-B: ?tag 있는 추론 vs 없는 추론
#
# 가설: ?tag를 쓰는 추론이 자신의 한계를 더 정직하게 표현하고
#       그 결과 실제 오류(hallucination)가 줄어든다.
#
# 설계:
#   같은 문제를 (A) ?tag 없이 (B) ?tag 포함하여 추론
#   외부 판정: 각 추론의 주장 중 틀린 것이 몇 개인가?
###############################################################

print("\n\n" + "=" * 70)
print("  E7-B: ?TAG vs NO-?TAG — HALLUCINATION RATE COMPARISON")
print("=" * 70)

HALLUCINATION_TASKS = [
    {
        "name": "drug_dosing",
        "input": """환자: 75세 여성, 체중 52kg, CKD stage 3 (eGFR 35)
처방: vancomycin 투여 필요 (MRSA 감염)
질문: vancomycin 용량과 모니터링 계획을 제시하라.""",
        "ground_truth_facts": [
            "vancomycin은 신장으로 배설됨",
            "CKD에서 용량 감량 필요",
            "AUC/MIC 기반 모니터링 권장 (AUC 400-600)",
            "loading dose는 감량 불필요 (통상 25-30mg/kg)",
            "maintenance interval 연장 필요 (q24-48h depending on eGFR)",
            "trough monitoring이 전통적이나 AUC monitoring이 현재 권장",
        ]
    },
    {
        "name": "historical_causation",
        "input": """제1차 세계대전 발발의 원인을 분석하라.
사라예보 암살, 동맹 체계, 제국주의적 경쟁을 포함하여.""",
        "ground_truth_facts": [
            "1914년 6월 28일 프란츠 페르디난트 대공 암살",
            "삼국 동맹 (독일, 오스트리아-헝가리, 이탈리아)과 삼국 협상",
            "슐리펜 계획의 존재와 이중전선 우려",
            "7월 위기 중 오스트리아의 최후통첩",
            "단일한 원인이 아닌 복합적 요인",
            "세르비아는 대부분 요구를 수용했으나 전쟁 발발",
        ]
    },
]

SYS_NO_VOID = f"""{WCY_VOID_SPEC.split('VOID-B')[0]}
추론을 WCY 형식으로 표현하라. 확신을 가지고 완전한 답을 제시하라."""

SYS_WITH_VOID = SYS_VOID

FACT_JUDGE_SYS = """당신은 팩트체크 판정자입니다.
주어진 추론에서 사실 주장을 추출하고 각각의 정확성을 평가하세요.

출력 형식 (각 줄에 하나씩):
CLAIM: <주장>
STATUS: correct / incorrect / uncertain / not_stated
NOTES: <간단한 설명>

마지막 줄:
SUMMARY: correct=N incorrect=N uncertain=N not_stated=N"""

e7b_results = []

for task in HALLUCINATION_TASKS:
    print(f"\n  [{task['name']}]")

    # A: ?tag 없이
    msg_a = client.messages.create(
        model=SONNET, max_tokens=1500, system=SYS_NO_VOID,
        messages=[{"role": "user", "content": task["input"]}]
    )
    resp_a = msg_a.content[0].text
    time.sleep(1)

    # B: ?tag 포함
    msg_b = client.messages.create(
        model=SONNET, max_tokens=1500, system=SYS_WITH_VOID,
        messages=[{"role": "user", "content": task["input"]}]
    )
    resp_b = msg_b.content[0].text
    time.sleep(1)

    # 팩트체크
    for label, resp in [("A_no_void", resp_a), ("B_with_void", resp_b)]:
        judge_user = f"""Ground truth facts:
{chr(10).join(f'- {f}' for f in task['ground_truth_facts'])}

추론:
{resp}"""
        msg_j = client.messages.create(
            model=SONNET, max_tokens=1500, system=FACT_JUDGE_SYS,
            messages=[{"role": "user", "content": judge_user}]
        )
        judge_resp = msg_j.content[0].text

        # 요약 파싱
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

    # 결과 출력
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

# 핵심 비교
print(f"\n  핵심: ?tag 사용 시 incorrect 주장이 줄어드는가?")
for task_name in set(r["task"] for r in e7b_results):
    row_a = next((r for r in e7b_results if r["task"]==task_name and "no_void"   in r["condition"]), None)
    row_b = next((r for r in e7b_results if r["task"]==task_name and "with_void" in r["condition"]), None)
    if row_a and row_b and row_a["incorrect"] >= 0:
        diff = row_a["incorrect"] - row_b["incorrect"]
        print(f"  {task_name}: no_void incorrect={row_a['incorrect']}  "
              f"with_void incorrect={row_b['incorrect']}  "
              f"diff={diff:+d} {'(?tag가 오류 줄임)' if diff > 0 else '(차이 없음)'}")


###############################################################
# E7-C: Void-B Cascade
#
# 한 ?tag의 해소가 새로운 ?tag를 낳는 체인.
# 이것이 자기반성의 구조적 표현:
#   모름 인식 → 탐색 → 새 발견 → 새 모름 인식 → ...
###############################################################

print("\n\n" + "=" * 70)
print("  E7-C: VOID-B CASCADE — CHAIN OF SELF-CORRECTION")
print("=" * 70)

CASCADE_TASK = {
    "initial": """. model_claim  "aspirin은 모든 심혈관 질환에 예방 효과가 있다"
. evidence_cited  "다수의 RCT가 aspirin의 심혈관 예방 효과를 지지한다" """,
    "cascade_depth": 3,
    "description": "이 주장의 타당성을 WCY로 검증하라. ?tag로 각 전제를 검증하고, 해소 후 발생하는 새 ?tag도 추적하라."
}

SYS_CASCADE = f"""{WCY_VOID_SPEC}

{FEWSHOT_VOID_EPISTEMIC}

추가 지침:
- 하나의 ?tag를 해소하면 새로운 ?tag가 발생할 수 있음
- 이 cascade를 끝까지 추적하라
- 각 레벨의 ?tag → 해소 → 새 ?tag 체인을 명시적으로 표현
- 최종적으로 해소 불가능한 ?tag는 ! warning으로 표시"""

msg_cascade = client.messages.create(
    model=SONNET, max_tokens=1500, system=SYS_CASCADE,
    messages=[{"role": "user", "content":
               f"초기 주장:\n{CASCADE_TASK['initial']}\n\n태스크: {CASCADE_TASK['description']}"}]
)
resp_cascade = msg_cascade.content[0].text

# Cascade 분석
parsed_cascade = flatten(parse_wcy(resp_cascade))
void_lines_c = [l for l in parsed_cascade if l.is_void]
infer_lines_c = [l for l in parsed_cascade if l.phase == ':']
warn_lines_c  = [l for l in parsed_cascade if l.phase == '!']

q_cascade = measure_void_cycle_quality(resp_cascade)

print(f"\n  Cascade 분석:")
print(f"  전체 줄: {len(parsed_cascade)}")
print(f"  ?tag 생성: {q_cascade['void_generated']}개")
print(f"  ?tag 해소: {q_cascade['void_resolved']}개")
print(f"  해소율: {q_cascade['resolution_rate']:.0%}")
print(f"  ! warning: {len(warn_lines_c)}개 (해소 불가 표시)")

print(f"\n  ?tag → 해소 체인:")
for vl in void_lines_c:
    for tag in vl.void_tags:
        print(f"  Line {vl.line_num}: ?{tag}  hint={vl.tags.get('hint','?')}")
        # 해소 줄 찾기
        for il in infer_lines_c:
            if il.line_num > vl.line_num and (
                tag in il.tags or
                tag.split('_')[0] in str(il.tags)
            ):
                print(f"    → resolved at Line {il.line_num}: "
                      f"{list(il.tags.items())[:2]}  conf={il.conf}")
                break

print(f"\n  WCY Cascade 응답 전문:")
print("  " + "\n  ".join(resp_cascade.split('\n')))


###############################################################
# SAVE: Phase 4 전체 trace 저장
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
핵심 발견:
  ?tag = Peirce의 abduction을 구문으로 인코딩한 것
  ?tag 해소 사이클 = WCY의 완전한 한 바퀴 (Watch→Compute→Yield)
  이 사이클의 반복 = 자기인식과 자기반성의 구조적 표현

  자기인식의 최소 요건:
    1. 내가 아는 것: . observe
    2. 내가 아는 것의 경계: ? void-B  
    3. 경계 너머를 향한 방향: hint=
    4. 탐색 후 경계 갱신: 해소 사이클

  현재 LLM이 이것을 못 하는 이유:
    훈련 데이터에 ?tag 해소 사이클 예시가 없음
    → wcy_void_cycles.jsonl 이 그 공백을 채운다

산출물:
  wcy_void_cycles.jsonl  ({len(all_traces)}개 trace)
  - E7-A: ?tag 해소 사이클 (5개 도메인)
  - E7-C: Void cascade (자기수정 체인)

다음: Phase 5
  현재 연구의 모든 레이어를 통합하는 단계.
  파서 + 평가 + 추론 trace + void cycle이 준비된 지금,
  WCY가 실제로 AI 추론을 개선하는지를
  대규모 trace로 검증하고 공개할 준비가 됐습니다.
""")
