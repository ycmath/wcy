# -*- coding: utf-8 -*-
"""
wcy_phase5_pipeline.py — WCY Trace Generation Pipeline v1.0
=============================================================

Phase 5: 훈련 데이터 대규모 생성

목표: 12개 seed trace → 500+ usable traces
방법: domain × difficulty × void_depth 조합 매트릭스로 체계적 생성

품질 게이트 (3단계):
  Gate 1 — 구조: parse_r ≥ 0.70  (파서가 읽을 수 있는가)
  Gate 2 — 공백: void_generated ≥ 1  (경계를 표시했는가)
  Gate 3 — 해소: resolution_rate ≥ 0.50  (탐색이 완결됐는가)

산출물:
  wcy_traces_v1.jsonl  ← 전체 생성 trace
  wcy_traces_v1_clean.jsonl  ← 3게이트 통과 trace만
  wcy_pipeline_report.txt  ← 생성 통계

실행: Colab Pro, wcy_parser.py 필요
주의: 많은 API 호출 발생 (50 tasks × ~2 calls = ~100 calls)
      약 $0.50-1.00 예상 비용
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
    print("⚠ wcy_parser.py를 같은 디렉토리에 업로드하세요")

# ↓↓↓ API 키 직접 입력 ↓↓↓
API_KEY = "sk-ant-api03-YOUR_KEY_HERE"
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

import anthropic
client = anthropic.Anthropic(api_key=API_KEY)
SONNET = "claude-sonnet-4-20250514"

enc = tiktoken.get_encoding("cl100k_base")
count = lambda t: len(enc.encode(str(t)))
print("✓ Setup complete\n")


###############################################################
# 매트릭스 정의
# domain(8) × difficulty(3) × void_depth(2) = 48 조합
# 각 조합에서 2개 생성 = 최대 96개 / 실행당
###############################################################

DOMAINS = {
    "medical":       "의학적 진단 및 치료 결정",
    "code":          "소프트웨어 디버깅 및 설계",
    "scientific":    "과학적 가설 검증",
    "legal":         "법적 추론 및 판단",
    "mathematical":  "수학적 증명 및 계산",
    "strategic":     "전략적 의사결정",
    "philosophical": "철학적/인식론적 분석",
    "engineering":   "시스템 설계 및 트레이드오프",
}

DIFFICULTIES = {
    "simple":  {"void_target": 2, "max_tokens": 400, "description": "단일 인과 체인, 명확한 해소"},
    "medium":  {"void_target": 4, "max_tokens": 700, "description": "다중 가설, 부분 해소"},
    "complex": {"void_target": 6, "max_tokens": 1200, "description": "cascade, 해소 불가 !warning 포함"},
}

VOID_DEPTHS = {
    "single":  "각 ?tag가 독립적으로 해소됨",
    "cascade": "?tag 해소가 새 ?tag를 낳는 체인",
}


###############################################################
# 태스크 템플릿 생성기
# 도메인별 태스크를 동적으로 생성
###############################################################

TASK_SEEDS = {
    "medical": [
        ("환자 {age}세 {sex}, {symptom} {duration}일, {lab_finding} 발견. 진단 및 치료 계획.",
         {"age": [28,45,62,71,38,55], "sex": ["남성","여성"],
          "symptom": ["급성 흉통","고열과 오한","호흡곤란","복통","두통"],
          "duration": [1,2,3,7,14], "lab_finding": ["troponin 상승","WBC 18000","CRP 45","D-dimer 양성"]}),
        ("약물 {drug_a}와 {drug_b} 동시 복용 환자. 상호작용 평가 및 대안.",
         {"drug_a": ["warfarin","metformin","lisinopril","atorvastatin"],
          "drug_b": ["ciprofloxacin","ibuprofen","amiodarone","fluconazole"]}),
    ],
    "code": [
        ("{service} 서비스에서 {symptom} 발생, {change} 이후 시작. 원인 분석 및 수정.",
         {"service": ["API 서버","배치 파이프라인","ML 추론 서버","데이터베이스"],
          "symptom": ["메모리 누수","응답 지연 급증","에러율 상승","CPU 과부하"],
          "change": ["배포 v3.1","인프라 업그레이드","트래픽 2배 증가","라이브러리 업데이트"]}),
        ("{algorithm}의 시간복잡도 {current}를 {target}로 최적화. 트레이드오프 분석.",
         {"algorithm": ["그래프 탐색","문자열 매칭","정렬","캐시 설계"],
          "current": ["O(n²)","O(n³)","O(2ⁿ)"],
          "target": ["O(n log n)","O(n)","O(1) amortized"]}),
    ],
    "scientific": [
        ("실험 {experiment}에서 예상과 다른 {observation} 관찰. 원인 가설 및 검증 방법.",
         {"experiment": ["단백질 결정화","세포 배양","화학 반응","물리 측정"],
          "observation": ["수율 급감","이상한 부산물 생성","반응 속도 변화","측정값 이상"]}),
        ("{phenomenon}과 {variable} 사이 상관관계 r={r} 발견. 인과 추론 평가.",
         {"phenomenon": ["소득 수준","운동 빈도","수면 시간","스트레스 지수"],
          "variable": ["수명","인지 기능","심혈관 건강","면역 수치"],
          "r": ["0.72","0.85","0.61","0.43"]}),
    ],
    "legal": [
        ("계약 {clause} 조항 해석 분쟁. {party_a} vs {party_b} 입장 분석.",
         {"clause": ["손해배상 한도","불가항력","지식재산 귀속","비밀유지"],
          "party_a": ["갑(발주자)","원고","구매자"],
          "party_b": ["을(수행자)","피고","판매자"]}),
        ("행위 {action}의 {law} 위반 여부 판단. 요건 충족 분석.",
         {"action": ["제3자에게 정보 공유","계약 미이행","경쟁사 직원 채용","가격 담합"],
          "law": ["개인정보보호법","공정거래법","노동법","민법 채무불이행"]}),
    ],
    "mathematical": [
        ("명제: {claim}. 증명 또는 반례 제시.",
         {"claim": ["소수는 무한히 많다","연속 함수는 적분 가능하다",
                    "P≠NP라면 RSA는 안전하다","모든 유리수는 실수다"]}),
        ("{problem_type} 문제: {setup}. 단계적 풀이.",
         {"problem_type": ["확률","최적화","통계 추정","미분방정식"],
          "setup": ["불완전 정보 게임에서 최적 전략","제약 조건 하 비용 최소화",
                    "표본 n=30에서 모평균 추정","SIR 모델 감염자 수 예측"]}),
    ],
    "strategic": [
        ("{company_type}가 {market}에 진출 검토. 리스크와 기회 분석.",
         {"company_type": ["스타트업","대기업","외국계 기업"],
          "market": ["동남아 시장","B2B SaaS","헬스케어 AI","리테일 핀테크"]}),
        ("{resource}가 제한된 상황에서 {goal} 달성. 우선순위 결정 프레임워크.",
         {"resource": ["6개월 런웨이","팀 5명","예산 1억원"],
          "goal": ["PMF 달성","시리즈A 준비","수익화","기술 특허 확보"]}),
    ],
    "philosophical": [
        ("주장: \"{claim}\". 전제 검증 및 반론 분석.",
         {"claim": ["자유의지는 존재한다","AI는 의식을 가질 수 없다",
                    "도덕은 문화 상대적이다","지식은 정당화된 참 믿음이다"]}),
        ("{scenario}에서 {ethical_framework} 관점의 판단과 한계.",
         {"scenario": ["자율주행차 트롤리 문제","AI 의사결정 공정성","개인정보 vs 공공안전"],
          "ethical_framework": ["공리주의","의무론","덕윤리학","계약론"]}),
    ],
    "engineering": [
        ("{system_type} 시스템 설계: {requirements}. 아키텍처 결정과 트레이드오프.",
         {"system_type": ["분산 캐시","메시지 큐","API 게이트웨이","ML 서빙"],
          "requirements": ["읽기 1M rps","99.99% 가용성","p99 <10ms","글로벌 분산"]}),
        ("{tech_a} vs {tech_b} 기술 선택. 평가 기준과 권고.",
         {"tech_a": ["PostgreSQL","Kafka","Kubernetes","REST"],
          "tech_b": ["MongoDB","RabbitMQ","Nomad","GraphQL"]}),
    ],
}

def generate_task(domain: str, difficulty: str, void_depth: str) -> dict:
    """도메인+난이도+void_depth 조합으로 태스크 생성."""
    templates = TASK_SEEDS.get(domain, TASK_SEEDS["medical"])
    template_text, params = random.choice(templates)

    # 파라미터 채우기
    filled = template_text
    for key, values in params.items():
        if f"{{{key}}}" in filled:
            filled = filled.replace(f"{{{key}}}", str(random.choice(values)))

    diff_cfg = DIFFICULTIES[difficulty]
    void_instruction = {
        "single":  f"?tag를 {diff_cfg['void_target']}개 생성하고 각각 독립적으로 해소하라.",
        "cascade": f"?tag를 {diff_cfg['void_target']}개 이상 생성하고, 해소 과정에서 새 ?tag가 발생하면 그것도 추적하라. 해소 불가능한 것은 ! warning으로 표시."
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
# WCY 추론 품질 측정
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

    # 해소율: ?tag 이름이 이후 : 줄에 등장하는지 (느슨한 매칭)
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
# 시스템 프롬프트 (Phase 4 few-shot 포함)
###############################################################

FEW_SHOT_COMPACT = dedent("""
예시 A (code/medium):
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

예시 B (philosophical/complex cascade):
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

SYS_PIPELINE = f"""WCY 추론 형식:
. observe  — 확인된 사실
: infer    — 추론 (conf= 권장, from= 로 근거 명시)
> act      — 탐색 행동 (reason=from=N)
~ meta     — 태스크 선언
! exception — 경고 또는 해소불가 표시

?tag 규칙:
  : ?unknown  hint=탐색방향  conf_range=L..H  ← 모름 표시
  > investigate  reason=from=?_line           ← 탐색
  . new_finding  result=value                  ← 관찰
  : resolved=value  conf=0.xx  from=?,obs      ← 해소

{FEW_SHOT_COMPACT}

규칙: ?tag 생성하면 반드시 해소 사이클 완성. 해소 불가면 ! warning.
마크다운, 자연어 설명 금지. WCY만 출력."""


###############################################################
# 파이프라인 실행
###############################################################

# 실행 계획 (조절 가능)
# 전체: 48 조합 × 1 = 48 태스크 (빠른 실행)
# 대규모: 48 × 2 = 96 태스크
RUNS_PER_COMBO = 1   # 빠른 실행. 2로 바꾸면 더 많은 trace

# 도메인 순서 (다양성 우선)
DOMAIN_ORDER  = list(DOMAINS.keys())
DIFF_ORDER    = ["simple", "medium", "complex"]
DEPTH_ORDER   = ["single", "cascade"]

# 생성할 조합 목록
combos = [
    (d, diff, depth)
    for d in DOMAIN_ORDER
    for diff in DIFF_ORDER
    for depth in DEPTH_ORDER
]
random.seed(42)
random.shuffle(combos)  # 다양성을 위해 섞기

print(f"계획: {len(combos)} 조합 × {RUNS_PER_COMBO} = {len(combos)*RUNS_PER_COMBO} tasks")
print(f"예상 시간: ~{len(combos)*RUNS_PER_COMBO*3//60}분\n")

all_traces = []
stats = {"total": 0, "usable": 0, "gate1_fail": 0, "gate2_fail": 0, "gate3_fail": 0}

OUTPUT_FILE = "wcy_traces_v1.jsonl"
CLEAN_FILE  = "wcy_traces_v1_clean.jsonl"

# 파일 초기화
open(OUTPUT_FILE, 'w').close()
open(CLEAN_FILE, 'w').close()

print(f"{'#':>4} {'Domain':<14} {'Diff':<8} {'Depth':<8} {'parse':>6} {'?gen':>5} {'?res%':>6} {'gate':>5}")
print("─" * 62)

for run_i in range(RUNS_PER_COMBO):
    for ci, (domain, difficulty, void_depth) in enumerate(combos):
        task = generate_task(domain, difficulty, void_depth)
        trace_id = f"v1_{ci:03d}_{run_i}"
        stats["total"] += 1

        user_msg = f"태스크: {task['task_text']}\n\n{task['void_instruction']}"

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

        # 품질 게이트
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

        # 실시간 저장
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(trace, ensure_ascii=False) + '\n')
        if trace["usable"]:
            with open(CLEAN_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(trace, ensure_ascii=False) + '\n')

        print(f"{stats['total']:>4} {domain:<14} {difficulty:<8} {void_depth:<8} "
              f"{q['parse_rate']:>6.0%} {q['void_generated']:>5} "
              f"{q['resolution_rate']:>5.0%} {gate_str:>5}")

        time.sleep(1.2)  # rate limit

# ── 최종 보고서 ────────────────────────────────────────────────
print(f"\n{'═'*62}")
print(f"  PIPELINE COMPLETE")
print(f"{'═'*62}")
print(f"  총 생성: {stats['total']}")
print(f"  Usable:  {stats['usable']} ({stats['usable']/max(stats['total'],1)*100:.0f}%)")
print(f"  Gate 1 실패 (parse_r): {stats['gate1_fail']}")
print(f"  Gate 2 실패 (void≥1):  {stats['gate2_fail']}")
print(f"  Gate 3 실패 (res≥50%): {stats['gate3_fail']}")

# 도메인별 통계
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

print(f"\n  도메인별:")
print(f"  {'Domain':<14} {'total':>6} {'usable':>7} {'?avg':>6} {'res%':>6}")
print(f"  {'─'*44}")
for dom, s in sorted(domain_stats.items()):
    n = max(s["total"], 1)
    print(f"  {dom:<14} {s['total']:>6} {s['usable']:>7} "
          f"{s['avg_void']/n:>6.1f} {s['avg_res']/n*100:>5.0f}%")

# 보고서 저장
report = {
    "stats": stats,
    "domain_stats": dict(domain_stats),
    "output_file": OUTPUT_FILE,
    "clean_file": CLEAN_FILE,
}
with open("wcy_pipeline_report.json", "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n  저장:")
print(f"  {OUTPUT_FILE}    — 전체 trace")
print(f"  {CLEAN_FILE} — usable trace만")
print(f"  wcy_pipeline_report.json — 통계")
print(f"\n  다음: Colab 파일 패널에서 세 파일 다운로드")
