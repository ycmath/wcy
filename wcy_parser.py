# -*- coding: utf-8 -*-
"""
wcy_parser.py — WCY Reference Parser v1.1
    Changelog:
      v1.1: | separator support
      v1.1.1: fix validate() dead code, fix reconstruct() block gap
==========================================
WCY (Watch → Compute → Yield) 포맷의 레퍼런스 구현.

Usage:
    from wcy_parser import parse_wcy, resolve_chain, extract_voids, WCYLine

    lines = parse_wcy(text)
    chain = resolve_chain(lines, target_line=8)
    voids = extract_voids(lines)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import re


# ── 상수 ──────────────────────────────────────────────────────────────────

PHASE_MARKERS = {
    '.': 'observe',
    ':': 'infer',
    '>': 'act',
    '~': 'meta',
    '!': 'exception',
}

MAX_DEPTH = 2  # WCY spec: 최대 중첩 깊이


# ── 데이터 클래스 ─────────────────────────────────────────────────────────

@dataclass
class WCYSlot:
    """단일 슬롯. tag=value 또는 positional bare_value."""
    key: str | None       # tag 이름 (없으면 None = positional)
    value: str            # 값
    is_void: bool = False # ?tag 마커 여부
    position: int = 0     # positional 슬롯의 순서 (0-indexed)


@dataclass
class WCYLine:
    """파싱된 WCY 줄 하나 (= 하나의 δ-expression)."""
    line_num: int                      # 1-indexed, 비어있지 않은 줄 기준
    raw: str                           # 원본 텍스트
    phase: str                         # '.', ':', '>', '~', '!'
    phase_name: str                    # 'observe', 'infer', 'act', 'meta', 'exception'
    depth: int                         # 들여쓰기 깊이 (0, 1, 2)
    slots: list[WCYSlot] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)    # tag → value
    void_tags: list[str] = field(default_factory=list)    # ?tag 이름 목록
    from_refs: list[int] = field(default_factory=list)    # from=N,M → [N, M]
    conf: float | None = None                             # conf=0.xx
    conf_range: tuple[float, float] | None = None         # conf_range=0.4..0.8
    block_index: int = 0                                  # 빈 줄로 구분된 블록 번호
    children: list[WCYLine] = field(default_factory=list) # 들여쓰기 자식

    @property
    def is_void(self) -> bool:
        """이 줄에 void-B 마커가 있는가."""
        return len(self.void_tags) > 0

    @property
    def positional_values(self) -> list[str]:
        """tag 없는 positional 슬롯 값들."""
        return [s.value for s in self.slots if s.key is None and not s.is_void]

    @property
    def first_value(self) -> str | None:
        """첫 번째 슬롯 값 (positional이든 tagged이든)."""
        if self.slots:
            return self.slots[0].value
        return None

    def to_dict(self) -> dict[str, Any]:
        """직렬화."""
        return {
            'line_num':    self.line_num,
            'phase':       self.phase,
            'phase_name':  self.phase_name,
            'depth':       self.depth,
            'slots':       [{'key': s.key, 'value': s.value, 'is_void': s.is_void}
                            for s in self.slots],
            'tags':        self.tags,
            'void_tags':   self.void_tags,
            'from_refs':   self.from_refs,
            'conf':        self.conf,
            'conf_range':  list(self.conf_range) if self.conf_range else None,
            'block_index': self.block_index,
        }

    def __repr__(self) -> str:
        parts = [f"WCYLine(#{self.line_num} '{self.phase}' depth={self.depth}"]
        if self.void_tags:
            parts.append(f" voids={self.void_tags}")
        if self.from_refs:
            parts.append(f" from={self.from_refs}")
        if self.conf is not None:
            parts.append(f" conf={self.conf}")
        parts.append(")")
        return "".join(parts)


# ── 슬롯 파서 ─────────────────────────────────────────────────────────────

_QUOTED_RE  = re.compile(r'^"([^"]*)"$')
_TAGVAL_RE  = re.compile(r'^(\w[\w\-\.]*)\s*=\s*(.+)$')
_VOID_RE    = re.compile(r'^\?(\w+)$')
_BACKREF_RE = re.compile(r'^\{(\d+)\}$')
_CONF_RE    = re.compile(r'^([\d.]+)\.\.([\d.]+)$')


def _parse_slot(token: str, pos_index: int) -> WCYSlot | None:
    """
    토큰 하나를 WCYSlot으로 파싱.
    특수 슬롯(from=, conf=, conf_range=)은 None 반환 (호출자가 별도 처리).
    """
    token = token.strip()
    if not token:
        return None

    # ?tag (void-B)
    m = _VOID_RE.match(token)
    if m:
        return WCYSlot(key=m.group(1), value='', is_void=True, position=pos_index)

    # tag=value
    m = _TAGVAL_RE.match(token)
    if m:
        key, val = m.group(1), m.group(2).strip()
        # 따옴표 제거
        qm = _QUOTED_RE.match(val)
        if qm:
            val = qm.group(1)
        return WCYSlot(key=key, value=val, position=pos_index)

    # {N} 역참조
    m = _BACKREF_RE.match(token)
    if m:
        return WCYSlot(key='__backref__', value=m.group(1), position=pos_index)

    # bare positional value (따옴표 포함 가능)
    qm = _QUOTED_RE.match(token)
    val = qm.group(1) if qm else token
    return WCYSlot(key=None, value=val, position=pos_index)


def _parse_slots(rest: str) -> tuple[
    list[WCYSlot], dict[str, str],
    list[str], list[int],
    float | None, tuple[float, float] | None
]:
    """
    phase marker 이후의 나머지 문자열을 파싱.
    반환: (slots, tags, void_tags, from_refs, conf, conf_range)
    """
    # 공백/| 구분 토큰화 (따옴표 내부 보호)
    # | = optional slot separator (spec v1.1)
    # 모델이 복잡한 슬롯에서 자발적으로 사용하는 패턴을 공식화
    # "a | b=c | d"  →  ["a", "b=c", "d"]
    tokens: list[str] = []
    current = []
    in_quote = False
    for ch in rest:
        if ch == '"':
            in_quote = not in_quote
            current.append(ch)
        elif (ch == ' ' or ch == '|') and not in_quote:
            tok = ''.join(current).strip()
            if tok:
                tokens.append(tok)
            current = []
        else:
            current.append(ch)
    tok = ''.join(current).strip()
    if tok:
        tokens.append(tok)

    slots: list[WCYSlot] = []
    tags: dict[str, str] = {}
    void_tags: list[str] = []
    from_refs: list[int] = []
    conf: float | None = None
    conf_range: tuple[float, float] | None = None
    pos_index = 0

    for token in tokens:
        if not token:
            continue

        # ── 특수 태그 우선 처리 ────────────────────────────────────────
        # from=N,M
        if token.startswith('from='):
            raw = token[5:]
            for n in raw.split(','):
                n = n.strip()
                if n.isdigit():
                    from_refs.append(int(n))
            continue

        # conf_range=0.4..0.8
        if token.startswith('conf_range='):
            raw = token[11:]
            m = _CONF_RE.match(raw)
            if m:
                conf_range = (float(m.group(1)), float(m.group(2)))
            continue

        # conf=0.88
        if token.startswith('conf='):
            try:
                conf = float(token[5:])
            except ValueError:
                pass
            continue

        # ?tag (void-B) — 줄 첫 실질 슬롯으로 오는 경우
        m = _VOID_RE.match(token)
        if m:
            tag_name = m.group(1)
            void_tags.append(tag_name)
            slots.append(WCYSlot(key=tag_name, value='', is_void=True, position=pos_index))
            pos_index += 1
            continue

        # 일반 슬롯
        slot = _parse_slot(token, pos_index)
        if slot:
            slots.append(slot)
            if slot.key and not slot.is_void and slot.key not in ('__backref__',):
                tags[slot.key] = slot.value
            pos_index += 1

    return slots, tags, void_tags, from_refs, conf, conf_range


# ── 줄 파서 ───────────────────────────────────────────────────────────────

def _measure_depth(raw_line: str) -> int:
    """들여쓰기 깊이 측정 (2칸 = 1 depth)."""
    spaces = len(raw_line) - len(raw_line.lstrip(' '))
    return min(spaces // 2, MAX_DEPTH)


def _parse_line(raw: str, line_num: int, block_index: int) -> WCYLine | None:
    """
    단일 줄을 WCYLine으로 파싱.
    빈 줄이거나 주석(# 시작)이면 None 반환.
    """
    stripped = raw.strip()
    if not stripped or stripped.startswith('#'):
        return None

    depth = _measure_depth(raw)

    # phase marker 확인 (첫 비공백 문자)
    if len(stripped) < 2:
        return None
    phase = stripped[0]
    if phase not in PHASE_MARKERS:
        return None
    if stripped[1] != ' ':  # phase marker 다음 반드시 공백
        return None

    rest = stripped[2:].strip()
    slots, tags, void_tags, from_refs, conf, conf_range = _parse_slots(rest)

    return WCYLine(
        line_num=line_num,
        raw=raw,
        phase=phase,
        phase_name=PHASE_MARKERS[phase],
        depth=depth,
        slots=slots,
        tags=tags,
        void_tags=void_tags,
        from_refs=from_refs,
        conf=conf,
        conf_range=conf_range,
        block_index=block_index,
    )


# ── 메인 파서 ─────────────────────────────────────────────────────────────

def parse_wcy(text: str) -> list[WCYLine]:
    """
    WCY 텍스트를 파싱하여 WCYLine 목록 반환.

    - line_num은 비어있지 않은 유효 줄 기준 1-indexed
    - 빈 줄은 block_index를 증가시킴
    - 들여쓰기 깊이 2 이하의 자식 관계를 children에 연결

    Args:
        text: WCY 형식의 텍스트

    Returns:
        최상위 WCYLine 목록 (depth=0인 줄들).
        children 필드로 하위 줄 접근 가능.
    """
    raw_lines = text.split('\n')
    parsed: list[WCYLine] = []
    line_num = 0
    block_index = 0

    for raw in raw_lines:
        stripped = raw.strip()
        if not stripped:
            block_index += 1
            continue
        if stripped.startswith('#'):
            continue

        line_num += 1
        wcy_line = _parse_line(raw, line_num, block_index)
        if wcy_line:
            parsed.append(wcy_line)

    # ── 들여쓰기 부모-자식 연결 ────────────────────────────────────────
    result: list[WCYLine] = []
    stack: list[WCYLine] = []  # (depth, line) 스택

    for line in parsed:
        # 현재 depth보다 깊거나 같은 것 스택에서 제거
        while stack and stack[-1].depth >= line.depth:
            stack.pop()

        if stack:
            # 부모의 children에 추가
            stack[-1].children.append(line)
        else:
            result.append(line)

        stack.append(line)

    return result


def flatten(lines: list[WCYLine]) -> list[WCYLine]:
    """children 포함하여 모든 줄을 line_num 순서로 평탄화."""
    result: list[WCYLine] = []

    def _collect(line: WCYLine):
        result.append(line)
        for child in line.children:
            _collect(child)

    for line in lines:
        _collect(line)

    return sorted(result, key=lambda l: l.line_num)


# ── 체인 분석 유틸리티 ─────────────────────────────────────────────────────

def resolve_chain(
    lines: list[WCYLine],
    target_line: int,
    max_depth: int = 20,
) -> list[WCYLine]:
    """
    from= 체인을 역추적하여 루트까지의 경로를 반환.

    Args:
        lines: parse_wcy() 결과 (flatten 후 사용)
        target_line: 역추적 시작 줄 번호 (1-indexed)
        max_depth: 최대 역추적 깊이 (순환 방지)

    Returns:
        [루트 줄, ..., 중간 줄들, ..., target 줄] 순서
    """
    flat = flatten(lines) if any(l.children for l in lines) else lines
    by_num = {l.line_num: l for l in flat}

    visited: set[int] = set()
    chain: list[WCYLine] = []

    def _trace(num: int, depth: int):
        if depth > max_depth or num in visited:
            return
        visited.add(num)
        line = by_num.get(num)
        if not line:
            return
        chain.append(line)
        for ref in line.from_refs:
            _trace(ref, depth + 1)

    _trace(target_line, 0)

    # 루트가 앞에 오도록 line_num 기준 정렬
    chain.sort(key=lambda l: l.line_num)
    return chain


def find_root(lines: list[WCYLine], target_line: int) -> WCYLine | None:
    """
    from= 체인의 최종 루트(from= 없는 최상위 근거)를 반환.
    """
    chain = resolve_chain(lines, target_line)
    for line in chain:
        if not line.from_refs:
            return line
    return chain[0] if chain else None


# ── void-B 유틸리티 ───────────────────────────────────────────────────────

def extract_voids(lines: list[WCYLine]) -> list[WCYLine]:
    """
    ?tag 마커가 있는 모든 줄을 반환.

    Returns:
        void-B 마커를 포함한 WCYLine 목록
    """
    flat = flatten(lines) if any(l.children for l in lines) else lines
    return [l for l in flat if l.is_void]


def void_summary(lines: list[WCYLine]) -> dict[str, Any]:
    """
    void-B 현황 요약.

    Returns:
        {
          'total': int,
          'voids': [{'tag': str, 'line': int, 'hint': str|None,
                     'conf_range': tuple|None}]
        }
    """
    voids = extract_voids(lines)
    summary = []
    for line in voids:
        for tag in line.void_tags:
            summary.append({
                'tag': tag,
                'line': line.line_num,
                'hint': line.tags.get('hint'),
                'conf_range': line.conf_range,
            })
    return {'total': len(summary), 'voids': summary}


# ── 블록/문맥 유틸리티 ────────────────────────────────────────────────────

def get_block(lines: list[WCYLine], block_index: int) -> list[WCYLine]:
    """특정 블록 번호의 줄만 반환 (빈 줄로 구분된 단위)."""
    flat = flatten(lines) if any(l.children for l in lines) else lines
    return [l for l in flat if l.block_index == block_index]


def get_phase(lines: list[WCYLine], phase: str) -> list[WCYLine]:
    """특정 phase marker의 줄만 반환."""
    flat = flatten(lines) if any(l.children for l in lines) else lines
    return [l for l in flat if l.phase == phase]


def get_by_tag(lines: list[WCYLine], tag: str) -> list[WCYLine]:
    """특정 태그 키가 있는 줄 반환."""
    flat = flatten(lines) if any(l.children for l in lines) else lines
    return [l for l in flat if tag in l.tags]


# ── 직렬화 ────────────────────────────────────────────────────────────────

def to_dict_list(lines: list[WCYLine]) -> list[dict]:
    """parse_wcy 결과를 JSON 직렬화 가능한 dict 목록으로 변환."""
    flat = flatten(lines) if any(l.children for l in lines) else lines
    return [l.to_dict() for l in flat]


def reconstruct(lines: list[WCYLine]) -> str:
    """
    WCYLine 목록을 WCY 텍스트로 재구성.
    (라운드트립 검증용)
    """
    flat = flatten(lines) if any(l.children for l in lines) else lines
    result = []
    prev_block = 0
    for line in flat:
        # 블록 경계마다 빈 줄 삽입 (블록 차이만큼)
        while prev_block < line.block_index:
            result.append('')
            prev_block += 1
        result.append(line.raw)
    return '\n'.join(result)


# ── 검증 ──────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def validate(lines: list[WCYLine]) -> ValidationResult:
    """
    파싱된 WCY 문서의 구조적 유효성 검사.

    검사 항목:
    - from= 참조가 유효한 줄 번호를 가리키는가
    - from= 전방 참조 금지 (from=N에서 N < 현재 줄)
    - 들여쓰기 깊이 ≤ 2
    - conf 값이 0.0~1.0 범위
    - conf_range에서 low ≤ high
    """
    flat = flatten(lines) if any(l.children for l in lines) else lines
    all_nums = {l.line_num for l in flat}

    errors: list[str] = []
    warnings: list[str] = []

    for line in flat:
        # from= 유효성
        for ref in line.from_refs:
            if ref not in all_nums:
                errors.append(
                    f"Line {line.line_num}: from={ref} references non-existent line"
                )
            if ref >= line.line_num:
                errors.append(
                    f"Line {line.line_num}: from={ref} is a forward reference (forbidden)"
                )

        # 들여쓰기
        if line.depth > MAX_DEPTH:
            errors.append(
                f"Line {line.line_num}: depth={line.depth} exceeds max ({MAX_DEPTH})"
            )

        # conf 범위
        if line.conf is not None and not (0.0 <= line.conf <= 1.0):
            errors.append(
                f"Line {line.line_num}: conf={line.conf} out of range [0, 1]"
            )

        # conf_range 논리
        if line.conf_range:
            lo, hi = line.conf_range
            if lo > hi:
                errors.append(
                    f"Line {line.line_num}: conf_range={lo}..{hi} invalid (low > high)"
                )

        # void-B에 hint= 권장
        if line.is_void and 'hint' not in line.tags:
            warnings.append(
                f"Line {line.line_num}: void-B ?{line.void_tags} without hint="
            )

    stats = {
        'total_lines': len(flat),
        'by_phase': {p: sum(1 for l in flat if l.phase == p)
                     for p in PHASE_MARKERS},
        'void_count': sum(1 for l in flat if l.is_void),
        'from_refs_total': sum(len(l.from_refs) for l in flat),
        'blocks': max((l.block_index for l in flat), default=0) + 1,
    }

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


# ── CLI / 빠른 테스트 ──────────────────────────────────────────────────────

if __name__ == '__main__':
    SAMPLE = """
~ context  case=E-2026-001
. patient=Kim  age=45  presented=clinic
. symptoms  fatigue  weight_loss  night_sweats  duration=3weeks
. vitals  temp=37.8  hr=95  bp=128/82
: ?primary_diagnosis  hint=fatigue,weight_loss,night_sweats  conf_range=0.3..0.8

~ context  case=E-2026-002
. patient=Lee  age=62  presented=ER
. chief_complaint  sudden_onset_weakness  left_side  duration=2h
. vitals  temp=36.9  hr=88  bp=178/96
: ?stroke_type  hint=sudden_onset,left_weakness,bp  conf_range=0.4..0.8
: diagnosis=STEMI  conf=0.92  from=8,9
> activate  cath_lab  priority=emergent  from=10
""".strip()

    print("=" * 60)
    print("  WCY Parser v1.1 — Quick Test")
    print("=" * 60)

    lines = parse_wcy(SAMPLE)
    flat = flatten(lines)

    print(f"\n[parse_wcy] → {len(flat)} lines, {len(lines)} top-level")
    for l in flat:
        vmark = f"  VOID={l.void_tags}" if l.is_void else ""
        fmark = f"  from={l.from_refs}" if l.from_refs else ""
        cmark = f"  conf={l.conf}" if l.conf is not None else ""
        print(f"  #{l.line_num:>2} {l.phase} [{l.phase_name:<9}] "
              f"depth={l.depth}  blk={l.block_index}"
              f"{vmark}{fmark}{cmark}")

    print(f"\n[extract_voids] → {len(extract_voids(lines))} void-B lines")
    vs = void_summary(lines)
    for v in vs['voids']:
        print(f"  ?{v['tag']:<25} line={v['line']}  "
              f"hint={v['hint']}  range={v['conf_range']}")

    print(f"\n[resolve_chain] from line 11 (diagnosis=STEMI):")
    chain = resolve_chain(flat, target_line=11)
    for l in chain:
        print(f"  #{l.line_num} {l.phase} {l.raw.strip()[:60]}")

    root = find_root(flat, target_line=11)
    print(f"\n[find_root] → #{root.line_num}: {root.raw.strip()}")

    print(f"\n[validate]")
    vr = validate(lines)
    print(f"  valid={vr.valid}  errors={len(vr.errors)}  warnings={len(vr.warnings)}")
    for e in vr.errors:
        print(f"  ERROR: {e}")
    for w in vr.warnings:
        print(f"  WARN:  {w}")
    print(f"  stats: {vr.stats}")

    print(f"\n[to_dict_list] sample (first line):")
    import json
    print(f"  {json.dumps(to_dict_list(flat)[:1], ensure_ascii=False, indent=2)}")
