# -*- coding: utf-8 -*-
"""
wcy_parser.py — WCY Reference Parser v2.0, WCY-2
    Changelog:
      v1.1:   | separator support
      v1.1.1: fix validate() dead code, fix reconstruct() block gap
      v2.0:   WCY-2 surface additions (SPEC.md section 4.2):
                - rail suffix on values:  tag=value^T | ^F | ^U | ^C
                - refutation slot:        tag!=value   (sugar for =value^F)
                - level tag:              lvl=m
                - resolution record:      : resolve tag was= now= from= why=
                - obstruction record:     ! obstruction tag from= note=
              plus non-numeric `from=` provenance labels (from_labels).
==========================================
Reference implementation of the WCY (Watch → Compute → Yield) format.

Backward compatibility (SPEC.md section 11.3): a v1 document is a Core
v2 document. Every v1 field of WCYSlot / WCYLine keeps its v1 meaning and
its v1 value; v2 adds fields only, all with defaults. The v2 additions
change a parse result only for documents that actually use v2 syntax.

Usage:
    from wcy_parser import parse_wcy, resolve_chain, extract_voids, WCYLine

    lines = parse_wcy(text)
    chain = resolve_chain(lines, target_line=8)
    voids = extract_voids(lines)

    # v2
    lines = parse_wcy(". allergy=penicillin^C\\n: resolve allergy was=C now=F")
    lines[0].slots[0].rail   # 'C'
    lines[1].kind            # 'resolve'
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import re


# ── Constants ───────────────────────────────────────────────────────────────

PHASE_MARKERS = {
    '.': 'observe',
    ':': 'infer',
    '>': 'act',
    '~': 'meta',
    '!': 'exception',
}

MAX_DEPTH = 2  # WCY spec: maximum nesting depth

# ── v2 constants (SPEC.md sections 3.1, 4.2) ────────────────────────────────

RAIL_STATES = ('T', 'F', 'U', 'C')  # assert / refute / unknown / conflict

DEFAULT_RAIL = 'T'   # a bare tag=value is asserting (SPEC 4.2)
VOID_RAIL    = 'U'   # ?tag is an explicitly represented unknown (SPEC 11.3)

# Reserved v2 keywords, per phase marker (Appendix A).
RESERVED_KEYWORDS = {
    ':': 'resolve',       # resolution record   (SPEC 7.1)
    '!': 'obstruction',   # obstruction record  (SPEC 7.3)
}


# ── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class WCYSlot:
    """A single slot: tag=value pair or a positional bare value.

    v2: `rail` records the dual-rail state of the slot's evidence
    (SPEC 3.1, 4.2). 'T' assert, 'F' refute, 'U' unknown, 'C' conflict.
    A bare `tag=value` is 'T'; `tag!=value` is 'F'; `?tag` is 'U'; an
    explicit `^X` suffix wins over all of these.
    """
    key: str | None       # tag name (None = positional slot)
    value: str            # slot value
    is_void: bool = False # True if this is a void-B ?tag marker
    position: int = 0     # order among positional slots (0-indexed)
    rail: str = DEFAULT_RAIL  # v2: 'T' | 'F' | 'U' | 'C'


@dataclass
class WCYRecord:
    """A v2 resolution or obstruction record (SPEC 7.1, 7.3).

    Attached to a WCYLine whose `kind` is 'resolve' or 'obstruction'.
    `was` / `now` are rail-state letters as written in the record.
    """
    kind: str                    # 'resolve' | 'obstruction'
    target: str | None = None    # the tag the record is about
    was: str | None = None       # resolve: prior state ('U' | 'C' | ...)
    now: str | None = None       # resolve: new state ('T' | 'F')
    why: str | None = None       # resolve: justification
    note: str | None = None      # obstruction: failure note

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            'kind':   self.kind,
            'target': self.target,
            'was':    self.was,
            'now':    self.now,
            'why':    self.why,
            'note':   self.note,
        }


@dataclass
class WCYLine:
    """A single parsed WCY line (one δ-expression)."""
    line_num: int                      # 1-indexed among non-blank lines
    raw: str                           # original source text
    phase: str                         # '.', ':', '>', '~', '!'
    phase_name: str                    # 'observe', 'infer', 'act', 'meta', 'exception'
    depth: int                         # indent depth (0, 1, or 2)
    slots: list[WCYSlot] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)    # tag → value mapping
    void_tags: list[str] = field(default_factory=list)    # list of void-B tag names
    from_refs: list[int] = field(default_factory=list)    # from=N,M parsed as [N, M]
    conf: float | None = None                             # conf=0.xx
    conf_range: tuple[float, float] | None = None         # conf_range=0.4..0.8
    block_index: int = 0                                  # block number (increments on blank line)
    children: list[WCYLine] = field(default_factory=list) # indented child lines
    # ── v2 additions (all defaulted; v1 documents keep v1 values) ──────────
    lvl: int = 1                                          # lvl=m level tag (SPEC 8)
    kind: str = 'normal'                                  # 'normal'|'resolve'|'obstruction'
    record: WCYRecord | None = None                       # parsed record fields
    from_labels: list[str] = field(default_factory=list)  # non-numeric from= sources

    @property
    def is_void(self) -> bool:
        """True if this line contains at least one void-B marker."""
        return len(self.void_tags) > 0

    @property
    def positional_values(self) -> list[str]:
        """Values of positional (unkeyed) slots."""
        return [s.value for s in self.slots if s.key is None and not s.is_void]

    @property
    def first_value(self) -> str | None:
        """First slot value, regardless of type."""
        if self.slots:
            return self.slots[0].value
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            'line_num':    self.line_num,
            'phase':       self.phase,
            'phase_name':  self.phase_name,
            'depth':       self.depth,
            'slots':       [{'key': s.key, 'value': s.value, 'is_void': s.is_void,
                             'rail': s.rail}
                            for s in self.slots],
            'tags':        self.tags,
            'void_tags':   self.void_tags,
            'from_refs':   self.from_refs,
            'conf':        self.conf,
            'conf_range':  list(self.conf_range) if self.conf_range else None,
            'block_index': self.block_index,
            # v2
            'lvl':         self.lvl,
            'kind':        self.kind,
            'record':      self.record.to_dict() if self.record else None,
            'from_labels': self.from_labels,
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


# ── Slot Parser ─────────────────────────────────────────────────────────────

_QUOTED_RE  = re.compile(r'^"([^"]*)"$')
_TAGVAL_RE  = re.compile(r'^(\w[\w\-\.]*)\s*=\s*(.+)$')
_VOID_RE    = re.compile(r'^\?(\w+)$')
_BACKREF_RE = re.compile(r'^\{(\d+)\}$')
_CONF_RE    = re.compile(r'^([\d.]+)\.\.([\d.]+)$')
# v2
_REFUTE_RE  = re.compile(r'^(\w[\w\-\.]*)\s*!=\s*(.+)$')   # tag!=value
_RAIL_RE    = re.compile(r'^(.+)\^([TFUC])$')              # value^T | ^F | ^U | ^C


def _split_rail(value: str) -> tuple[str, str | None]:
    """
    Split a trailing v2 rail suffix off a value (SPEC 4.2).

    'penicillin^C'  → ('penicillin', 'C')
    'A^T*lambda'    → ('A^T*lambda', None)   (suffix must end the value)
    '"a^T"'         → ('"a^T"', None)        (fully quoted values are opaque)
    """
    if _QUOTED_RE.match(value):
        return value, None
    m = _RAIL_RE.match(value)
    if m:
        return m.group(1), m.group(2)
    return value, None


def _unquote(value: str) -> str:
    """Strip one layer of surrounding double quotes, if present."""
    qm = _QUOTED_RE.match(value)
    return qm.group(1) if qm else value


def _parse_slot(token: str, pos_index: int) -> WCYSlot | None:
    """
    Parse a single token into a WCYSlot.
    Special slots (from=, conf=, conf_range=) return None; caller handles them.
    """
    token = token.strip()
    if not token:
        return None

    # ?tag (void-B)
    m = _VOID_RE.match(token)
    if m:
        return WCYSlot(key=m.group(1), value='', is_void=True,
                       position=pos_index, rail=VOID_RAIL)

    # tag!=value  (v2 refutation sugar for tag=value^F)
    m = _REFUTE_RE.match(token)
    if m:
        key, val = m.group(1), m.group(2).strip()
        val, rail = _split_rail(val)
        return WCYSlot(key=key, value=_unquote(val), position=pos_index,
                       rail=rail or 'F')

    # tag=value
    m = _TAGVAL_RE.match(token)
    if m:
        key, val = m.group(1), m.group(2).strip()
        val, rail = _split_rail(val)
        # strip surrounding quotes
        return WCYSlot(key=key, value=_unquote(val), position=pos_index,
                       rail=rail or DEFAULT_RAIL)

    # {N} back-reference
    m = _BACKREF_RE.match(token)
    if m:
        return WCYSlot(key='__backref__', value=m.group(1), position=pos_index)

    # bare positional value (may be quoted, may carry a v2 rail suffix)
    val, rail = _split_rail(token)
    return WCYSlot(key=None, value=_unquote(val), position=pos_index,
                   rail=rail or DEFAULT_RAIL)


def _parse_slots(rest: str) -> tuple[
    list[WCYSlot], dict[str, str],
    list[str], list[int],
    float | None, tuple[float, float] | None,
    list[str]
]:
    """
    Parse the remainder of a line after the phase marker.
    Returns: (slots, tags, void_tags, from_refs, conf, conf_range, from_labels)

    v2: non-numeric `from=` sources (e.g. `from=intake_form`) are collected
    into from_labels; from_refs keeps its v1 meaning (line numbers only).
    """
    # Tokenize on whitespace / pipe, protecting quoted strings.
    # | = optional slot separator (spec v1.1)
    # Formalizes a pattern models use spontaneously on dense slot lines.
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
    from_labels: list[str] = []
    conf: float | None = None
    conf_range: tuple[float, float] | None = None
    pos_index = 0

    for token in tokens:
        if not token:
            continue

        # ── Handle special tags first ────────────────────────────────────────
        # from=N,M
        if token.startswith('from='):
            raw = token[5:]
            for n in raw.split(','):
                n = n.strip()
                if n.isdigit():
                    from_refs.append(int(n))
                elif n:
                    from_labels.append(n)   # v2: named provenance source
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

        # ?tag (void-B) — may appear as the first real slot on a : line
        m = _VOID_RE.match(token)
        if m:
            tag_name = m.group(1)
            void_tags.append(tag_name)
            slots.append(WCYSlot(key=tag_name, value='', is_void=True,
                                 position=pos_index, rail=VOID_RAIL))
            pos_index += 1
            continue

        # Regular slot
        slot = _parse_slot(token, pos_index)
        if slot:
            slots.append(slot)
            if slot.key and not slot.is_void and slot.key not in ('__backref__',):
                tags[slot.key] = slot.value
            pos_index += 1

    return slots, tags, void_tags, from_refs, conf, conf_range, from_labels


# ── Line Parser ─────────────────────────────────────────────────────────────

def _parse_level(tags: dict[str, str]) -> int:
    """Read the v2 `lvl=m` grade off a line's tags (SPEC 4.2, 8). Default 1."""
    raw = tags.get('lvl')
    if raw is not None and raw.isdigit():
        m = int(raw)
        if m >= 1:
            return m
    return 1


def _parse_record(phase: str, slots: list[WCYSlot],
                  tags: dict[str, str]) -> WCYRecord | None:
    """
    Detect a v2 resolution / obstruction record (SPEC 7.1, 7.3).

    A record is an infer line whose first positional slot is the reserved
    keyword `resolve`, or an exception line whose first positional slot is
    `obstruction`. The keyword is reserved per phase marker, so v1's
    `> resolve ...` act lines and `: resolved=...` tags are untouched.
    The record's target tag is the next positional slot (or `tag=`).
    """
    keyword = RESERVED_KEYWORDS.get(phase)
    if keyword is None:
        return None

    positional = [s for s in slots if s.key is None and not s.is_void]
    if not positional or positional[0].value != keyword:
        return None

    target = positional[1].value if len(positional) > 1 else tags.get('tag')

    if keyword == 'resolve':
        return WCYRecord(
            kind='resolve',
            target=target,
            was=tags.get('was'),
            now=tags.get('now'),
            why=tags.get('why'),
        )
    return WCYRecord(
        kind='obstruction',
        target=target,
        note=tags.get('note'),
    )


def _measure_depth(raw_line: str) -> int:
    """Measure indent depth (2 spaces = 1 level)."""
    spaces = len(raw_line) - len(raw_line.lstrip(' '))
    return min(spaces // 2, MAX_DEPTH)


def _parse_line(raw: str, line_num: int, block_index: int) -> WCYLine | None:
    """
    Parse a single raw line into a WCYLine.
    Returns None for blank lines and comment lines (starting with #).
    """
    stripped = raw.strip()
    if not stripped or stripped.startswith('#'):
        return None

    depth = _measure_depth(raw)

    # validate phase marker (first non-whitespace character)
    if len(stripped) < 2:
        return None
    phase = stripped[0]
    if phase not in PHASE_MARKERS:
        return None
    if stripped[1] != ' ':  # phase marker must be followed by a space
        return None

    rest = stripped[2:].strip()
    (slots, tags, void_tags, from_refs,
     conf, conf_range, from_labels) = _parse_slots(rest)

    record = _parse_record(phase, slots, tags)

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
        lvl=_parse_level(tags),
        kind=record.kind if record else 'normal',
        record=record,
        from_labels=from_labels,
    )


# ── Main Parser ─────────────────────────────────────────────────────────────

def parse_wcy(text: str) -> list[WCYLine]:
    """
    Parse WCY text into a list of WCYLine objects.

    - line_num is 1-indexed over non-blank valid lines
    - blank lines increment block_index
    - lines indented up to depth 2 are linked via children

    Args:
        text: WCY-formatted string

    Returns:
        List of top-level WCYLine objects (depth=0).
        Deeper lines are accessible via the children field.
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

    # ── Build parent-child tree from indentation ────────────────────────────────
    result: list[WCYLine] = []
    stack: list[WCYLine] = []  # (depth, line) stack for parent tracking

    for line in parsed:
        # pop entries at same or deeper depth
        while stack and stack[-1].depth >= line.depth:
            stack.pop()

        if stack:
            # attach to parent's children list
            stack[-1].children.append(line)
        else:
            result.append(line)

        stack.append(line)

    return result


def flatten(lines: list[WCYLine]) -> list[WCYLine]:
    """Flatten all lines (including children) in line_num order."""
    result: list[WCYLine] = []

    def _collect(line: WCYLine):
        result.append(line)
        for child in line.children:
            _collect(child)

    for line in lines:
        _collect(line)

    return sorted(result, key=lambda l: l.line_num)


# ── Chain Analysis Utilities ────────────────────────────────────────────────

def resolve_chain(
    lines: list[WCYLine],
    target_line: int,
    max_depth: int = 20,
) -> list[WCYLine]:
    """
    Walk the from= provenance chain back to the root.

    Args:
        lines: result of parse_wcy() (use after flatten)
        target_line: starting line number to trace back from (1-indexed)
        max_depth: maximum chain depth to follow (prevents cycles)

    Returns:
        Lines sorted by line_num from root to target.
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

    # sort ascending so root appears first
    chain.sort(key=lambda l: l.line_num)
    return chain


def find_root(lines: list[WCYLine], target_line: int) -> WCYLine | None:
    """
    Return the root line of the from= chain (first line with no from= refs).
    """
    chain = resolve_chain(lines, target_line)
    for line in chain:
        if not line.from_refs:
            return line
    return chain[0] if chain else None


# ── Void-B Utilities ────────────────────────────────────────────────────────

def extract_voids(lines: list[WCYLine]) -> list[WCYLine]:
    """
    Return all lines containing at least one void-B (?tag) marker.

    Returns:
        List of WCYLine objects that have void_tags.
    """
    flat = flatten(lines) if any(l.children for l in lines) else lines
    return [l for l in flat if l.is_void]


def void_summary(lines: list[WCYLine]) -> dict[str, Any]:
    """
    Summarize void-B state for a parsed document.

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


# ── v2 Record Utilities (SPEC 7) ────────────────────────────────────────────

def extract_records(lines: list[WCYLine], kind: str | None = None) -> list[WCYLine]:
    """
    Return all lines carrying a v2 record (SPEC 7.1, 7.3).

    Args:
        lines: result of parse_wcy()
        kind:  'resolve' | 'obstruction' | None (both)

    Returns:
        List of WCYLine objects whose kind is a record kind.
    """
    flat = flatten(lines) if any(l.children for l in lines) else lines
    if kind is None:
        return [l for l in flat if l.kind != 'normal']
    return [l for l in flat if l.kind == kind]


def audit_trail(lines: list[WCYLine]) -> list[dict[str, Any]]:
    """
    The minimal audit surface of a document (SPEC 7.2): the resolution and
    obstruction records, in document order, with their provenance.
    """
    trail: list[dict[str, Any]] = []
    for line in extract_records(lines):
        entry = line.record.to_dict() if line.record else {'kind': line.kind}
        entry['line'] = line.line_num
        entry['from_refs'] = list(line.from_refs)
        entry['from_labels'] = list(line.from_labels)
        trail.append(entry)
    return trail


# ── Block / Context Utilities ───────────────────────────────────────────────

def get_block(lines: list[WCYLine], block_index: int) -> list[WCYLine]:
    """Return only lines belonging to the given block_index."""
    flat = flatten(lines) if any(l.children for l in lines) else lines
    return [l for l in flat if l.block_index == block_index]


def get_phase(lines: list[WCYLine], phase: str) -> list[WCYLine]:
    """Return only lines with the given phase marker."""
    flat = flatten(lines) if any(l.children for l in lines) else lines
    return [l for l in flat if l.phase == phase]


def get_by_tag(lines: list[WCYLine], tag: str) -> list[WCYLine]:
    """Return lines that contain the given tag key."""
    flat = flatten(lines) if any(l.children for l in lines) else lines
    return [l for l in flat if tag in l.tags]


# ── Serialization ───────────────────────────────────────────────────────────

def to_dict_list(lines: list[WCYLine]) -> list[dict]:
    """Convert parse_wcy results to a JSON-serialisable list of dicts."""
    flat = flatten(lines) if any(l.children for l in lines) else lines
    return [l.to_dict() for l in flat]


def reconstruct(lines: list[WCYLine]) -> str:
    """
    Reconstruct WCY text from a list of WCYLine objects.
    Useful for round-trip validation.
    """
    flat = flatten(lines) if any(l.children for l in lines) else lines
    result = []
    prev_block = 0
    for line in flat:
        # insert one blank line per block boundary crossed
        while prev_block < line.block_index:
            result.append('')
            prev_block += 1
        result.append(line.raw)
    return '\n'.join(result)


# ── Validation ──────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def validate(lines: list[WCYLine]) -> ValidationResult:
    """
    Validate the structural correctness of a parsed WCY document.

    Checks:
    - from= references point to existing line numbers
    - from= forward references are forbidden (N < current line)
    - indent depth ≤ 2
    - conf values in [0.0, 1.0]
    - conf_range satisfies low ≤ high
    """
    flat = flatten(lines) if any(l.children for l in lines) else lines
    all_nums = {l.line_num for l in flat}

    errors: list[str] = []
    warnings: list[str] = []

    for line in flat:
        # validate from= references
        for ref in line.from_refs:
            if ref not in all_nums:
                errors.append(
                    f"Line {line.line_num}: from={ref} references non-existent line"
                )
            if ref >= line.line_num:
                errors.append(
                    f"Line {line.line_num}: from={ref} is a forward reference (forbidden)"
                )

        # indent depth
        if line.depth > MAX_DEPTH:
            errors.append(
                f"Line {line.line_num}: depth={line.depth} exceeds max ({MAX_DEPTH})"
            )

        # conf range
        if line.conf is not None and not (0.0 <= line.conf <= 1.0):
            errors.append(
                f"Line {line.line_num}: conf={line.conf} out of range [0, 1]"
            )

        # conf_range logic
        if line.conf_range:
            lo, hi = line.conf_range
            if lo > hi:
                errors.append(
                    f"Line {line.line_num}: conf_range={lo}..{hi} invalid (low > high)"
                )

        # recommend hint= on void-B lines
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


# ── CLI / Quick Test ────────────────────────────────────────────────────────

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

    SAMPLE_V2 = """
~ schema patient  fields=id,dx,allergy   lvl=1
. allergy=penicillin^C  from=2,5
. sensor_ok!=timeout
: resolve allergy  was=C  now=F  from=3  why=skin_test_negative
! obstruction dx  from=3  note=imaging_unavailable
""".strip()

    print("=" * 60)
    print("  WCY Parser v2.0 (WCY-2) — Quick Test")
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

    print("\n" + "=" * 60)
    print("  v2 additions (SPEC section 4.2)")
    print("=" * 60)

    v2 = flatten(parse_wcy(SAMPLE_V2))
    for l in v2:
        rails = " ".join(f"{s.key or s.value}^{s.rail}"
                         for s in l.slots if not s.is_void)
        print(f"  #{l.line_num} {l.phase} lvl={l.lvl} kind={l.kind:<11} {rails}")

    print(f"\n[audit_trail] → {len(audit_trail(v2))} record(s)")
    for entry in audit_trail(v2):
        print(f"  {entry}")
