# -*- coding: utf-8 -*-
"""
wcy_merge.py — WCY-2 Reference Merge Semantics v2.0
===================================================
Reference implementation of SPEC.md sections 3 (value model), 5 (merge
semantics), 6 (mutation discipline), 7 (resolution records) and 11 (JSON
interoperability).

An atomic value is a dual-rail evidence pair `(assert_rail, refute_rail)`
with four states — UNK (0,0), TRU (1,0), FAL (0,1), CON (1,1) — and merge
is the railwise join. `verify_laws()` re-checks, exhaustively and in pure
Python, the same laws that lean/wcymerge/WcyMerge.lean checks in the Lean
kernel, under the same names.

Usage:
    from wcy_parser import parse_wcy
    from wcy_merge import (
        document_from_lines, merge_documents, apply_resolutions,
        embed_json, project_json, verify_laws,
    )

    a = document_from_lines(parse_wcy(text_a), label='A')
    b = document_from_lines(parse_wcy(text_b), label='B')
    m = merge_documents(a, b)
    m.atom('allergy').state        # 'CON'
    final = apply_resolutions(m)
    final.atom('allergy').state    # 'FAL'


MERGE POLICY (exact, as implemented)
------------------------------------
SPEC section 5 fixes the merge on atoms but leaves the document-level
bookkeeping to the implementation. This module implements exactly:

1. *Atom identity.* An atom is identified by (block ordinal, tag). Tags
   come from keyed slots (`tag=value`, `tag!=value`, `?tag`) and from bare
   positional slots, which become flag atoms (tag = the slot text, no
   value). Only observe (`.`) and infer (`:`) lines carry evidence; meta,
   act and exception lines contribute records and provenance, not atoms.
   The bookkeeping tags `hint` and `lvl` are never atoms.

2. *Block matching.* Blocks are matched by **ordinal position among the
   atom-bearing blocks** of each document (the first block that contains
   an atom is ordinal 0, the next is 1, ...). Blank-line blocks that carry
   only meta/act lines therefore do not shift the alignment.

3. *Atom merge.* Railwise join per level: for each level m present in
   either side, `(a1|a2, r1|r2)`. Levels are never collapsed into one
   another (SPEC section 8: level families are irreducible). A level
   present on only one side is carried over unchanged.

4. *Values.* Rails carry the evidence; the value text is data. Distinct
   values under one tag are kept as an ordered value set (first-seen
   order, duplicates dropped) — never last-writer-wins, never dropped.

5. *Provenance.* `from=` sources union, first-seen order, deduplicated.
   Numeric `from=N` line references are qualified by the source document
   label (`A#2`) because line numbers are document-local; named sources
   (`from=intake_form`) are kept verbatim.

6. *Unmatched tags.* Tags present in only one document are carried into
   the merge unchanged (union of key sets, per block ordinal).

7. *Records.* Resolution and obstruction records are **appended**, never
   merged, deduplicated or dropped: `merge_documents(a, b).records ==
   a.records + b.records`. They are the audit surface (SPEC section 7.2)
   and applying them is a separate, explicit step (`apply_resolutions`).

8. *Conflict.* Merge never raises on conflicting evidence and never
   prefers a side: `TRU join FAL = CON`, surfaced as an ordinary state
   (SPEC section 5.2).

Merge is therefore associative, commutative and idempotent on documents
because it is so on atoms (verify_laws) and the block/tag keying is a
plain union of finite maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from wcy_parser import WCYLine, flatten


# ── Section 3: the dual-rail value model ────────────────────────────────────

@dataclass(frozen=True)
class Atom:
    """A dual-rail atom: (assert_rail, refute_rail) — SPEC section 3.1."""
    assert_rail: bool = False
    refute_rail: bool = False

    @property
    def state(self) -> str:
        """'UNK' | 'TRU' | 'FAL' | 'CON'."""
        return _STATE_TABLE[(self.assert_rail, self.refute_rail)]

    @property
    def rail(self) -> str:
        """The rail letter used by the surface syntax: 'U'|'T'|'F'|'C'."""
        return STATE_TO_RAIL[self.state]

    @property
    def is_resolved(self) -> bool:
        """True on the resolved face R = {TRU, FAL} (SPEC section 3.1)."""
        return self.state in ('TRU', 'FAL')

    def __str__(self) -> str:
        return self.state


UNK = Atom(False, False)
TRU = Atom(True,  False)
FAL = Atom(False, True)
CON = Atom(True,  True)

ATOMS: tuple[Atom, ...] = (UNK, TRU, FAL, CON)

_STATE_TABLE = {
    (False, False): 'UNK',
    (True,  False): 'TRU',
    (False, True):  'FAL',
    (True,  True):  'CON',
}

RAIL_TO_STATE = {'U': 'UNK', 'T': 'TRU', 'F': 'FAL', 'C': 'CON'}
STATE_TO_RAIL = {v: k for k, v in RAIL_TO_STATE.items()}

_RAIL_TO_ATOM = {'U': UNK, 'T': TRU, 'F': FAL, 'C': CON}


def atom_for_rail(rail: str) -> Atom:
    """Map a surface rail letter ('T'|'F'|'U'|'C') to its atom."""
    try:
        return _RAIL_TO_ATOM[rail.upper()]
    except (KeyError, AttributeError):
        raise ValueError(f"unknown rail letter: {rail!r}")


def atom_for_state(state: str) -> Atom:
    """Map a state name ('TRU'|'FAL'|'UNK'|'CON') to its atom."""
    try:
        return _RAIL_TO_ATOM[STATE_TO_RAIL[state.upper()]]
    except (KeyError, AttributeError):
        raise ValueError(f"unknown state: {state!r}")


# ── Section 5.1: the merge, the meet, and the information order ─────────────

def merge(x: Atom, y: Atom) -> Atom:
    """Railwise join — the document merge on atoms (SPEC section 5.1)."""
    return Atom(x.assert_rail or y.assert_rail,
                x.refute_rail or y.refute_rail)


def meet(x: Atom, y: Atom) -> Atom:
    """Railwise meet (evidence intersection; used by absorption)."""
    return Atom(x.assert_rail and y.assert_rail,
                x.refute_rail and y.refute_rail)


def le(x: Atom, y: Atom) -> bool:
    """Information order: x <= y iff y has at least x's evidence."""
    return ((not x.assert_rail) or y.assert_rail) and \
           ((not x.refute_rail) or y.refute_rail)


def merge_all(atoms: Iterable[Atom]) -> Atom:
    """Fold the merge over any number of atoms (order-irrelevant)."""
    result = UNK
    for a in atoms:
        result = merge(result, a)
    return result


# ── The Lean-checked laws, mirrored in pure Python ──────────────────────────

@dataclass
class LawReport:
    """Result of verify_laws(): one entry per kernel-checked theorem."""
    laws: dict[str, bool] = field(default_factory=dict)
    cases: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True iff every law held on every case."""
        return all(self.laws.values())

    @property
    def failures(self) -> list[str]:
        return sorted(name for name, held in self.laws.items() if not held)

    @property
    def total_cases(self) -> int:
        return sum(self.cases.values())

    def __str__(self) -> str:
        status = 'PASS' if self.ok else 'FAIL'
        return (f"LawReport({status}: {len(self.laws)} laws, "
                f"{self.total_cases} cases, failures={self.failures})")


def verify_laws() -> LawReport:
    """
    Exhaustively re-check, in pure Python, the merge laws that
    lean/wcymerge/WcyMerge.lean checks in the Lean kernel. Theorem names
    match that file one for one:

      merge_assoc, merge_comm, merge_idem, merge_unk, meet_assoc,
      absorption, conflict_surfaces, merge_monotone, merge_lub,
      retraction_is_monotone

    The four-element carrier makes exhaustion cheap: 4 unary, 16 binary
    and 64 ternary cases.
    """
    laws: dict[str, bool] = {}
    cases: dict[str, int] = {}

    def record(name: str, results: list[bool]) -> None:
        laws[name] = all(results)
        cases[name] = len(results)

    pairs = [(x, y) for x in ATOMS for y in ATOMS]
    triples = [(x, y, z) for x in ATOMS for y in ATOMS for z in ATOMS]

    record('merge_assoc',
           [merge(merge(x, y), z) == merge(x, merge(y, z))
            for x, y, z in triples])
    record('merge_comm',
           [merge(x, y) == merge(y, x) for x, y in pairs])
    record('merge_idem',
           [merge(x, x) == x for x in ATOMS])
    record('merge_unk',
           [merge(UNK, x) == x for x in ATOMS])
    record('meet_assoc',
           [meet(meet(x, y), z) == meet(x, meet(y, z))
            for x, y, z in triples])
    record('absorption',
           [merge(x, meet(x, y)) == x and meet(x, merge(x, y)) == x
            for x, y in pairs])
    record('conflict_surfaces',
           [merge(TRU, FAL) == CON])
    record('merge_monotone',
           [le(x, merge(x, y)) and le(y, merge(x, y)) for x, y in pairs])
    record('merge_lub',
           [(not (le(x, z) and le(y, z))) or le(merge(x, y), z)
            for x, y, z in triples])
    record('retraction_is_monotone',
           [merge(TRU, Atom(False, True)) == CON])

    return LawReport(laws=laws, cases=cases)


# ── Document model ──────────────────────────────────────────────────────────

# Tags that are line bookkeeping, never evidence atoms.
BOOKKEEPING_TAGS = frozenset({'hint', 'lvl'})

# Only these phases carry evidence (SPEC section 4.2).
EVIDENCE_PHASES = frozenset({'.', ':'})


@dataclass
class AtomRecord:
    """
    One tag's evidence inside a document block.

    `levels` is the level-indexed family of SPEC section 8: level m maps
    to the atom asserted at that level. It is never compressed to a single
    summary; `atom` merely *derives* the coarsest level's atom for display
    and for the resolved-face projection.
    """
    tag: str
    path: tuple[Any, ...] = ()
    values: list[Any] = field(default_factory=list)
    levels: dict[int, Atom] = field(default_factory=dict)
    provenance: list[str] = field(default_factory=list)

    @property
    def level(self) -> int:
        """The coarsest level carrying evidence for this tag."""
        return min(self.levels) if self.levels else 1

    @property
    def atom(self) -> Atom:
        """Derived display atom: the atom at the coarsest recorded level."""
        return self.levels.get(self.level, UNK)

    @property
    def state(self) -> str:
        return self.atom.state

    @property
    def value(self) -> Any:
        """First value of the value set (None if the atom carries no value)."""
        return self.values[0] if self.values else None

    def __str__(self) -> str:
        return f"{self.tag}={self.values}^{self.atom.rail}"


@dataclass
class DocRecord:
    """A resolution or obstruction record carried by a document (SPEC 7)."""
    kind: str                         # 'resolve' | 'obstruction'
    target: str | None = None
    was: str | None = None
    now: str | None = None
    why: str | None = None
    note: str | None = None
    provenance: list[str] = field(default_factory=list)
    line_num: int | None = None
    source: str | None = None         # label of the document it came from

    def to_dict(self) -> dict[str, Any]:
        return {
            'kind': self.kind, 'target': self.target,
            'was': self.was, 'now': self.now, 'why': self.why,
            'note': self.note, 'provenance': list(self.provenance),
            'line_num': self.line_num, 'source': self.source,
        }


@dataclass
class WCY2Document:
    """
    A parsed WCY-2 document reduced to its merge-relevant content:
    atom-bearing blocks (by ordinal) plus the appended audit records.
    """
    blocks: dict[int, dict[str, AtomRecord]] = field(default_factory=dict)
    records: list[DocRecord] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    # ── access helpers ──────────────────────────────────────────────────
    @property
    def resolutions(self) -> list[DocRecord]:
        return [r for r in self.records if r.kind == 'resolve']

    @property
    def obstructions(self) -> list[DocRecord]:
        return [r for r in self.records if r.kind == 'obstruction']

    def tags(self, block: int | None = None) -> list[str]:
        """All tags, in block then insertion order."""
        out: list[str] = []
        for idx in sorted(self.blocks):
            if block is not None and idx != block:
                continue
            for tag in self.blocks[idx]:
                if tag not in out:
                    out.append(tag)
        return out

    def atom(self, tag: str, block: int | None = None) -> AtomRecord | None:
        """
        The AtomRecord for `tag`. With block=None, searches every block in
        order and returns the first match.
        """
        for idx in sorted(self.blocks):
            if block is not None and idx != block:
                continue
            rec = self.blocks[idx].get(tag)
            if rec is not None:
                return rec
        return None

    def state(self, tag: str, block: int | None = None) -> str | None:
        rec = self.atom(tag, block)
        return rec.state if rec else None

    def unresolved(self) -> list[tuple[int, str, str]]:
        """(block, tag, state) for every atom off the resolved face."""
        out = []
        for idx in sorted(self.blocks):
            for tag, rec in self.blocks[idx].items():
                if not rec.atom.is_resolved:
                    out.append((idx, tag, rec.state))
        return out


# ── Building a document from parsed lines ───────────────────────────────────

def _provenance_of(line: WCYLine, label: str | None) -> list[str]:
    """
    Provenance tokens for a line: numeric `from=N` refs qualified by the
    document label (line numbers are document-local), named `from=` sources
    verbatim.
    """
    prefix = f"{label}#" if label else "#"
    out = [f"{prefix}{n}" for n in line.from_refs]
    out.extend(line.from_labels)
    return out


def _atom_slots(line: WCYLine) -> list[tuple[str, Any, Atom]]:
    """(tag, value, atom) triples contributed by one evidence line."""
    out: list[tuple[str, Any, Atom]] = []
    for slot in line.slots:
        if slot.key == '__backref__':
            continue
        if slot.is_void:
            out.append((slot.key or slot.value, None, UNK))
            continue
        if slot.key is None:
            # bare positional value -> flag atom, no value text
            out.append((slot.value, None, atom_for_rail(slot.rail)))
            continue
        if slot.key in BOOKKEEPING_TAGS:
            continue
        out.append((slot.key, slot.value, atom_for_rail(slot.rail)))
    return out


def _add_value(values: list[Any], value: Any) -> None:
    """Append a value to the ordered value set (dedup by equality)."""
    if value is None:
        return
    for existing in values:
        if existing == value and type(existing) is type(value):
            return
    values.append(value)


def _add_provenance(target: list[str], sources: Iterable[str]) -> None:
    for src in sources:
        if src not in target:
            target.append(src)


def document_from_lines(lines: list[WCYLine],
                        label: str | None = None) -> WCY2Document:
    """
    Reduce parsed WCY lines to a WCY2Document (see MERGE POLICY above).

    Args:
        lines: result of parse_wcy()
        label: optional document/agent label, used to qualify numeric
               `from=` line references in provenance.

    Returns:
        WCY2Document with atom-bearing blocks numbered by ordinal.
    """
    flat = flatten(lines) if any(l.children for l in lines) else list(lines)

    blocks: dict[int, dict[str, AtomRecord]] = {}
    records: list[DocRecord] = []
    ordinal_of: dict[int, int] = {}   # source block_index -> block ordinal

    for line in flat:
        provenance = _provenance_of(line, label)

        if line.kind != 'normal' and line.record is not None:
            records.append(DocRecord(
                kind=line.record.kind,
                target=line.record.target,
                was=line.record.was,
                now=line.record.now,
                why=line.record.why,
                note=line.record.note,
                provenance=provenance,
                line_num=line.line_num,
                source=label,
            ))
            continue

        if line.phase not in EVIDENCE_PHASES:
            continue

        contributions = _atom_slots(line)
        if not contributions:
            continue

        if line.block_index not in ordinal_of:
            ordinal_of[line.block_index] = len(ordinal_of)
        ordinal = ordinal_of[line.block_index]
        block = blocks.setdefault(ordinal, {})

        for tag, value, atom in contributions:
            rec = block.get(tag)
            if rec is None:
                rec = AtomRecord(tag=tag, path=(tag,))
                block[tag] = rec
            _add_value(rec.values, value)
            rec.levels[line.lvl] = merge(rec.levels.get(line.lvl, UNK), atom)
            _add_provenance(rec.provenance, provenance)

    return WCY2Document(blocks=blocks, records=records,
                        labels=[label] if label else [])


# ── Section 5: document merge ───────────────────────────────────────────────

def merge_atoms(a: AtomRecord, b: AtomRecord) -> AtomRecord:
    """Merge two AtomRecords for the same tag (railwise join per level)."""
    out = AtomRecord(tag=a.tag, path=a.path or b.path,
                     values=list(a.values),
                     levels=dict(a.levels),
                     provenance=list(a.provenance))
    for value in b.values:
        _add_value(out.values, value)
    for lvl, atom in b.levels.items():
        out.levels[lvl] = merge(out.levels.get(lvl, UNK), atom)
    _add_provenance(out.provenance, b.provenance)
    return out


def merge_documents(a: WCY2Document, b: WCY2Document) -> WCY2Document:
    """
    Merge two WCY-2 documents (SPEC section 5) under the policy documented
    at the top of this module. Never raises on conflict, never prefers a
    side, never drops a record.
    """
    blocks: dict[int, dict[str, AtomRecord]] = {}

    for idx in sorted(set(a.blocks) | set(b.blocks)):
        left = a.blocks.get(idx, {})
        right = b.blocks.get(idx, {})
        out: dict[str, AtomRecord] = {}
        for tag, rec in left.items():
            other = right.get(tag)
            out[tag] = merge_atoms(rec, other) if other else _copy_atom(rec)
        for tag, rec in right.items():
            if tag not in out:            # unmatched tags union in
                out[tag] = _copy_atom(rec)
        blocks[idx] = out

    labels = list(a.labels)
    _add_provenance(labels, b.labels)

    return WCY2Document(
        blocks=blocks,
        records=list(a.records) + list(b.records),   # appended, never merged
        labels=labels,
    )


def merge_many(docs: Iterable[WCY2Document]) -> WCY2Document:
    """Fold merge_documents over any number of documents (order-irrelevant)."""
    result = WCY2Document()
    for doc in docs:
        result = merge_documents(result, doc)
    return result


def _copy_atom(rec: AtomRecord) -> AtomRecord:
    return AtomRecord(tag=rec.tag, path=rec.path, values=list(rec.values),
                      levels=dict(rec.levels), provenance=list(rec.provenance))


def copy_document(doc: WCY2Document) -> WCY2Document:
    """Deep-enough copy: atoms and record lists are fresh."""
    return WCY2Document(
        blocks={idx: {tag: _copy_atom(rec) for tag, rec in block.items()}
                for idx, block in doc.blocks.items()},
        records=[DocRecord(**r.to_dict()) for r in doc.records],
        labels=list(doc.labels),
    )


# ── Section 7: applying resolution records ──────────────────────────────────

def apply_resolutions(doc: WCY2Document, strict: bool = False,
                      level: int | None = None) -> WCY2Document:
    """
    Apply the document's resolution records to its atoms (SPEC section 7.1).

    Resolution is the one non-monotone act WCY-2 allows (SPEC section 6.3):
    it *replaces* the rails of the target atom with the record's `now`
    state instead of joining. The records themselves are kept — applying a
    resolution never consumes the audit trail. Obstruction records are
    never applied; they are recorded failures (SPEC section 7.3).

    A record with an unknown target tag is left standing (nothing to move).
    A record is applied to every block that carries the target tag.

    Args:
        strict: raise ValueError when a record's `was=` disagrees with the
                atom's current state.
        level:  level to write the resolved state at; default = the atom's
                coarsest recorded level (its other levels are untouched,
                SPEC section 8).
    """
    out = copy_document(doc)

    for rec in out.records:
        if rec.kind != 'resolve' or not rec.target or not rec.now:
            continue
        new_atom = atom_for_rail(rec.now)
        for block in out.blocks.values():
            atom_rec = block.get(rec.target)
            if atom_rec is None:
                continue
            if strict and rec.was and atom_rec.atom.rail != rec.was.upper():
                raise ValueError(
                    f"resolution for {rec.target!r} says was={rec.was} but "
                    f"the atom is {atom_rec.atom.rail}")
            target_level = level if level is not None else atom_rec.level
            atom_rec.levels[target_level] = new_atom
            _add_provenance(atom_rec.provenance, rec.provenance)

    return out


# ── Section 11: JSON interoperability ───────────────────────────────────────

ROOT_TAG = '$'

_SCALARS = (str, int, float, bool)


@dataclass
class ProjectionSidecar:
    """
    The soft-failure record of a resolved-face projection (SPEC 11.1).

    unresolved: paths whose atom is UNK or CON (emitted as JSON null)
    refuted:    paths whose atom is FAL (omitted from the JSON)
    ambiguous:  TRU paths carrying more than one value (emitted as null;
                the merge kept every value rather than picking one)
    """
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    refuted: list[dict[str, Any]] = field(default_factory=list)
    ambiguous: list[dict[str, Any]] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.unresolved or self.refuted or self.ambiguous)

    @property
    def unresolved_paths(self) -> list[str]:
        return [entry['path'] for entry in self.unresolved]

    def to_dict(self) -> dict[str, Any]:
        return {'unresolved': self.unresolved,
                'refuted': self.refuted,
                'ambiguous': self.ambiguous}


def path_to_tag(path: tuple[Any, ...]) -> str:
    """Render a JSON path as a WCY tag: ('a', 0, 'b') -> 'a.0.b'."""
    return '.'.join(str(p) for p in path) if path else ROOT_TAG


def _is_leaf(value: Any) -> bool:
    if value is None or isinstance(value, _SCALARS):
        return True
    if isinstance(value, (dict, list)) and not value:
        return True      # empty container is an asserted-empty leaf
    return False


def embed_json(obj: Any, label: str | None = None) -> WCY2Document:
    """
    Embed a JSON value as a resolved-face WCY-2 document (SPEC 11.2).

    Every leaf becomes a `^T` atom keyed by its dotted path; `null`
    becomes UNK. Empty objects/arrays are asserted-empty leaves (SPEC 3.1
    distinguishes these from UNK). The embedding is lossless: projection
    is its left inverse (`project_json(embed_json(x))[0] == x`).
    """
    block: dict[str, AtomRecord] = {}

    def walk(value: Any, path: tuple[Any, ...]) -> None:
        if _is_leaf(value):
            tag = path_to_tag(path)
            atom = UNK if value is None else TRU
            block[tag] = AtomRecord(
                tag=tag, path=path,
                values=[] if value is None else [value],
                levels={1: atom},
                provenance=[label] if label else [],
            )
            return
        if isinstance(value, dict):
            for key, sub in value.items():
                walk(sub, path + (str(key),))
        elif isinstance(value, list):
            for i, sub in enumerate(value):
                walk(sub, path + (i,))
        else:
            raise TypeError(f"not a JSON value: {type(value).__name__}")

    walk(obj, ())
    return WCY2Document(blocks={0: block} if block else {},
                        labels=[label] if label else [])


def _flatten_atoms(doc: WCY2Document) -> dict[str, AtomRecord]:
    """
    Collapse the document's blocks to one tag -> AtomRecord map for
    projection; identical tags in different blocks are joined railwise
    (the same merge, so the projection is block-order independent).
    """
    out: dict[str, AtomRecord] = {}
    for idx in sorted(doc.blocks):
        for tag, rec in doc.blocks[idx].items():
            out[tag] = merge_atoms(out[tag], rec) if tag in out else rec
    return out


def _insert(root: dict[str, Any], path: tuple[Any, ...], value: Any) -> Any:
    """
    Insert `value` at `path` inside the container held by root['v'],
    growing dicts for str segments and lists for int segments.
    """
    if not path:
        root['v'] = value
        return root['v']

    if root.get('v') is None:
        root['v'] = [] if isinstance(path[0], int) else {}

    node = root['v']
    for i, seg in enumerate(path[:-1]):
        nxt = path[i + 1]
        child_default: Any = [] if isinstance(nxt, int) else {}
        if isinstance(seg, int):
            while len(node) <= seg:
                node.append(None)
            if node[seg] is None:
                node[seg] = child_default
            node = node[seg]
        else:
            if seg not in node or node[seg] is None:
                node[seg] = child_default
            node = node[seg]

    last = path[-1]
    if isinstance(last, int):
        while len(node) <= last:
            node.append(None)
        node[last] = value
    else:
        node[last] = value
    return root['v']


def project_json(doc: WCY2Document) -> tuple[Any, ProjectionSidecar]:
    """
    Resolved-face projection (SPEC 11.1): WCY-2 document -> JSON + sidecar.

      TRU  -> the value (a valueless flag atom projects to `true`)
      FAL  -> omitted from the JSON, listed in sidecar.refuted
      UNK  -> `null`, listed in sidecar.unresolved
      CON  -> `null`, listed in sidecar.unresolved (never coerced to a
              value, never resolved silently — SPEC section 5.2)

    Returns:
        (json_value, ProjectionSidecar)
    """
    sidecar = ProjectionSidecar()
    holder: dict[str, Any] = {'v': None}
    saw_any = False

    for tag, rec in _flatten_atoms(doc).items():
        path = rec.path if rec.path else (tag,)
        if path == (ROOT_TAG,):
            path = ()
        state = rec.state
        entry = {'path': path_to_tag(path), 'tag': tag, 'state': state}

        if state == 'FAL':
            entry['value'] = rec.value
            sidecar.refuted.append(entry)
            continue

        saw_any = True
        if state == 'TRU':
            if len(rec.values) > 1:
                entry['values'] = list(rec.values)
                sidecar.ambiguous.append(entry)
                _insert(holder, path, None)
            else:
                value = rec.values[0] if rec.values else True
                _insert(holder, path, value)
        else:                                   # UNK or CON
            sidecar.unresolved.append(entry)
            _insert(holder, path, None)

    if not saw_any and holder['v'] is None:
        holder['v'] = {}

    return holder['v'], sidecar


# ── CLI / Quick Test ────────────────────────────────────────────────────────

if __name__ == '__main__':
    from wcy_parser import parse_wcy

    DOC_A = """
~ schema patient  fields=id,dx,allergy   lvl=1

. id=P-1043
. allergy=penicillin            | from=intake_form
: ?dx  hint=labs+imaging  conf_range=0.3..0.7
""".strip()

    DOC_B = """
. allergy!=penicillin           | from=lab_panel_2
. labs  wbc=14.2  crp=high
""".strip()

    RESOLUTION = """
. skin_test=negative
: resolve allergy  was=C  now=F  from=skin_test  why=skin_test_negative
""".strip()

    print("=" * 60)
    print("  WCY-2 Merge Semantics — Quick Test")
    print("=" * 60)

    report = verify_laws()
    print(f"\n[verify_laws] {report}")
    for name in sorted(report.laws):
        print(f"  {name:<24} {'ok' if report.laws[name] else 'FAILED':<6} "
              f"({report.cases[name]} cases)")

    a = document_from_lines(parse_wcy(DOC_A), label='A')
    b = document_from_lines(parse_wcy(DOC_B), label='B')
    m = merge_documents(a, merge_documents(b, document_from_lines(
        parse_wcy(RESOLUTION), label='C')))

    print("\n[merge_documents] merged atoms:")
    for tag in m.tags():
        rec = m.atom(tag)
        print(f"  {tag:<12} {rec.state:<4} values={rec.values} "
              f"from={rec.provenance}")

    print(f"\n[records] {len(m.records)} appended record(s)")
    for rec in m.records:
        print(f"  {rec.to_dict()}")

    final = apply_resolutions(m)
    print(f"\n[apply_resolutions] allergy -> {final.state('allergy')} "
          f"(records kept: {len(final.records)})")

    value, sidecar = project_json(final)
    print(f"\n[project_json] {value}")
    print(f"  sidecar unresolved={sidecar.unresolved_paths} "
          f"refuted={[e['path'] for e in sidecar.refuted]}")

    sample = {'id': 'P-1043', 'labs': {'wbc': 14.2, 'crp': 'high'},
              'dx': None, 'tags': ['a', 'b']}
    round_trip, side2 = project_json(embed_json(sample))
    print(f"\n[embed_json/project_json] round-trip identical: "
          f"{round_trip == sample}")
    print(f"  sidecar unresolved={side2.unresolved_paths}")
