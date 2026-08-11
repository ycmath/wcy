# -*- coding: utf-8 -*-
"""
test_wcy2.py — conformance tests for the WCY-2 reference modules.

Run from the repository root:

    python -X utf8 -m unittest discover tests

Covers:
  * v1 compatibility — the whole shipped corpus re-parsed and compared,
    field by field, against an independent re-implementation of the v1
    slot grammar embedded in this file (SPEC.md section 11.3);
  * v2 syntax — rail suffixes, refutation sugar, level tags, resolution
    and obstruction records (SPEC.md section 4.2);
  * merge laws — verify_laws() against the theorem names kernel-checked in
    lean/wcymerge/WcyMerge.lean, plus document-level associativity,
    commutativity and idempotence (SPEC.md section 5);
  * the SPEC Appendix B worked example, end to end;
  * JSON embedding / resolved-face projection and sidecars (section 11).
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from typing import Any

from wcy_parser import (
    parse_wcy, flatten, validate, extract_voids, extract_records,
    void_summary, resolve_chain, audit_trail, WCYLine,
)
from wcy_merge import (
    Atom, UNK, TRU, FAL, CON, ATOMS, merge, meet, le, merge_all,
    atom_for_rail, verify_laws, document_from_lines, merge_documents,
    merge_many, apply_resolutions, embed_json, project_json, path_to_tag,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / 'data'
LEAN_FILE = REPO_ROOT / 'lean' / 'wcymerge' / 'WcyMerge.lean'

WCY_TEXT_FIELDS = ('wcy_reasoning', 'input', 'context')


# ── An independent re-implementation of the v1 slot grammar ─────────────────
# Deliberately a separate implementation, so the compatibility test compares
# the v2 parser against the v1 rules rather than against itself.

_V1_QUOTED = re.compile(r'^"([^"]*)"$')
_V1_TAGVAL = re.compile(r'^(\w[\w\-\.]*)\s*=\s*(.+)$')
_V1_VOID = re.compile(r'^\?(\w+)$')
_V1_BACKREF = re.compile(r'^\{(\d+)\}$')
_V1_CONF = re.compile(r'^([\d.]+)\.\.([\d.]+)$')
_V1_PHASES = '.:>~!'


def _v1_tokenize(rest: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    in_quote = False
    for ch in rest:
        if ch == '"':
            in_quote = not in_quote
            current.append(ch)
        elif ch in (' ', '|') and not in_quote:
            tok = ''.join(current).strip()
            if tok:
                tokens.append(tok)
            current = []
        else:
            current.append(ch)
    tok = ''.join(current).strip()
    if tok:
        tokens.append(tok)
    return tokens


def _v1_parse(text: str) -> list[dict[str, Any]]:
    """Parse WCY text under the v1.1 rules; returns comparable dicts."""
    out: list[dict[str, Any]] = []
    line_num = 0
    block_index = 0

    for raw in text.split('\n'):
        stripped = raw.strip()
        if not stripped:
            block_index += 1
            continue
        if stripped.startswith('#'):
            continue
        line_num += 1
        if len(stripped) < 2 or stripped[0] not in _V1_PHASES:
            continue
        if stripped[1] != ' ':
            continue

        depth = min((len(raw) - len(raw.lstrip(' '))) // 2, 2)
        slots: list[tuple[str | None, str, bool]] = []
        tags: dict[str, str] = {}
        void_tags: list[str] = []
        from_refs: list[int] = []
        conf: float | None = None
        conf_range: tuple[float, float] | None = None

        for token in _v1_tokenize(stripped[2:].strip()):
            if token.startswith('from='):
                for n in token[5:].split(','):
                    n = n.strip()
                    if n.isdigit():
                        from_refs.append(int(n))
                continue
            if token.startswith('conf_range='):
                m = _V1_CONF.match(token[11:])
                if m:
                    conf_range = (float(m.group(1)), float(m.group(2)))
                continue
            if token.startswith('conf='):
                try:
                    conf = float(token[5:])
                except ValueError:
                    pass
                continue
            m = _V1_VOID.match(token)
            if m:
                void_tags.append(m.group(1))
                slots.append((m.group(1), '', True))
                continue
            m = _V1_TAGVAL.match(token)
            if m:
                key, val = m.group(1), m.group(2).strip()
                qm = _V1_QUOTED.match(val)
                if qm:
                    val = qm.group(1)
                slots.append((key, val, False))
                tags[key] = val
                continue
            m = _V1_BACKREF.match(token)
            if m:
                slots.append(('__backref__', m.group(1), False))
                continue
            qm = _V1_QUOTED.match(token)
            slots.append((None, qm.group(1) if qm else token, False))

        out.append({
            'line_num': line_num, 'phase': stripped[0], 'depth': depth,
            'slots': slots, 'tags': tags, 'void_tags': void_tags,
            'from_refs': from_refs, 'conf': conf, 'conf_range': conf_range,
            'block_index': block_index,
        })
    return out


def _v2_shape(line: WCYLine) -> dict[str, Any]:
    """Project a v2 WCYLine onto the v1 field set, for comparison."""
    return {
        'line_num': line.line_num, 'phase': line.phase, 'depth': line.depth,
        'slots': [(s.key, s.value, s.is_void) for s in line.slots],
        'tags': line.tags, 'void_tags': line.void_tags,
        'from_refs': line.from_refs, 'conf': line.conf,
        'conf_range': line.conf_range, 'block_index': line.block_index,
    }


def _load_corpus() -> list[tuple[str, str]]:
    """[(trace_id, wcy_text)] over every shipped .jsonl document."""
    corpus: list[tuple[str, str]] = []
    for path in sorted(DATA_DIR.glob('*.jsonl')):
        with path.open(encoding='utf-8') as fh:
            for i, raw in enumerate(fh):
                raw = raw.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                ident = row.get('id', f'{path.stem}:{i}')
                for fieldname in WCY_TEXT_FIELDS:
                    text = row.get(fieldname)
                    if isinstance(text, str) and text.strip():
                        corpus.append((f'{ident}.{fieldname}', text))
    return corpus


CORPUS = _load_corpus()


# ── v1 compatibility ────────────────────────────────────────────────────────

class TestV1Compatibility(unittest.TestCase):
    """SPEC 11.3: a v1 document is a Core-conformant v2 document."""

    def test_corpus_is_present(self):
        self.assertGreaterEqual(len(CORPUS), 540)

    def test_every_trace_still_parses(self):
        empty = []
        for ident, text in CORPUS:
            lines = flatten(parse_wcy(text))
            if not lines:
                empty.append(ident)
        self.assertEqual(empty, [], 'traces that produced no lines')

    def test_v1_fields_unchanged(self):
        """Every v1 field of every line matches the v1 grammar exactly."""
        checked = 0
        for ident, text in CORPUS:
            expected = _v1_parse(text)
            actual = [_v2_shape(l) for l in flatten(parse_wcy(text))]
            self.assertEqual(len(actual), len(expected),
                             f'line count differs for {ident}')
            for exp, act in zip(expected, actual):
                self.assertEqual(act, exp, f'field mismatch in {ident}')
                checked += 1
        self.assertGreater(checked, 10000)

    def test_v2_fields_take_defaults_on_v1_documents(self):
        for ident, text in CORPUS:
            for line in flatten(parse_wcy(text)):
                self.assertEqual(line.lvl, 1, ident)
                self.assertEqual(line.kind, 'normal', ident)
                self.assertIsNone(line.record, ident)
                for slot in line.slots:
                    self.assertEqual(slot.rail, 'U' if slot.is_void else 'T',
                                     f'{ident} slot {slot.key}')

    def test_v1_utilities_still_work(self):
        text = CORPUS[0][1]
        lines = parse_wcy(text)
        self.assertTrue(validate(lines).stats['total_lines'] > 0)
        self.assertIsInstance(extract_voids(lines), list)
        self.assertIn('total', void_summary(lines))
        flat = flatten(lines)
        self.assertIsInstance(resolve_chain(flat, flat[-1].line_num), list)

    def test_corpus_carries_voids(self):
        total = sum(void_summary(parse_wcy(t))['total'] for _, t in CORPUS)
        self.assertGreater(total, 0)


# ── v2 syntax (SPEC 4.2) ────────────────────────────────────────────────────

class TestV2Syntax(unittest.TestCase):

    def _one(self, text: str) -> WCYLine:
        lines = flatten(parse_wcy(text))
        self.assertEqual(len(lines), 1)
        return lines[0]

    def test_rail_suffix_on_tag_value(self):
        line = self._one('. allergy=penicillin^C  from=2,5')
        slot = line.slots[0]
        self.assertEqual((slot.key, slot.value, slot.rail),
                         ('allergy', 'penicillin', 'C'))
        self.assertEqual(line.tags['allergy'], 'penicillin')
        self.assertEqual(line.from_refs, [2, 5])

    def test_all_four_rails(self):
        line = self._one('. a=1^T  b=2^F  c=3^U  d=4^C')
        self.assertEqual([s.rail for s in line.slots], ['T', 'F', 'U', 'C'])
        self.assertEqual([s.value for s in line.slots], ['1', '2', '3', '4'])

    def test_bare_value_is_asserting(self):
        line = self._one('. sensor_ok=true')
        self.assertEqual(line.slots[0].rail, 'T')

    def test_rail_suffix_on_positional_slot(self):
        line = self._one('. sensor_ok^C  from=A1,B1')
        self.assertEqual((line.slots[0].key, line.slots[0].value,
                          line.slots[0].rail), (None, 'sensor_ok', 'C'))
        self.assertEqual(line.from_refs, [])
        self.assertEqual(line.from_labels, ['A1', 'B1'])

    def test_rail_marker_must_end_the_token(self):
        """Math text such as A^T*lambda is not a rail suffix (v1 corpus)."""
        line = self._one('. derivation  grad_f+A^T*lambda=0')
        self.assertEqual([(s.key, s.value, s.rail) for s in line.slots],
                         [(None, 'derivation', 'T'),
                          (None, 'grad_f+A^T*lambda=0', 'T')])
        line = self._one('. formula=x^T*y')
        self.assertEqual((line.slots[0].value, line.slots[0].rail),
                         ('x^T*y', 'T'))

    def test_quoted_value_is_opaque(self):
        line = self._one('. note="already^T"')
        self.assertEqual(line.slots[0].value, 'already^T')
        self.assertEqual(line.slots[0].rail, 'T')

    def test_refutation_sugar_equals_rail_f(self):
        sugar = self._one('. allergy!=penicillin')
        explicit = self._one('. allergy=penicillin^F')
        self.assertEqual((sugar.slots[0].key, sugar.slots[0].value,
                          sugar.slots[0].rail),
                         (explicit.slots[0].key, explicit.slots[0].value,
                          explicit.slots[0].rail))
        self.assertEqual(sugar.tags['allergy'], 'penicillin')

    def test_level_tag(self):
        self.assertEqual(self._one('. a=1').lvl, 1)
        line = self._one('~ schema patient  fields=id,dx  lvl=3')
        self.assertEqual(line.lvl, 3)
        self.assertEqual(line.tags['lvl'], '3')   # still an ordinary tag

    def test_resolve_record(self):
        line = self._one(
            ': resolve allergy  was=C  now=F  from=8  why=skin_test_negative')
        self.assertEqual(line.kind, 'resolve')
        rec = line.record
        self.assertIsNotNone(rec)
        self.assertEqual((rec.target, rec.was, rec.now, rec.why),
                         ('allergy', 'C', 'F', 'skin_test_negative'))
        self.assertEqual(line.from_refs, [8])

    def test_obstruction_record(self):
        line = self._one('! obstruction dx  from=3  note=imaging_unavailable')
        self.assertEqual(line.kind, 'obstruction')
        self.assertEqual((line.record.target, line.record.note),
                         ('dx', 'imaging_unavailable'))

    def test_keywords_are_phase_scoped(self):
        """v1's `> resolve ...` act lines keep their v1 meaning."""
        line = self._one('> resolve  ambiguity_resolution  reason=from=11')
        self.assertEqual(line.kind, 'normal')
        self.assertIsNone(line.record)
        self.assertEqual(self._one(': resolved=logging_overflow  conf=0.9').kind,
                         'normal')
        self.assertEqual(self._one('! obstruction_like=x').kind, 'normal')

    def test_record_utilities(self):
        text = (': resolve allergy  was=C  now=F  why=skin_test\n'
                '! obstruction dx  note=no_imaging\n'
                '. a=1')
        lines = parse_wcy(text)
        self.assertEqual(len(extract_records(lines)), 2)
        self.assertEqual(len(extract_records(lines, 'resolve')), 1)
        self.assertEqual(len(extract_records(lines, 'obstruction')), 1)
        self.assertEqual([e['kind'] for e in audit_trail(lines)],
                         ['resolve', 'obstruction'])

    def test_void_slot_is_unknown_rail(self):
        line = self._one(': ?dx  hint=labs+imaging  conf_range=0.3..0.7')
        void_slot = line.slots[0]
        self.assertTrue(void_slot.is_void)
        self.assertEqual(void_slot.rail, 'U')
        self.assertEqual(line.void_tags, ['dx'])


# ── Merge laws (SPEC 5, mirroring lean/wcymerge/WcyMerge.lean) ──────────────

class TestMergeLaws(unittest.TestCase):

    def test_verify_laws_passes(self):
        report = verify_laws()
        self.assertTrue(report.ok, f'failed laws: {report.failures}')
        self.assertEqual(report.failures, [])
        self.assertGreaterEqual(report.total_cases, 250)

    def test_law_names_match_the_lean_theorems(self):
        if not LEAN_FILE.exists():                      # pragma: no cover
            self.skipTest('Lean source not present')
        source = LEAN_FILE.read_text(encoding='utf-8')
        lean_names = set(re.findall(r'^theorem\s+(\w+)', source, re.M))
        self.assertEqual(set(verify_laws().laws), lean_names)

    def test_state_table(self):
        self.assertEqual([a.state for a in ATOMS],
                         ['UNK', 'TRU', 'FAL', 'CON'])
        self.assertEqual([a.rail for a in ATOMS], ['U', 'T', 'F', 'C'])
        self.assertEqual([atom_for_rail(r) for r in 'TFUC'],
                         [TRU, FAL, UNK, CON])
        self.assertTrue(TRU.is_resolved and FAL.is_resolved)
        self.assertFalse(UNK.is_resolved or CON.is_resolved)

    def test_conflict_is_a_value_not_an_exception(self):
        self.assertEqual(merge(TRU, FAL), CON)
        self.assertEqual(merge(FAL, TRU), CON)          # no last-writer-wins
        self.assertEqual(merge_all([TRU, FAL, UNK, TRU]), CON)

    def test_information_order(self):
        self.assertTrue(le(UNK, CON) and le(TRU, CON) and le(FAL, CON))
        self.assertFalse(le(TRU, FAL))
        self.assertTrue(all(le(a, a) for a in ATOMS))

    def test_meet_is_the_dual(self):
        self.assertEqual(meet(TRU, FAL), UNK)
        self.assertEqual(meet(CON, TRU), TRU)

    # document level
    DOC_A = '. x=1\n. y=2'
    DOC_B = '. x!=1\n. z=3'
    DOC_C = '. y!=2\n. w=4'

    def _docs(self):
        return [document_from_lines(parse_wcy(t), label=lab)
                for t, lab in ((self.DOC_A, 'A'), (self.DOC_B, 'B'),
                               (self.DOC_C, 'C'))]

    @staticmethod
    def _states(doc) -> dict[str, str]:
        return {tag: doc.atom(tag).state for tag in doc.tags()}

    def test_document_merge_is_commutative_and_associative(self):
        a, b, c = self._docs()
        self.assertEqual(self._states(merge_documents(a, b)),
                         self._states(merge_documents(b, a)))
        left = merge_documents(merge_documents(a, b), c)
        right = merge_documents(a, merge_documents(b, c))
        self.assertEqual(self._states(left), self._states(right))
        self.assertEqual(self._states(merge_many([a, b, c])),
                         self._states(merge_many([c, b, a])))

    def test_document_merge_is_idempotent(self):
        a, _, _ = self._docs()
        self.assertEqual(self._states(merge_documents(a, a)), self._states(a))

    def test_unmatched_tags_union(self):
        a, b, c = self._docs()
        merged = merge_many([a, b, c])
        self.assertEqual(self._states(merged),
                         {'x': 'CON', 'y': 'CON', 'z': 'TRU', 'w': 'TRU'})

    def test_levels_are_not_compressed(self):
        doc = document_from_lines(parse_wcy('. a=1  lvl=1\n. a=1^F  lvl=2'))
        rec = doc.atom('a')
        self.assertEqual(sorted(rec.levels), [1, 2])
        self.assertEqual(rec.levels[1], TRU)
        self.assertEqual(rec.levels[2], FAL)
        self.assertEqual(rec.state, 'TRU')      # derived from coarsest level


# ── SPEC Appendix B, end to end ─────────────────────────────────────────────

AGENT_A = """
~ schema patient  fields=id,dx,allergy   lvl=1

. id=P-1043
. allergy=penicillin            | from=intake_form
: ?dx  hint=labs+imaging  conf_range=0.3..0.7
""".strip()

AGENT_B = """
. allergy!=penicillin           | from=lab_panel_2
. labs  wbc=14.2  crp=high
""".strip()

RESOLUTION = """
> order  skin_test  reason=resolve_allergy
. skin_test=negative
: resolve allergy  was=C  now=F  from=skin_test  why=skin_test_negative
: dx=viral_syndrome  conf=0.78  from=lab_panel_2
""".strip()


class TestAppendixB(unittest.TestCase):

    def setUp(self):
        self.a = document_from_lines(parse_wcy(AGENT_A), label='A')
        self.b = document_from_lines(parse_wcy(AGENT_B), label='B')
        self.merged = merge_documents(self.a, self.b)

    def test_agent_documents(self):
        self.assertEqual(self.a.state('allergy'), 'TRU')
        self.assertEqual(self.a.state('dx'), 'UNK')
        self.assertEqual(self.b.state('allergy'), 'FAL')

    def test_conflict_is_surfaced_with_provenance_union(self):
        allergy = self.merged.atom('allergy')
        self.assertEqual(allergy.state, 'CON')
        self.assertEqual(allergy.values, ['penicillin'])
        self.assertEqual(sorted(allergy.provenance),
                         ['intake_form', 'lab_panel_2'])

    def test_nothing_is_lost_by_the_merge(self):
        states = {tag: self.merged.atom(tag).state
                  for tag in self.merged.tags()}
        self.assertEqual(states['id'], 'TRU')       # only in A
        self.assertEqual(states['wbc'], 'TRU')      # only in B
        self.assertEqual(states['crp'], 'TRU')
        self.assertEqual(states['dx'], 'UNK')       # the void survives

    def test_resolution_moves_the_atom_and_keeps_the_record(self):
        store = merge_documents(
            self.merged,
            document_from_lines(parse_wcy(RESOLUTION), label='C'))
        self.assertEqual(store.state('allergy'), 'CON')   # not yet applied
        self.assertEqual(len(store.resolutions), 1)

        final = apply_resolutions(store)
        self.assertEqual(final.state('allergy'), 'FAL')
        self.assertEqual(len(final.resolutions), 1)       # audit trail kept
        self.assertEqual(final.resolutions[0].why, 'skin_test_negative')
        self.assertIn('skin_test', final.atom('allergy').provenance)
        self.assertEqual(final.state('dx'), 'TRU')        # resolved by agent C
        self.assertEqual(final.atom('dx').values, ['viral_syndrome'])

    def test_records_are_appended_never_merged(self):
        rec_doc = document_from_lines(parse_wcy(RESOLUTION), label='C')
        obstruction = document_from_lines(
            parse_wcy('! obstruction dx  from=lab_panel_2  note=no_imaging'),
            label='D')
        store = merge_many([self.merged, rec_doc, rec_doc, obstruction])
        self.assertEqual(len(store.resolutions), 2)   # duplicate kept
        self.assertEqual(len(store.obstructions), 1)
        self.assertEqual([r.kind for r in store.records],
                         ['resolve', 'resolve', 'obstruction'])

    def test_strict_resolution_checks_the_was_field(self):
        bad = document_from_lines(
            parse_wcy(': resolve id  was=C  now=T  why=typo'), label='X')
        store = merge_documents(self.merged, bad)
        with self.assertRaises(ValueError):
            apply_resolutions(store, strict=True)
        self.assertEqual(apply_resolutions(store).state('id'), 'TRU')

    def test_merge_order_does_not_matter(self):
        forward = merge_documents(self.a, self.b)
        backward = merge_documents(self.b, self.a)
        self.assertEqual(
            {t: forward.atom(t).state for t in forward.tags()},
            {t: backward.atom(t).state for t in backward.tags()})
        self.assertEqual(sorted(backward.atom('allergy').provenance),
                         ['intake_form', 'lab_panel_2'])


# ── JSON interoperability (SPEC 11) ─────────────────────────────────────────

SAMPLE_JSON = {
    'id': 'P-1043',
    'age': 45,
    'admitted': True,
    'labs': {'wbc': 14.2, 'crp': 'high'},
    'dx': None,
    'tags': ['a', 'b'],
    'notes': {},
    'history': [],
}


class TestJSONInterop(unittest.TestCase):

    def test_projection_is_left_inverse_of_embedding(self):
        value, sidecar = project_json(embed_json(SAMPLE_JSON))
        self.assertEqual(value, SAMPLE_JSON)
        self.assertEqual(sidecar.unresolved_paths, ['dx'])
        self.assertEqual(sidecar.refuted, [])

    def test_embedding_round_trips_through_projection(self):
        doc = embed_json(SAMPLE_JSON)
        value, _ = project_json(doc)
        self.assertEqual(embed_json(value), doc)

    def test_null_becomes_unknown_and_leaves_become_asserted(self):
        doc = embed_json(SAMPLE_JSON)
        self.assertEqual(doc.atom('dx').state, 'UNK')
        self.assertEqual(doc.atom('id').state, 'TRU')
        self.assertEqual(doc.atom('labs.wbc').value, 14.2)
        self.assertEqual(doc.atom('tags.0').value, 'a')
        self.assertEqual(doc.atom('notes').value, {})

    def test_scalar_and_empty_roots(self):
        for value in ('x', 7, None, [], {}):
            with self.subTest(value=value):
                self.assertEqual(project_json(embed_json(value))[0], value)
        self.assertEqual(path_to_tag(()), '$')
        self.assertEqual(path_to_tag(('a', 0, 'b')), 'a.0.b')

    def test_conflict_is_never_coerced(self):
        doc = merge_documents(
            document_from_lines(parse_wcy('. allergy=penicillin'), label='A'),
            document_from_lines(parse_wcy('. allergy!=penicillin'), label='B'))
        value, sidecar = project_json(doc)
        self.assertEqual(doc.state('allergy'), 'CON')
        self.assertIsNone(value['allergy'])
        self.assertEqual([e['state'] for e in sidecar.unresolved], ['CON'])
        self.assertEqual(sidecar.unresolved_paths, ['allergy'])
        self.assertNotIn('penicillin', json.dumps(value))

    def test_unknown_projects_to_null_with_sidecar(self):
        doc = document_from_lines(parse_wcy(': ?dx  hint=labs'), label='A')
        value, sidecar = project_json(doc)
        self.assertEqual(value, {'dx': None})
        self.assertEqual(sidecar.unresolved,
                         [{'path': 'dx', 'tag': 'dx', 'state': 'UNK'}])
        self.assertFalse(sidecar.clean)

    def test_refuted_is_omitted_with_sidecar(self):
        doc = document_from_lines(parse_wcy('. a=1\n. b!=2'), label='A')
        value, sidecar = project_json(doc)
        self.assertEqual(value, {'a': '1'})
        self.assertEqual(sidecar.refuted[0]['path'], 'b')
        self.assertEqual(sidecar.refuted[0]['value'], '2')

    def test_multiple_asserted_values_are_flagged_not_picked(self):
        doc = merge_documents(
            document_from_lines(parse_wcy('. dx=flu'), label='A'),
            document_from_lines(parse_wcy('. dx=covid'), label='B'))
        value, sidecar = project_json(doc)
        self.assertEqual(doc.atom('dx').values, ['flu', 'covid'])
        self.assertIsNone(value['dx'])
        self.assertEqual(sidecar.ambiguous[0]['values'], ['flu', 'covid'])

    def test_projection_after_resolution_is_clean(self):
        store = merge_many([
            document_from_lines(parse_wcy(AGENT_A), label='A'),
            document_from_lines(parse_wcy(AGENT_B), label='B'),
            document_from_lines(parse_wcy(RESOLUTION), label='C'),
        ])
        value, sidecar = project_json(apply_resolutions(store))
        self.assertNotIn('allergy', value)                 # FAL -> omitted
        self.assertEqual([e['path'] for e in sidecar.refuted], ['allergy'])
        self.assertEqual(sidecar.unresolved, [])           # dx was resolved
        self.assertEqual(value['dx'], 'viral_syndrome')


if __name__ == '__main__':
    unittest.main(verbosity=2)
