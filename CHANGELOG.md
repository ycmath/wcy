# Changelog

## v2.0 — WCY-2 (2026-08)

The surface syntax of v1 is unchanged; v2 adds a theorem-backed semantic
layer. See SPEC.md for the full specification and its normative
references (the Dual-Rail Carrier Program DOIs).

Added:
- SPEC.md — the WCY-2 specification, draft v0.1: dual-rail value model
  (unknown ≠ conflict), railwise-join merge (CRDT laws), monotone
  mutation discipline (retraction = refuting evidence; no deletion),
  resolution/obstruction records as the (provably sufficient) audit
  surface, level-indexed confidence families, the self-dual safe query
  fragment and the mandatory-`?` rule, conservative schema evolution,
  JSON embedding and resolved-face projection.
- lean/wcymerge/ — kernel-checked merge laws (Lean 4 core only;
  10 theorems, all axiom-free; frozen axcheck.log).
- wcy_merge.py — reference merge, JSON embed/project, and a pure-Python
  mirror of the Lean-checked laws.
- wcy_parser.py v2.0 — rail suffixes (^T/^F/^U/^C), refutation sugar
  (tag!=value), lvl=, and resolve/obstruction records; fully
  v1-compatible (v1 documents parse with v1 semantics).
- tests/ — v1-compatibility and v2-semantics suite.

Changed:
- README.md rewritten around the v2 semantics; v1 empirical results
  retained (they concern the unchanged surface syntax).

Pending:
- papers/: v2 revision of the position paper (Zenodo record
  10.5281/zenodo.19068379 will be updated to reference SPEC.md and the
  carrier series).

## v1.1 (2026-03)

- `|` separator support; parser fixes (validate() dead code,
  reconstruct() block gap).

## v1.0 (2026-03)

- Initial public release: format, parser, evaluator, 540-trace dataset,
  nine experiments.
