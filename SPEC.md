# WCY-2 Specification — Draft v0.1

**A token-native data and reasoning format with theorem-backed semantics.**

Won Chul Yang — wcy0969@gmail.com — 2026-08

Status: DRAFT v0.1, staged for the upgrade of https://github.com/ycmath/wcy
(supersedes the semantics-free portions of WCY v1; the v1 surface syntax is
inherited unchanged). This document uses MUST / SHOULD / MAY in the RFC-2119
sense.

---

## 1. Introduction and lineage

WCY v1 (DOI 10.5281/zenodo.19068379) established a line-oriented,
phase-tagged format for agent reasoning: five phase markers
(`.` observe, `:` infer, `>` act, `~` meta, `!` exception), slot syntax
`tag=value`, the void marker `?tag` for explicit unknowns, `from=`
provenance, and `conf=`/`conf_range=`. Its empirical results — 50–71%
token reduction against JSON, 100% provenance validity in a 3-agent
pipeline, and a 528/528 quality-gate corpus — validated the *surface*.
Three things remained conventions rather than guarantees: what values
*mean* (one `null`-like hole with many readings), what *merging* two
documents means (undefined, as in JSON), and *when* an agent must emit
`?` (a style norm; 0-shot emission rate was 0%).

WCY-2 closes those gaps with a semantic layer derived from the Dual-Rail
Carrier Program (12 machine-verified releases; §13). Every normative rule
in this specification that is marked **[T]** is backed by a published,
kernel-checked theorem, cited inline by DOI. The surface syntax of v1 is
unchanged; v2 adds a value model, merge semantics, a mutation discipline,
an audit rule, and an emission rule for `?`.

**Design summary.** Atomic values are *dual-rail evidence pairs*; merging
is the railwise join, which is associative, commutative, and idempotent
by kernel-checked theorem — so documents merge in any order and conflict
is a *value*, never an exception; retraction is a monotone act (add
refuting evidence), and only *resolution* — the step that moves a value
onto the resolved face — carries an essential obstruction, so resolution
steps are exactly the audit surface; confidence and versioning are
level-indexed and MUST NOT be compressed to a single summary; a query
that stays inside the self-dual fragment never forces resolution, and a
query that leaves it MUST either log a resolution or emit `?`.

## 2. Conformance classes

- **Core** — value model (§3), syntax (§4), JSON projection (§11).
- **Merge** — Core + merge semantics (§5) + mutation discipline (§6).
- **Audit** — Merge + resolution records (§7) + level families (§8).
- **Full** — Audit + safe-fragment query discipline (§9) + schema
  evolution (§10).

A v1 document is a valid Core document (§11.3).

## 3. Value model

### 3.1 Dual-rail atoms

An atomic value is a pair of evidence rails `(a, r)`:

| state | (a, r) | reading |
|---|---|---|
| `UNK` | (0, 0) | no evidence either way |
| `TRU` | (1, 0) | asserting evidence only |
| `FAL` | (0, 1) | refuting evidence only |
| `CON` | (1, 1) | conflicting evidence |

`TRU` and `FAL` form the **resolved face** R; `UNK` and `CON` are
unresolved. The four states form the dual-rail carrier D4 with railwise
meet/join and the De Morgan involution; this is the substrate of the
carrier series and of Belnap's four-valued logic. **[T]** The full logic
of these states — including why `UNK` and `CON` must be distinguished
(the T₀ obstruction) — is doi:10.5281/zenodo.21800031.

This resolves the `null` ambiguity of JSON: *unknown* is `UNK`,
*conflict* is `CON`, *inapplicable* is a schema-level absence (§10), and
*asserted-empty* is a resolved value.

### 3.2 Graded rails (levels)

A rail MAY be graded by a level `m ≥ 1` (confidence depth, version
depth): a level-m atom takes rail values in ℤ/2^m rather than bits.
Level-1 atoms are the four states of §3.1. Refining level m to m+1 is a
*lifting problem*; §8 states the (theorem-backed) rules levels obey.
Implementations without graded rails simply fix m = 1.

## 4. Syntax

### 4.1 Inherited from v1 (normative, unchanged)

Line = optional indent (depth ≤ 2) + phase marker + slots. Slots are
positional values or `tag=value`, separated by whitespace or `|`. Blank
line = block boundary. `?tag` marks a void (an explicitly represented
unknown), optionally with `hint=` and `conf_range=`. `from=N,M` cites
source lines. `conf=0.xx` on inferences. (Reference grammar: Appendix A;
reference implementation: `wcy_parser.py` v1.1.)

### 4.2 New in v2

- **Rail literal**: a value slot MAY carry an explicit state suffix
  `tag=value^T | ^F | ^U | ^C` (assert / refute / unknown / conflict).
  A bare `tag=value` in observe/infer lines is `^T` (asserting).
- **Refutation slot**: `tag!=value` asserts the refuting rail for
  `tag=value` (sugar for `tag=value^F`). This is the *retraction* form
  (§6): v2 has no deletion syntax.
- **Level tag**: `lvl=m` grades the line's evidence at level m (§8);
  absent means m = 1.
- **Resolution record**: a line of the form
  `: resolve tag from=... was=U|C now=T|F why=...` (§7). The `resolve`
  keyword in an infer line is reserved by v2.
- **Obstruction record**: `! obstruction tag from=... note=...` — the
  recorded failure of a resolution/lifting attempt (§7.3, §8).

All v2 additions are ordinary slots under the v1 grammar, so v1 parsers
read v2 documents (they see the suffixes as opaque value text).

## 5. Merge semantics

### 5.1 The merge

The merge of two documents is slotwise, and on each shared atom it is the
**railwise join**: `(a₁,r₁) ⊔ (a₂,r₂) = (a₁∨a₂, r₁∨r₂)`. Evidence
accumulates; nothing is lost.

**[T] Laws.** Railwise meet and join on D4 are associative — all 64
triples, zero violations, kernel-checked in core Lean 4 with an
axiom-free proof: doi:10.5281/zenodo.21870654 (library `RFlip`, theorems
`meet_assoc`, `join_assoc`). Commutativity, idempotence, the UNK
identity, absorption, and the least-upper-bound law are kernel-checked
in this repository: lean/wcymerge/ (10 theorems, all axiom-free; see
axcheck.log). Consequently a Merge-conformant store is a bounded
join-semilattice: merges MAY be applied in any order, repeated, or
batched, with identical results — the defining property of a state-based
CRDT.

### 5.2 Conflict is a value

`TRU ⊔ FAL = CON`. A Merge-conformant implementation MUST NOT throw on
conflicting merges, MUST NOT silently prefer either side (no
last-writer-wins), and MUST surface `CON` as an ordinary queryable
state. Conflict repair is *resolution* (§7), a separate, audited act.

### 5.3 Example

```
agent A:   . sensor_ok            ← TRU
agent B:   . sensor_ok!=timeout   ← FAL (refuting evidence, reason slot)
merged:    . sensor_ok^C  from=A1,B1
```

## 6. Mutation discipline

**[T] Monotone is free; negation is priced.** On the resolved face the
minimal number of negations a function needs equals its chain-decrease
number, ν(f) = dec(f), and dec(f) = 0 exactly for monotone f:
doi:10.5281/zenodo.21800033. WCY-2 turns this into a discipline:

1. **Adding evidence (either rail) is monotone** and therefore free:
   unrestricted, merge-safe, no audit obligation.
2. **Retraction is not deletion.** To retract `tag=v`, write `tag!=v`
   (add refuting evidence). The store's history is append-only *as a
   theorem consequence*, not as a policy: every free operation is a
   join, and joins only go up the lattice.
3. **Non-monotone operations** (overwriting a resolution, downgrading
   evidence) have dec ≥ 1: they MUST be expressed as resolution records
   (§7) and are the only operations that consume audit budget.

## 7. Resolution and the audit surface

### 7.1 The resolution record

Resolution moves an atom off `UNK`/`CON` onto the resolved face (or
overrides a prior resolution). Audit-conformant implementations MUST
record each resolution:

```
: resolve diagnosis  was=C  now=T  from=3,5  why=biopsy_confirms
```

### 7.2 Minimality guarantee

**[T]** The carrier's symmetry receptor H¹(V₄;𝔽₂) has exactly three
nonzero characters, realized by: the negation boundary (priced by dec,
§6), the rail swap (monotone bookkeeping — free, no audit), and the
**R-flip** — the character detecting exactly the operations that move
the resolved face. The R-flip is the *unique* essential degree-one
obstruction, and every higher or parallel structure reduces to it or
separates from it (Theorem 0): doi:10.5281/zenodo.21870654, with the
three separations at doi:10.5281/zenodo.21868976, 10.5281/zenodo.21869440,
10.5281/zenodo.21869871. Consequence: **logging resolutions is not just
necessary but sufficient** — an audit trail of R-flip records captures
the only obstruction-carrying steps a WCY-2 store performs. WCY-2's
audit surface is minimal by theorem, not by design taste.

### 7.3 The void cycle as a lifting problem

v1's cycle `?tag → > investigate → . observe → : resolve` is, in v2
semantics, a lifting problem: raise an unresolved atom to the resolved
face (or a level-m atom to level m+1, §8). A cycle that fails MUST NOT
silently drop: it records `! obstruction tag ...`. Failed resolutions
are data — they are the levelwise obstruction values of §8.

## 8. Confidence and versioning: level-indexed families

**[T] No compression.** For the carrier's internal level tower, a
levelwise obstruction value exists at every level while no nonzero
reduction-persistent class exists — the level index is irreducible data:
doi:10.5281/zenodo.21871946 (and the exactness principle at
doi:10.5281/zenodo.21869871). WCY-2 therefore REQUIRES (Audit class):

1. Confidence/version history is stored as a **level-indexed family**
   `(m, value_m, obstruction_m)` — v1's `conf_range=` maps to the
   coarsest such family. A single scalar "current confidence" MAY be
   *derived* for display but MUST NOT replace the family.
2. **No persistent-trust shortcut**: a mark that survives every
   refinement level is, by exactness, a mark that was never obstructed —
   "verified once, trusted forever" flags across levels are unsound
   unless each level's check is recorded. **[T]** doi:10.5281/zenodo.21869871 §8.

## 9. The safe query fragment and the `?`-emission rule

**[T]** On the resolved face, the operations expressible by the closed
core are exactly the **self-dual** Boolean functions — 2^(2^(n−1)) of
them — and adjoining a single resolved constant completes the fragment
to all functions: doi:10.5281/zenodo.21866741.

Discipline (Full class): a query touching only resolved atoms through
self-dual combinators is **safe** — it can be answered with no
resolution, no audit, no `?`. A query that (i) touches an unresolved
atom, or (ii) requires a non-self-dual combinator without its
constant-completion being resolved, is **unsafe**: the implementation
MUST either perform and log a resolution (§7) or emit a void:

```
: ?tag  hint=<what would resolve it>  conf_range=...
```

This converts v1's weakest point — `?` emission as an unenforced style
(0% zero-shot) — into a mechanical rule: `?` appears exactly where the
mathematics says an answer does not exist yet.

## 10. Schema evolution

Adding fields, tags, or types is a **conservative pointed extension**
(typed open update): new unknowns may be introduced, but resolved data
MUST NOT change state. **[T]** The extension calculus, and the
obstruction that detects non-conservative extensions, are
doi:10.5281/zenodo.21800031; the boundary of how much operational
structure can be added before collapse is delimited by the maximality
result doi:10.5281/zenodo.21866478. Inapplicable fields are schema-level
absences, distinct from `UNK` (§3.1).

## 11. JSON interoperability

### 11.1 Projection (v2 → JSON)

The **resolved-face projection** maps `TRU`-valued atoms to their
values, `FAL` atoms to explicit negative fields (or omission, per
schema), and MUST fail soft on `UNK`/`CON` (emit `null` plus a sidecar
listing the unresolved paths — never silently coerce `CON` to a value).

### 11.2 Embedding (JSON → v2)

Every JSON document embeds as a resolved-face WCY-2 document: each leaf
becomes a `^T` atom; `null` becomes `UNK`. The embedding is lossless and
the projection is its left inverse on resolved documents.

### 11.3 v1 compatibility

A v1 document is a Core-conformant v2 document with all atoms at `^T`,
voids as `UNK` atoms, and `conf_range` as the coarsest level family.

## 12. Conformance summary

| # | Requirement | Class | Backing |
|---|---|---|---|
| 1 | dual-rail atoms; UNK ≠ CON | Core | [T] 21800031 |
| 2 | merge = railwise join; no LWW; CON surfaced | Merge | [T] 21870654 |
| 3 | retraction = refuting rail; no deletion op | Merge | [T] 21800033 |
| 4 | non-monotone ops only via resolution records | Audit | [T] 21800033 |
| 5 | every resolution logged; nothing else need be | Audit | [T] 21870654 (Thm 0) |
| 6 | failed resolutions recorded as obstructions | Audit | [T] 21871946 |
| 7 | level families never compressed | Audit | [T] 21871946, 21869871 |
| 8 | unsafe queries ⟹ resolve-and-log or emit `?` | Full | [T] 21866741 |
| 9 | schema changes conservative-pointed | Full | [T] 21800031, 21866478 |
| 10 | JSON projection fails soft on UNK/CON | Core | §11.1 |

## 12a. Draft notes (implementation-defined in v0.1, to be normativized in v0.2)

Two behaviors are implementation-defined in this draft and are pinned by
the reference implementation (`wcy_merge.py`); v0.2 will make them
normative:

1. **Block matching under merge**: blocks are matched by ordinal among
   *atom-bearing* blocks (blocks containing only meta/act lines do not
   shift alignment); atom identity is (block ordinal, tag).
2. **Multi-valued assertions**: when two documents assert *different
   values* for the same tag (both `^T`), the merge keeps an ordered set
   of values — never last-writer-wins — and the JSON projection emits
   `null` plus an `ambiguous` sidecar entry; nothing is coerced.

## 13. Normative references

The Dual-Rail Carrier Program (series hub, map and DAG:
https://github.com/ycmath/dual-rail-carrier-program):

1. Finite-Energy Epistemic Logic… — doi:10.5281/zenodo.21800031
2. The Price of NOT on D4 — doi:10.5281/zenodo.21800033
3. The Cohomological Price of NOT — doi:10.5281/zenodo.21775055
4. T₀ Is a Maximal Clone on the Three-Element Domain — doi:10.5281/zenodo.21866478
5. The Resolved-Face Crown of the Dual-Rail Carrier — doi:10.5281/zenodo.21866741
6. Exact Selector Counting on the Odd Flat Carrier — doi:10.5281/zenodo.21868044
7. No Genuine Degree-Three De Morgan Cohomology… — doi:10.5281/zenodo.21868976
8. Carry and Associator Decouple by Bidegree — doi:10.5281/zenodo.21869440
9. The Dual-Rail Carry and the Bockstein Obstructions of CSS Codes… — doi:10.5281/zenodo.21869871
10. The Carrier R-Flip (capstone) — doi:10.5281/zenodo.21870654
11. The Carrier's Internal 2-Adic Tower Is Split-Positive — doi:10.5281/zenodo.21871946
12. WCY v1 (format + dataset) — doi:10.5281/zenodo.19068379, 10.5281/zenodo.19068769

---

## Appendix A. Line grammar (v1-inherited, informative)

```
document   := block (BLANK_LINE block)*
block      := line+
line       := INDENT{0..2} phase slot (SEP slot)*
phase      := "." | ":" | ">" | "~" | "!"
slot       := VOID | pair | refute | VALUE
pair       := TAG "=" VALUE railmark?
refute     := TAG "!=" VALUE            ; sugar for =VALUE^F
railmark   := "^" ("T" | "F" | "U" | "C")
VOID       := "?" TAG (SEP "hint=" VALUE)? (SEP "conf_range=" RANGE)?
SEP        := WSP+ | WSP* "|" WSP*
```

Reserved v2 keywords in infer/exception lines: `resolve`, `obstruction`.

## Appendix B. Worked example (merge, conflict, resolution, audit)

```
~ schema patient  fields=id,dx,allergy   lvl=1

. id=P-1043
. allergy=penicillin            | from=intake_form
: ?dx  hint=labs+imaging  conf_range=0.3..0.7

--- merge: lab agent ---
. allergy!=penicillin           | from=lab_panel_2
. labs  wbc=14.2  crp=high

--- merged store ---
. allergy=penicillin^C  from=2,5      ← conflict surfaced, nothing lost
: ?dx  hint=labs+imaging

--- resolution (audited) ---
> order  skin_test  reason=resolve_allergy
. skin_test=negative
: resolve allergy  was=C  now=F  from=8  why=skin_test_negative
: dx=viral_syndrome  conf=0.78  from=6,8
```

The only line the audit trail needs is the `resolve` line — by §7.2 that
is sufficient, not merely necessary.
