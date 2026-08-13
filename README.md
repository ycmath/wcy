# WCY — Watch → Compute → Yield

**A token-native reasoning format with theorem-backed semantics.**

[![DOI (Paper)](https://zenodo.org/badge/DOI/10.5281/zenodo.19068378.svg)](https://doi.org/10.5281/zenodo.19068378)
[![DOI (Dataset)](https://zenodo.org/badge/DOI/10.5281/zenodo.19068768.svg)](https://doi.org/10.5281/zenodo.19068768)
[![DOI (Repository, v2)](https://zenodo.org/badge/DOI/10.5281/zenodo.21886552.svg)](https://doi.org/10.5281/zenodo.21886552)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

> *The name is the grammar. The grammar encodes the capacity to not-know.*

---

## What is WCY?

WCY is a line-oriented, phase-tagged format for AI reasoning and
agent-to-agent data exchange, designed for how transformer-based LLMs
actually process information. Every line begins with a phase marker
(`.` observe, `:` infer, `>` act, `~` meta, `!` exception), and the `?`
marker gives the unknown an explicit, structural representation.

**WCY-2** (this release) keeps that surface unchanged and adds what v1
left as convention: a value model, merge semantics, a mutation
discipline, an audit rule, and a mechanical rule for when `?` must
appear. Each of these is backed by a published, machine-verified theorem
from the [Dual-Rail Carrier Program](https://github.com/ycmath/dual-rail-carrier-program)
— a 12-release series of kernel-checked mathematics that grew out of
this format's original design questions and now, closing the circle,
answers them. The full specification is [SPEC.md](SPEC.md).

## Why WCY-2 (the one-paragraph pitch)

JSON's `null` conflates *unknown*, *inapplicable*, and *conflicting*;
JSON has no defined merge; deletion destroys history; and nothing tells
an agent when it should admit it doesn't know. In WCY-2: every atom is a
pair of evidence rails, so unknown (no evidence) and conflict (opposing
evidence) are different values; merge is the railwise join — associative,
commutative, idempotent **by kernel-checked proof** (`lean/wcymerge/`,
10 theorems, all axiom-free), so replicas merge in any order and
conflict surfaces as a value instead of a lost update; retraction adds
refuting evidence rather than deleting, making stores append-only as a
*theorem consequence*; only resolution steps (unknown/conflict →
resolved) carry an essential obstruction — so logging exactly those is a
provably sufficient audit trail; and a query that leaves the safe
(self-dual) fragment must either resolve-and-log or emit `?` — turning
v1's weakest point (`?` as unenforced style) into grammar.

## The theorem-backed rules

| Rule | Backing theorem (DOI) |
|---|---|
| unknown ≠ conflict: dual-rail atoms | [10.5281/zenodo.21800031](https://doi.org/10.5281/zenodo.21800031) |
| merge = railwise join, CRDT laws kernel-checked | this repo (`lean/wcymerge/`) + [10.5281/zenodo.21870654](https://doi.org/10.5281/zenodo.21870654) |
| retraction is monotone; append-only derived, not decreed | [10.5281/zenodo.21800033](https://doi.org/10.5281/zenodo.21800033) |
| audit = resolution records only, and that is *sufficient* | [10.5281/zenodo.21870654](https://doi.org/10.5281/zenodo.21870654) (Theorem 0) |
| confidence/version history is level-indexed, never compressed | [10.5281/zenodo.21871946](https://doi.org/10.5281/zenodo.21871946) |
| safe query fragment = self-dual; outside it, `?` is mandatory | [10.5281/zenodo.21866741](https://doi.org/10.5281/zenodo.21866741) |
| schema evolution = conservative pointed extension | [10.5281/zenodo.21800031](https://doi.org/10.5281/zenodo.21800031), [21866478](https://doi.org/10.5281/zenodo.21866478) |
| no "verified once, trusted forever" across levels | [10.5281/zenodo.21869871](https://doi.org/10.5281/zenodo.21869871) |

## Empirical results (v1 series, unchanged by the upgrade)

| Experiment | Finding |
|-----------|---------|
| Token reduction vs JSON (structured data) | 50–60% |
| Token reduction for tool-call schemas | 65–71% |
| Full MCP protocol exchange reduction | 61% |
| Agent output token reduction | 40% |
| `from=` provenance validity (3-agent pipeline) | 45/45 (100%) |
| WCY format acquisition (0-shot → 3-shot) | parse_r: 0.29 → 1.00 |
| Void-B resolution rate | 67–97% |
| Pipeline quality gate pass rate (528 traces) | 528/528 (100%) |

The v2 semantic layer adds no surface tokens to resolved-face documents,
so these figures carry over; conflict and level annotations cost tokens
only where the corresponding information exists.

## Dataset

Two corpora ship with this repository:

- **WCY v1 — 540 reasoning traces** (`data/`). The original corpus:
  line-oriented, phase-tagged agent reasoning, 100% parse rate, published as
  [10.5281/zenodo.19068769](https://doi.org/10.5281/zenodo.19068769) (v1.1).
- **WCY-2 — 379 traces** (`data_v2/`). v1 traces re-encoded under the
  theorem-backed v2 semantics (317), plus natively-authored v2 traces
  exercising merge-conflict resolution (18), level refinement (9),
  safe-fragment query discipline (14), and schema evolution (21).
  Published as [10.5281/zenodo.19089743](https://doi.org/10.5281/zenodo.19089743)
  (v2.0; concept DOI [10.5281/zenodo.19068768](https://doi.org/10.5281/zenodo.19068768)
  resolves to the latest version). Also on
  [Hugging Face](https://huggingface.co/datasets/ycmath/wcy-reasoning-traces).

Every v2 trace passed nine mechanical gates and an LLM audit.
**145 of the 379 v2 traces (38%) have `corrected_from_v1: true`** — their
domain content was altered relative to the v1 source, not merely re-encoded,
because the audit found factual defects (some inherited from the v1 source
text, some introduced during re-encoding). The v2 corpus is consequently
*not* a faithful re-encoding of v1, and any comparison between the two must
account for those rows. Full composition, gate definitions, and the
correction disclosure are in [`data_v2/DATASET_v2.md`](data_v2/DATASET_v2.md).

## Quick example (merge, conflict, resolution)

```
agent A:   . allergy=penicillin        | from=intake_form
agent B:   . allergy!=penicillin       | from=lab_panel_2

merged:    . allergy=penicillin^C  from=A1,B1     ← conflict is a value
           > order  skin_test  reason=resolve_allergy
           . skin_test=negative
           : resolve allergy  was=C  now=F  from=3  why=skin_test_negative
```

Nothing was overwritten, nothing was lost, and the single `resolve` line
is the entire audit obligation for this exchange.

## Repository contents

```
SPEC.md                         The WCY-2 specification (v0.1 draft):
                                value model, merge, mutation, audit,
                                levels, safe fragment, JSON interop;
                                normative references = the series DOIs
wcy_parser.py                   Reference parser v2 (v1-compatible)
wcy_merge.py                    Merge / JSON embed & project / law mirror
wcy_eval.py                     3-axis evaluation (v1)
lean/wcymerge/                  Kernel-checked merge laws: 10 theorems,
                                all axiom-free (Lean 4, core only)
tests/                          v1-compat + v2 semantics test suite
data/                           v1 trace corpus (540 traces)
data_v2/                        WCY-2 corpus (379 traces, 9-gate + audited;
                                see DATASET_v2.md for the correction
                                disclosure)
experiments/                    v1 experiment scripts
papers/                         papers; second_edition/ = the v2 paper
                                (Zenodo: 10.5281/zenodo.19068378, latest)
```

## Conformance

Four classes — **Core** (value model + syntax + JSON projection),
**Merge** (+ join semantics + mutation discipline), **Audit**
(+ resolution records + level families), **Full** (+ safe-fragment
query discipline + conservative schema evolution). A v1 document is a
valid Core document. See SPEC.md §2, §12.

## Authorship & provenance

Won Chul Yang, independent researcher (wcy0969@gmail.com). The format
and the underlying mathematics are the author's; the carrier-theory
series backing the v2 semantics is published with complete proofs,
kernel-checked Lean artifacts, and independent replays (series hub:
https://github.com/ycmath/dual-rail-carrier-program). AI assistance
(Anthropic Claude family) was used for the machine-verification layer
and implementation, with the Lean 4 kernel as the acceptance gate for
verified components. Corrections are invited.

## License

CC BY 4.0 (text, data); code files carry their headers.
