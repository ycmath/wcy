# WCY — Watch → Compute → Yield

**A token-native reasoning format and epistemic substrate for AI systems.**

[![DOI (Paper)](https://zenodo.org/badge/DOI/10.5281/zenodo.19068379.svg)](https://doi.org/10.5281/zenodo.19068379)
[![DOI (Dataset)](https://zenodo.org/badge/DOI/10.5281/zenodo.19068769.svg)](https://doi.org/10.5281/zenodo.19068769)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

> *The name is the grammar. The grammar encodes the capacity to not-know.*

---

## What is WCY?

WCY is a line-oriented, phase-tagged format for AI reasoning. It was designed from first principles for how transformer-based LLMs actually process information — not for human readers.

Every WCY line begins with one of five phase markers:

```
.  observe    — confirmed fact
:  infer      — derived conclusion  (conf=, from=)
>  act        — output or action
~  meta       — schema / context
!  exception  — error or unresolvable state
```

And one special slot prefix that defines the entire project:

```
?tag          — void-B: explicit representation of the unknown
               hint= points the direction. conf_range= bounds the expectation.
```

---

## The core claim

Current LLMs can represent what they know. They cannot represent **the boundary of what they know** — not structurally, not in a way that generates directed exploration.

The `?` marker is the minimum syntax for that.

```
: ?diagnosis  hint=labs+imaging  conf_range=0.4..0.8    ← mark the unknown
> order  CT_scan  reason=from=3                          ← directed action
. CT_result  mass_in_RUL  size=2.3cm                    ← new observation
: diagnosis=adenocarcinoma  conf=0.82  from=3,5         ← resolved
```

This four-step cycle — mark unknown → investigate → observe → resolve — is the structural minimum for directed learning. A system without this cycle can process information. It cannot know that it is learning.

---

## Empirical results

Nine experiments across four research phases:

| Experiment | Finding |
|-----------|---------|
| Token reduction vs JSON (structured data) | 50–60% |
| Token reduction for tool-call schemas | 65–71% |
| Full MCP protocol exchange reduction | 61% |
| Agent output token reduction | 40% |
| `from=` provenance validity (3-agent pipeline) | 45/45 (100%) |
| WCY format acquisition (0-shot → 3-shot) | parse_r: 0.29 → 1.00 |
| Void-B generation (0-shot) | 0% |
| Void-B generation (with examples) | 5.4 markers/trace |
| Void-B resolution rate | 67–97% |
| Pipeline quality gate pass rate (528 traces) | 528/528 (100%) |

---

## Repository contents

```
wcy_parser.py                   Reference parser (v1.1, | separator support)
wcy_eval.py                     3-axis evaluation: Structural / Meaning / Provenance
data/
  wcy_reasoning_traces.jsonl    Reasoning traces (6 domains)
  wcy_void_cycles.jsonl         Void-B resolution cycle traces (6 traces)
  wcy_traces_v1_clean.jsonl     Pipeline-generated traces (48, all usable)
papers/
  WCY_Position_Paper_v2_0.docx  Full position paper
experiments/
  wcy_phase2_sprint1.py         Phase 2 Sprint 1 (scale + multi-agent)
  wcy_phase2_sprint2.py         Phase 2 Sprint 2 (void-B + shared state)
  wcy_phase2_sprint3.py         Phase 2 Sprint 3 (verification + MCP + cross-model)
  wcy_phase4_void_cycle.py      Phase 4 (void-B resolution cycles)
  wcy_phase5_pipeline.py        Phase 5 (automated trace generation)
```

---

## Quick start

```python
from wcy_parser import parse_wcy, flatten, extract_voids, validate

text = """
~ context  case=example
. patient=Kim  age=45  temp=38.5
. symptoms  fever  cough  duration=7days
: ?diagnosis  hint=fever+cough+duration  conf_range=0.4..0.8
: diagnosis=influenza  conf=0.78  from=2,3
> prescribe  oseltamivir  75mg  q12h  days=5  from=4
"""

lines = flatten(parse_wcy(text))
voids = extract_voids(lines)
result = validate(lines)

print(f"Lines: {len(lines)}, Voids: {len(voids)}, Valid: {result.valid}")
# → Lines: 6, Voids: 1, Valid: True
```

---

## The training data hypothesis

Phase 3 experiments showed that zero-shot WCY reasoning fails on complex tasks (parse_r = 0.29) but succeeds with three few-shot examples (parse_r = 1.00). The capacity was present; the training signal was not.

This repository contains 540 high-quality WCY reasoning traces across 8 domains, each with explicit void-B generation and resolution cycles. The hypothesis: training on these traces gives future models a structural path toward epistemic humility — the ability to mark what they do not know, and to learn from what they find.

**The traces are the contribution.** The format is the vehicle.

---

## Format specification

### Phase markers

| Marker | Phase | Rules |
|--------|-------|-------|
| `.` | observe | Confirmed fact. No forward references. |
| `:` | infer | Derived conclusion. `conf=` recommended. `from=` for traceability. |
| `>` | act | Output or external call. Should derive from `:` not directly from `.` |
| `~` | meta | Schema or context declaration. |
| `!` | exception | Error, warning, or unresolvable void. |
| `?tag` | void-B | Slot prefix in `:` lines. `hint=` required. `conf_range=` recommended. |

### Slot syntax

```
tag=value           named slot
bare_value          positional slot
?tag                void-B marker
from=N,M            provenance (derives from lines N and M)
conf=0.xx           confidence (0.0–1.0)
conf_range=L..H     expected range for void-B
a | b=c | d         pipe separator (v1.1, equivalent to spaces)
```

### Three modes

| Mode | Syntax | vs JSON | Use case |
|------|--------|---------|----------|
| Positional | bare slots | −54% | AI-to-AI, schema pre-shared |
| Tagged | `tag=value` | −40% | Semantic search |
| Hybrid (recommended) | `~` schema + positional | −50% | Balanced |

### Structural rules

- One phase marker per line, followed by a space
- `\n` = δ boundary (information firewall)
- Blank line = semantic block boundary
- 2-space indent = subscope, max depth 2
- No forward references (`from=N` requires `N < current line`)

---

## Theoretical grounding

WCY's design is consistent across three independent theoretical frameworks:

**Semiotics (Peirce):** The six markers cover all three sign types (icon, index, symbol) and all three reasoning modes (deduction, induction, abduction). The `?` marker uniquely encodes abductive stance — the only reasoning mode capable of generating new knowledge.

**Category theory:** WCY implements a monad transformer stack:
`WriterT(from=) ∘ ReaderT(~meta) ∘ EitherT(!) ∘ ContT(?)`. The `?` marker corresponds to `callCC` — a request for a continuation that will fill this position. This is why `?` cannot exist in JSON: JSON describes present values; WCY describes present values and deferred computations.

**Epistemology:** The void-B resolution cycle satisfies the four conditions for epistemic self-awareness: (1) representation of known states, (2) representation of the boundary, (3) directed exploration from the boundary, (4) integration of new observations. These conditions are necessary; no subset is sufficient.

---

## Citation

```bibtex
@misc{yang2026wcy,
  title     = {WCY: Watch $\to$ Compute $\to$ Yield ---
               A Token-Native Reasoning Format and Epistemic Substrate for AI Systems},
  author    = {Yang, Won Chul},
  year      = {2026},
  month     = {March},
  doi       = {10.5281/zenodo.19068379},
  url       = {https://doi.org/10.5281/zenodo.19068379},
  note      = {Zenodo preprint. Dataset: doi:10.5281/zenodo.19068769.
               Code: https://github.com/ycmath/wcy}
}
```

---

## License

Data (`.jsonl` files): CC BY 4.0 — use freely, cite the source.  
Code (`.py` files): MIT License.  
Papers (`.docx`): CC BY 4.0.

---

*Watch what is. Compute what follows. Yield — and mark what remains unknown.*
