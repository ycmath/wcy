# WCY Dataset Manifest v1.0
# Generated: 2026-03-17

## Overview

| File | Traces | Type | Avg ?tags | Avg Resolution |
|------|--------|------|-----------|----------------|
| wcy_reasoning_traces.jsonl | 6 | Reasoning (no void-B) | 0.0 | — |
| wcy_void_cycles.jsonl | 6 | Void-B resolution cycle | 6.2 | 61% |
| wcy_traces_v1_clean.jsonl | 48 | Pipeline (all types) | 5.2 | 95% |
| **Total** | **60** | | **4.5** | **90%** |

## Quality Gates (all traces passed)

- Gate 1: parse_rate ≥ 0.70 (WCY parser can read ≥70% of lines)
- Gate 2: void_generated ≥ 1 (at least one ?tag present)
- Gate 3: resolution_rate ≥ 0.50 (≥50% of ?tags resolved)

## Domain Coverage

### wcy_reasoning_traces.jsonl
General reasoning traces without void-B. Used as seed examples
for demonstrating structured WCY reasoning.

Domains: medical (2), logic (1), code (1), math (1), architecture (1)

### wcy_void_cycles.jsonl
Hand-crafted void-B resolution cycles. Each trace demonstrates
the full ?tag → > → . → : cycle across diverse domains.

Domains: medical, code, scientific, strategic, philosophical, epistemology

### wcy_traces_v1_clean.jsonl
Pipeline-generated traces. 48 combinations from an 8-domain ×
3-difficulty × 2-void-depth matrix. All 48 passed quality gates.

Domain × Difficulty × Void-depth breakdown:
- 8 domains: mathematical, legal, code, strategic,
             philosophical, scientific, medical, engineering
- 3 difficulties: simple (2 void target), medium (4), complex (6+)
- 2 void depths: single (independent resolution),
                cascade (resolution generates new ?tags)

## Trace Schema

Each line in the JSONL files is a JSON object:

```json
{
  "id": "v1_042_0",
  "domain": "medical",
  "difficulty": "complex",
  "void_depth": "cascade",
  "task": "Task description text",
  "void_instruction": "Void-B specific instruction",
  "wcy_reasoning": "Full WCY trace text",
  "quality": {
    "parse_rate": 1.0,
    "void_generated": 6,
    "void_resolved": 6,
    "resolution_rate": 1.0,
    "infer_count": 12,
    "act_count": 7,
    "warning_count": 1,
    "usable": true
  },
  "model": "claude-sonnet-4-20250514",
  "spec_version": "1.1",
  "type": "void_resolution_cycle",
  "usable": true
}
```

## Key field: wcy_reasoning

The `wcy_reasoning` field contains the actual WCY trace.
This is the primary training signal. Example (abbreviated):

```
~ task  sepsis_evaluation  domain=emergency_medicine
. patient=Im  age=78  nursing_home_resident
. vitals  temp=38.9  hr=112  rr=22  bp=94/60
: ?sepsis_criteria  hint=SIRS+organ_dysfunction  conf_range=0.7..0.95
> assess  SOFA_score  reason=from=3
. SOFA_result  score=6  organ_dysfunction=respiratory+renal
: sepsis_criteria=met  conf=0.94  from=3,5
: ?source_control  hint=UTI_3days+culture_pending  conf_range=0.6..0.85
> order  blood_cx_urine_cx  before_antibiotics  reason=from=7
. cultures_ordered  blood=2sets  urine=midstream
: ?antibiotic_choice  hint=nursing_home_gram_neg_coverage  conf_range=0.6..0.8
> start  piperacillin_tazobactam  4.5g_q6h  reason=from=10
: source_control=broad_spectrum_antibiotics  conf=0.87  from=7,12
! note  deescalate_based_on_culture_results  from=13
```

## Usage

```python
import json

with open('wcy_traces_v1_clean.jsonl') as f:
    traces = [json.loads(line) for line in f]

# Filter by domain
medical = [t for t in traces if t['domain'] == 'medical']

# Filter by void depth
cascade = [t for t in traces if t.get('void_depth') == 'cascade']

# Get reasoning text only
reasoning_texts = [t['wcy_reasoning'] for t in traces]

# Quality stats
avg_resolution = sum(
    t['quality']['resolution_rate'] for t in traces
) / len(traces)
```

## Version history

- v1.0 (2026-03-17): Initial release. 60 traces, 8 domains.
  Pipeline: 48 traces (Phase 5). Hand-crafted: 12 traces (Phase 4).

## License

CC BY 4.0 — Use freely, cite the source.
