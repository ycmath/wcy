# WCY-2 dataset (v2)

- generated: 2026-08-13T07:54:20+00:00
- **total adopted traces: 379**
- spec: WCY-2 draft v0.1 (SPEC.md sections 3-11 + Appendix A/B)
- generation: Stage A `claude-sonnet-5`; Stage B b1/b2 `claude-opus-5`, b3/b4 `claude-sonnet-5`
- audit: `claude-sonnet-5` single pass + 10% adversarial re-audit on `claude-opus-5`
- repair: opus for content defects, sonnet for provenance/structural

## Composition

| file | kind | traces |
|---|---|---|
| `wcy2_semantized_540.jsonl` | semantize | 317 |
| `wcy2_native_b1.jsonl` | b1 | 18 |
| `wcy2_native_b2.jsonl` | b2 | 9 |
| `wcy2_native_b3.jsonl` | b3 | 14 |
| `wcy2_native_b4.jsonl` | b4 | 21 |
| **total** | | **379** |

| domain | traces |
|---|---|
| medical | 55 |
| scientific | 51 |
| engineering | 49 |
| mathematical | 48 |
| code | 46 |
| strategic | 46 |
| legal | 42 |
| philosophical | 40 |
| math | 1 |
| epistemology | 1 |

## Yield

| stage | requested | adopted | rate |
|---|---|---|---|
| Stage A (semantize v1) | 540 | 317 | 59% |
| Stage B b1 | 48 | 18 | 38% |
| Stage B b2 | 24 | 9 | 38% |
| Stage B b3 | 24 | 14 | 58% |
| Stage B b4 | 24 | 21 | 88% |

## Disclosure: corrected_from_v1

**145 of 379 adopted traces (38%) had their DOMAIN CONTENT altered** relative to their v1 source, not merely re-encoded. Repair-with-disclosure was authorised after attribution analysis. Corrected defects come from **both origins**: some are inherited verbatim from the v1 corpus (measured inherited domain-defect rate: ~7% of v1 traces; e.g. a hallucinated API name and a mis-scored clinical criterion present in the v1 source text) and some were introduced during re-encoding. Both classes were repaired, and every repaired row is flagged.

- `corrected_from_v1: true` — content changed by an opus repair: **145**
- `was_fixed: true` — any repair applied (incl. provenance/structural): 259
- `audit_notes` present — adopted with minor issues recorded: 68

This corpus is therefore **not** a faithful re-encoding of WCY v1. Any v1-vs-v2 comparison must exclude or account for the corrected rows.

## Gate definitions

Every adopted trace passed all nine local gates AND an audit verdict of `pass`, or `fix` whose repaired trace re-passed all nine.

| gate | name | definition |
|---|---|---|
| G1 | parse | wcy_parser parse_rate == 1.0 over every candidate line |
| G2 | laws | verify_laws() passes; document build + merge raise nothing |
| G3 | con-discipline | every ^C atom is later resolved or obstructed |
| G4 | void-cycle | every ?tag ends in resolve, obstruction, or a tail note naming it with hint= preserved |
| G5 | resolve-integrity | was= equals a state the atom actually held; now= is on the resolved face; from= points at real earlier lines/labels |
| G6 | retraction-form | no deletion expression; retraction only as tag!=v or an explicit ^F rail |
| G7 | rail-sanity | no ^F/^C atom without from= or a prior observation of that tag |
| G8 | merge-replay | (merge kind) the trace's merged block agrees atom-for-atom with wcy_merge's own merge of the two sources |
| G9 | provenance | each inference's `from=` cites a line that actually contains the tag or measurement being reasoned about; unique-candidate mis-citations are repaired deterministically before audit |

## Cost ledger

| model | calls | USD |
|---|---|---|
| `claude-sonnet-5` | 1717 | 59.89 |
| `claude-opus-5` | 395 | 24.73 |
| `claude-haiku-4-5` | 6 | 0.02 |
| **total** | **2118** | **84.64** |

| phase | authorized | spent |
|---|---|---|
| recovery_finish | 4.0 | 4.14 |
| stage_b | 13.5 | 5.60 |

Hard cap $95.00; final spend **$84.64**.

## Prompt hashes

| prompt | sha256 |
|---|---|
| `SYSTEM` (FROZEN) | `4678bfb42ac267e1333d4efe3cfd69ce8900f67165cab5f0410cd4f5787e8f60` |
| `SEMANTIZE_PROMPT` | `df7cc3e2b7c8e04c52aa73a869ff933809ca8dba8fe15a6ba31a56af9697ea9e` |
| `GEN_B1` | `53a6e35198ed55479df0e59fdbd394f3c4b31f20b383655ca256abb399bb8bd7` |
| `GEN_B2` | `79cda72bd5b2a4c07b05e00eb8fe12b31d48abf7caf9fbe28dea2f53010d573b` |
| `GEN_B3` | `8227505ae988d1024fb17dbd067b7aac6019a80fb67c6a2b5197bd38e2066957` |
| `GEN_B4` | `5c815b58af8d1660c962cb8f58b587a06a5e7f9ac4774e2a5da2cb8b0e194fac` |
| `AUDIT_PROMPT` | `a113cf05e59c38139b43768a19729c8ce3c126d0b717c431892ec9375968deb1` |
| `FIX_PROMPT` | `0e2f1ca71ec47fcaaf7fab08f111a524b6e26cfd43871a4ceae369c924dccf2d` |

## Scope reduction — reason

Planned scope was Stage A 540 + Stage B 600 = 1,140 traces. Delivered 379. The reduction was driven by **sustained Batch API degradation** over the build window, not by quality or budget policy:

- Batch throughput oscillated between healthy (180/180 in minutes) and fully stalled (0/170 for 68 minutes; 0/18 for 60 minutes).
- A cancel-flush technique was discovered: cancelling a stalled batch forces finalisation and typically returns 60-95% of its requests. Most recovered work in this corpus came through that path.
- Non-batch fallback is reliable but ~2x the price, so it is capped at 15 requests and must fit the phase's remaining authorization.
- Stage B was consequently re-scoped from 600 to 120 requested, proportionally across types.

## Provenance and reuse

Stage A semantizes the 540-trace v1 corpus in `data/*.jsonl`; each row keeps `source_id`. Stage B is native v2 against locally-built scaffolds — for B1 the two source documents and the merged block are computed by `wcy_merge` itself and injected into the prompt, so the merged block is kernel output and G8 re-derives it independently.

> Not promoted into the repo's `data/` directory and not published. Both remain owner gates.
