# -*- coding: utf-8 -*-
"""
test_wcy_bilattice.py — tests for the carrier bilattice coordinates.

Runnable two ways (no third-party dependency required):
    python test_wcy_bilattice.py      # runs every test_* function
    pytest test_wcy_bilattice.py      # standard discovery
"""

from __future__ import annotations

from wcy_bilattice import (
    BilatticeCoord, UNK, TRU, FAL, CON,
    COORD_UNK, COORD_TRU, COORD_FAL, COORD_CON, CORNER_COORD,
    from_point, from_conf_range, classify_void, accumulate,
    coord_of_line,
)
from wcy_parser import parse_wcy, flatten


# ── corners & derived axes ────────────────────────────────────────────────────

def test_corners_round_trip():
    for name, (t, f) in CORNER_COORD.items():
        assert BilatticeCoord(t, f).corner() == name


def test_derived_axes():
    assert COORD_UNK.knowledge == 0.0
    assert COORD_CON.conflict == 1.0
    assert COORD_TRU.truth == 1.0 and COORD_FAL.truth == -1.0
    assert COORD_TRU.knowledge == COORD_FAL.knowledge == 1.0
    assert COORD_UNK.info == 0.0 and COORD_CON.info == 2.0


def test_clamping():
    c = BilatticeCoord(1.5, -0.3)
    assert c.t == 1.0 and c.f == 0.0


# ── the two orders ────────────────────────────────────────────────────────────

def test_knowledge_order_bounds_lattice():
    for c in (COORD_TRU, COORD_FAL):
        assert COORD_UNK.leq_k(c) and c.leq_k(COORD_CON)
    # TRU and FAL are incomparable in the knowledge order
    assert not COORD_TRU.leq_k(COORD_FAL) and not COORD_FAL.leq_k(COORD_TRU)


def test_truth_order_bounds_lattice():
    assert COORD_FAL.leq_t(COORD_TRU)
    for c in (COORD_UNK, COORD_CON):
        assert COORD_FAL.leq_t(c) and c.leq_t(COORD_TRU)
    # UNK and CON are incomparable in the truth order
    assert not COORD_UNK.leq_t(COORD_CON) and not COORD_CON.leq_t(COORD_UNK)


# ── the four lattice operations ───────────────────────────────────────────────

def test_gullible_join_makes_conflict():
    # accepting independent for/against evidence yields contradiction
    assert COORD_TRU.gullible(COORD_FAL) == COORD_CON


def test_consensus_meet_makes_ignorance():
    assert COORD_TRU.consensus(COORD_FAL) == COORD_UNK


def test_truth_meet_join():
    assert COORD_TRU.tjoin(COORD_FAL) == COORD_TRU
    assert COORD_TRU.tmeet(COORD_FAL) == COORD_FAL


def test_negation():
    assert COORD_TRU.neg() == COORD_FAL
    assert COORD_FAL.neg() == COORD_TRU
    assert COORD_UNK.neg() == COORD_UNK   # negation fixes the k-extremes
    assert COORD_CON.neg() == COORD_CON


def test_operations_commute():
    a, b = BilatticeCoord(0.7, 0.2), BilatticeCoord(0.3, 0.9)
    assert a.gullible(b) == b.gullible(a)
    assert a.consensus(b) == b.consensus(a)
    assert a.tjoin(b) == b.tjoin(a)
    assert a.tmeet(b) == b.tmeet(a)


# ── confidence → carrier mapping ──────────────────────────────────────────────

def test_point_belief():
    assert from_point(0.9).corner() == TRU
    assert from_point(0.1).corner() == FAL
    # a confident point has full knowledge (no ignorance)
    assert abs(from_point(0.9).info - 1.0) < 1e-9


def test_conf_range_is_belief_interval():
    # info = 1 - width; a single interval never reaches CON on its own
    c = from_conf_range(0.4, 0.8)
    assert abs(c.info - (1 - 0.4)) < 1e-9
    assert c.corner() != CON
    # swapped bounds are tolerated
    assert from_conf_range(0.8, 0.4) == from_conf_range(0.4, 0.8)


def test_single_interval_cannot_be_conflict():
    # exhaustive-ish sweep: no [lo,hi] interval lands on CON except the
    # degenerate point lo=hi=0.5 (perfect ambivalence on the boundary).
    for i in range(0, 101):
        for j in range(i, 101):
            lo, hi = i / 100, j / 100
            c = from_conf_range(lo, hi)
            if c.corner() == CON:
                assert lo == hi == 0.5


def test_classify_void_unknown_vs_conflict():
    # skewed midpoint -> unknown (explore); balanced -> conflict (judge)
    assert classify_void(from_conf_range(0.4, 0.8)) == "unknown"
    assert classify_void(from_conf_range(0.3, 0.7)) == "conflict"
    assert classify_void(COORD_CON) == "conflict"


# ── s8 accumulator duality ────────────────────────────────────────────────────

def test_max_is_count_invariant():
    one = accumulate([COORD_FAL], mode="max", alpha=1.0)[-1]
    five = accumulate([COORD_FAL] * 5, mode="max", alpha=1.0)[-1]
    assert one == five == COORD_FAL


def test_sum_is_count_sensitive():
    q1 = accumulate([COORD_FAL], mode="sum", alpha=1.0, gamma=0.1)[-1]
    q3 = accumulate([COORD_FAL] * 3, mode="sum", alpha=1.0, gamma=0.1)[-1]
    assert q3.f > q1.f > 0


def test_max_reads_con_regardless_of_imbalance():
    balanced = accumulate([COORD_TRU, COORD_FAL], mode="max", alpha=1.0)[-1]
    skewed = accumulate([COORD_FAL] + [COORD_TRU] * 9, mode="max", alpha=1.0)[-1]
    assert balanced.corner() == CON == skewed.corner()


def test_accumulate_rejects_bad_mode():
    try:
        accumulate([COORD_TRU], mode="avg")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown mode")


# ── line-level integration ────────────────────────────────────────────────────

def test_coord_of_line_dispatch():
    text = (
        "~ context  case=x\n"
        ". fact  observed\n"
        ": ?diag  hint=h  conf_range=0.4..0.8\n"
        ": diag=influenza  conf=0.94  from=2\n"
        "! note  unresolved  from=4\n"
    )
    lines = flatten(parse_wcy(text))
    by_num = {ln.line_num: coord_of_line(ln) for ln in lines}
    assert by_num[2].corner() == TRU          # bare observe = asserted fact
    assert by_num[3].corner() == UNK          # void-B interval = ignorance
    assert classify_void(by_num[3]) == "unknown"
    assert by_num[4].corner() == TRU          # conf=0.94 inference
    assert by_num[5].corner() == CON          # exception = contradiction (⊤)


# ── runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  ok  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed.")
