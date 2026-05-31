# -*- coding: utf-8 -*-
"""
wcy_bilattice.py — Carrier Bilattice Coordinates for WCY v1
============================================================
Reference implementation that places every WCY value on the **Belnap
bilattice** (Belnap's FOUR / the D4 "carrier" of Carrier Theory) and gives
it a coordinate in the continuous (truth x knowledge) square.

Why this exists
---------------
WCY already represents confidence (`conf=`) and an explicit unknown
(`?tag` + `conf_range=`). The Position Paper v3.0 showed that
`conf_range=(lo,hi)` is a Dempster-Shafer belief interval `[bel, pl]`, and
that the void-B marker `?` corresponds to unallocated mass `m(Theta) > 0`.

A single interval, however, cannot tell **UNKNOWN** ("no evidence either
way — go explore") apart from **CONFLICT** ("evidence on *both* sides — make
a judgement"). Those are two different kinds of not-knowing. The bilattice
is the minimal structure that separates them, because it carries *two*
independent orderings:

  - knowledge / information order  (<=_k):  UNK  <=  {TRU, FAL}  <=  CON
  - truth order                    (<=_t):  FAL  <=  {UNK, CON}  <=  TRU

The four corners are Belnap's four values:

        knowledge (t v f)
              ^
       CON =  T  (1,1)          <- both true and false seen  (contradiction)
             / \
        TRU /   \ FAL           <- (1,0) only-true , (0,1) only-false
             \ /
       UNK =  _|_ (0,0)         <- nothing seen                (ignorance)
              |
              +--------> truth (t - f)

Coordinate convention (chosen for v1: "Belnap corners + (t,k) square")
----------------------------------------------------------------------
A value is a *presence pair* ``(t, f)`` in the unit square [0,1]^2:

    t = evidence-for      ("saw_true"  energy, e+)
    f = evidence-against  ("saw_false" energy, e-)

Derived views (the two bilattice axes plus the carrier diagonals):

    truth      = t - f          in [-1, 1]   (truth-order coordinate)
    knowledge  = max(t, f)      in [0, 1]    (presence; the t v f join)
    conflict   = min(t, f)      in [0, 1]    (contradiction degree; t ^ f)
    info       = t + f          in [0, 2]    (count / information height)

`knowledge = t v f` separates UNK (0) from everything else (1);
`conflict = t ^ f` separates CON (1) from everything else (0). Together they
locate which corner a coordinate is nearest.

The s8 connection (sum vs max accumulator)
------------------------------------------
The experiment `s8_v1_bilattice_conflict` showed that *where* you put the
lattice matters: accumulating evidence across time with a plain decaying
**sum** (linear attention) tracks *count*, whereas a tropical **max** (the
bilattice join) tracks *presence* and is count-invariant. In the neural
model that count-invariance makes CON robust under count-imbalance — a lone
minority assertion is swamped by the majority in the *shared* recurrent
state under `sum`, but preserved under `max`.

`accumulate(..., mode="sum"|"max")` exposes the underlying count-vs-presence
distinction on WCY coordinates. Note the corner-flip robustness result is a
property of s8's *shared-state* model; on clean, separated (t, f) channels
this fold demonstrates count-invariance (max) vs count-sensitivity (sum),
which is the mechanism behind it, not the full OOD effect itself.

Usage
-----
    from wcy_parser import parse_wcy, flatten
    from wcy_bilattice import coord_of_line, classify_void, BilatticeCoord

    lines = flatten(parse_wcy(text))
    for ln in lines:
        c = coord_of_line(ln)
        print(ln.line_num, c.corner, round(c.truth, 2), round(c.knowledge, 2))

No third-party dependencies. Pure standard library; `wcy_parser` is only
needed for the line-level helpers.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

# wcy_parser is optional: the algebra works standalone. The line-level
# helpers (coord_of_line / coords_of) need it.
try:
    from wcy_parser import WCYLine  # noqa: F401
    _HAS_PARSER = True
except Exception:  # pragma: no cover - parser absent
    WCYLine = object  # type: ignore
    _HAS_PARSER = False


# ── Belnap corners ────────────────────────────────────────────────────────────

#: The four Belnap values, with their canonical presence-pair (t, f) corners.
UNK = "UNK"   # _|_  (0, 0)  ignorance      — nothing seen          → explore
TRU = "TRU"   #  t   (1, 0)  only-true                              → assert true
FAL = "FAL"   #  f   (0, 1)  only-false                             → assert false
CON = "CON"   #  T   (1, 1)  contradiction  — both seen             → judge

CORNER_COORD: dict[str, tuple[float, float]] = {
    UNK: (0.0, 0.0),
    TRU: (1.0, 0.0),
    FAL: (0.0, 1.0),
    CON: (1.0, 1.0),
}

#: Unicode Belnap symbols, handy for printing.
CORNER_SYMBOL: dict[str, str] = {UNK: "⊥", TRU: "T", FAL: "F", CON: "⊤"}


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


# ── The coordinate ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BilatticeCoord:
    """
    A point on the Belnap bilattice, stored as a presence pair (t, f) in
    [0,1]^2 and exposing both bilattice axes plus the four lattice meet/join
    operations.
    """
    t: float  # evidence-for      (e+, "saw_true")
    f: float  # evidence-against  (e-, "saw_false")

    def __post_init__(self):
        # frozen dataclass: clamp via object.__setattr__
        object.__setattr__(self, "t", _clamp01(self.t))
        object.__setattr__(self, "f", _clamp01(self.f))

    # — derived axes ———————————————————————————————————————————————————————
    @property
    def truth(self) -> float:
        """Truth-order coordinate t - f, in [-1, 1]. 0 = balanced/undecided."""
        return self.t - self.f

    @property
    def knowledge(self) -> float:
        """Presence (the t v f join), in [0, 1]. 0 only at UNK."""
        return max(self.t, self.f)

    @property
    def conflict(self) -> float:
        """Contradiction degree (the t ^ f meet), in [0, 1]. 1 only at CON."""
        return min(self.t, self.f)

    @property
    def info(self) -> float:
        """Information height t + f, in [0, 2]: 0 at UNK, 2 at CON."""
        return self.t + self.f

    # — corner classification ——————————————————————————————————————————————
    def corner(self, tau: float = 0.5) -> str:
        """
        Nearest Belnap corner under presence threshold `tau`:
        a channel "fired" when its energy >= tau.
        """
        saw_t = self.t >= tau
        saw_f = self.f >= tau
        if saw_t and saw_f:
            return CON
        if saw_t:
            return TRU
        if saw_f:
            return FAL
        return UNK

    # — the two bilattice orders ———————————————————————————————————————————
    def leq_k(self, other: "BilatticeCoord") -> bool:
        """Knowledge order: more evidence on both channels (componentwise)."""
        return self.t <= other.t and self.f <= other.f

    def leq_t(self, other: "BilatticeCoord") -> bool:
        """Truth order: more for, less against."""
        return self.t <= other.t and self.f >= other.f

    # — the four lattice operations ————————————————————————————————————————
    def consensus(self, other: "BilatticeCoord") -> "BilatticeCoord":
        """k-meet (x) — keep only what *both* agree is present (cautious)."""
        return BilatticeCoord(min(self.t, other.t), min(self.f, other.f))

    def gullible(self, other: "BilatticeCoord") -> "BilatticeCoord":
        """k-join (+) — accept evidence from *either* source (accumulation).

        TRU + FAL = CON: independent for/against evidence becomes conflict.
        """
        return BilatticeCoord(max(self.t, other.t), max(self.f, other.f))

    def tmeet(self, other: "BilatticeCoord") -> "BilatticeCoord":
        """truth-meet (^) — the 'more false' combination."""
        return BilatticeCoord(min(self.t, other.t), max(self.f, other.f))

    def tjoin(self, other: "BilatticeCoord") -> "BilatticeCoord":
        """truth-join (v) — the 'more true' combination."""
        return BilatticeCoord(max(self.t, other.t), min(self.f, other.f))

    # — negation (truth flip, knowledge-preserving) ————————————————————————
    def neg(self) -> "BilatticeCoord":
        """Bilattice negation: swap for/against. Fixes UNK and CON."""
        return BilatticeCoord(self.f, self.t)

    def __repr__(self) -> str:
        return (f"BilatticeCoord(t={self.t:.3f}, f={self.f:.3f}, "
                f"{CORNER_SYMBOL[self.corner()]} {self.corner()}, "
                f"truth={self.truth:+.3f}, know={self.knowledge:.3f})")


# Canonical corner constants (after the class is defined).
COORD_UNK = BilatticeCoord(0.0, 0.0)
COORD_TRU = BilatticeCoord(1.0, 0.0)
COORD_FAL = BilatticeCoord(0.0, 1.0)
COORD_CON = BilatticeCoord(1.0, 1.0)


# ── Mapping WCY confidence onto the carrier ───────────────────────────────────

def from_point(conf: float) -> BilatticeCoord:
    """
    A committed point belief (`conf=c`, no range) as a degenerate interval
    [c, c]: bel = pl = c, so t = c, f = 1 - c. Full knowledge, no ignorance.
    Note c == 0.5 lands exactly on the CON/UNK boundary (perfect ambivalence).
    """
    c = _clamp01(conf)
    return BilatticeCoord(c, 1.0 - c)


def from_conf_range(lo: float, hi: float) -> BilatticeCoord:
    """
    A void-B / interval belief `conf_range=lo..hi` read as a Dempster-Shafer
    belief interval [bel, pl] = [lo, hi]:

        t = bel        = lo            (mass committed *for*)
        f = 1 - pl     = 1 - hi        (mass committed *against*)
        info = t + f   = 1 - (hi - lo) = 1 - ignorance

    Consequence (the theoretical payoff): a *single* interval always has
    info <= 1 — it lives in the consistent lower-left triangle and can never
    reach CON=(1,1) on its own. Conflict only arises by *combining* evidence
    (see `gullible`/`accumulate`). The midpoint still hints which kind of
    not-knowing it is — see `classify_void`.
    """
    lo, hi = _clamp01(lo), _clamp01(hi)
    if lo > hi:
        lo, hi = hi, lo
    return BilatticeCoord(lo, 1.0 - hi)


def classify_void(coord: BilatticeCoord, mid_tol: float = 0.15) -> str:
    """
    Distinguish the two kinds of not-knowing for a low-knowledge coordinate,
    following the conf_range-midpoint rule (WCY-OC §2.4):

      - 'conflict' : balanced (|truth| <= mid_tol, midpoint ~ 0.5)
                     -> evidence pulls both ways -> needs a JUDGEMENT.
      - 'unknown'  : skewed   (|truth| >  mid_tol)
                     -> direction is set, magnitude isn't -> needs EXPLORATION.

    For an already-saturated CON corner this returns 'conflict' directly.
    """
    if coord.corner() == CON:
        return "conflict"
    return "conflict" if abs(coord.truth) <= mid_tol else "unknown"


# ── Line-level helpers (need wcy_parser) ──────────────────────────────────────

def coord_of_line(line: "WCYLine") -> BilatticeCoord:
    """
    Map one parsed WCY line to a bilattice coordinate by phase + confidence:

        '!' exception        -> CON  (an unresolved/contradictory state, T)
        ':' / '.' with range -> from_conf_range  (interval belief)
        ':' / '.' with conf  -> from_point(conf)
        void-B, no range/conf -> UNK  (_|_, pure ignorance)
        '.' observe (no conf) -> TRU  (an asserted fact, point at conf=1)
        otherwise (`~`/`>`)   -> UNK  (carries no truth claim)
    """
    if not _HAS_PARSER:  # pragma: no cover
        raise RuntimeError("wcy_parser is required for coord_of_line()")

    phase = getattr(line, "phase", None)
    if phase == "!":
        return COORD_CON
    if line.conf_range is not None:
        return from_conf_range(*line.conf_range)
    if line.conf is not None:
        return from_point(line.conf)
    if getattr(line, "is_void", False):
        return COORD_UNK
    if phase == ".":
        return COORD_TRU
    return COORD_UNK


def coords_of(lines: Iterable["WCYLine"]) -> list[tuple[int, BilatticeCoord]]:
    """(line_num, coord) for every line. Convenience over `coord_of_line`."""
    return [(ln.line_num, coord_of_line(ln)) for ln in lines]


# ── Temporal accumulation: the s8 sum-vs-max duality ──────────────────────────

def accumulate(
    evidence: Iterable[BilatticeCoord],
    mode: str = "max",
    alpha: float = 0.9,
    gamma: float = 1.0,
) -> list[BilatticeCoord]:
    """
    Fold a stream of evidence coordinates into a running carrier state,
    returning the state after each step.

      mode='max'  (tropical join / presence accumulator):
            S <- ( max(alpha*S.t, gamma*e.t), max(alpha*S.f, gamma*e.f) )
        Count-invariant: repeating an item does not change the state (max is
        idempotent). This is the bilattice operation in the temporal update —
        where Carrier Theory says the lattice belongs.

      mode='sum'  (decaying sum / linear attention = count accumulator):
            S <- ( alpha*S.t + gamma*e.t, alpha*S.f + gamma*e.f )  [clamped]
        Count-sensitive: the channel grows with the number of assertions.

    On a *shared* recurrent state (s8), count-sensitivity is what lets a
    majority swamp a lone minority so CON is missed; the count-invariant
    `max` avoids it. On these clean (t, f) channels the two modes differ only
    in count-invariance vs count-sensitivity (see module docstring).

    Both start from UNK = (0, 0).
    """
    if mode not in ("max", "sum"):
        raise ValueError("mode must be 'max' or 'sum'")
    out: list[BilatticeCoord] = []
    st, sf = 0.0, 0.0
    for e in evidence:
        if mode == "max":
            st = max(alpha * st, gamma * e.t)
            sf = max(alpha * sf, gamma * e.f)
        else:
            st = alpha * st + gamma * e.t
            sf = alpha * sf + gamma * e.f
        s = BilatticeCoord(st, sf)  # clamps to [0,1]
        out.append(s)
    return out


# ── CLI / self-test ───────────────────────────────────────────────────────────

def _selftest() -> None:
    eps = 1e-9

    # corners round-trip through (t, f)
    for name, (t, f) in CORNER_COORD.items():
        assert BilatticeCoord(t, f).corner() == name, name

    # derived axes at the corners
    assert COORD_UNK.knowledge == 0.0 and COORD_CON.conflict == 1.0
    assert abs(COORD_TRU.truth - 1.0) < eps and abs(COORD_FAL.truth + 1.0) < eps
    assert COORD_TRU.knowledge == 1.0 and COORD_FAL.knowledge == 1.0
    assert COORD_UNK.info == 0.0 and COORD_CON.info == 2.0

    # the two orders bound the lattice
    for c in (COORD_TRU, COORD_FAL):
        assert COORD_UNK.leq_k(c) and c.leq_k(COORD_CON)   # knowledge order
    assert COORD_FAL.leq_t(COORD_TRU)                       # truth order
    assert COORD_FAL.leq_t(COORD_UNK) and COORD_UNK.leq_t(COORD_TRU)

    # gullible (k-join) of independent for/against evidence = contradiction
    assert COORD_TRU.gullible(COORD_FAL) == COORD_CON
    # consensus (k-meet) of the same = ignorance
    assert COORD_TRU.consensus(COORD_FAL) == COORD_UNK
    # negation swaps TRU/FAL, fixes UNK and CON
    assert COORD_TRU.neg() == COORD_FAL and COORD_CON.neg() == COORD_CON
    assert COORD_UNK.neg() == COORD_UNK

    # conf_range = belief interval: a single interval never reaches CON,
    # and its info equals 1 - width.
    wide = from_conf_range(0.4, 0.8)        # UNKNOWN-type void (skewed mid 0.6)
    bal  = from_conf_range(0.3, 0.7)        # CONFLICT-type void (mid 0.5)
    assert wide.corner() != CON and bal.corner() != CON
    assert abs(wide.info - (1 - 0.4)) < 1e-6
    assert classify_void(wide) == "unknown"
    assert classify_void(bal) == "conflict"

    # point belief: confident inference is TRU, ambivalent (0.5) is the boundary
    assert from_point(0.9).corner() == TRU
    assert from_point(0.1).corner() == FAL

    # the s8 knob at coordinate level: count-invariance (max) vs
    # count-sensitivity (sum) — the mechanism behind the OOD result.
    # (a) max is idempotent: repeating evidence does not change the state.
    one = accumulate([COORD_FAL], mode="max", alpha=1.0, gamma=1.0)[-1]
    five = accumulate([COORD_FAL] * 5, mode="max", alpha=1.0, gamma=1.0)[-1]
    assert one == five == COORD_FAL                       # count-invariant
    # (b) sum grows with count (below clamp): presence becomes count.
    q1 = accumulate([COORD_FAL], mode="sum", alpha=1.0, gamma=0.1)[-1]
    q3 = accumulate([COORD_FAL] * 3, mode="sum", alpha=1.0, gamma=0.1)[-1]
    assert q3.f > q1.f > 0                                # count-sensitive
    # (c) with both polarities present and no decay, max reads CON regardless
    # of how lop-sided the counts are (the count-invariance that makes CON
    # robust in s8's shared-state model).
    balanced = accumulate([COORD_TRU, COORD_FAL], mode="max", alpha=1.0)[-1]
    skewed = accumulate([COORD_FAL] + [COORD_TRU] * 9, mode="max", alpha=1.0)[-1]
    assert balanced.corner() == CON and skewed.corner() == CON
    print(f"  s8 knob: max idempotent (count-invariant), "
          f"sum f-channel {q1.f:.1f}->{q3.f:.1f} (count-sensitive)")

    print("  all assertions passed.")


if __name__ == "__main__":
    print("=" * 64)
    print("  WCY Bilattice Coordinates v1 — self-test + demo")
    print("=" * 64)

    print("\n[corners]")
    for name in (UNK, TRU, FAL, CON):
        c = BilatticeCoord(*CORNER_COORD[name])
        print(f"  {CORNER_SYMBOL[name]} {name}: (t,f)={CORNER_COORD[name]}  "
              f"truth={c.truth:+.1f}  knowledge={c.knowledge:.1f}  "
              f"conflict={c.conflict:.1f}")

    print("\n[self-test]")
    _selftest()

    # Demo on a WCY trace, if the parser is available.
    if _HAS_PARSER:
        from wcy_parser import parse_wcy, flatten
        SAMPLE = """
~ context  case=example
. patient=Kim  age=45  temp=38.5
. symptoms  fever  cough  duration=7days
: ?diagnosis  hint=fever+cough+duration  conf_range=0.4..0.8
> order  rapid_flu_test  reason=from=4
. test_result=positive
: diagnosis=influenza  conf=0.94  from=4,6
! note  monitor_for_complications  from=7
""".strip()
        print("\n[trace coordinates]")
        print(f"  {'#':>2}  {'phase':<9} {'corner':<5} {'truth':>6} "
              f"{'know':>5} {'kind':<9}")
        print("  " + "-" * 44)
        for ln in flatten(parse_wcy(SAMPLE)):
            c = coord_of_line(ln)
            # 'kind' is only meaningful for genuine epistemic voids / CON,
            # not for ~meta / >act lines that are UNK merely for lack of a claim.
            is_void = getattr(ln, "is_void", False)
            kind = classify_void(c) if (is_void or c.corner() == CON) else "-"
            print(f"  {ln.line_num:>2}  {ln.phase_name:<9} "
                  f"{CORNER_SYMBOL[c.corner()]:<5} {c.truth:>+6.2f} "
                  f"{c.knowledge:>5.2f} {kind:<9}")
    else:
        print("\n[trace demo skipped — wcy_parser.py not importable]")
