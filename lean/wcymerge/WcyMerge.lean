/-
  WcyMerge — kernel-checked merge laws for the WCY-2 value model.

  The atomic value model is the dual-rail carrier D4: an atom is a pair
  of evidence rails (assert, refute) over F2, with
    UNK = (0,0), TRU = (1,0), FAL = (0,1), CON = (1,1).
  Document merge is the railwise join (SPEC.md section 5). This file
  kernel-checks the semilattice laws the specification relies on:
  associativity, commutativity, idempotence, identity (UNK), and
  absorption with the railwise meet — plus the conflict-surfacing law
  TRU join FAL = CON and the monotonicity of merge.

  Core Lean 4 only; every theorem is axiom-free (decided by the kernel).
  The carrier-theory provenance of this model is the Dual-Rail Carrier
  Program (see SPEC.md section 13); associativity is also published,
  with the same kernel discipline, at doi:10.5281/zenodo.21870654.
-/

namespace WcyMerge

/-- A dual-rail atom: (assert-rail, refute-rail). -/
abbrev Atom := Bool × Bool

def UNK : Atom := (false, false)
def TRU : Atom := (true,  false)
def FAL : Atom := (false, true)
def CON : Atom := (true,  true)

/-- Document merge on atoms: railwise join (evidence union). -/
def merge (x y : Atom) : Atom := (x.1 || y.1, x.2 || y.2)

/-- Railwise meet (evidence intersection; used by absorption). -/
def meet (x y : Atom) : Atom := (x.1 && y.1, x.2 && y.2)

/-- The information order: x ≤ y iff y has at least x's evidence. -/
def le (x y : Atom) : Bool := (!x.1 || y.1) && (!x.2 || y.2)

instance decForallAtom {p : Atom → Prop} [DecidablePred p] :
    Decidable (∀ x : Atom, p x) :=
  decidable_of_iff
      (p (false, false) ∧ p (false, true) ∧ p (true, false) ∧ p (true, true))
    ⟨fun ⟨h1, h2, h3, h4⟩ x =>
       match x with
       | (false, false) => h1
       | (false, true)  => h2
       | (true, false)  => h3
       | (true, true)   => h4,
     fun h => ⟨h _, h _, h _, h _⟩⟩

/-! ## The semilattice laws (SPEC section 5.1) -/

theorem merge_assoc : ∀ x y z : Atom, merge (merge x y) z = merge x (merge y z) := by
  decide

theorem merge_comm : ∀ x y : Atom, merge x y = merge y x := by decide

theorem merge_idem : ∀ x : Atom, merge x x = x := by decide

theorem merge_unk : ∀ x : Atom, merge UNK x = x := by decide

theorem meet_assoc : ∀ x y z : Atom, meet (meet x y) z = meet x (meet y z) := by
  decide

theorem absorption :
    ∀ x y : Atom, merge x (meet x y) = x ∧ meet x (merge x y) = x := by
  decide

/-! ## Conflict surfacing and monotonicity (SPEC sections 5.2, 6) -/

/-- Conflict is a value: merging opposing resolved evidence yields CON,
    losing neither side. -/
theorem conflict_surfaces : merge TRU FAL = CON := by decide

/-- Merge never destroys evidence: both inputs lie below the merge in the
    information order (append-only as a lattice fact). -/
theorem merge_monotone :
    ∀ x y : Atom, le x (merge x y) = true ∧ le y (merge x y) = true := by
  decide

/-- Merge is the least upper bound: anything above both inputs is above
    the merge. -/
theorem merge_lub :
    ∀ x y z : Atom, le x z = true → le y z = true → le (merge x y) z = true := by
  decide

/-- Retraction is monotone: adding refuting evidence to a resolved TRU
    yields CON (the conflict is surfaced), never a silent deletion. -/
theorem retraction_is_monotone : merge TRU (false, true) = CON := by decide

end WcyMerge
