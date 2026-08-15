# Open task: verify RULA Tables A and B

**Status: not done. No RULA integer from this system may appear in the paper
until it is.**

`tests/test_all.py::test_rula_tables_flagged_unverified` fails if anyone flips
`rula.TABLES_VERIFIED` without recording that this task was completed.

---

## Why this is open

`rula.py` needs three lookup tables from McAtamney and Corlett (1993):

- **Table C** (grand score from Score A and Score B): verified against published
  reproductions. Anchor cells are asserted in the test suite. High confidence.
- **Table A** (upper arm, lower arm, wrist, wrist twist) and **Table B** (neck,
  trunk, legs): transcribed, but the sources available were image-based and did
  not extract reliably as text. Transcribing a numeric grid cell by cell from a
  mangled extraction and calling it verified would be exactly the kind of
  unchecked claim this rebuild exists to eliminate.

The pipeline runs correctly either way. Only the absolute RULA integers depend
on these two grids.

---

## How to close it

1. Get a printed or clean PDF RULA worksheet (McAtamney, L. and Corlett, E.N.,
   1993, *RULA: a survey method for the investigation of work-related upper limb
   disorders*, Applied Ergonomics 24(2), 91-99, or an authorised reproduction).
2. Compare cell by cell against `_TABLE_A` and `_TABLE_B` in `src/rula.py`.
   Note that scoring only ever indexes a fixed slice of these tables, because
   wrist, wrist twist, and legs are assumed constants, so start with the rows
   and columns actually reachable given `RulaAssumptions`.
3. Either correct the literals in place, or inject validated grids without
   touching the source:

```python
import rula
rula.set_tables(table_a=my_table_a, table_b=my_table_b, verified=True)
```

4. Update `test_rula_tables_flagged_unverified` to assert `True`, and record who
   verified it and against which edition.

---

## Related: the assumptions the expert coder must match

Wrist posture, wrist twist, and legs are not observable from two cameras with a
tool in hand. `RulaAssumptions` fixes them:

```
wrist = 2, wrist twist = 1, legs = 1, muscle use = 0, force = 0
```

Assuming a near-neutral wrist **under**-estimates strain for fine conservation
work, so the automated grand score is a conservative estimate. State this
wherever the number appears.

When the expert codes the same clips, they must use the same assumptions, or the
Bland-Altman comparison measures the disagreement between two different
definitions rather than between automated and human scoring. Give them
`RulaAssumptions.note()` output before they start.

Because Score B (neck, trunk, legs) is fully measurable from the rig, it is the
cleaner apples-to-apples comparison. `validate_rula` computes both.
