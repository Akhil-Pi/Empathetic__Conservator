"""
rula.py
=======
Part 5, component 1: automated RULA scoring from measured joint angles.

Why this exists
---------------
The original paper promised a validation of the strain score against expert RULA
but never ran it. Part 5 runs it. To do that we need to turn measured angles into
a RULA grand score using the published lookup tables.

Honest limitations, stated up front
------------------------------------
A front + side camera with MediaPipe CANNOT observe three of RULA's inputs:
  - wrist posture (Step 3)
  - wrist twist  (Step 4)
  - legs         (Step 11)
Conservators also hold tools, so hands are frequently occluded even when visible.
Those three inputs are therefore ASSUMED, not measured. Each assumption is a named
constant below with its bias direction documented. Assuming a near-neutral wrist
UNDER-estimates true strain for fine conservation work, so the automated grand
score is a conservative (low) estimate. This must be disclosed wherever the number
appears.

Because wrist/twist/legs are fixed by assumption, scoring only ever indexes a fixed
slice of Table A and Table B. The far corners of those tables are never reached in
practice, so table-transcription risk is confined to the slice actually used, which
the self-test checks.

Table provenance
----------------
  Table C  : verified against published reproductions (Ergonomics Plus worksheet;
             Scribd RULA worksheet). High confidence.
  Table A/B: transcribed from the standard McAtamney & Corlett (1993) worksheet as
             reproduced by Ergonomics Plus / Osmond Ergonomics. PDF table extraction
             is unreliable, so these are flagged TABLES_VERIFIED = False. Before any
             RULA integer goes into the paper, verify Table A and Table B against a
             printed worksheet, or inject a validated grid via set_tables().

Reference: McAtamney, L. & Corlett, E.N. (1993). RULA: a survey method for the
investigation of work-related upper limb disorders. Applied Ergonomics 24(2), 91-99.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


TABLES_VERIFIED = False  # Table A/B not yet checked against a printed worksheet.

RULA_SOURCE = ("McAtamney & Corlett 1993; tables via Ergonomics Plus / Osmond "
               "Ergonomics reproductions. Table C verified; Table A/B pending "
               "worksheet verification.")


# --------------------------------------------------------------------------
# Segment scoring: angle -> base segment score. These angle bands are the
# published RULA thresholds and match the anchors already used in pss_v2.
# --------------------------------------------------------------------------

def upper_arm_score(flexion_deg: float, shoulder_raised: bool = False,
                    abducted: bool = False, supported_or_leaning: bool = False) -> int:
    """Shoulder flexion(+)/extension(-). Extension is negative flexion."""
    a = flexion_deg
    if -20 <= a <= 20:
        s = 1
    elif a < -20 or (20 < a <= 45):
        s = 2
    elif 45 < a <= 90:
        s = 3
    else:  # > 90
        s = 4
    if shoulder_raised:
        s += 1
    if abducted:
        s += 1
    if supported_or_leaning:
        s -= 1
    return max(1, min(6, s))


def lower_arm_score(flexion_deg: float, across_midline_or_out: bool = False) -> int:
    """Elbow flexion. 60-100 deg is the low-risk band."""
    s = 1 if (60 <= flexion_deg <= 100) else 2
    if across_midline_or_out:
        s += 1
    return max(1, min(3, s))


# Genuine extension starts beyond this deadband; small negative flexion is
# calibration jitter around neutral, not extension.
# KEEP IN SYNC with pss_v2.PSSv2Config.NECK_EXTENSION_DEADBAND_DEG.
EXTENSION_DEADBAND_DEG = 5.0


def neck_score(flexion_deg: float, twisted: bool = False,
               side_bending: bool = False) -> int:
    """Neck flexion(+)/extension(-). Extension needs to exceed the deadband;
    within it, negative values score as upright (1)."""
    a = flexion_deg
    if a < -EXTENSION_DEADBAND_DEG:
        s = 4  # genuine extension
    elif a <= 10:
        s = 1
    elif 10 < a <= 20:
        s = 2
    else:  # > 20
        s = 3
    if twisted:
        s += 1
    if side_bending:
        s += 1
    return max(1, min(6, s))


def trunk_score(flexion_deg: float, twisted: bool = False,
                side_bending: bool = False) -> int:
    """Trunk flexion. 0 deg assumes a supported, upright, seated/standing trunk."""
    a = abs(flexion_deg)
    if a <= 5:
        s = 1
    elif a <= 20:
        s = 2
    elif a <= 60:
        s = 3
    else:
        s = 4
    if twisted:
        s += 1
    if side_bending:
        s += 1
    return max(1, min(6, s))


# --------------------------------------------------------------------------
# Combination tables. Table C verified; Table A/B flagged.
# --------------------------------------------------------------------------

# Table C: rows = Score A (Wrist&Arm, 1..8), cols = Score B (Neck/Trunk/Leg, 1..7).
_TABLE_C = [
    #  B1 B2 B3 B4 B5 B6 B7
    [1, 2, 3, 3, 4, 5, 5],   # A1
    [2, 2, 3, 4, 4, 5, 5],   # A2
    [3, 3, 3, 4, 4, 5, 6],   # A3
    [3, 3, 3, 4, 5, 6, 6],   # A4
    [4, 4, 4, 5, 6, 7, 7],   # A5
    [4, 4, 5, 6, 6, 7, 7],   # A6
    [5, 5, 6, 6, 7, 7, 7],   # A7
    [5, 5, 6, 7, 7, 7, 7],   # A8+
]

# Table A: [upper_arm 1..6][lower_arm 1..3] -> 8 cols
# ordered (wrist,twist): W1T1 W1T2 W2T1 W2T2 W3T1 W3T2 W4T1 W4T2
_TABLE_A = {
    (1, 1): [1, 2, 2, 2, 2, 3, 3, 3], (1, 2): [2, 2, 2, 2, 3, 3, 3, 3], (1, 3): [2, 3, 3, 3, 3, 3, 4, 4],
    (2, 1): [2, 3, 3, 3, 3, 4, 4, 4], (2, 2): [3, 3, 3, 3, 3, 4, 4, 4], (2, 3): [3, 4, 4, 4, 4, 4, 5, 5],
    (3, 1): [3, 3, 4, 4, 4, 4, 5, 5], (3, 2): [3, 4, 4, 4, 4, 4, 5, 5], (3, 3): [4, 4, 4, 4, 4, 5, 5, 5],
    (4, 1): [4, 4, 4, 4, 4, 5, 5, 5], (4, 2): [4, 4, 4, 4, 4, 5, 5, 5], (4, 3): [4, 4, 4, 5, 5, 5, 6, 6],
    (5, 1): [5, 5, 5, 5, 5, 6, 6, 7], (5, 2): [5, 6, 6, 6, 6, 7, 7, 7], (5, 3): [6, 6, 6, 7, 7, 7, 7, 8],
    (6, 1): [7, 7, 7, 7, 7, 8, 8, 9], (6, 2): [8, 8, 8, 8, 8, 9, 9, 9], (6, 3): [9, 9, 9, 9, 9, 9, 9, 9],
}

# Table B: [neck 1..6] -> 12 cols ordered (trunk,legs):
# T1L1 T1L2 T2L1 T2L2 T3L1 T3L2 T4L1 T4L2 T5L1 T5L2 T6L1 T6L2
_TABLE_B = {
    1: [1, 3, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7],
    2: [2, 3, 2, 3, 4, 5, 5, 5, 6, 7, 7, 7],
    3: [3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7],
    4: [5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 8, 8],
    5: [7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8],
    6: [8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9],
}


def set_tables(table_a: Optional[dict] = None, table_b: Optional[dict] = None,
               table_c: Optional[list] = None, verified: bool = True) -> None:
    """Inject validated grids (e.g. transcribed from a printed worksheet)."""
    global _TABLE_A, _TABLE_B, _TABLE_C, TABLES_VERIFIED
    if table_a is not None:
        _TABLE_A = table_a
    if table_b is not None:
        _TABLE_B = table_b
    if table_c is not None:
        _TABLE_C = table_c
    TABLES_VERIFIED = verified


def _score_a(upper_arm: int, lower_arm: int, wrist: int, wrist_twist: int) -> int:
    col = (wrist - 1) * 2 + (wrist_twist - 1)
    return _TABLE_A[(min(6, upper_arm), min(3, lower_arm))][col]


def _score_b(neck: int, trunk: int, legs: int) -> int:
    col = (trunk - 1) * 2 + (legs - 1)
    return _TABLE_B[min(6, neck)][col]


def _table_c(score_a: int, score_b: int) -> int:
    r = min(8, score_a) - 1
    c = min(7, score_b) - 1
    return _TABLE_C[r][c]


# --------------------------------------------------------------------------
# Assumptions for unmeasurable inputs (documented, overrideable).
# --------------------------------------------------------------------------

@dataclass
class RulaAssumptions:
    # Unmeasurable from the camera rig. Documented bias direction in comments.
    wrist_score: int = 2         # 0-15 deg band. Neutral-ish. UNDER-estimates fine work.
    wrist_twist_score: int = 1   # mainly mid-range. UNDER-estimates twisted tool grips.
    leg_score: int = 1           # seated/standing, supported and balanced.
    # Task-level modifiers. Set to match how the expert coded the same clip so the
    # comparison is fair; defaults suit light tools held with brief static holds.
    muscle_use_a: int = 0
    force_a: int = 0
    muscle_use_b: int = 0
    force_b: int = 0

    def note(self) -> str:
        return (f"assumed wrist={self.wrist_score}, twist={self.wrist_twist_score}, "
                f"legs={self.leg_score}, muscle(A/B)={self.muscle_use_a}/{self.muscle_use_b}, "
                f"force(A/B)={self.force_a}/{self.force_b} "
                f"[wrist/twist/legs not measurable from rig]")


DEFAULT_ASSUMPTIONS = RulaAssumptions()


def rula_from_angles(angles, assumptions: RulaAssumptions = DEFAULT_ASSUMPTIONS,
                     sidebend_deg_thresh: float = 8.0,
                     twist_deg_thresh: float = 10.0) -> dict:
    """
    Compute RULA from a PostureAngles-like object (attributes used, all optional):
      trunk_flexion_deg, neck_flexion_deg, upper_arm_flexion_deg, lower_arm_flexion_deg,
      trunk_sidebend_deg, neck_sidebend_deg, trunk_twist_deg, neck_twist_deg
    Returns sub-scores and the grand score, plus a flag list of assumed inputs.
    """
    def g(name, default=0.0):
        return getattr(angles, name, default) or 0.0

    ua = upper_arm_score(g("upper_arm_flexion_deg"),
                         supported_or_leaning=False)
    la = lower_arm_score(g("lower_arm_flexion_deg"))
    nk = neck_score(g("neck_flexion_deg"),
                    twisted=abs(g("neck_twist_deg")) > twist_deg_thresh,
                    side_bending=abs(g("neck_sidebend_deg")) > sidebend_deg_thresh)
    tr = trunk_score(g("trunk_flexion_deg"),
                     twisted=abs(g("trunk_twist_deg")) > twist_deg_thresh,
                     side_bending=abs(g("trunk_sidebend_deg")) > sidebend_deg_thresh)

    posture_a = _score_a(ua, la, assumptions.wrist_score, assumptions.wrist_twist_score)
    posture_b = _score_b(nk, tr, assumptions.leg_score)

    score_a = posture_a + assumptions.muscle_use_a + assumptions.force_a
    score_b = posture_b + assumptions.muscle_use_b + assumptions.force_b
    grand = _table_c(score_a, score_b)

    return {
        "upper_arm": ua, "lower_arm": la, "neck": nk, "trunk": tr,
        "posture_a": posture_a, "posture_b": posture_b,
        "score_a": score_a, "score_b": score_b,
        "grand": grand,
        "assumed": ["wrist", "wrist_twist", "legs"],
        "tables_verified": TABLES_VERIFIED,
    }


# --------------------------------------------------------------------------
# Self-test: structural checks + the verified Table C anchors. Does not assert
# unverified Table A/B corner cells; asserts the slice actually used is sane.
# --------------------------------------------------------------------------

def verify_tables() -> None:
    assert len(_TABLE_C) == 8 and all(len(r) == 7 for r in _TABLE_C), "Table C shape"
    assert _TABLE_C[0][0] == 1, "Table C A1,B1 should be 1"
    assert _TABLE_C[7][6] == 7, "Table C A8,B7 should be 7"
    assert _TABLE_C[4][5] == 7, "Table C A5,B6 should be 7"
    assert len(_TABLE_A) == 18 and all(len(v) == 8 for v in _TABLE_A.values()), "Table A shape"
    assert len(_TABLE_B) == 6 and all(len(v) == 12 for v in _TABLE_B.values()), "Table B shape"


if __name__ == "__main__":
    from dataclasses import dataclass as _dc

    @_dc
    class A:
        trunk_flexion_deg: float = 0.0
        neck_flexion_deg: float = 0.0
        upper_arm_flexion_deg: float = 0.0
        lower_arm_flexion_deg: float = 80.0
        trunk_sidebend_deg: float = 0.0
        neck_sidebend_deg: float = 0.0
        trunk_twist_deg: float = 0.0
        neck_twist_deg: float = 0.0

    verify_tables()
    print("table shapes + Table C anchors OK")
    print(f"TABLES_VERIFIED = {TABLES_VERIFIED} (Table A/B need worksheet check)\n")

    neutral = rula_from_angles(A())
    print(f"neutral posture: segment(ua,la,nk,tr)="
          f"({neutral['upper_arm']},{neutral['lower_arm']},{neutral['neck']},{neutral['trunk']}) "
          f"A={neutral['score_a']} B={neutral['score_b']} grand={neutral['grand']}")
    assert neutral["grand"] <= 2, "neutral should be acceptable (1-2)"

    severe = rula_from_angles(A(trunk_flexion_deg=65, neck_flexion_deg=25,
                                upper_arm_flexion_deg=95, lower_arm_flexion_deg=50,
                                trunk_twist_deg=20, neck_sidebend_deg=15))
    print(f"severe posture:  segment(ua,la,nk,tr)="
          f"({severe['upper_arm']},{severe['lower_arm']},{severe['neck']},{severe['trunk']}) "
          f"A={severe['score_a']} B={severe['score_b']} grand={severe['grand']}")
    assert severe["grand"] >= 6, "severe should demand change (6-7)"

    # monotonicity in trunk flexion
    grands = [rula_from_angles(A(trunk_flexion_deg=t))["grand"] for t in (0, 15, 40, 70)]
    print(f"grand vs trunk flexion [0,15,40,70]: {grands}")
    assert grands == sorted(grands), "grand should be non-decreasing in trunk flexion"
    print(f"\nassumption note: {DEFAULT_ASSUMPTIONS.note()}")
    print("rula.py self-test passed")
