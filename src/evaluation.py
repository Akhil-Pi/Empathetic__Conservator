"""
evaluation.py
=============
Part 5, component 2: the evaluation pipeline.

Runs on files produced by session_logger_v2 (frames + events + meta). Four analyses:

  A. RULA validation       automated RULA vs expert RULA per session:
                           Bland-Altman (bias, limits of agreement), quadratic-
                           weighted Cohen's kappa, Spearman(auto, expert), and
                           Spearman(PSS_v2, expert). This is the validation the
                           original paper promised but never ran.

  B. H1 test               PSS lower in experimental than control, per participant.
                           Shapiro-Wilk on paired differences FIRST (the original
                           skipped this), then paired t (if normal) AND Wilcoxon,
                           with effect sizes. Honest about N.

  C. Paradox analysis      why H1 can be null even though interventions work:
                           unsupervised clustering of strain into posture modes,
                           then per-mode achievable PSS reduction using the actual
                           controller optimizer. Arm-dominated strain is not
                           reducible by moving the artifact, so it is residual the
                           cobot cannot fix. Quantifies that residual.

  D. Latency vs recovery   Spearman(intervention latency, recovery time). Recovery
                           is DERIVED here from event timestamps + the PSS series,
                           so there is a single source of truth (the paper's
                           headline result, recomputed on clean data).

All analyses are keyed to the logger v2 columns. Run main() to produce a report
and figures. Import the individual functions to run them piecewise.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pss_v2 import PSSv2Calculator, PostureAngles
from goal_controller import optimize_target, predict_angles, _score_once, ControllerConfig
from rula import rula_from_angles, RulaAssumptions, TABLES_VERIFIED, RULA_SOURCE


THRESHOLD = ControllerConfig.THRESHOLD          # 0.25
HYSTERESIS = ControllerConfig.HYSTERESIS        # 0.08
RECOVERY_TARGET = THRESHOLD - HYSTERESIS         # 0.17


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _canon_participant(raw: str) -> str:
    """Normalize inconsistent IDs (P001, P02, P6) to canonical P1..Pn."""
    digits = re.sub(r"[^0-9]", "", raw)
    return f"P{int(digits)}" if digits else raw


@dataclass
class Session:
    participant: str
    condition: str
    frames: pd.DataFrame
    events: pd.DataFrame
    meta: dict


def load_sessions(sessions_dir: str) -> list[Session]:
    """Load every *_frames.csv and pair it with its _events.csv and _meta.txt."""
    sessions: list[Session] = []
    for fpath in sorted(glob.glob(os.path.join(sessions_dir, "*_frames.csv"))):
        base = fpath[: -len("_frames.csv")]
        fname = os.path.basename(base)
        m = re.match(r"(P[0-9]+)_(control|experimental)_", fname)
        if not m:
            continue
        participant = _canon_participant(m.group(1))
        condition = m.group(2)
        frames = pd.read_csv(fpath)
        epath = base + "_events.csv"
        events = pd.read_csv(epath) if os.path.exists(epath) else pd.DataFrame()
        mpath = base + "_meta.txt"
        meta = _parse_meta(mpath) if os.path.exists(mpath) else {}
        sessions.append(Session(participant, condition, frames, events, meta))
    return sessions


def _parse_meta(path: str) -> dict:
    meta = {}
    for line in open(path):
        if ":" in line and not line.startswith("["):
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta


def _pool_by_participant_condition(sessions: list[Session]) -> dict:
    """Pool frames/events across runs sharing (participant, condition)."""
    pooled: dict = {}
    for s in sessions:
        key = (s.participant, s.condition)
        if key not in pooled:
            pooled[key] = {"frames": [], "events": []}
        pooled[key]["frames"].append(s.frames)
        if not s.events.empty:
            pooled[key]["events"].append(s.events)
    out = {}
    for key, d in pooled.items():
        out[key] = {
            "frames": pd.concat(d["frames"], ignore_index=True) if d["frames"] else pd.DataFrame(),
            "events": pd.concat(d["events"], ignore_index=True) if d["events"] else pd.DataFrame(),
        }
    return out


def _row_to_angles(row) -> PostureAngles:
    def gv(c):
        v = row.get(c, 0.0)
        return float(v) if pd.notna(v) else 0.0
    return PostureAngles(
        trunk_flexion_deg=gv("trunk_flexion_deg"),
        neck_flexion_deg=gv("neck_flexion_deg"),
        upper_arm_flexion_deg=gv("upper_arm_flexion_deg"),
        lower_arm_flexion_deg=gv("lower_arm_flexion_deg"),
        trunk_sidebend_deg=gv("trunk_sidebend_deg"),
        neck_sidebend_deg=gv("neck_sidebend_deg"),
        trunk_twist_deg=gv("trunk_twist_deg"),
        neck_twist_deg=gv("neck_twist_deg"),
    )


# --------------------------------------------------------------------------
# A. RULA validation
# --------------------------------------------------------------------------

def _weighted_kappa(a: np.ndarray, b: np.ndarray, max_score: int = 7) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal RULA scores."""
    a = np.asarray(a, int); b = np.asarray(b, int)
    cats = list(range(1, max_score + 1))
    k = len(cats)
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x - 1, y - 1] += 1
    if O.sum() == 0:
        return float("nan")
    W = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            W[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)
    row = O.sum(1); col = O.sum(0); n = O.sum()
    E = np.outer(row, col) / n
    num = (W * O).sum(); den = (W * E).sum()
    return 1.0 - num / den if den > 0 else float("nan")


def validate_rula(sessions: list[Session], expert_df: pd.DataFrame,
                  assumptions: RulaAssumptions = RulaAssumptions(),
                  aggregate: str = "p90") -> dict:
    """
    Compare automated RULA to expert RULA per (participant, condition).
    expert_df columns: participant, condition, expert_rula[, expert_score_b].
    aggregate: how to summarize per-frame automated RULA to a session score
               ('p90' = worst-sustained proxy, 'median', 'max').
    """
    pooled = _pool_by_participant_condition(sessions)
    agg = {"p90": lambda x: np.percentile(x, 90), "median": np.median, "max": np.max}[aggregate]

    rows = []
    for (participant, condition), d in pooled.items():
        fr = d["frames"]
        if fr.empty:
            continue
        grands, score_bs, pss = [], [], []
        for r in fr.to_dict("records"):
            res = rula_from_angles(_row_to_angles(r), assumptions)
            grands.append(res["grand"]); score_bs.append(res["score_b"])
            pss.append(float(r.get("pss_smooth", np.nan)))
        rows.append({
            "participant": participant, "condition": condition,
            "auto_grand": int(round(agg(grands))),
            "auto_score_b": int(round(agg(score_bs))),
            "pss_summary": float(np.nanpercentile(pss, 90)),
        })
    auto_df = pd.DataFrame(rows)
    merged = auto_df.merge(expert_df, on=["participant", "condition"], how="inner")
    if merged.empty:
        return {"error": "no overlap between sessions and expert scores"}

    diff = merged["auto_grand"].to_numpy() - merged["expert_rula"].to_numpy()
    bias = float(np.mean(diff)); sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    loa = (bias - 1.96 * sd, bias + 1.96 * sd)
    kappa = _weighted_kappa(merged["auto_grand"], merged["expert_rula"])
    rho_ag, p_ag = stats.spearmanr(merged["auto_grand"], merged["expert_rula"])
    rho_ps, p_ps = stats.spearmanr(merged["pss_summary"], merged["expert_rula"])

    return {
        "n_pairs": int(len(merged)),
        "bland_altman": {"bias": bias, "sd": sd, "loa_lower": loa[0], "loa_upper": loa[1]},
        "weighted_kappa": float(kappa),
        "spearman_auto_vs_expert": {"rho": float(rho_ag), "p": float(p_ag)},
        "spearman_pss_vs_expert": {"rho": float(rho_ps), "p": float(p_ps)},
        "tables_verified": TABLES_VERIFIED,
        "assumptions": assumptions.note(),
        "merged": merged,
    }


def plot_bland_altman(result: dict, out_path: str) -> None:
    m = result["merged"]; ba = result["bland_altman"]
    mean = (m["auto_grand"] + m["expert_rula"]) / 2
    diff = m["auto_grand"] - m["expert_rula"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(mean, diff, s=42, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax.axhline(ba["bias"], color="C0", label=f"bias {ba['bias']:+.2f}")
    ax.axhline(ba["loa_upper"], color="C3", ls="--", label=f"+1.96 SD {ba['loa_upper']:+.2f}")
    ax.axhline(ba["loa_lower"], color="C3", ls="--", label=f"-1.96 SD {ba['loa_lower']:+.2f}")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("mean of automated and expert RULA")
    ax.set_ylabel("automated - expert")
    ax.set_title("Bland-Altman: automated vs expert RULA grand score")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------
# B. H1 test
# --------------------------------------------------------------------------

def _session_pss_summary(frames: pd.DataFrame) -> dict:
    pss = frames["pss_smooth"].to_numpy(dtype=float)
    pss = pss[~np.isnan(pss)]
    return {
        "mean_pss": float(np.mean(pss)) if len(pss) else np.nan,
        "frac_above_threshold": float(np.mean(pss > THRESHOLD)) if len(pss) else np.nan,
    }


def test_h1(sessions: list[Session]) -> dict:
    pooled = _pool_by_participant_condition(sessions)
    per_p: dict = {}
    for (participant, condition), d in pooled.items():
        if d["frames"].empty:
            continue
        per_p.setdefault(participant, {})[condition] = _session_pss_summary(d["frames"])

    paired = [(p, v["control"], v["experimental"]) for p, v in per_p.items()
              if "control" in v and "experimental" in v]
    paired.sort()
    if len(paired) < 3:
        return {"error": f"only {len(paired)} complete pairs; need >= 3"}

    ctrl = np.array([c["mean_pss"] for _, c, _ in paired])
    exp = np.array([e["mean_pss"] for _, _, e in paired])
    diff = exp - ctrl  # H1 predicts negative (experimental lower)

    # Secondary endpoint: fraction of time above the intervention threshold.
    # Mean PSS is diluted by strain the cobot cannot address (arm-dominated
    # baseline); time-above-threshold is closer to what interventions target.
    ctrl_f = np.array([c["frac_above_threshold"] for _, c, _ in paired])
    exp_f = np.array([e["frac_above_threshold"] for _, _, e in paired])
    try:
        wf_stat, wf_p = stats.wilcoxon(exp_f, ctrl_f)
    except ValueError:
        wf_stat, wf_p = float("nan"), float("nan")
    secondary = {
        "frac_above_control": float(np.mean(ctrl_f)),
        "frac_above_experimental": float(np.mean(exp_f)),
        "wilcoxon_p": float(wf_p),
        "rank_biserial_r": _wilcoxon_rank_biserial(exp_f, ctrl_f),
    }

    sh_w, sh_p = stats.shapiro(diff)
    normal = sh_p > 0.05
    # paired t (report regardless; valid if normal)
    t_stat, t_p = stats.ttest_rel(exp, ctrl)
    dz = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0
    # Wilcoxon (nonparametric fallback / always reported)
    try:
        w_stat, w_p = stats.wilcoxon(exp, ctrl)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    n = len(diff)
    rank_biserial = _wilcoxon_rank_biserial(exp, ctrl)

    return {
        "n_pairs": n,
        "mean_pss_control": float(np.mean(ctrl)),
        "mean_pss_experimental": float(np.mean(exp)),
        "mean_diff": float(np.mean(diff)),
        "direction": "experimental lower" if np.mean(diff) < 0 else "experimental higher",
        "shapiro": {"W": float(sh_w), "p": float(sh_p), "normal": bool(normal)},
        "paired_t": {"t": float(t_stat), "p": float(t_p), "cohens_dz": dz,
                     "valid": bool(normal)},
        "wilcoxon": {"W": float(w_stat), "p": float(w_p), "rank_biserial_r": rank_biserial},
        "secondary_frac_above": secondary,
        "supported": bool((np.mean(diff) < 0) and
                          ((t_p < 0.05 and normal) or (w_p < 0.05))),
        "power_note": f"N={n} is a small pilot; a null result is inconclusive, not "
                      f"evidence of no effect.",
    }


def _wilcoxon_rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a) - np.asarray(b)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    r_plus = ranks[d > 0].sum(); r_minus = ranks[d < 0].sum()
    total = r_plus + r_minus
    return float((r_plus - r_minus) / total) if total > 0 else 0.0


# --------------------------------------------------------------------------
# C. Paradox analysis
# --------------------------------------------------------------------------

_BAND_COLS = ["trunk_band", "neck_band", "upper_arm_band", "lower_arm_band"]
_BAND_TO_SEGMENT = {"trunk_band": "trunk", "neck_band": "neck",
                    "upper_arm_band": "upper_arm", "lower_arm_band": "lower_arm"}
_REDUCIBLE_SEGMENTS = {"trunk", "neck"}  # artifact repositioning can help these


def analyze_paradox(sessions: list[Session]) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    pooled = _pool_by_participant_condition(sessions)
    exp_frames = [d["frames"] for (p, c), d in pooled.items()
                  if c == "experimental" and not d["frames"].empty]
    if not exp_frames:
        return {"error": "no experimental frames"}
    allf = pd.concat(exp_frames, ignore_index=True)

    hi = allf[allf["pss_raw"].astype(float) > THRESHOLD].copy()
    if len(hi) < 20:
        return {"error": f"too few high-strain frames ({len(hi)})"}
    X = hi[_BAND_COLS].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)

    best = None
    for k in range(2, 6):
        if len(hi) <= k:
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
        try:
            sil = silhouette_score(Xs, km.labels_)
        except Exception:
            sil = -1
        if best is None or sil > best["sil"]:
            best = {"k": k, "sil": sil, "labels": km.labels_}
    labels = best["labels"]
    hi = hi.assign(_cluster=labels)

    pss_calc = PSSv2Calculator()
    cfg = ControllerConfig()
    clusters = []
    for cid in sorted(np.unique(labels)):
        sub = hi[hi["_cluster"] == cid]
        mean_bands = sub[_BAND_COLS].mean()
        dominant = _BAND_TO_SEGMENT[mean_bands.idxmax()]
        # achievable PSS reduction: run the real controller optimizer on the
        # mode's mean posture.
        centroid = _row_to_angles(sub[
            ["trunk_flexion_deg", "neck_flexion_deg", "upper_arm_flexion_deg",
             "lower_arm_flexion_deg", "trunk_sidebend_deg", "neck_sidebend_deg",
             "trunk_twist_deg", "neck_twist_deg"]].mean())
        pss_before = _score_once(centroid, pss_calc)
        delta, _, _ = optimize_target(centroid, pss_calc, cfg)
        pss_after = _score_once(predict_angles(centroid, delta, cfg), pss_calc)
        reduction = pss_before - pss_after
        clusters.append({
            "cluster": int(cid),
            "frac_of_highstrain": float(len(sub) / len(hi)),
            "dominant_segment": dominant,
            "reducible_by_repositioning": dominant in _REDUCIBLE_SEGMENTS,
            "mean_pss": float(sub["pss_raw"].astype(float).mean()),
            "achievable_pss_reduction": float(reduction),
        })

    residual_frac = sum(c["frac_of_highstrain"] for c in clusters
                        if not c["reducible_by_repositioning"])
    restrain = _restrain_rate(pooled)

    return {
        "k_selected": best["k"], "silhouette": float(best["sil"]),
        "n_highstrain_frames": int(len(hi)),
        "clusters": clusters,
        "residual_unfixable_fraction": float(residual_frac),
        "restrain_rate_per_min": restrain,
        "interpretation": (
            f"{residual_frac*100:.0f}% of high-strain time is dominated by arm "
            f"posture, which repositioning the artifact cannot relieve. This is the "
            f"residual load that keeps mean PSS from dropping, which explains a null H1 "
            f"even when interventions reduce trunk/neck strain."),
    }


def _restrain_rate(pooled: dict) -> dict:
    """Threshold up-crossings per minute, control vs experimental."""
    out = {}
    for cond in ("control", "experimental"):
        frames = [d["frames"] for (p, c), d in pooled.items()
                  if c == cond and not d["frames"].empty]
        if not frames:
            out[cond] = float("nan"); continue
        rates = []
        for fr in frames:
            fr = fr.sort_values("timestamp_s")
            pss = fr["pss_smooth"].to_numpy(dtype=float)
            t = fr["timestamp_s"].to_numpy(dtype=float)
            if len(pss) < 2 or (t[-1] - t[0]) <= 0:
                continue
            up = 0; armed = True
            for v in pss:
                if armed and v > THRESHOLD:
                    up += 1; armed = False
                elif not armed and v < RECOVERY_TARGET:
                    armed = True
            rates.append(up / ((t[-1] - t[0]) / 60.0))
        out[cond] = float(np.mean(rates)) if rates else float("nan")
    return out


# --------------------------------------------------------------------------
# D. Latency vs recovery (recovery derived here = single source of truth)
# --------------------------------------------------------------------------

def _recovery_after(frames: pd.DataFrame, t0: float, dwell_s: float = 0.5,
                    horizon_s: float = 20.0) -> float:
    """
    Time from t0 until PSS first drops below the recovery target AND stays
    below it for dwell_s (so a single dipping frame does not count). If the
    data ends while still below target before dwell_s elapses, that counts as
    recovered. Returns NaN if never recovered within the horizon (censored).
    """
    fr = frames[(frames["timestamp_s"] > t0) &
                (frames["timestamp_s"] <= t0 + horizon_s)].sort_values("timestamp_s")
    if fr.empty:
        return float("nan")
    t = fr["timestamp_s"].to_numpy(dtype=float)
    pss = fr["pss_smooth"].to_numpy(dtype=float)
    i = 0
    while i < len(t):
        if pss[i] >= RECOVERY_TARGET:
            i += 1
            continue
        # candidate recovery at i: scan forward until dwell confirmed or broken
        j = i
        while j < len(t) and (t[j] - t[i]) < dwell_s:
            if pss[j] >= RECOVERY_TARGET:
                break
            j += 1
        else:
            return float(t[i] - t0)  # dwell held (or data ended while below)
        i = j  # dwell broken at j; resume scanning from there
    return float("nan")  # censored: never recovered within horizon


def latency_vs_recovery(sessions: list[Session]) -> dict:
    pooled = _pool_by_participant_condition(sessions)
    lat, rec = [], []
    censored = 0
    for (participant, condition), d in pooled.items():
        if condition != "experimental" or d["events"].empty or d["frames"].empty:
            continue
        ev = d["events"]
        ev = ev[ev["event_type"] == "intervention"] if "event_type" in ev else ev
        for _, e in ev.iterrows():
            t0 = float(e["timestamp_s"])
            latency = e.get("latency_s", np.nan)
            latency = float(latency) if pd.notna(latency) else np.nan
            r = _recovery_after(d["frames"], t0)
            if np.isnan(r):
                censored += 1; continue
            if np.isnan(latency):
                continue
            lat.append(latency); rec.append(r)
    if len(lat) < 3:
        return {"error": f"only {len(lat)} usable intervention pairs", "censored": censored}
    rho, p = stats.spearmanr(lat, rec)
    return {
        "n_interventions": len(lat), "censored": censored,
        "spearman": {"rho": float(rho), "p": float(p)},
        "latency_s": {"mean": float(np.mean(lat)), "median": float(np.median(lat))},
        "recovery_s": {"mean": float(np.mean(rec)), "median": float(np.median(rec))},
        "_lat": lat, "_rec": rec,
    }


def plot_latency_recovery(result: dict, out_path: str) -> None:
    lat = result["_lat"]; rec = result["_rec"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(lat, rec, s=40, alpha=0.75, edgecolor="k", linewidth=0.5)
    ax.set_xlabel("intervention latency (s)")
    ax.set_ylabel("recovery time (s)")
    sp = result["spearman"]
    ax.set_title(f"Latency vs recovery  (Spearman rho={sp['rho']:.2f}, p={sp['p']:.3g}, "
                 f"n={result['n_interventions']})")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------
# E. Response-gain fit (refines ControllerConfig.GAINS from data)
# --------------------------------------------------------------------------

_GAIN_ANGLES = [
    "trunk_flexion_deg", "neck_flexion_deg",
    "trunk_sidebend_deg", "neck_sidebend_deg",
    "trunk_twist_deg", "neck_twist_deg",
]
_DOF_COLS = ["dz", "dy", "dtilt", "drot", "dx"]


def fit_response_gains(sessions: list[Session], pre_s: float = 2.0,
                       post_start_s: float = 0.5, post_s: float = 2.5,
                       min_events: int = 8) -> dict:
    """
    Estimate the actual human-response gains from logged interventions:
    regress the observed angle reduction (pre-window mean minus post-window
    mean) on the applied DOF delta, per scored angle, using only the couplings
    ControllerConfig.GAINS declares. This is the data-driven refinement of the
    analytical gains the controller ships with.

    Interventions whose windows overlap a neighbouring intervention are
    dropped (contaminated). Angles whose delta columns show too little
    variation to identify a slope are reported as unfittable rather than
    fitted to noise.
    """
    rows = []
    pooled = _pool_by_participant_condition(sessions)
    for (participant, condition), d in pooled.items():
        if condition != "experimental" or d["events"].empty or d["frames"].empty:
            continue
        ev = d["events"]
        ev = ev[ev["event_type"] == "intervention"] if "event_type" in ev else ev
        ev = ev.sort_values("timestamp_s").reset_index(drop=True)
        fr = d["frames"].sort_values("timestamp_s")
        times = ev["timestamp_s"].astype(float).to_numpy()
        for i in range(len(ev)):
            e = ev.iloc[i]
            t0 = float(e["timestamp_s"])
            # contamination guard: neighbouring interventions inside the windows
            if i > 0 and (t0 - times[i - 1]) < (pre_s + post_start_s + post_s):
                continue
            if i < len(times) - 1 and (times[i + 1] - t0) < (post_start_s + post_s):
                continue
            pre = fr[(fr["timestamp_s"] >= t0 - pre_s) & (fr["timestamp_s"] < t0 - 0.1)]
            post = fr[(fr["timestamp_s"] >= t0 + post_start_s) &
                      (fr["timestamp_s"] <= t0 + post_start_s + post_s)]
            if len(pre) < 3 or len(post) < 3:
                continue
            row, ok = {}, True
            for dof in _DOF_COLS:
                try:
                    row[dof] = float(e.get(dof))
                except (TypeError, ValueError):
                    ok = False
                    break
                if np.isnan(row[dof]):
                    ok = False
                    break
            if not ok:
                continue
            for ang in _GAIN_ANGLES:
                if "sidebend" in ang or "twist" in ang:
                    # lateral quantities are signed; the response model works
                    # in magnitudes
                    row[f"red_{ang}"] = float(pre[ang].abs().mean() - post[ang].abs().mean())
                else:
                    row[f"red_{ang}"] = float(pre[ang].mean() - post[ang].mean())
            rows.append(row)

    if len(rows) < min_events:
        return {"error": f"only {len(rows)} uncontaminated interventions; "
                         f"need >= {min_events}"}

    df = pd.DataFrame(rows)
    out = {"n_events_used": int(len(df)), "per_angle": {}}
    for ang in _GAIN_ANGLES:
        couplings = ControllerConfig.GAINS.get(ang, {})
        if not couplings:
            continue
        dofs = list(couplings.keys())
        X = df[dofs].to_numpy(dtype=float)
        y = df[f"red_{ang}"].to_numpy(dtype=float)
        # identifiability guard: a DOF column with (near-)zero variance cannot
        # be fitted; fitting it anyway returns an arbitrary large slope.
        stds = X.std(axis=0)
        if np.any(stds < 1e-4) or np.linalg.matrix_rank(X - X.mean(0)) < len(dofs):
            out["per_angle"][ang] = {
                "fitted": None, "config": couplings, "n": int(len(y)),
                "note": "insufficient delta variation to identify gains"}
            continue
        Xc = np.column_stack([X, np.ones(len(X))])   # intercept absorbs drift
        coef, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        pred = Xc @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        out["per_angle"][ang] = {
            "fitted": {dof: round(float(c), 1) for dof, c in zip(dofs, coef[:-1])},
            "config": couplings,
            "r2": round(r2, 3),
            "n": int(len(y)),
        }
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(sessions_dir: str, expert_csv: str = "", out_dir: str = "eval_out") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    sessions = load_sessions(sessions_dir)
    report = [f"# Empathetic Conservator - Part 5 evaluation",
              f"sessions loaded: {len(sessions)}",
              f"RULA source: {RULA_SOURCE}",
              f"RULA tables verified: {TABLES_VERIFIED}", ""]

    results = {}

    # B. H1
    h1 = test_h1(sessions); results["h1"] = h1
    report.append("## B. H1 (PSS lower in experimental)")
    if "error" in h1:
        report.append(f"skipped: {h1['error']}")
    else:
        report += [
            f"- pairs: {h1['n_pairs']}",
            f"- mean PSS control={h1['mean_pss_control']:.3f}, "
            f"experimental={h1['mean_pss_experimental']:.3f} ({h1['direction']})",
            f"- Shapiro-Wilk on diffs: W={h1['shapiro']['W']:.3f} p={h1['shapiro']['p']:.3f} "
            f"({'normal' if h1['shapiro']['normal'] else 'non-normal'})",
            f"- paired t: t={h1['paired_t']['t']:.3f} p={h1['paired_t']['p']:.3f} "
            f"dz={h1['paired_t']['cohens_dz']:.3f} (valid={h1['paired_t']['valid']})",
            f"- Wilcoxon: W={h1['wilcoxon']['W']:.1f} p={h1['wilcoxon']['p']:.3f} "
            f"r={h1['wilcoxon']['rank_biserial_r']:.3f}",
            f"- secondary (frac of time above threshold): "
            f"control={h1['secondary_frac_above']['frac_above_control']:.3f}, "
            f"experimental={h1['secondary_frac_above']['frac_above_experimental']:.3f}, "
            f"Wilcoxon p={h1['secondary_frac_above']['wilcoxon_p']:.3f}, "
            f"r={h1['secondary_frac_above']['rank_biserial_r']:.3f}",
            f"- H1 supported: {h1['supported']}. {h1['power_note']}", ""]

    # C. Paradox
    par = analyze_paradox(sessions); results["paradox"] = par
    report.append("## C. Paradox analysis")
    if "error" in par:
        report.append(f"skipped: {par['error']}")
    else:
        report += [f"- clusters k={par['k_selected']} (silhouette {par['silhouette']:.2f}), "
                   f"{par['n_highstrain_frames']} high-strain frames"]
        for c in par["clusters"]:
            report.append(
                f"  - cluster {c['cluster']}: {c['frac_of_highstrain']*100:.0f}% of high-strain, "
                f"dominated by {c['dominant_segment']}, "
                f"reducible={c['reducible_by_repositioning']}, "
                f"achievable PSS reduction={c['achievable_pss_reduction']:.3f}")
        report += [f"- residual unfixable fraction: {par['residual_unfixable_fraction']*100:.0f}%",
                   f"- re-strain rate/min: control={par['restrain_rate_per_min'].get('control', float('nan')):.2f}, "
                   f"experimental={par['restrain_rate_per_min'].get('experimental', float('nan')):.2f}",
                   f"- {par['interpretation']}", ""]

    # D. Latency vs recovery
    lr = latency_vs_recovery(sessions); results["latency_recovery"] = lr
    report.append("## D. Latency vs recovery")
    if "error" in lr:
        report.append(f"skipped: {lr['error']}")
    else:
        report += [
            f"- interventions used: {lr['n_interventions']} (censored: {lr['censored']})",
            f"- Spearman rho={lr['spearman']['rho']:.3f} p={lr['spearman']['p']:.4f}",
            f"- latency median={lr['latency_s']['median']:.2f}s, "
            f"recovery median={lr['recovery_s']['median']:.2f}s", ""]
        plot_latency_recovery(lr, os.path.join(out_dir, "latency_recovery.png"))

    # E. Response-gain fit
    gains = fit_response_gains(sessions); results["gain_fit"] = gains
    report.append("## E. Response-gain fit (ControllerConfig.GAINS refinement)")
    if "error" in gains:
        report.append(f"skipped: {gains['error']}")
        report.append("")
    else:
        report.append(f"- uncontaminated interventions used: {gains['n_events_used']}")
        for ang, g in gains["per_angle"].items():
            if g["fitted"] is None:
                report.append(f"  - {ang}: {g['note']} (config {g['config']})")
            else:
                report.append(f"  - {ang}: fitted {g['fitted']} vs config {g['config']} "
                              f"(R2={g['r2']}, n={g['n']})")
        report.append("- update ControllerConfig.GAINS only from fits with "
                      "reasonable R2 and enough delta variation.")
        report.append("")

    # A. RULA validation (needs expert scores)
    report.append("## A. RULA validation")
    if expert_csv and os.path.exists(expert_csv):
        expert_df = pd.read_csv(expert_csv)
        expert_df["participant"] = expert_df["participant"].map(_canon_participant)
        val = validate_rula(sessions, expert_df); results["rula_validation"] = val
        if "error" in val:
            report.append(f"skipped: {val['error']}")
        else:
            ba = val["bland_altman"]
            report += [
                f"- pairs: {val['n_pairs']}   (tables_verified={val['tables_verified']})",
                f"- Bland-Altman bias={ba['bias']:+.2f}, "
                f"LoA [{ba['loa_lower']:+.2f}, {ba['loa_upper']:+.2f}]",
                f"- weighted kappa={val['weighted_kappa']:.3f}",
                f"- Spearman auto vs expert: rho={val['spearman_auto_vs_expert']['rho']:.3f} "
                f"p={val['spearman_auto_vs_expert']['p']:.4f}",
                f"- Spearman PSS vs expert: rho={val['spearman_pss_vs_expert']['rho']:.3f} "
                f"p={val['spearman_pss_vs_expert']['p']:.4f}",
                f"- assumptions: {val['assumptions']}", ""]
            plot_bland_altman(val, os.path.join(out_dir, "bland_altman.png"))
    else:
        report.append("skipped: no expert_csv provided (expert RULA coding happens in August). "
                      "Pipeline is ready; drop in a CSV with columns "
                      "participant,condition,expert_rula.")
        report.append("")

    report_text = "\n".join(report)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(report_text)
    results["report"] = report_text
    return results


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "data/sessions"
    ecsv = sys.argv[2] if len(sys.argv) > 2 else ""
    res = main(d, ecsv)
    print(res["report"])
