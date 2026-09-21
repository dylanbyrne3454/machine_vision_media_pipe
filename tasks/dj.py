"""Drop-jump pipeline (MediaPipe port) — jump height, flight time, contact time.

Reuses the CMJ module's loader, pixel conversion, and scaling. Differs from CMJ
in two ways:
  1. Segmentation is MANUAL (hand-verified rep-start frames), not auto get_reps.
     Because MediaPipe ran on the same videos as OpenPose, the original op segs
     transfer; they are inlined below as OPENPOSE_OP_SEGS (verified with the
     preview tools in dj_segment.py).
  2. Adds flight time and contact time, from toe-velocity peaks (MMC) and force
     thresholds (force plate), mirroring the original dj.py.

Faithful to the original method: each 30fps MMC toe window is resampled to the
1.5s @ 100fps OMC length (150 samples) before the velocity/peak timing, so the
peak thresholds and the /100 in time_from_velocity stay valid.

Flight/contact time are reported for HEIGHT ptm only (as in the original);
gravity returns jump height alone.

    from tasks import dj
    import glob, utils
    dj_bl = sorted(p for p in glob.glob("data/keypoints/dj/*.txt")
                   if "world" not in p.lower() and "UL" not in p.upper())
    dj_ul = sorted(p for p in glob.glob("data/keypoints/dj/*.txt")
                   if "world" not in p.lower() and "UL" in p.upper())
    subjects = utils.read_list('dj_data')

    jh, ft, ct = dj.get_metrics(dj_bl, dj_ul, subjects, ptm='height')   # 3 frames
    jh_g       = dj.get_metrics(dj_bl, dj_ul, subjects, ptm='gravity')  # 1 frame
"""

import os
import numpy as np
import pandas as pd
import pingouin as pg

from scipy.signal import savgol_filter, resample
from sklearn.metrics import mean_absolute_error as MAE
import matplotlib.pyplot as plt
import seaborn as sns

from utilities import utils
from utilities.utils import flip_axis, jump_heights, time_from_velocity, \
    scale_filter, plot_ba
import utilities.PTM as PTM

from tasks import cmj


SUBJECT_IDS = [f"P{i:02d}" for i in range(3, 19)]   # P03..P18, pickle order
OPENPOSE_OP_SEGS = {
    "bl": np.array([
        [356, 668, 1025], [408, 871, 1214], [339, 669, 966], [298, 553, 802],
        [290, 702, 1058], [724, 1296, 1637], [416, 855, 1155], [1699, 1987, 2246],
        [384, 837, 1213], [393, 707, 1118], [473, 788, 1081], [391, 622, 819],
        [504, 1139, 1511], [459, 1227, 1554], [484, 895, 1176], [349, 623, 853],
    ]),
    "ul": np.array([
        [502, 803, 1083], [364, 713, 1035], [870, 1351, 1718], [413, 803, 1076],
        [1095, 1515, 2029], [343, 629, 902], [394, 713, 1056], [399, 991, 1300],
        [443, 816, 1182], [440, 774, 1086], [367, 965, 1514], [417, 667, 922],
        [459, 902, 1211], [540, 1273, 1574], [474, 813, 1124], [162, 409, 665],
    ]),
}
SKIP_SUBJECTS = set()       

ZERO_AFTER = {}         


def _toe_rep_window(toe_y, start, fps=30, win_sec=1.5):
    """Slice a rep window from the toe signal, zeroed to its own min."""
    w = int(win_sec * fps)
    rep = toe_y[start:start + w].copy()
    rep -= rep.min()
    return rep


def _mmc_perf(files, task, ptm, clamp_gravity=True, width=1920, height=1080,
              fps=30):
    ptm_scale = _ptm_scale(files, ptm, clamp_gravity, width, height, fps)

    jh, ct, ft = {}, {}, {}
    h_thr, d_thr = (0.5, 15) if task == "bl" else (0.2, 10)
    for fp in files:
        sid = _sid(fp)
        if sid is None or sid in SKIP_SUBJECTS:
            continue
        toe_y = _toe_signal(fp, width, height)
        starts = OPENPOSE_OP_SEGS[task][SUBJECT_IDS.index(sid)]
        jh[sid], ct[sid], ft[sid] = [], [], []
        for i, s0 in enumerate(starts):
            rep = _toe_rep_window(toe_y, int(s0), fps=fps)
            rep = rep * ptm_scale[sid]                 # px -> mm
            rep = resample(rep, 150)                   # to 1.5s @ 100fps
            rep = np.where(rep > 0, rep, 0)
            key = (task, sid, i)
            if key in ZERO_AFTER:
                rep[ZERO_AFTER[key]:] = 0
            # --- jump height (unchanged; this works) ---
            vel = abs(savgol_filter(np.gradient(rep), 21, 2))[:-10]
            vel, _ = scale_filter(vel)
            times, peaks = time_from_velocity(vel, h_thr, d_thr)
            if len(peaks) >= 2:
                o1, o2 = peaks[-2:]
                lo = max(o1 - 5, 0)
                seg = rep[lo:o2 + 1]
                pk_rel = int(np.argmax(seg))
                jump_peak = seg[pk_rel]
                takeoff = seg[:pk_rel + 1].min() if pk_rel > 0 else seg.min()
                jump_h = np.round((jump_peak - takeoff) / 10, 2)  # mm->cm
            else:
                jump_h = np.nan
            if len(peaks) >= 2:
                takeoff_idx = lo + int(np.argmax(seg[:pk_rel + 1] == takeoff)) \
                    if pk_rel > 0 else lo
                ct_val, ft_val = _threshold_timing(rep, takeoff_idx, fps_eff=100)
            else:
                ct_val, ft_val = np.nan, np.nan

            jh[sid].append(jump_h)
            ct[sid].append(ct_val)
            ft[sid].append(ft_val)
    return jh, ct, ft


def _threshold_timing(rep, takeoff_idx, fps_eff=100):
    r = np.asarray(rep, dtype=float)
    if r.size < 10 or not np.isfinite(r).any():
        return np.nan, np.nan
    peak = np.nanmax(r)
    if peak <= 0:
        return np.nan, np.nan
    thr = 0.25 * peak
    airborne = r > thr
    idx = np.where(airborne)[0]
    if len(idx) == 0:
        return np.nan, np.nan
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    MIN_FLIGHT = 8                      # frames (~0.08s @100fps); reject noise runs
    if len(runs) == 1:
        flight_run = runs[0]
    else:
        later = runs[1:]
        sized = [run for run in later if len(run) >= MIN_FLIGHT]
        if sized:
            flight_run = max(sized, key=lambda run: r[run].max())
        elif any(len(run) >= MIN_FLIGHT for run in runs):
            flight_run = max([run for run in runs if len(run) >= MIN_FLIGHT],
                             key=lambda run: r[run].max())
        else:
            flight_run = max(runs, key=len)
    flight = np.round(len(flight_run) / fps_eff, 2)
    # contact = ground run immediately before the flight run
    f0 = flight_run[0]
    before = np.where(r[:f0] <= thr)[0]
    if len(before) == 0:
        # no sub-threshold ground before flight: use all pre-flight frames as a
        # fallback rather than dropping the rep entirely.
        contact = np.round(f0 / fps_eff, 2) if f0 > 0 else np.nan
        return contact, flight
    gruns = np.split(before, np.where(np.diff(before) > 1)[0] + 1)
    contact = np.round(len(gruns[-1]) / fps_eff, 2)
    return contact, flight


def _sid(fp):
    return next((s for s in SUBJECT_IDS if s in os.path.basename(fp)), None)


def _toe_signal(fp, width=1920, height=1080):
    df = cmj.load_pose_file(fp)
    px = cmj.to_pixels(df, width, height)
    toe = px[["RIGHT_FOOT_INDEX_x", "RIGHT_FOOT_INDEX_y"]].to_numpy()
    return savgol_filter(flip_axis(toe.copy())[:, 1], 11, 2)


def _ptm_scale(files, ptm, clamp_gravity, width, height, fps):
    scales = {}
    if ptm == "height":
        for fp in files:
            sid = _sid(fp)
            if sid is None:
                continue
            df = cmj.load_pose_file(fp); px = cmj.to_pixels(df, width, height)
            rest_n = int(0.5 * fps)
            scales[sid] = cmj.height_scale(px, cmj.SUBJECT_HEIGHTS[sid], rest_n)
        return scales

    raw = {}
    for fp in files:
        sid = _sid(fp)
        if sid is None:
            continue
        toe_y = _toe_signal(fp, width, height)
        starts = OPENPOSE_OP_SEGS["bl" if "UL" not in os.path.basename(fp).upper()
                                  else "ul"][SUBJECT_IDS.index(sid)]
        vals = []
        for s0 in starts:
            rep = _toe_rep_window(toe_y, int(s0), fps=fps)
            try:
                vals.append(float(PTM.mm_per_px(rep, fps=fps)))
            except Exception:
                pass
        raw[sid] = np.nanmedian(vals) if vals else np.nan
    if clamp_gravity:
        m = np.nanmean(list(raw.values()))
        scales = {s: (max(v, m) if np.isfinite(v) else m) for s, v in raw.items()}
    else:
        scales = raw
    return scales


def _fp_perf(subjects, task):
    jh, ct, ft = {}, {}, {}
    coda_segs = _CODA_SEGS[task]
    for a, subject in enumerate(subjects):
        sid = SUBJECT_IDS[a]
        if sid in SKIP_SUBJECTS:
            continue
        F = subject[task]["force"]
        f1 = F[1] + F[3]; f2 = F[2] + F[4]
        f = f1 if f1.max() > f2.max() else f2
        force = f - f.min()
        fp_segs = coda_segs[a] * 10          # 100fps coda -> 1000fps force
        jh[sid], ct[sid], ft[sid] = [], [], []
        for j in range(3):
            fp0 = int(fp_segs[j]); fp1 = fp0 + int(1.5 * 1000)
            grf = force[fp0:fp1].copy(); grf -= grf.min()
            thresh = grf[:20].mean()
            below = np.diff(np.where(grf <= thresh + 15)[0])
            above = np.diff(np.where(grf > thresh + 15)[0])
            try:
                contact_time = np.round(below[below > 20][0] / 1000, 2)
                flight_time = np.round(above[above > 20][0] / 1000, 2)
                jump_height = np.round(jump_heights(flight_time) * 100, 2)
            except Exception:
                contact_time = flight_time = jump_height = np.nan
            jh[sid].append(jump_height)
            ct[sid].append(contact_time)
            ft[sid].append(flight_time)
    return jh, ct, ft

_CODA_SEGS = {
    "bl": np.array([
        [405, 1455, 2633], [651, 2201, 3338], [575, 1669, 2663], [605, 1480, 2301],
        [588, 1970, 3166], [2042, 3953, 5088], [1146, 2602, 3608], [4997, 5960, 6814],
        [1050, 2547, 3817], [914, 1958, 3314], [1242, 2259, 3235], [973, 1732, 2397],
        [1220, 3329, 4576], [1131, 3691, 4781], [1266, 2636, 3570], [833, 1750, 2515],
    ]),
    "ul": np.array([
        [379, 1386, 2328], [501, 1682, 2757], [2408, 4011, 5234], [646, 1947, 2858],
        [3306, 4705, 6410], [876, 1825, 2731], [1039, 2082, 3243], [1030, 3005, 4031],
        [1222, 2471, 3669], [1033, 2139, 3182], [813, 2807, 4629], [901, 1730, 2586],
        [1218, 2695, 3723], [1367, 3806, 4826], [1254, 2392, 3419], [916, 1741, 2592],
    ]),
}

def _paired_arrays(mmc, fp):
 
    sids = sorted(set(mmc) & set(fp))
    fp_arr, mmc_arr = [], []
    for sid in sids:
        m, f = mmc[sid], fp[sid]
        if len(m) == 3 and len(f) == 3 and all(np.isfinite(m)) and all(np.isfinite(f)):
            fp_arr.extend(f); mmc_arr.extend(m)
    return np.array(fp_arr), np.array(mmc_arr), sids


def _retest(mmc):
  
    rows = [m for m in mmc.values() if len(m) == 3 and all(np.isfinite(m))]
    if len(rows) < 2:
        return np.nan
    arr = np.array(rows)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return np.nan
    L = len(arr)
    reps = [pd.DataFrame({"sn": np.arange(1, L + 1), "score": arr[:, k],
                          "rater": [f"rep{k+1}"] * L}) for k in range(3)]
    try:
        icc = pg.intraclass_corr(data=pd.concat(reps), targets="sn",
                                 raters="rater", ratings="score")
        return np.round(icc.set_index("Type")["ICC"].iloc[1], 2)
    except Exception:
        return np.nan


def _icc(fp_arr, mmc_arr):
    L = len(fp_arr)
    a = pd.DataFrame({"sn": np.arange(1, L + 1), "score": fp_arr, "rater": ["FP"] * L})
    b = pd.DataFrame({"sn": np.arange(1, L + 1), "score": mmc_arr, "rater": ["MMC"] * L})
    icc = pg.intraclass_corr(data=pd.concat([a, b]), targets="sn",
                             raters="rater", ratings="score")
    return np.round(icc.set_index("Type")["ICC"].iloc[1], 2)


def _metric_rows(metric, mmc_bl, fp_bl, mmc_ul, fp_ul, ptm):
    out = []
    for taskname, mmc, fp in [("Drop jump bilateral", mmc_bl, fp_bl),
                              ("Drop jump unilateral", mmc_ul, fp_ul)]:
        fa, ma, _ = _paired_arrays(mmc, fp)
        out.append({"Metric": metric, "Task": taskname, "PTM Ref": ptm,
                    "Ground Truth": "Force plates",
                    "MAE": np.round(MAE(fa, ma), 2) if len(fa) else np.nan,
                    "Reliability": _retest(mmc),
                    "ICC": _icc(fa, ma) if len(fa) else np.nan})
    return pd.DataFrame(out)


def get_metrics(bl_files, ul_files, subjects, ptm="height", clamp_gravity=True):
 
    mmc_jh_bl, mmc_ct_bl, mmc_ft_bl = _mmc_perf(bl_files, "bl", ptm, clamp_gravity)
    mmc_jh_ul, mmc_ct_ul, mmc_ft_ul = _mmc_perf(ul_files, "ul", ptm, clamp_gravity)
    fp_jh_bl, fp_ct_bl, fp_ft_bl = _fp_perf(subjects, "bl")
    fp_jh_ul, fp_ct_ul, fp_ft_ul = _fp_perf(subjects, "ul")

    jh_df = _metric_rows("Jump height", mmc_jh_bl, fp_jh_bl, mmc_jh_ul, fp_jh_ul, ptm)
    if ptm != "height":
        return jh_df
    ft_df = _metric_rows("Flight time", mmc_ft_bl, fp_ft_bl, mmc_ft_ul, fp_ft_ul, ptm)
    ct_df = _metric_rows("Contact time", mmc_ct_bl, fp_ct_bl, mmc_ct_ul, fp_ct_ul, ptm)
    return jh_df, ft_df, ct_df

def ba_plots(bl_files, ul_files, subjects, ptm="height", plot="jump height",
             clamp_gravity=True):
    mmc_jh_bl, mmc_ct_bl, mmc_ft_bl = _mmc_perf(bl_files, "bl", ptm, clamp_gravity)
    mmc_jh_ul, mmc_ct_ul, mmc_ft_ul = _mmc_perf(ul_files, "ul", ptm, clamp_gravity)
    fp_jh_bl, fp_ct_bl, fp_ft_bl = _fp_perf(subjects, "bl")
    fp_jh_ul, fp_ct_ul, fp_ft_ul = _fp_perf(subjects, "ul")

    def _plot(mb, fb, mu, fu, name, unit):
        fb_a, mb_a, _ = _paired_arrays(mb, fb)
        fu_a, mu_a, _ = _paired_arrays(mu, fu)
        fig, axs = plt.subplots(1, 2, figsize=(10, 3), dpi=100)
        plot_ba(fb_a, mb_a, title="bilateral", ax=axs[0])
        plot_ba(fu_a, mu_a, title="unilateral", ax=axs[1], label_y=False)
        fig.suptitle(f"Drop jump {name} ({unit}) - {ptm} PTM")
        fig.supxlabel("Means"); plt.tight_layout(); plt.show()

    if "time" in plot:
        _plot(mmc_ft_bl, fp_ft_bl, mmc_ft_ul, fp_ft_ul, "flight time", "secs")
        _plot(mmc_ct_bl, fp_ct_bl, mmc_ct_ul, fp_ct_ul, "contact time", "secs")
    else:
        _plot(mmc_jh_bl, fp_jh_bl, mmc_jh_ul, fp_jh_ul, "jump height", "cm")