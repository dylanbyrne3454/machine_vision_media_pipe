

import os
import numpy as np
import pandas as pd
import pingouin as pg

from scipy.signal import find_peaks
from sklearn.metrics import mean_absolute_error as MAE
import matplotlib.pyplot as plt
import seaborn as sns

from utilities import utils
from utilities.utils import smoothen, rjt_flight_times, rjt_contact_times, \
    jump_heights, get_ratings, plot_ba, flip_axis
from tasks import cmj


SUBJECT_IDS = [f"P{i:02d}" for i in range(3, 19)]
N_JUMPS = 10


def _sid(fp):
    return next((s for s in SUBJECT_IDS if s in os.path.basename(fp)), None)


def _hip_signal(fp, width=1920, height=1080):
    """Hip vertical (up-positive) in pixels from the txt."""
    df = cmj.load_pose_file(fp)
    px = cmj.to_pixels(df, width, height)
    hip = np.column_stack([(px["LEFT_HIP_x"] + px["RIGHT_HIP_x"]) / 2,
                           (px["LEFT_HIP_y"] + px["RIGHT_HIP_y"]) / 2])
    return flip_axis(hip.copy())[:, 1]


def _ptm_scale(fp, ptm, width, height, fps=30):
    if ptm == "height":
        df = cmj.load_pose_file(fp); px = cmj.to_pixels(df, width, height)
        rest_n = int(0.5 * fps)
        sid = _sid(fp)
        return cmj.height_scale(px, cmj.SUBJECT_HEIGHTS[sid], rest_n)
    # gravity fallback: single value from a peak fit (best-effort)
    sig = _hip_signal(fp, width, height)
    try:
        import utilities.PTM as PTM
        return float(PTM.mm_per_px(sig, fps=fps))
    except Exception:
        return np.nan


def get_mmc_values(fp, ptm, width=1920, height=1080, fps=30):
    mm_px = _ptm_scale(fp, ptm, width, height, fps)
    op = smoothen(_hip_signal(fp, width, height)) * mm_px
    op -= op.min()
    op_rest = op[:50].mean()
    op_peaks, _ = find_peaks(op, height=op[0] + 60, distance=10)
    op_jh = (op[op_peaks][:N_JUMPS] - op_rest) / 10
    op_ft = rjt_flight_times(op, op_rest + 15, fps=fps)[:N_JUMPS]
    op_ct = rjt_contact_times(op, op_peaks, op_rest, fps=fps)[:N_JUMPS]
    return op_ft, op_ct, np.round(op_jh, 2)


def get_force_plate_values(p):
    F = p['rjt']['force']
    f1 = F[1] + F[3]; f2 = F[2] + F[4]
    f = f1 if f1.max() > f2.max() else f2
    force = f - f.min()
    thresh = min(force)
    ct = np.diff(np.where(force <= thresh + 10)[0])
    ft = np.diff(np.where(force > thresh + 30)[0])
    fp_ct = np.round(ct[ct > 100] / 1000, 2)[:N_JUMPS]
    fp_ft = np.round(ft[ft > 100] / 1000, 2)[:N_JUMPS]
    fp_jh = np.round(jump_heights(fp_ft) * 100, 2)
    return fp_ft, fp_ct, fp_jh


def get_performance(files, subjects, ptm):
    OP_FT, OP_CT, OP_JH, FP_FT, FP_CT, FP_JH = [], [], [], [], [], []
    file_by_sid = {_sid(fp): fp for fp in files if _sid(fp)}
    kept = []
    for a in range(len(subjects)):
        sid = SUBJECT_IDS[a]
        if sid not in file_by_sid:
            continue
        try:
            op_ft, op_ct, op_jh = get_mmc_values(file_by_sid[sid], ptm)
            fp_ft, fp_ct, fp_jh = get_force_plate_values(subjects[a])
        except Exception as ex:
            print(f"[rjt] {sid}: {type(ex).__name__}: {str(ex)[:50]}, skipped")
            continue
        # require enough jumps on both sides; pad/truncate to a common count
        n = min(len(op_ft), len(op_ct), len(fp_ft), len(fp_ct))
        if n < 3:
            print(f"[rjt] {sid}: only {n} usable jumps, skipped")
            continue
        OP_FT.append(op_ft[:n]); OP_CT.append(op_ct[:n])
        FP_FT.append(fp_ft[:n]); FP_CT.append(fp_ct[:n])
        kept.append(sid)
    return {"op_ft": OP_FT, "op_ct": OP_CT, "fp_ft": FP_FT, "fp_ct": FP_CT,
            "subjects": kept}


def _flatten(list_of_arrays):
    return np.concatenate([np.asarray(a) for a in list_of_arrays]) if list_of_arrays else np.array([])


def ba_plots(files, subjects, ptm="height"):
    d = get_performance(files, subjects, ptm)
    ft = {"fp": _flatten(d["fp_ft"]), "mmc": _flatten(d["op_ft"])}
    ct = {"fp": _flatten(d["fp_ct"]), "mmc": _flatten(d["op_ct"])}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3), dpi=100)
    plot_ba(ft["fp"], ft["mmc"], title='flight times (secs)', ax=axs[0])
    plot_ba(ct["fp"], ct["mmc"], title='contact times (secs)', ax=axs[1], label_y=False)
    fig.suptitle('Repeated jump test'); fig.supxlabel('Means'); plt.show()
    return ft, ct, d


def _icc(x, y):
    L = min(len(x), len(y)); x, y = x[:L], y[:L]
    a = pd.DataFrame({"rep": np.arange(L), "score": x, "rater": ["FP"] * L})
    b = pd.DataFrame({"rep": np.arange(L), "score": y, "rater": ["MMC"] * L})
    icc = pg.intraclass_corr(data=pd.concat([a, b]), targets="rep",
                             raters="rater", ratings="score")
    return np.round(icc.set_index("Type")["ICC"].iloc[1], 2)


def _retest(list_of_arrays, n_reps=3):
    rows = [np.asarray(a)[:n_reps] for a in list_of_arrays
            if len(a) >= n_reps and np.all(np.isfinite(np.asarray(a)[:n_reps]))]
    if len(rows) < 2:
        return np.nan
    arr = np.array(rows)
    L = len(arr)
    reps = [pd.DataFrame({"sn": np.arange(1, L + 1), "score": arr[:, k],
                          "rater": [f"rep{k+1}"] * L}) for k in range(n_reps)]
    icc = pg.intraclass_corr(data=pd.concat(reps), targets="sn",
                             raters="rater", ratings="score")
    return np.round(icc.set_index("Type")["ICC"].iloc[1], 2)


def get_metrics(files, subjects, ptm="height"):
    ft, ct, d = ba_plots(files, subjects, ptm)
    n = min(len(ft["fp"]), len(ft["mmc"]))
    m = min(len(ct["fp"]), len(ct["mmc"]))
    rows = [
        {"Metric": "Flight time", "Task": "Repeated jump test", "PTM Ref": ptm,
         "Ground Truth": "Force plates",
         "MAE": np.round(MAE(ft["fp"][:n], ft["mmc"][:n]), 2),
         "Reliability": _retest(d["op_ft"]),
         "ICC": _icc(ft["fp"], ft["mmc"])},
        {"Metric": "Contact time", "Task": "Repeated jump test", "PTM Ref": ptm,
         "Ground Truth": "Force plates",
         "MAE": np.round(MAE(ct["fp"][:m], ct["mmc"][:m]), 2),
         "Reliability": _retest(d["op_ct"]),
         "ICC": _icc(ct["fp"], ct["mmc"])},
    ]
    return pd.DataFrame(rows)