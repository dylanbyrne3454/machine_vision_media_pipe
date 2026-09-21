"""Velocity-based tests (MediaPipe port) — mean & peak barbell velocity.

For back squat (bsq) and overhead press (ohp). Ground truth is OMC (the pickle's
'coda'), NOT force plates. Two scalings: 'barbell' (PTM.barbell) and 'height'.

Bar proxy: LEFT_WRIST_y. Confirmed by correlating the velocity pickle's
p['left']['op'] signal against every txt landmark -> wrist/hand/arm all match at
|r|=0.997 (they move as one rigid unit with the bar); wrist is the natural,
low-noise left-side choice matching the original's p['left'].

Velocity pipeline (faithful to the original velocity.py):
  position (mm) -> *0.001 (m) -> savgol(gradient * fps) -> velocity (m/s)
  take the positive lifting-phase velocity up to the position peak;
  peak velocity = max of it; mean velocity = mean from zero-crossing to peak.

    from tasks import velocity
    import glob, utils
    bsq = sorted(p for p in glob.glob("data/keypoints/bsq/*.txt")
                 if "world" not in p.lower())
    subjects = utils.read_list('bsq_data')
    m = velocity.get_metrics(bsq, subjects, task='bsq',
                             task_label='Back Squat', ptm='height')
"""

import os
import numpy as np
import pandas as pd
import pingouin as pg
from copy import deepcopy

from scipy.signal import savgol_filter, resample
from sklearn.metrics import mean_absolute_error as MAE
import matplotlib.pyplot as plt
import seaborn as sns

from utilities import utils
from utilities.utils import get_reps, plot_ba, get_ratings, smooth_by_resampling
import utilities.PTM as PTM

from tasks import cmj


SUBJECT_IDS = [f"P{i:02d}" for i in range(3, 19)]
BAR_LANDMARK = "LEFT_WRIST"          # bar proxy (see module docstring)


def _sid(fp):
    return next((s for s in SUBJECT_IDS if s in os.path.basename(fp)), None)


def _bar_signal(fp, width=1920, height=1080, sign=1):
    """LEFT_WRIST vertical in pixels. `sign` flips orientation per task:
    BSQ segments cleanly as-is (sign=1); OHP needs sign=-1 because the press is
    an upward bar path that get_reps otherwise can't latch onto."""
    df = cmj.load_pose_file(fp)
    px = cmj.to_pixels(df, width, height)
    return sign * px[f"{BAR_LANDMARK}_y"].to_numpy()


# per-task orientation for get_reps segmentation
TASK_SIGN = {"bsq": 1, "ohp": -1}


def _ptm_scale(files, ptm, task, width, height):
    """{sid: mm_per_px}. 'height' uses anthropometric scale; 'barbell' uses
    PTM.barbell() (one value per subject), matching the original."""
    if ptm == "barbell":
        vals = PTM.barbell()
        return {SUBJECT_IDS[i]: vals[i] for i in range(len(vals))}
    # height
    scales = {}
    for fp in files:
        sid = _sid(fp)
        if sid is None:
            continue
        df = cmj.load_pose_file(fp); px = cmj.to_pixels(df, width, height)
        rest_n = int(0.5 * 30)
        scales[sid] = cmj.height_scale(px, cmj.SUBJECT_HEIGHTS[sid], rest_n)
    return scales


def _velocity_metrics(signal_m):
    """From a 1-D position signal in metres, return (peak_vel, mean_vel) for the
    lifting (concentric) phase, faithful to the original velocity calc. The rep
    is assumed oriented as an upward arc (handled by per-task signs upstream)."""
    s = np.asarray(signal_m, dtype=float)
    if len(s) < 6:
        return np.nan, np.nan
    vel = savgol_filter(np.gradient(s) * 100, 21, 2)          # 100fps timebase
    vel_pos = vel[4:np.argmax(s)]
    if len(vel_pos) < 2:
        return np.nan, np.nan
    zero_vel = np.argmin(abs(vel_pos[:np.argmax(vel_pos)])) if np.argmax(vel_pos) > 0 else 0
    peak = np.round(float(np.max(vel_pos)), 2)
    mean = np.round(float(vel_pos[zero_vel:].mean()), 2)
    return peak, mean


def get_performance(files, subjects, task, ptm="barbell", width=1920, height=1080):
    """Per-rep peak & mean velocity for MMC (txt) and OMC (pickle coda).
    Returns peak_vel_df, mean_vel_df keyed like the original
    (coda_ohp_{i}_pv / op_ohp_{i}_pv etc.) so downstream funcs port unchanged.
    """
    scale = _ptm_scale(files, ptm, task, width, height)
    mean_vel_dict, peak_vel_dict = {}, {}

    # map subject id -> file for the MMC side
    file_by_sid = {}
    for fp in files:
        sid = _sid(fp)
        if sid is not None:
            file_by_sid[sid] = fp

    for a in range(len(subjects)):
        sid = SUBJECT_IDS[a]
        if sid not in file_by_sid:
            continue
        p = subjects[a]

        # --- OMC (ground truth) from pickle ---
        omc = savgol_filter(p['left']['coda'][:, 2], 21, 2)
        coda_seg, coda_reps = get_reps(omc, fps=100, plot=False, d=1,
                                       forward=1.5, rewind=1.5)
        # --- MMC from txt (bar proxy), oriented per task for get_reps ---
        sign = TASK_SIGN.get(task, 1)
        mmc_raw = _bar_signal(file_by_sid[sid], width, height, sign=sign)
        mmc = savgol_filter(mmc_raw, 21, 2)
        op_seg, op_reps = get_reps(mmc, fps=30, plot=False, d=1,
                                   forward=1.5, rewind=1.5)

        # last three reps each
        coda_reps = coda_reps[-3:]
        op_reps = op_reps[-3:]
        if len(coda_reps) < 3 or len(op_reps) < 3:
            continue

        for i in range(3):
            coda_rep = deepcopy(coda_reps[i])[:-20]
            op_rep = deepcopy(op_reps[i])
            coda_rep = coda_rep - coda_rep.min()
            op_rep = op_rep - op_rep.min()

            op_conv = deepcopy(op_rep) * scale[sid]
            op_conv = resample(op_conv, len(coda_rep))[:-20]
            op_conv = smooth_by_resampling(op_conv)
            op_conv = op_conv * 0.001                 # mm -> m
            op_peak, op_mean = _velocity_metrics(op_conv)

            coda_rep = coda_rep * 0.001               # mm -> m
            coda_peak, coda_mean = _velocity_metrics(coda_rep)

            ck, ok = f"coda_ohp_{i+1}", f"op_ohp_{i+1}"
            mean_vel_dict.setdefault(f"{ck}_mv", []).append(coda_mean)
            peak_vel_dict.setdefault(f"{ck}_pv", []).append(coda_peak)
            mean_vel_dict.setdefault(f"{ok}_mv", []).append(op_mean)
            peak_vel_dict.setdefault(f"{ok}_pv", []).append(op_peak)

    return pd.DataFrame(peak_vel_dict), pd.DataFrame(mean_vel_dict)


def flatten_reps(df, device, variant):
    return np.array([df[f'{device}_ohp_{i}_{variant}'] for i in [1, 2, 3]]).flatten()


def get_velocity(files, subjects, task, ptm="barbell"):
    peak_vel_df, mean_vel_df = get_performance(files, subjects, task=task, ptm=ptm)
    peak = {'omc': flatten_reps(peak_vel_df, 'coda', 'pv'),
            'mmc': flatten_reps(peak_vel_df, 'op', 'pv')}
    mean = {'omc': flatten_reps(mean_vel_df, 'coda', 'mv'),
            'mmc': flatten_reps(mean_vel_df, 'op', 'mv')}
    return peak, mean, peak_vel_df, mean_vel_df


def ba_plots(files, subjects, task, task_label, ptm):
    peak, mean, *dfs = get_velocity(files, subjects, task, ptm=ptm)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3), dpi=100)
    plot_ba(mean['omc'], mean['mmc'], title='mean velocities (m/s)', ax=axs[0])
    plot_ba(peak['omc'], peak['mmc'], title='peak velocities (m/s)', ax=axs[1],
            label_y=False)
    fig.suptitle(f'{task_label} - {ptm} PTM')
    fig.supxlabel('Means')
    plt.show()
    return peak, mean, dfs


def get_retest(df, device, metric_variant):
    L = df.shape[0]
    reps = []
    for i in range(1, 4):
        reps.append(pd.DataFrame({
            'sn': np.arange(1, L + 1),
            'score': df[f'{device}_ohp_{i}_{metric_variant}'],
            'rater': [f'rep{i}'] * L}))
    icc = pg.intraclass_corr(data=pd.concat(reps), targets='sn',
                             raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'].iloc[1], 2)


def get_icc(x, y, devices=['OMC', 'MMC']):
    df = pd.concat([get_ratings(x, devices[0]), get_ratings(y, devices[1])])
    icc = pg.intraclass_corr(data=df, targets='rep', raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'].iloc[1], 2)


def get_metrics(files, subjects, task, task_label, ptm='barbell'):
    peak, mean, dfs = ba_plots(files, subjects, task, task_label, ptm=ptm)
    peak_df, mean_df = dfs
    peak_vel = {'Metric': 'Peak velocity', 'Task': task_label, 'PTM Ref': ptm,
                'Ground Truth': 'Optical motion capture',
                'MAE': np.round(MAE(peak['omc'], peak['mmc']), 2),
                'Reliability': get_retest(peak_df, 'op', 'pv'),
                'ICC': get_icc(peak['omc'], peak['mmc'])}
    mean_vel = {'Metric': 'Mean velocity', 'Task': task_label, 'PTM Ref': ptm,
                'Ground Truth': 'Optical motion capture',
                'MAE': np.round(MAE(mean['omc'], mean['mmc']), 2),
                'Reliability': get_retest(mean_df, 'op', 'mv'),
                'ICC': get_icc(mean['omc'], mean['mmc'])}
    metrics_df = pd.DataFrame(peak_vel, index=[0])
    metrics_df.loc[1] = mean_vel
    return metrics_df