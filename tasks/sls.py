"""Single leg squat (MediaPipe port) — range of motion + angular velocity.

The hardest angle task: MANUAL segmentation (4 frames -> 3 reps per subject via
segment_squat), knee angle (hip-knee-ankle), then per rep: flip the angle,
find the squat-bottom peak, window back ~0.8s, extract ROM (max) and mean
angular velocity (positive gradient).

Same videos as OpenPose -> the original op segs transfer (inlined below).
Scale-invariant -> no PTM. Ground truth = OMC (pickle 'coda').
Side camera -> right-side joints. Pickle: p['slsquat'][part]['coda'|'op'].

Quirks preserved from the original:
  - ankle gets interpolate() cleaning (clips above-mean spikes to min)
  - MMC uses ndims=2, OMC uses ndims=3
  - per-subject coda peak overrides (OpenPose-specific; yours may differ) live
    in CODA_PEAK_FIX and are easy to edit.

    from tasks import sls
    import glob, utils
    files = sorted(p for p in glob.glob("data/keypoints/slsq/*.txt")
                   if "world" not in p.lower())
    subjects = utils.read_list('sls_data')
    m = sls.get_metrics(files, subjects)
"""

import os
import numpy as np
import pandas as pd
import pingouin as pg

from scipy.signal import savgol_filter, resample, find_peaks
from sklearn.metrics import mean_absolute_error as MAE
import matplotlib.pyplot as plt
import seaborn as sns

from utilities import utils
from utilities.utils import get_angle, get_ratings, plot_ba, fill_nan, \
    flip_axis, segment_squat
from tasks import cmj


SUBJECT_IDS = [f"P{i:02d}" for i in range(3, 19)]
HIP, KNEE, ANKLE = "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"

# OpenPose op segs (same videos -> transfer). 4 frames -> 3 reps per subject.
OP_SEGS = [
    [300, 390, 450, 530], [250, 380, 480, 580], [280, 400, 470, 570],
    [200, 260, 310, 380], [200, 340, 440, 550], [200, 290, 355, 430],
    [160, 275, 375, 500], [175, 238, 288, 348], [280, 420, 555, 750],
    [225, 300, 350, 425], [225, 325, 405, 500], [200, 280, 390, 490],
    [270, 410, 510, 620], [170, 290, 400, 500], [260, 360, 445, 540],
    [180, 325, 445, 560],
]

CODA_SEGS = [
    [400, 650, 860, 1100], [700, 1050, 1350, 1650], [650, 980, 1250, 1500],
    [450, 650, 810, 1150], [650, 1050, 1400, 1800], [500, 800, 1020, 1250],
    [700, 1050, 1400, 1750], [600, 800, 950, 1150], [650, 1100, 1550, 2100],
    [650, 800, 1000, 1200], [620, 920, 1200, 1500], [620, 950, 1300, 1600],
    [880, 1250, 1600, 1950], [500, 880, 1220, 1570], [700, 1010, 1290, 1580],
    [600, 1000, 1400, 1750],
]


CODA_PEAK_FIX = {
    ("P04", 2): 156,
    ("P05", 0): 177, ("P05", 1): 140,
    ("P14", 1): 152, ("P14", 2): 165,
    ("P17", 0): 198,
}


def _sid(fp):
    return next((s for s in SUBJECT_IDS if s in os.path.basename(fp)), None)


def _interpolate(arr):
    """Clip above-mean spikes to the min, per axis (ankle cleaning)."""
    if arr.ndim == 1:
        return np.where(arr > arr.mean(), arr.min(), arr)
    out = np.zeros_like(arr)
    for i in range(arr.shape[1]):
        x = arr[:, i]
        out[:, i] = np.where(x > x.mean(), x.min(), x)
    return out


def _mmc_parts(fp):
    df = cmj.load_pose_file(fp)
    def part(name):
        return df[[f"{name}_x", f"{name}_y", f"{name}_z"]].to_numpy().copy()
    return part(HIP), part(KNEE), part(ANKLE)


def _roms_for_subject(fp, p, s, ndims=2):
    """Return (coda_angle_reps, op_angle_reps) -- 3 each, as angle arrays."""
    op_hip, op_knee, op_ankle = _mmc_parts(fp)
    co_hip = fill_nan(p['slsquat']['hip']['coda'])
    co_knee = fill_nan(p['slsquat']['knee']['coda'])
    co_ankle = fill_nan(p['slsquat']['ankle']['coda'])

    op_segs, co_segs = OP_SEGS[s], CODA_SEGS[s]
    op_hip_r = segment_squat(op_hip, op_segs)
    op_knee_r = segment_squat(op_knee, op_segs)
    op_ankle_r = segment_squat(op_ankle, op_segs)
    co_hip_r = segment_squat(co_hip, co_segs)
    co_knee_r = segment_squat(co_knee, co_segs)
    co_ankle_r = segment_squat(co_ankle, co_segs)

    op_roms = [get_angle(op_hip_r[i], op_knee_r[i],
                         _interpolate(op_ankle_r[i]), ndims=ndims) for i in range(3)]
    co_roms = [get_angle(co_hip_r[i], co_knee_r[i],
                         co_ankle_r[i], ndims=3) for i in range(3)]
    return co_roms, op_roms


def get_performance(files, subjects, ndims=2):
    rom_dict, vel_dict = {}, {}
    file_by_sid = {_sid(fp): fp for fp in files if _sid(fp)}
    for s in range(len(subjects)):
        sid = SUBJECT_IDS[s]
        if sid not in file_by_sid:
            continue
        co_roms, op_roms = _roms_for_subject(file_by_sid[sid], subjects[s], s, ndims=ndims)
        for k in range(3):
            op_rom = flip_axis(savgol_filter(op_roms[k], 21, 2))
            coda_rom = flip_axis(savgol_filter(co_roms[k], 5, 2))
            op_peaks, _ = find_peaks(op_rom, prominence=0.9, distance=len(op_rom) / 1.5)
            coda_peaks, _ = find_peaks(coda_rom, prominence=0.9, distance=len(coda_rom) / 1.5)
            if (sid, k) in CODA_PEAK_FIX:
                coda_peaks = np.array([CODA_PEAK_FIX[(sid, k)]])
            if len(op_peaks) == 0 or len(coda_peaks) == 0:
                print(f"[sls] {sid} rep {k+1}: no peak found, skipped")
                continue
            coda_start = max(0, coda_peaks[0] - 80)
            coda_rom = coda_rom[coda_start:coda_peaks[0]]
            op_start = max(0, op_peaks[0] - 24)
            op_rom = op_rom[op_start:op_peaks[0]]
            if len(op_rom) < 5 or len(coda_rom) < 5:
                print(f"[sls] {sid} rep {k+1}: window too short, skipped")
                continue
            coda_rom = resample(coda_rom, len(op_rom))
            op_rom = savgol_filter(op_rom - min(op_rom), 5, 2)
            coda_rom = coda_rom - min(coda_rom)
            op_vel = np.where(np.gradient(op_rom) * 30 < 0, 0, np.gradient(op_rom) * 30)
            coda_vel = np.where(np.gradient(coda_rom) * 30 < 0, 0, np.gradient(coda_rom) * 30)
            ck, ok = f"coda_rep_{k+1}", f"op_rep_{k+1}"
            vel_dict.setdefault(ck, []).append(np.round(coda_vel[2:-2].mean(), 2))
            rom_dict.setdefault(ck, []).append(np.round(max(coda_rom[2:-2]), 2))
            vel_dict.setdefault(ok, []).append(np.round(op_vel[2:-2].mean(), 2))
            rom_dict.setdefault(ok, []).append(np.round(max(op_rom[2:-2]), 2))
    return pd.DataFrame(rom_dict), pd.DataFrame(vel_dict)


def flatten_reps(df, device):
    return np.array([df[f'{device}_rep_{i}'] for i in [1, 2, 3]]).flatten()


def ba_plots(files, subjects, ndims=2):
    rom_df, vel_df = get_performance(files, subjects, ndims=ndims)
    roms = {'omc': flatten_reps(rom_df, 'coda'), 'mmc': flatten_reps(rom_df, 'op')}
    vels = {'omc': flatten_reps(vel_df, 'coda'), 'mmc': flatten_reps(vel_df, 'op')}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3), dpi=100)
    plot_ba(roms['omc'], roms['mmc'], title='range of motion (degs)', ax=axs[0])
    plot_ba(vels['omc'], vels['mmc'], title='mean angular velocity (degs/s)',
            ax=axs[1], label_y=False)
    fig.suptitle('Single leg squat'); fig.supxlabel('Means'); plt.show()
    return roms, vels, rom_df, vel_df


def get_icc(x, y, devices=['OMC', 'MMC']):
    df = pd.concat([get_ratings(x, devices[0]), get_ratings(y, devices[1])])
    icc = pg.intraclass_corr(data=df, targets='rep', raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'].iloc[1], 2)


def get_retest(df, device):
    L = df.shape[0]
    reps = [pd.DataFrame({'sn': np.arange(1, L + 1), 'score': df[f'{device}_rep_{i}'],
                          'rater': [f'rep{i}'] * L}) for i in range(1, 4)]
    icc = pg.intraclass_corr(data=pd.concat(reps), targets='sn',
                             raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'].iloc[1], 2)



def get_metrics(files, subjects, ndims=2):
    roms, vels, rom_df, vel_df = ba_plots(files, subjects, ndims=ndims)
    rom = {'Metric': 'Range of motion', 'Task': 'Single leg squat', 'PTM Ref': 'N/A',
           'Ground Truth': 'Optical motion capture',
           'MAE': np.round(MAE(roms['omc'], roms['mmc']), 2),
           'Reliability': get_retest(rom_df, 'op'),
           'ICC': get_icc(roms['omc'], roms['mmc'])}
    vel = {'Metric': 'Angular velocity', 'Task': 'Single leg squat', 'PTM Ref': 'N/A',
           'Ground Truth': 'Optical motion capture',
           'MAE': np.round(MAE(vels['omc'], vels['mmc']), 2),
           'Reliability': get_retest(vel_df, 'op'),
           'ICC': get_icc(vels['omc'], vels['mmc'])}
    metrics_df = pd.DataFrame(rom, index=[0]); metrics_df.loc[1] = vel
    return metrics_df