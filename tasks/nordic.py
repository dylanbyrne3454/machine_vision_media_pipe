
import os
import numpy as np
import pandas as pd
import pingouin as pg

from scipy.signal import savgol_filter, resample
from sklearn.metrics import mean_absolute_error as MAE
import matplotlib.pyplot as plt
import seaborn as sns

from utilities import utils
from utilities.utils import get_angle, get_reps, get_ratings, plot_ba, \
    fill_nan, smooth_array
from tasks import cmj


SUBJECT_IDS = [f"P{i:02d}" for i in range(3, 19)]
JOINTS = ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE")   # side camera

# editable trim offsets (frames) into the MMC txt, per subject if needed.
# default: a single offset applied to all; override specific subjects as found.
MMC_START_DEFAULT = 600
MMC_START = {}                 # e.g. {'P08': 700} if a subject needs a different trim
# coda offsets come from the original (start 2500 for P08 i.e. idx 5, else 1500)
CODA_START_DEFAULT = 1500
CODA_START = {"P08": 2500}     # idx==5 -> P08 in the original


def _sid(fp):
    return next((s for s in SUBJECT_IDS if s in os.path.basename(fp)), None)


def _mmc_angle(fp, start, ndims=2):
    """hip-knee-ankle angle from the MediaPipe txt.
    ndims=2 (default) = current validated behaviour (x,y).
    ndims=3 uses x,y,z. The file (fp) determines normalized vs world."""
    df = cmj.load_pose_file(fp)
    pts = []
    for j in JOINTS:
        xyz = df[[f"{j}_x", f"{j}_y", f"{j}_z"]].to_numpy().copy()
        pts.append(smooth_array(xyz, tuple(range(ndims))))
    hip, knee, ankle = [p[start:] for p in pts]
    return get_angle(hip, knee, ankle, ndims=ndims)

def _omc_angle(p, start):
    """hip-knee-ankle angle from the pickle OMC 'coda' (ndims=3)."""
    hip = fill_nan(p['nordic']['hip']['coda'])[start:]
    knee = fill_nan(p['nordic']['knee']['coda'])[start:]
    ankle = fill_nan(p['nordic']['ankle']['coda'])[start:]
    return get_angle(hip, knee, ankle, ndims=3)


def _participant_reps(fp, p, sid, ndims=2):
    omc = savgol_filter(_omc_angle(p, CODA_START.get(sid, CODA_START_DEFAULT)), 5, 2)
    mmc = savgol_filter(_mmc_angle(fp, MMC_START.get(sid, MMC_START_DEFAULT), ndims=ndims), 5, 2)
    _, coda_reps = get_reps(omc, fps=100, plot=False, d=5, forward=0.1, rewind=2.5)
    _, op_reps = get_reps(mmc, fps=30, plot=False, d=5, forward=0.1, rewind=2.5)
    return coda_reps[:3], op_reps[:3]

def get_performance(files, subjects, ndims=2):
    rom_dict, vel_dict = {}, {}
    file_by_sid = {_sid(fp): fp for fp in files if _sid(fp)}
    for a in range(len(subjects)):
        sid = SUBJECT_IDS[a]
        if sid not in file_by_sid:
            continue
        coda_reps, op_reps = _participant_reps(file_by_sid[sid], subjects[a], sid, ndims=ndims)
        if len(coda_reps) < 3 or len(op_reps) < 3:
            continue
        for i in range(3):
            coda_rep = coda_reps[i] - min(coda_reps[i])
            op_rep = op_reps[i] - min(op_reps[i])
            coda_rom = np.round(max(coda_rep), 2)
            op_rep = resample(op_rep, len(coda_rep))[:-5]
            op_rep = savgol_filter(op_rep, 21, 2)
            op_rom = np.round(max(op_rep), 2)
            coda_vel = savgol_filter(np.gradient(coda_rep) * 100, 5, 2)[10:]
            op_vel = savgol_filter(np.gradient(op_rep) * 100, 21, 2)[10:]
            ck, ok = f"coda_rep_{i+1}", f"op_rep_{i+1}"
            vel_dict.setdefault(ck, []).append(np.round(coda_vel.mean(), 2))
            rom_dict.setdefault(ck, []).append(coda_rom)
            vel_dict.setdefault(ok, []).append(np.round(op_vel.mean(), 2))
            rom_dict.setdefault(ok, []).append(op_rom)
    return pd.DataFrame(rom_dict), pd.DataFrame(vel_dict)


def flatten_reps(df, device):
    return np.array([df[f'{device}_rep_{i}'] for i in [1, 2, 3]]).flatten()


def ba_plots(files, subjects, ndims=2):
    rom_df, vel_df = get_performance(files, subjects, ndims=ndims)
    roms = {'omc': flatten_reps(rom_df, 'coda'), 'mmc': flatten_reps(rom_df, 'op')}
    vel = {'omc': flatten_reps(vel_df, 'coda'), 'mmc': flatten_reps(vel_df, 'op')}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3), dpi=100)
    plot_ba(roms['omc'], roms['mmc'], title='range of motion (degs)', ax=axs[0])
    plot_ba(vel['omc'], vel['mmc'], title='angular velocity (deg/s)',
            ax=axs[1], label_y=False)
    fig.suptitle('Nordic curl'); fig.supxlabel('Means'); plt.show()
    return roms, vel, rom_df, vel_df


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
    rom = {'Metric': 'Range of motion', 'Task': 'Nordic curl', 'PTM Ref': 'N/A',
           'Ground Truth': 'Optical motion capture',
           'MAE': np.round(MAE(roms['omc'], roms['mmc']), 2),
           'Reliability': get_retest(rom_df, 'op'),
           'ICC': get_icc(roms['omc'], roms['mmc'])}
    vel = {'Metric': 'Angular velocity', 'Task': 'Nordic curl', 'PTM Ref': 'N/A',
           'Ground Truth': 'Optical motion capture',
           'MAE': np.round(MAE(vels['omc'], vels['mmc']), 2),
           'Reliability': get_retest(vel_df, 'op'),
           'ICC': get_icc(vels['omc'], vels['mmc'])}
    metrics_df = pd.DataFrame(rom, index=[0]); metrics_df.loc[1] = vel
    return metrics_df