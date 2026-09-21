import os
import numpy as np
import pandas as pd
import pingouin as pg

from scipy.signal import savgol_filter, resample
from sklearn.metrics import mean_absolute_error as MAE
import matplotlib.pyplot as plt
import seaborn as sns

from utilities import utils
from utilities.utils import get_angle, get_reps, get_ratings, plot_ba, smooth_array
from tasks import cmj


SUBJECT_IDS = [f"P{i:02d}" for i in range(3, 19)]
HIP, ANKLE = "RIGHT_HIP", "RIGHT_ANKLE"

EXCLUDE = {"P05", "P07", "P09", "P10", "P12", "P13", "P16", "P17", "P18"}

TRIM = {"P11": (1000, 400), "P15": (2500, 750)}


def _sid(fp):
    return next((s for s in SUBJECT_IDS if s in os.path.basename(fp)), None)


def _omc_angle(p, omc_start):
    hip = smooth_array(p['hip']['coda'].copy(), [0, 1, 2])
    ankle = smooth_array(p['ankle']['coda'].copy(), [0, 1, 2])
    ankle_ref = np.ones_like(ankle) * ankle[0]           
    ang = get_angle(ankle_ref, hip, ankle, ndims=3)  
    return savgol_filter(ang, 21, 2)[omc_start:]


def _mmc_angle(fp, mmc_start, ndims=2):
    df = cmj.load_pose_file(fp)
    hip = smooth_array(df[[f"{HIP}_x", f"{HIP}_y", f"{HIP}_z"]].to_numpy().copy(), tuple(range(ndims)))
    ankle = smooth_array(df[[f"{ANKLE}_x", f"{ANKLE}_y", f"{ANKLE}_z"]].to_numpy().copy(), tuple(range(ndims)))
    ankle_ref = np.ones_like(ankle) * ankle[0]
    ang = get_angle(ankle_ref, hip, ankle, ndims=ndims)
    return savgol_filter(ang, 21, 2)[mmc_start:]


def _participant_reps(fp, p, sid, ndims=2):
    omc_s, mmc_s = TRIM.get(sid, (0, 0))
    omc = _omc_angle(p, omc_s)
    mmc = _mmc_angle(fp, mmc_s, ndims=ndims)
    _, coda_reps = get_reps(omc, fps=100, plot=False, d=6, forward=3.7, rewind=3.7)
    _, op_reps = get_reps(mmc, fps=30, plot=False, d=6, forward=3.7, rewind=3.7)
    return coda_reps[-3:], op_reps[-3:]           


def get_ranges_of_motion(files, subjects, ndims=2):
    rom_dict = {}
    file_by_sid = {_sid(fp): fp for fp in files if _sid(fp)}
    for a in range(len(subjects)):
        sid = SUBJECT_IDS[a]
        if sid in EXCLUDE or sid not in file_by_sid:
            continue
        coda_reps, op_reps = _participant_reps(file_by_sid[sid], subjects[a], sid, ndims=ndims)
        if len(coda_reps) < 3 or len(op_reps) < 3:
            print(f"[slr] {sid}: only {len(coda_reps)}/{len(op_reps)} reps, skipped")
            continue
        for i in range(3):
            coda_rep = coda_reps[i] - min(coda_reps[i])
            op_rep = resample(op_reps[i] - min(op_reps[i]), len(coda_rep))
            rom_dict.setdefault(f"coda_{i+1}", []).append(np.round(max(coda_rep), 2))
            rom_dict.setdefault(f"op_{i+1}", []).append(np.round(max(op_rep), 2))
    return pd.DataFrame(rom_dict)


def flatten_reps(df, device):
    return np.array([df[f'{device}_{i}'] for i in [1, 2, 3]]).flatten()


def ba_plots(files, subjects, ndims=2):
    rom_df = get_ranges_of_motion(files, subjects, ndims=ndims)
    roms = {'omc': flatten_reps(rom_df, 'coda'), 'mmc': flatten_reps(rom_df, 'op')}
    fig, ax = plt.subplots(1, 1, figsize=(10, 3), dpi=100)
    plot_ba(roms['omc'], roms['mmc'],
            title='Straight leg raise range of motion (degs)', ax=ax)
    fig.supxlabel('Means'); plt.show()
    return roms, rom_df


def get_icc(x, y, devices=['OMC', 'MMC']):
    df = pd.concat([get_ratings(x, devices[0]), get_ratings(y, devices[1])])
    icc = pg.intraclass_corr(data=df, targets='rep', raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'].iloc[1], 2)


def get_retest(df, device):
    L = df.shape[0]
    reps = [pd.DataFrame({'sn': np.arange(1, L + 1), 'score': df[f'{device}_{i}'],
                          'rater': [f'rep{i}'] * L}) for i in range(1, 4)]
    icc = pg.intraclass_corr(data=pd.concat(reps), targets='sn',
                             raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'].iloc[1], 2)


def get_metrics(files, subjects, ndims=2):
    roms, rom_df = ba_plots(files, subjects, ndims=ndims)
    rom = {'Metric': 'Range of motion', 'Task': 'Straight leg raise', 'PTM Ref': 'N/A',
           'Ground Truth': 'Optical motion capture',
           'MAE': np.round(MAE(roms['omc'], roms['mmc']), 2),
           'Reliability': get_retest(rom_df, 'op'),
           'ICC': get_icc(roms['omc'], roms['mmc'])}
    return pd.DataFrame(rom, index=[0])