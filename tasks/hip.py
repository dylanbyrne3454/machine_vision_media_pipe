"""Hip rotation (MediaPipe port) — range of motion, HER & HIR.

Different from the other angle tasks:
  - TWO files per subject: one HER/ER capture, one HIR/IR capture.
  - FRONT camera (not side).
  - Uses intersection_angle: the angle between the knee->ankle segment at rest
    (frame 0) and its moving position -- i.e. how far the lower leg has rotated.
    Pure 2D, so MediaPipe x,y works directly.
  - Intentional her<->hir label swap in reporting "to correct a flip during
    data collection" -- preserved from the original.

Scale-invariant -> no PTM. Ground truth = OMC (pickle 'coda').
The pickle is p['her'|'hir']['knee'|'ankle']['coda'|'op'].

    from tasks import hip
    import utils
    subjects = utils.read_list('hip_data')
    m = hip.get_metrics("data/keypoints/hip", subjects)
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import pingouin as pg

from scipy.signal import resample
from sklearn.metrics import mean_absolute_error as MAE
import matplotlib.pyplot as plt
import seaborn as sns

from utilities import utils
from utilities.utils import intersection_angle, get_reps, get_ratings, plot_ba
from tasks import cmj


SUBJECT_IDS = [f"P{i:02d}" for i in range(3, 19)]
KNEE, ANKLE = "RIGHT_KNEE", "RIGHT_ANKLE"      # front camera lower-leg
EXCLUDE = {("P09", "her"), ("P12", "her"), ("P12", "hir")}


def _task_token(task):
    return ("HER", "HIP_ER", "ER") if task == "her" else ("HIR", "HIP_IR", "IR")


def _find_file(folder, sid, task):
    cands = [p for p in glob.glob(f"{folder}/*.txt")
             if "world" not in p.lower() and sid in os.path.basename(p)]
    toks = _task_token(task)
    # prefer the most specific token match; ER must not match HER-as-substring wrongly
    for p in cands:
        name = os.path.basename(p).upper()
        # build a token check that distinguishes ER from IR cleanly
        if task == "her" and re.search(r"(^|_)H?ER(_|\b)|HIP_ER", name):
            return p
        if task == "hir" and re.search(r"(^|_)H?IR(_|\b)|HIP_IR", name):
            return p
    return None

def _rotation_angle_3d(knee, ankle):
    """3D angle between the knee->ankle segment and its rest orientation.
    The 3D generalization of intersection_angle (which does this in 2D via slopes).
    """
    seg = ankle - knee
    rest = seg[0]
    cos = (seg @ rest) / (np.linalg.norm(seg, axis=1) * np.linalg.norm(rest) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def _mmc_angle(fp, ndims=2):
    df = cmj.load_pose_file(fp)
    if ndims == 2:
        knee = df[[f"{KNEE}_x", f"{KNEE}_y"]].to_numpy()
        ankle = df[[f"{ANKLE}_x", f"{ANKLE}_y"]].to_numpy()
        knee_ref = np.ones_like(knee) * knee[0]
        ankle_ref = np.ones_like(ankle) * ankle[0]
        return intersection_angle(line1=[ankle_ref, knee_ref], line2=[ankle, knee])
    else:
        knee = df[[f"{KNEE}_x", f"{KNEE}_y", f"{KNEE}_z"]].to_numpy()
        ankle = df[[f"{ANKLE}_x", f"{ANKLE}_y", f"{ANKLE}_z"]].to_numpy()
        return _rotation_angle_3d(knee, ankle)
    
def _omc_angle(p, task):
    knee = p[task]['knee']['coda'][:, [1, 2]]
    ankle = p[task]['ankle']['coda'][:, [1, 2]]
    knee_ref = np.ones_like(knee) * knee[0]
    ankle_ref = np.ones_like(ankle) * ankle[0]
    return intersection_angle(line1=[ankle_ref, knee_ref], line2=[ankle, knee])

def _reps(omc, mmc, a):
    _, coda_reps = get_reps(omc, fps=100, plot=False,
                            d=6 if a in [4, 5] else 8, forward=3.7, rewind=3.7)
    _, op_reps = get_reps(mmc, fps=30, plot=False, d=7, forward=3.7, rewind=3.7)
    return coda_reps[-3:], op_reps[-3:]


def get_range_of_motion(folder, subjects, ndims=2):
    folder = folder.rstrip("/")
    rom = {'hir': {}, 'her': {}}
    for a in range(len(subjects)):
        sid = SUBJECT_IDS[a]
        p = subjects[a]
        for task in ('hir', 'her'):
            if (sid, task) in EXCLUDE:
                continue
            fp = _find_file(folder, sid, task)
            if fp is None:
                print(f"[hip] no {task} file for {sid}, skipped")
                continue
            omc = _omc_angle(p, task)
            mmc = _mmc_angle(fp, ndims=ndims)
            coda_reps, op_reps = _reps(omc, mmc, a)
            if len(coda_reps) < 3 or len(op_reps) < 3:
                print(f"[hip] {sid} {task}: {len(coda_reps)}/{len(op_reps)} reps, skipped")
                continue
            for i in range(3):
                coda_rep = coda_reps[i] - min(coda_reps[i])
                op_rep = resample(op_reps[i] - min(op_reps[i]), len(coda_rep))
                rom[task].setdefault(f"coda_{task}_{i+1}", []).append(np.round(max(coda_rep), 2))
                rom[task].setdefault(f"op_{task}_{i+1}", []).append(np.round(max(op_rep), 2))
    return pd.DataFrame(rom['her']), pd.DataFrame(rom['hir'])


def flatten_reps(df, device, task):
    return np.array([df[f'{device}_{task}_{i}'] for i in [1, 2, 3]]).flatten()


def ba_plots(folder, subjects, ndims=2):
    her_df, hir_df = get_range_of_motion(folder, subjects, ndims=ndims)
    hers = {'omc': flatten_reps(her_df, 'coda', 'her'),
            'mmc': flatten_reps(her_df, 'op', 'her')}
    hirs = {'omc': flatten_reps(hir_df, 'coda', 'hir'),
            'mmc': flatten_reps(hir_df, 'op', 'hir')}
    # the her<->hir title swap is intentional (corrects a data-collection flip)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3), dpi=100)
    plot_ba(hers['omc'], hers['mmc'], title='internal rotation (degs)', ax=axs[0])
    plot_ba(hirs['omc'], hirs['mmc'], title='external rotation', ax=axs[1], label_y=False)
    fig.suptitle('Hip rotation'); fig.supxlabel('Means'); plt.show()
    return hers, hirs, her_df, hir_df


def get_icc(x, y, devices=['OMC', 'MMC']):
    df = pd.concat([get_ratings(x, devices[0]), get_ratings(y, devices[1])])
    icc = pg.intraclass_corr(data=df, targets='rep', raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'].iloc[1], 2)


def get_retest(df, device, task):
    L = df.shape[0]
    reps = [pd.DataFrame({'sn': np.arange(1, L + 1), 'score': df[f'{device}_{task}_{i}'],
                          'rater': [f'rep{i}'] * L}) for i in range(1, 4)]
    icc = pg.intraclass_corr(data=pd.concat(reps), targets='sn',
                             raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'].iloc[1], 2)


def get_metrics(folder, subjects, ndims=2):
    hers, hirs, her_df, hir_df = ba_plots(folder, subjects, ndims=ndims)
    her = {'Metric': 'Range of motion', 'Task': 'Hip internal rotation', 'PTM Ref': 'N/A',
           'Ground Truth': 'Optical motion capture',
           'MAE': np.round(MAE(hers['omc'], hers['mmc']), 2),
           'Reliability': get_retest(her_df, 'op', 'her'),
           'ICC': get_icc(hers['omc'], hers['mmc'])}
    hir = {'Metric': 'Range of motion', 'Task': 'Hip external rotation', 'PTM Ref': 'N/A',
           'Ground Truth': 'Optical motion capture',
           'MAE': np.round(MAE(hirs['omc'], hirs['mmc']), 2),
           'Reliability': get_retest(hir_df, 'op', 'hir'),
           'ICC': get_icc(hirs['omc'], hirs['mmc'])}
    metrics_df = pd.DataFrame(her, index=[0]); metrics_df.loc[1] = hir
    return metrics_df