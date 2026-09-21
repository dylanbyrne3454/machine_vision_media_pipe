import os
import re
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
from sklearn.metrics import mean_absolute_error as MAE
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt

from utilities import utils
from utilities.utils import read_list, force_flight_time, jump_heights, flip_axis, \
    get_reps, ts_jump_height, plot_ba
import utilities.PTM as PTM

METRE_HEIGHTS = [1.84, 1.5, 1.725, 1.73, 1.635, 1.67, 1.71, 1.65,
                 1.8, 1.88, 1.62, 1.85, 1.915, 1.78, 1.84, 1.76]
SUBJECT_HEIGHTS = {f"P{i:02d}": h for i, h in zip(range(3, 19), METRE_HEIGHTS)}

POSE_LANDMARKS = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER",
    "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT",
    "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
    "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL",
    "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]
COORDS = ("x", "y", "z")
PER_FRAME = len(POSE_LANDMARKS) * len(COORDS)   # 99

def subject_from_filename(filepath):
    m = re.search(r"P\d{2}", os.path.basename(filepath))
    return m.group(0) if m else None


def load_pose_file(path):
    raw = open(path).read().replace(",", " ")
    frames, cur = [], []
    for tok in raw.split():
        if tok == "-":
            if cur:
                frames.append(cur); cur = []
        else:
            cur.append(float(tok))
    if cur:
        frames.append(cur)
    bad = [(i, len(f)) for i, f in enumerate(frames) if len(f) != PER_FRAME]
    if bad:
        raise ValueError(f"{len(bad)} frame(s) not {PER_FRAME} long: {bad[:5]}")
    arr = np.array(frames).reshape(len(frames), len(POSE_LANDMARKS), len(COORDS))
    return pd.DataFrame({f"{n}_{c}": arr[:, i, j]
                         for i, n in enumerate(POSE_LANDMARKS)
                         for j, c in enumerate(COORDS)})


def to_pixels(df, width, height):
    px = df.copy()
    for c in px.columns:
        if c.endswith("_x"):
            px[c] = px[c] * width
        elif c.endswith("_y"):
            px[c] = px[c] * height
    return px


def pixel_body_height(px, rest_frames):
    def pt(name):
        return np.array([np.median(px[f"{name}_x"][:rest_frames]),
                         np.median(px[f"{name}_y"][:rest_frames])])
    heel = pt("RIGHT_HEEL"); knee = pt("RIGHT_KNEE"); hip = pt("RIGHT_HIP")
    midhip = (pt("LEFT_HIP") + pt("RIGHT_HIP")) / 2
    neck = (pt("LEFT_SHOULDER") + pt("RIGHT_SHOULDER")) / 2
    tibia = np.linalg.norm(heel - knee)
    femur = np.linalg.norm(hip - knee)
    trunk = np.linalg.norm(midhip - neck)
    return (8 * (tibia + femur + trunk)) / 6


def height_scale(px, height_m, rest_frames):
    return (height_m * 1000) / pixel_body_height(px, rest_frames)


def calculate_cmj(filepath, height_m=None, fps=30, width=1920, height=1080,
                  rest_sec=0.5, rewind=1, forward=1, d_sec=5, expected_reps=3):
    if height_m is None:
        subj = subject_from_filename(filepath)
        height_m = SUBJECT_HEIGHTS.get(subj)
        if height_m is None:
            raise ValueError(f"no height for '{subj}' from {os.path.basename(filepath)}")

    df = load_pose_file(filepath)
    px = to_pixels(df, width, height)

    hip = np.column_stack([(px["LEFT_HIP_x"] + px["RIGHT_HIP_x"]) / 2,
                           (px["LEFT_HIP_y"] + px["RIGHT_HIP_y"]) / 2])
    toe = px[["RIGHT_FOOT_INDEX_x", "RIGHT_FOOT_INDEX_y"]].to_numpy()
    hip_y = savgol_filter(flip_axis(hip.copy())[:, 1], 11, 2)
    toe_y = savgol_filter(flip_axis(toe.copy())[:, 1], 11, 2)

    segs, reps = get_reps(hip_y, fps=fps, rewind=rewind, forward=forward,
                          d=d_sec, expected_reps=expected_reps)

    rest_frames = int(rest_sec * fps)
    scale_h = height_scale(px, height_m, rest_frames)

    jumps_mm_unit, scales_g = [], []
    for seg, rep in zip(segs, reps):
        s, e = int(seg[0]), int(seg[1])
        jump = toe_y[s:e].copy(); jump -= jump.min()
        jumps_mm_unit.append(jump)
        try:
            scales_g.append(float(PTM.mm_per_px(rep, fps=fps)))
        except Exception as ex:
            print(f"[calculate_cmj] gravity fit failed on a rep "
                  f"({subject_from_filename(filepath)}): {type(ex).__name__}: {str(ex)[:80]}")
            scales_g.append(np.nan)

    return {"subject": subject_from_filename(filepath), "reps": len(reps),
            "segs": segs, "scale_height": float(scale_h),
            "scales_gravity": scales_g, "_jumps": jumps_mm_unit}


def _mmc_heights_by_subject(files, ptm, clamp_gravity=True, trim_first=None, **kw):

    trim_first = set(trim_first or [])
    records = [calculate_cmj(fp, **kw) for fp in files]
    if ptm == "gravity" and clamp_gravity:
        allg = [g for r in records for g in r["scales_gravity"] if np.isfinite(g)]
        cohort_mean_g = float(np.mean(allg)) if allg else np.nan
    else:
        cohort_mean_g = None

    out = {}
    for r in records:
        hs = []
        for j, jump in enumerate(r["_jumps"]):
            if ptm == "height":
                scale = r["scale_height"]
            else:
                scale = r["scales_gravity"][j]
                if clamp_gravity and np.isfinite(scale) and np.isfinite(cohort_mean_g):
                    scale = max(scale, cohort_mean_g)
            hs.append(np.nan if not np.isfinite(scale)
                      else ts_jump_height(jump * scale, fps=kw.get("fps", 30)))
        out[r["subject"]] = [round(float(x), 2) if np.isfinite(x) else np.nan for x in hs]

    for sid in trim_first:
        if sid in out and len(out[sid]) == 4:
            out[sid] = out[sid][1:]
    return out


def get_jump_heights(cmj_data, ptm="height", clamp_gravity=True, **kw):
    """Heights-only view (no ground truth): one row per file. Kept for quick looks."""
    by_subj = _mmc_heights_by_subject(cmj_data, ptm, clamp_gravity, **kw)
    rows = [{"subject": s, "ptm": ptm, "reps": len(h), "heights": h,
             "mean": round(float(np.nanmean(h)), 2),
             "best": round(float(np.nanmax(h)), 2)} for s, h in by_subj.items()]
    return pd.DataFrame(rows).set_index("subject")

def _fp_heights_by_subject(subjects, subject_ids=None, thresh="auto"):
    """{Pxx: {'bl': [...], 'ul': [...]}} of force-plate heights (cm).

    subject_ids: list aligning pickle index -> 'Pxx'. Defaults to P03.. order.
    """
    if subject_ids is None:
        subject_ids = [f"P{i:02d}" for i in range(3, 3 + len(subjects))]

    force_ft = force_flight_time(subjects, thresh=thresh)   # {'bl':[...], 'ul':[...]}
    out = {}
    for s, sid in enumerate(subject_ids):
        d = {}
        for task in ("bl", "ul"):
            ft = np.asarray(force_ft[task][s], dtype=float) / 1000.0   # ms -> s
            jh = np.round(jump_heights(ft) * 100.0, 2)                 # m  -> cm
            d[task] = jh.tolist()
        out[sid] = d
    return out

def assemble_jumps(bl_files, ul_files, subjects, ptm="height",
                   subject_ids=None, thresh="auto", clamp_gravity=True,
                   trim_first_bl=None, trim_first_ul=None, **kw):
  
    mmc_bl = _mmc_heights_by_subject(bl_files, ptm, clamp_gravity,
                                     trim_first=trim_first_bl, **kw)
    mmc_ul = _mmc_heights_by_subject(ul_files, ptm, clamp_gravity,
                                     trim_first=trim_first_ul, **kw)
    fp = _fp_heights_by_subject(subjects, subject_ids=subject_ids, thresh=thresh)

    def ok(lst):
        return isinstance(lst, list) and len(lst) == 3 and all(np.isfinite(lst))

    rows, dropped = [], []
    for sid in sorted(set(mmc_bl) & set(mmc_ul) & set(fp)):
        ob, ou = mmc_bl[sid], mmc_ul[sid]
        fb, fu = fp[sid]["bl"], fp[sid]["ul"]
        if all(ok(x) for x in (ob, ou, fb, fu)):
            rows.append({"subject": sid, "fp_bl": fb, "op_bl_toe_3": ob,
                         "fp_ul": fu, "op_ul_toe_3": ou})
        else:
            dropped.append(sid)

    present = set(mmc_bl) | set(mmc_ul) | set(fp)
    missing = sorted(present - (set(mmc_bl) & set(mmc_ul) & set(fp)))
    if dropped:
        print(f"[assemble_jumps] dropped (rep count != 3 somewhere): {dropped}")
    if missing:
        print(f"[assemble_jumps] subject not in all three sources: {missing}")

    if not rows:
        print("[assemble_jumps] no subjects survived the join -- returning empty frame. "
              "If ptm='gravity', this usually means the PTM.mm_per_px fit produced NaNs.")
        return pd.DataFrame(columns=["fp_bl", "op_bl_toe_3", "fp_ul", "op_ul_toe_3"])

    return pd.DataFrame(rows).set_index("subject")


def get_all_jumps(df, cols):
    jumps_lst = []
    # pandas >=2.1 removed DataFrame.applymap in favour of .map
    sub = df[cols]
    (sub.map if hasattr(sub, "map") else sub.applymap)(lambda x: jumps_lst.extend(x))
    return np.array(jumps_lst).flatten()


def flatten_jump_reps(df):
    fp_jumps_bl = get_all_jumps(df, cols=['fp_bl'])
    fp_jumps_ul = get_all_jumps(df, cols=['fp_ul'])
    fp_jumps_arr = np.concatenate([fp_jumps_bl, fp_jumps_ul])
    ptm_jumps_arr = get_all_jumps(df, cols=['op_bl_toe_3', 'op_ul_toe_3'])
    bls = ['bilateral' for _ in fp_jumps_bl]
    uls = ['unilateral' for _ in fp_jumps_ul]
    return pd.DataFrame({'task': bls + uls, 'FP': fp_jumps_arr, 'PTM': ptm_jumps_arr})


def ba_plots(df, title):
    ba_df = flatten_jump_reps(df)
    sns.set_style('white')
    fig, axs = plt.subplots(1, 2, figsize=(10, 3), dpi=100)
    for t, task in enumerate(['bilateral', 'unilateral']):
        task_df = ba_df[ba_df['task'] == task]
        plot_ba(task_df['FP'], task_df['PTM'], title=task, ax=axs[t], label_y=t == 0)
    fig.supxlabel('Means')
    fig.suptitle(f'Countermovement jump - {title}')
    fig.tight_layout()
    sns.despine()
    plt.show()


def get_mae(ba_df, task):
    task_df = ba_df[ba_df['task'] == task]
    return np.round(MAE(task_df['FP'], task_df['PTM']), 2)


def get_retest(df, task):
    df = df[[f'op_{task}_toe_3']]
    lst = []
    for item in df.itertuples():
        item = item[1]
        if len(item) > 2:
            lst.append(item)
    three_reps = np.array(lst)
    L, _ = three_reps.shape
    reps = [pd.DataFrame({'jump': np.arange(1, L + 1), 'score': three_reps[:, k],
                          'rater': [f'rep{k+1}'] * L}) for k in range(3)]
    jumps_icc = pd.concat(reps)
    icc = pg.intraclass_corr(data=jumps_icc, targets='jump',
                             raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'].iloc[1], 2)


def get_ratings(ba_df, task):
    task_df = ba_df[ba_df['task'] == task]
    L = len(task_df) + 1
    fp_ratings = pd.DataFrame({'jump': np.arange(1, L), 'score': task_df['FP'].to_numpy(),
                               'rater': ['FP'] * (L - 1)})
    ptm_ratings = pd.DataFrame({'jump': np.arange(1, L), 'score': task_df['PTM'].to_numpy(),
                                'rater': ['PTM'] * (L - 1)})
    return fp_ratings, ptm_ratings


def get_icc(ba_df, task):
    r1, r2 = get_ratings(ba_df, task)
    jumps_icc = pd.concat([r1, r2])
    icc = pg.intraclass_corr(data=jumps_icc, targets='jump',
                             raters='rater', ratings='score')
    return np.round(icc.set_index('Type')['ICC'].iloc[1], 2)


def get_metrics(df, ptm):
    ba_df = flatten_jump_reps(df)
    bl = {'Metric': 'Jump height', 'Task': 'CMJ bilateral', 'PTM Ref': ptm,
          'Ground Truth': 'Force plates', 'MAE': get_mae(ba_df, 'bilateral'),
          'Reliability': get_retest(df, 'bl'), 'ICC': get_icc(ba_df, 'bilateral')}
    ul = {'Metric': 'Jump height', 'Task': 'CMJ unilateral', 'PTM Ref': ptm,
          'Ground Truth': 'Force plates', 'MAE': get_mae(ba_df, 'unilateral'),
          'Reliability': get_retest(df, 'ul'), 'ICC': get_icc(ba_df, 'unilateral')}
    metrics_df = pd.DataFrame(bl, index=[0])
    metrics_df.loc[1] = ul
    return metrics_df