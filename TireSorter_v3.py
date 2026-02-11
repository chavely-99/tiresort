"""
TRK Tire Sorter v3.0
Trackhouse Racing — Tire Set Optimization Tool
Multi-start Simulated Annealing optimizer with interactive results.
"""

import math
import random
from collections import Counter
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit_sortables import sort_items
    HAS_SORTABLES = True
except ImportError:
    HAS_SORTABLES = False

# ============================================================
# CONSTANTS
# ============================================================

APP_TITLE = "TRK Tire Sorter v3"
N_RESTARTS = 8
N_ITERATIONS = 100_000
T_START = 50.0
T_END = 0.01
INTER_SET_PROB = 0.70
STAGGER_WEIGHT = 1_000_000.0
PRIORITY_WEIGHTS = [10_000.0, 1_000.0, 100.0, 10.0, 1.0]
# Quick variant runs for pre-computed solutions
QUICK_RESTARTS = 4
QUICK_ITERATIONS = 50_000
PRIORITY_OPTIONS = ["Cross Weight", "Common RR Rollout", "Soft Rear", "Shift Match", "Date Match"]
PRIORITY_KEYS = {
    "Cross Weight": "cross_weight",
    "Common RR Rollout": "rr_common",
    "Soft Rear": "soft_rear",
    "Shift Match": "shift",
    "Date Match": "date",
}
# Short labels for variant buttons (keeps buttons single-line)
_BTN_LABELS = {"Common RR Rollout": "RR"}

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="\U0001F3CE",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
/* Tighten up default Streamlit padding */
.block-container { padding-top: 1.25rem; padding-bottom: 0.5rem; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
header[data-testid="stHeader"] { height: 1rem; }
.stTabs [data-baseweb="tab-list"] { gap: 2px; margin-bottom: 0.25rem; }

/* Set card */
.set-card {
    border: 2px solid #bbb;
    border-radius: 8px;
    padding: 8px;
    margin-bottom: 6px;
    background: #fafafa;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.set-header {
    text-align: center;
    font-weight: 800;
    font-size: 15px;
    color: #222;
    margin-bottom: 3px;
}

/* Metrics bar within each set card */
.metrics-bar {
    display: flex;
    justify-content: space-around;
    padding: 3px 6px;
    margin-bottom: 4px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
}
.metrics-bar.good { background: #c8e6c9; border: 1px solid #66bb6a; }
.metrics-bar.ok   { background: #fff9c4; border: 1px solid #ffee58; }
.metrics-bar.bad  { background: #ffcdd2; border: 1px solid #ef5350; }
.metrics-bar span { white-space: nowrap; }

/* 2x2 tire grid */
.car-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px;
}

/* Individual tire cell */
.tire-cell {
    border-radius: 4px;
    padding: 5px 7px;
    line-height: 1.25;
}
.tire-cell.left-side {
    background: #f0e6ff;
    border-left: 4px solid #7c3aed;
}
.tire-cell.right-side {
    background: #e6f0ff;
    border-left: 4px solid #2563eb;
}
.tire-num {
    font-weight: 800; font-size: 20px; color: #1e293b;
    margin-bottom: 2px;
    border-bottom: 1px solid rgba(0,0,0,0.08);
    padding-bottom: 1px;
}
.tire-stats {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 0;
}
.tire-stat {
    text-align: center;
    line-height: 1.3;
}
.tire-stat .label {
    display: block;
    font-size: 11px;
    color: #94a3b8;
    font-weight: 500;
}
.tire-stat .val {
    display: block;
    font-size: 14px;
    color: #1e293b;
    font-weight: 700;
}

/* Summary metric tiles */
.summary-tile {
    text-align: center;
    padding: 12px 8px;
    border-radius: 8px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}
.summary-tile .val {
    font-size: 26px; font-weight: 800; color: #1e293b;
}
.summary-tile .lbl {
    font-size: 11px; color: #64748b; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px;
}

/* Priority list styling */
.priority-fixed {
    padding: 8px 12px;
    border-radius: 6px;
    background: #fee2e2;
    border: 2px solid #ef4444;
    font-weight: 700;
    margin-bottom: 6px;
    text-align: center;
    font-size: 14px;
    color: #991b1b;
}

/* Raw tire data table */
.tire-data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
.tire-data-table thead th {
    background: #f1f5f9;
    color: #334155;
    font-weight: 700;
    padding: 8px 10px;
    border-bottom: 2px solid #cbd5e1;
    text-align: left;
    position: sticky;
    top: 0;
}
.tire-data-table tbody td {
    padding: 6px 10px;
    border-bottom: 1px solid #e2e8f0;
    color: #475569;
}
.tire-data-table tbody tr:hover {
    background: #f8fafc;
}
.tire-data-table tbody tr:nth-child(even) {
    background: #fafbfc;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE DEFAULTS
# ============================================================

_SS_DEFAULTS = {
    "tire_df": None,
    "lefts": None,
    "rights": None,
    "ls_dcode": None,
    "rs_dcode": None,
    "stagger_info": None,
    "target_stagger": None,
    "priority_order": list(PRIORITY_OPTIONS),
    "left_perm": None,
    "right_perm": None,
    "set_metrics": None,
    "solution_score": None,
    "mode_rr": 0.0,
    "n_sets": 0,
    "data_loaded": False,
    "_upload_token": None,
    "variant_solutions": {},   # {"Original Sort": {...}, "Optimal Cross": {...}, ...}
    "active_variant": "Original Sort",
}

for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ============================================================
# DATA LOADING
# ============================================================

def load_scan_data(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """
    Parse 'Scan Data' sheet.  Returns (tires_df, ls_dcode, rs_dcode).
    ls_dcode / rs_dcode may be None if not found in the header rows.
    """
    try:
        raw = pd.read_excel(uploaded_file, sheet_name="Scan Data",
                            header=None, nrows=4, engine="openpyxl")
    except Exception as e:
        st.error(f"Could not read 'Scan Data' sheet: {e}")
        return None, None, None

    # Extract embedded LS / RS codes from rows 1-2, column D (index 3)
    ls_code = None
    rs_code = None
    try:
        ls_val = raw.iloc[1, 3]
        rs_val = raw.iloc[2, 3]
        if pd.notna(ls_val):
            ls_code = str(int(float(ls_val)))
        if pd.notna(rs_val):
            rs_code = str(int(float(rs_val)))
    except Exception:
        pass

    # Read full data with header at row 3
    uploaded_file.seek(0)
    try:
        df = pd.read_excel(uploaded_file, sheet_name="Scan Data",
                           header=3, engine="openpyxl")
    except Exception as e:
        st.error(f"Error reading data: {e}")
        return None, None, None

    # Find required columns
    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "d-code":
            col_map["dcode"] = c
        elif cl == "spring rate":
            col_map["spring_rate"] = c
        elif cl == "shift" and "shift" not in col_map:
            col_map["shift"] = c
        elif cl == "date" and "date" not in col_map:
            col_map["date"] = c
        elif cl == "size" and "size" not in col_map:
            col_map["size"] = c

    # Tire ID: second 'Number' column → pandas names it 'Number.1'
    if "Number.1" in df.columns:
        col_map["tire_id"] = "Number.1"
    elif "Number" in df.columns:
        col_map["tire_id"] = "Number"

    required = ["dcode", "spring_rate", "size", "tire_id"]
    missing = [k for k in required if k not in col_map]
    if missing:
        st.error(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")
        return None, None, None

    # Build clean DataFrame
    tires = pd.DataFrame()
    tires["Tire_ID"] = pd.to_numeric(df[col_map["tire_id"]], errors="coerce")
    tires["D-Code"] = df[col_map["dcode"]].astype(str).str.strip().str.replace(".0", "", regex=False)
    tires["Spring Rate"] = pd.to_numeric(df[col_map["spring_rate"]], errors="coerce")
    tires["Size"] = pd.to_numeric(df[col_map["size"]], errors="coerce")

    if "shift" in col_map:
        tires["Shift"] = df[col_map["shift"]].astype(str).str.strip()
    else:
        tires["Shift"] = ""

    if "date" in col_map:
        tires["Date_Raw"] = df[col_map["date"]]
        tires["Date"] = tires["Date_Raw"].apply(_format_date)
    else:
        tires["Date_Raw"] = 0
        tires["Date"] = ""

    # Drop rows missing critical data
    before = len(tires)
    tires = tires.dropna(subset=["Tire_ID", "D-Code", "Spring Rate", "Size"])
    tires = tires[tires["D-Code"] != "nan"].copy()
    tires["Tire_ID"] = tires["Tire_ID"].astype(float).astype(int).astype(str)
    tires = tires.reset_index(drop=True)
    dropped = before - len(tires)
    if dropped > 0:
        st.info(f"Dropped {dropped} rows with missing data. {len(tires)} valid tires loaded.")

    return tires, ls_code, rs_code


def _format_date(val) -> str:
    """Convert MMDD integer (e.g. 925 → '9/25', 1025 → '10/25') or pass through."""
    if pd.isna(val):
        return ""
    try:
        v = int(float(val))
        if 100 <= v <= 1231:
            month = v // 100
            day = v % 100
            if 1 <= month <= 12 and 1 <= day <= 31:
                return f"{month}/{day}"
        return str(v)
    except (ValueError, TypeError):
        return str(val)


def _date_numeric(val) -> int:
    """Convert date raw value to integer for spread computation."""
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


# ============================================================
# STAGGER ANALYSIS
# ============================================================

def _max_matching(adj: Dict[int, List[int]]) -> int:
    """Maximum bipartite matching via augmenting paths."""
    match_right: Dict[int, int] = {}

    def _augment(u: int, visited: set) -> bool:
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                if v not in match_right or _augment(match_right[v], visited):
                    match_right[v] = u
                    return True
        return False

    for u in adj:
        _augment(u, set())
    return len(match_right)


def analyze_stagger(lefts: pd.DataFrame, rights: pd.DataFrame) -> dict:
    """
    Determine achievable stagger range across all sets.
    Returns dict with min_achievable, max_achievable, n_sets, etc.
    """
    n_left = len(lefts)
    n_right = len(rights)
    n_sets = min(n_left, n_right) // 2

    if n_sets == 0:
        return {"min_achievable": 0, "max_achievable": 0, "n_sets": 0,
                "n_lefts": n_left, "n_rights": n_right}

    left_sizes = lefts["Size"].values
    right_sizes = rights["Size"].values

    # Compute possible stagger range
    min_possible = float(right_sizes.min() - left_sizes.max())
    max_possible = float(right_sizes.max() - left_sizes.min())

    if min_possible < 0:
        min_possible = 0.0

    # Sweep in 1mm steps, check feasibility with EXACT matching (tolerance=0)
    # This finds stagger values where ALL sets can hit the same target exactly.
    step = 1.0
    candidates = np.arange(min_possible, max_possible + step, step)

    achievable_exact = []
    for target in candidates:
        if _check_feasibility(left_sizes, right_sizes, target, n_sets, tolerance=0.0):
            achievable_exact.append(float(target))

    # Also check with small tolerance for a wider usable range
    achievable_loose = []
    for target in candidates:
        if _check_feasibility(left_sizes, right_sizes, target, n_sets, tolerance=1.0):
            achievable_loose.append(float(target))

    # Default to max exact achievable (all sets at same stagger)
    # Fall back to loose range if no exact matches found
    if achievable_exact:
        min_ach = min(achievable_exact)
        max_ach = max(achievable_exact)
    elif achievable_loose:
        min_ach = min(achievable_loose)
        max_ach = max(achievable_loose)
    else:
        min_ach = float(min_possible)
        max_ach = float(max_possible)

    return {
        "min_achievable": min_ach,
        "max_achievable": max_ach,
        "n_sets": n_sets,
        "n_lefts": n_left,
        "n_rights": n_right,
        "unused_lefts": n_left - n_sets * 2,
        "unused_rights": n_right - n_sets * 2,
    }


def _check_feasibility(left_sizes, right_sizes, target, n_sets, tolerance):
    """Check if target stagger is feasible via bipartite matching."""
    adj: Dict[int, List[int]] = {}
    for i, ls in enumerate(left_sizes):
        compat = []
        for j, rs in enumerate(right_sizes):
            if abs((rs - ls) - target) <= tolerance:
                compat.append(j)
        if compat:
            adj[i] = compat
    return _max_matching(adj) >= n_sets


# ============================================================
# SCORING & METRICS
# ============================================================

def _reorder_sets_by_rr(lp, rp, metrics, n_sets):
    """Reorder sets so RR rollouts are grouped, least common rollout first."""
    rr_sizes = [m["rr_size"] for m in metrics]
    size_counts = Counter(rr_sizes)
    # Sort: least common RR rollout first, then by rollout value, then original index
    order = sorted(range(n_sets),
                   key=lambda i: (size_counts[rr_sizes[i]], rr_sizes[i], i))

    new_lp = lp.copy()
    new_rp = rp.copy()
    new_metrics = [None] * n_sets
    for new_idx, old_idx in enumerate(order):
        new_lp[2 * new_idx] = lp[2 * old_idx]
        new_lp[2 * new_idx + 1] = lp[2 * old_idx + 1]
        new_rp[2 * new_idx] = rp[2 * old_idx]
        new_rp[2 * new_idx + 1] = rp[2 * old_idx + 1]
        new_metrics[new_idx] = metrics[old_idx]
    return new_lp, new_rp, new_metrics


def _reorder_sets_by_sr(lp, rp, metrics, n_sets):
    """Reorder sets by average spring rate, softest first."""
    order = sorted(range(n_sets), key=lambda i: metrics[i].get("avg_sr", 0))
    new_lp = lp.copy()
    new_rp = rp.copy()
    new_metrics = [None] * n_sets
    for new_idx, old_idx in enumerate(order):
        new_lp[2 * new_idx] = lp[2 * old_idx]
        new_lp[2 * new_idx + 1] = lp[2 * old_idx + 1]
        new_rp[2 * new_idx] = rp[2 * old_idx]
        new_rp[2 * new_idx + 1] = rp[2 * old_idx + 1]
        new_metrics[new_idx] = metrics[old_idx]
    return new_lp, new_rp, new_metrics


def _extract_arrays(lefts: pd.DataFrame, rights: pd.DataFrame):
    """Pre-extract numpy arrays from DataFrames for fast SA inner loop."""
    l_sizes = lefts["Size"].values.astype(np.float64)
    l_springs = lefts["Spring Rate"].values.astype(np.float64)
    r_sizes = rights["Size"].values.astype(np.float64)
    r_springs = rights["Spring Rate"].values.astype(np.float64)

    # Encode shifts as integers for fast comparison
    all_shifts = list(set(lefts["Shift"].tolist() + rights["Shift"].tolist()) - {"", "nan"})
    shift_map = {s: i + 1 for i, s in enumerate(all_shifts)}
    shift_map[""] = 0
    shift_map["nan"] = 0

    l_shifts = np.array([shift_map.get(str(s), 0) for s in lefts["Shift"]], dtype=np.int32)
    r_shifts = np.array([shift_map.get(str(s), 0) for s in rights["Shift"]], dtype=np.int32)

    l_dates = np.array([_date_numeric(v) for v in lefts["Date_Raw"]], dtype=np.int32)
    r_dates = np.array([_date_numeric(v) for v in rights["Date_Raw"]], dtype=np.int32)

    return l_sizes, l_springs, l_shifts, l_dates, r_sizes, r_springs, r_shifts, r_dates


def _fast_set_metrics(lp, rp, si, l_sz, l_sr, l_sh, l_dt, r_sz, r_sr, r_sh, r_dt, mode_rr=0.0):
    """Compute metrics for set si using pre-extracted arrays.
    Returns (stag, cross, cross_dev, shift_count, date_spread, rr_dev, sr_dev)."""
    lf_i = lp[2 * si]
    lr_i = lp[2 * si + 1]
    rf_i = rp[2 * si]
    rr_i = rp[2 * si + 1]

    rr_size = r_sz[rr_i]
    stag = rr_size - l_sz[lr_i]
    total_sr = l_sr[lf_i] + r_sr[rf_i] + l_sr[lr_i] + r_sr[rr_i]
    cross = (r_sr[rf_i] + l_sr[lr_i]) / total_sr * 100.0 if total_sr else 50.0
    cross_dev = abs(cross - 50.0)

    # RR rollout deviation from mode (for common RR rollout priority)
    rr_dev = abs(rr_size - mode_rr)

    # Soft rear: penalty when rear avg SR is stiffer than front avg SR
    avg_rear_sr = (l_sr[lr_i] + r_sr[rr_i]) / 2.0
    avg_front_sr = (l_sr[lf_i] + r_sr[rf_i]) / 2.0
    sr_dev = max(0.0, avg_rear_sr - avg_front_sr)

    # Shift count: unique non-zero shifts
    shifts = set()
    for sv in (l_sh[lf_i], l_sh[lr_i], r_sh[rf_i], r_sh[rr_i]):
        if sv > 0:
            shifts.add(sv)
    shift_count = max(len(shifts), 1)

    # Date spread
    dates = []
    for dv in (l_dt[lf_i], l_dt[lr_i], r_dt[rf_i], r_dt[rr_i]):
        if dv > 0:
            dates.append(dv)
    date_spread = (max(dates) - min(dates)) if len(dates) >= 2 else 0

    return stag, cross, cross_dev, shift_count, date_spread, rr_dev, sr_dev


def _fast_set_score(stag, cross_dev, shift_count, date_spread, rr_dev, sr_dev,
                     target, w_cross, w_shift, w_date, w_rr, w_sr):
    """Fast inline score for one set."""
    return (STAGGER_WEIGHT * abs(stag - target)
            + w_cross * cross_dev
            + w_rr * rr_dev
            + w_sr * sr_dev
            + w_shift * (shift_count - 1)
            + w_date * date_spread)


def compute_set_metrics(lf, rf, lr, rr, mode_rr=0.0) -> dict:
    """Compute all metrics for one 4-tire set. Inputs are DataFrame rows."""
    rr_size = rr["Size"]
    stagger = rr_size - lr["Size"]
    total_sr = lf["Spring Rate"] + rf["Spring Rate"] + lr["Spring Rate"] + rr["Spring Rate"]
    cross_weight = (rf["Spring Rate"] + lr["Spring Rate"]) / total_sr * 100 if total_sr else 50.0

    shifts = {lf["Shift"], rf["Shift"], lr["Shift"], rr["Shift"]}
    shifts.discard("")
    shifts.discard("nan")
    shift_count = max(len(shifts), 1)

    dates = [_date_numeric(lf["Date_Raw"]), _date_numeric(rf["Date_Raw"]),
             _date_numeric(lr["Date_Raw"]), _date_numeric(rr["Date_Raw"])]
    valid_dates = [d for d in dates if d > 0]
    date_spread = (max(valid_dates) - min(valid_dates)) if len(valid_dates) >= 2 else 0

    # Soft rear: penalty when rear avg SR is stiffer than front
    avg_rear_sr = (lr["Spring Rate"] + rr["Spring Rate"]) / 2.0
    avg_front_sr = (lf["Spring Rate"] + rf["Spring Rate"]) / 2.0
    soft_rear_dev = max(0.0, avg_rear_sr - avg_front_sr)
    avg_sr = total_sr / 4.0

    return {
        "stagger": stagger,
        "cross_weight": cross_weight,
        "cross_dev": abs(cross_weight - 50.0),
        "shift_count": shift_count,
        "date_spread": date_spread,
        "rr_dev": abs(rr_size - mode_rr),
        "rr_size": rr_size,
        "soft_rear_dev": soft_rear_dev,
        "avg_sr": avg_sr,
    }


def _set_score(metrics: dict, target_stagger: float, weights: dict) -> float:
    """Weighted score for a single set. Lower is better."""
    s = STAGGER_WEIGHT * abs(metrics["stagger"] - target_stagger)
    s += weights.get("cross_weight", 0) * metrics["cross_dev"]
    s += weights.get("rr_common", 0) * metrics.get("rr_dev", 0)
    s += weights.get("soft_rear", 0) * metrics.get("soft_rear_dev", 0)
    s += weights.get("shift", 0) * (metrics["shift_count"] - 1)
    s += weights.get("date", 0) * metrics["date_spread"]
    return s


def _compute_from_perms(left_perm, right_perm, lefts, rights, set_idx, mode_rr=0.0):
    """Compute set metrics for set_idx from permutation arrays."""
    lf = lefts.iloc[left_perm[2 * set_idx]]
    lr = lefts.iloc[left_perm[2 * set_idx + 1]]
    rf = rights.iloc[right_perm[2 * set_idx]]
    rr = rights.iloc[right_perm[2 * set_idx + 1]]
    return compute_set_metrics(lf, rf, lr, rr, mode_rr=mode_rr)


def _full_score(left_perm, right_perm, lefts, rights, n_sets,
                target_stagger, weights, mode_rr=0.0):
    """Full solution score + list of per-set metrics."""
    metrics_list = []
    total = 0.0
    for s in range(n_sets):
        m = _compute_from_perms(left_perm, right_perm, lefts, rights, s, mode_rr=mode_rr)
        metrics_list.append(m)
        total += _set_score(m, target_stagger, weights)
    return total, metrics_list


# ============================================================
# OPTIMIZER: MULTI-START SIMULATED ANNEALING
# ============================================================

def build_weights(priority_order: List[str]) -> dict:
    """Convert user-ranked priority names to weight dict."""
    w = {}
    for i, name in enumerate(priority_order):
        key = PRIORITY_KEYS.get(name)
        if key and i < len(PRIORITY_WEIGHTS):
            w[key] = PRIORITY_WEIGHTS[i]
    return w


def _greedy_match(left_sizes, right_sizes, target, n_pairs):
    """Greedy minimum-cost matching for LR-RR stagger pairs. Returns (lr_indices, rr_indices)."""
    # Build all possible pairs sorted by cost
    pairs = []
    for li in range(len(left_sizes)):
        for ri in range(len(right_sizes)):
            cost = abs((right_sizes[ri] - left_sizes[li]) - target)
            pairs.append((cost, li, ri))
    pairs.sort(key=lambda x: x[0])

    used_left = set()
    used_right = set()
    lr_indices = []
    rr_indices = []
    for cost, li, ri in pairs:
        if li not in used_left and ri not in used_right:
            lr_indices.append(li)
            rr_indices.append(ri)
            used_left.add(li)
            used_right.add(ri)
            if len(lr_indices) >= n_pairs:
                break
    return lr_indices, rr_indices


def _smart_init(lefts, rights, n_sets, target_stagger, rng):
    """Create an initial solution using greedy matching for stagger."""
    n_left = len(lefts)
    n_right = len(rights)
    left_sizes = lefts["Size"].values
    right_sizes = rights["Size"].values

    # Greedy match n_sets LR-RR pairs closest to target stagger
    lr_indices, rr_indices = _greedy_match(left_sizes, right_sizes,
                                            target_stagger, n_sets)

    # Remaining tires become front (LF, RF)
    used_left = set(lr_indices)
    used_right = set(rr_indices)
    remaining_left = [i for i in range(n_left) if i not in used_left]
    remaining_right = [i for i in range(n_right) if i not in used_right]
    rng.shuffle(remaining_left)
    rng.shuffle(remaining_right)

    # Build permutation arrays:
    # Set i: left_perm[2i]=LF, left_perm[2i+1]=LR, right_perm[2i]=RF, right_perm[2i+1]=RR
    left_perm = np.zeros(n_left, dtype=int)
    right_perm = np.zeros(n_right, dtype=int)

    for k in range(n_sets):
        left_perm[2 * k] = remaining_left[k] if k < len(remaining_left) else lr_indices[k]
        left_perm[2 * k + 1] = lr_indices[k]
        right_perm[2 * k] = remaining_right[k] if k < len(remaining_right) else rr_indices[k]
        right_perm[2 * k + 1] = rr_indices[k]

    # Fill remaining slots (unused tires beyond 2*n_sets)
    for i in range(n_sets, len(remaining_left)):
        pos = 2 * n_sets + (i - n_sets)
        if pos < n_left:
            left_perm[pos] = remaining_left[i]
    for i in range(n_sets, len(remaining_right)):
        pos = 2 * n_sets + (i - n_sets)
        if pos < n_right:
            right_perm[pos] = remaining_right[i]

    return left_perm, right_perm


def run_optimizer(lefts, rights, n_sets, target_stagger, priority_order,
                  progress_callback=None, n_restarts=None, n_iterations=None,
                  _precomputed_arrays=None):
    """Multi-start SA optimizer using pre-extracted arrays for speed.
    Returns (left_perm, right_perm, score, metrics_list, rr_mode)."""
    if n_restarts is None:
        n_restarts = N_RESTARTS
    if n_iterations is None:
        n_iterations = N_ITERATIONS

    weights = build_weights(priority_order)
    w_cross = weights.get("cross_weight", 0.0)
    w_rr = weights.get("rr_common", 0.0)
    w_sr = weights.get("soft_rear", 0.0)
    w_shift = weights.get("shift", 0.0)
    w_date = weights.get("date", 0.0)
    n_left = len(lefts)
    n_right = len(rights)

    # Pre-extract arrays for fast inner loop (avoids DataFrame .iloc overhead)
    if _precomputed_arrays is not None:
        l_sz, l_sr, l_sh, l_dt, r_sz, r_sr, r_sh, r_dt = _precomputed_arrays
    else:
        l_sz, l_sr, l_sh, l_dt, r_sz, r_sr, r_sh, r_dt = _extract_arrays(lefts, rights)

    # Mode RR rollout: most common right-side size in the pool
    rr_mode = Counter(r_sz).most_common(1)[0][0]

    best_score = float("inf")
    best_lp = None
    best_rp = None
    best_metrics = None

    ns2 = 2 * n_sets  # pre-compute

    for restart in range(n_restarts):
        rng = np.random.default_rng(seed=restart * 42 + 7)

        lp, rp = _smart_init(lefts, rights, n_sets, target_stagger, rng)

        # Compute initial scores using fast arrays
        cur_scores = np.zeros(n_sets)
        cur_stags = np.zeros(n_sets)
        cur_cross = np.zeros(n_sets)
        cur_cdev = np.zeros(n_sets)
        cur_shcnt = np.zeros(n_sets, dtype=int)
        cur_dtspd = np.zeros(n_sets, dtype=int)
        cur_rrdev = np.zeros(n_sets)
        cur_srdev = np.zeros(n_sets)
        for s in range(n_sets):
            stag, cross, cdev, shcnt, dtspd, rrdev, srdev = _fast_set_metrics(
                lp, rp, s, l_sz, l_sr, l_sh, l_dt, r_sz, r_sr, r_sh, r_dt, mode_rr=rr_mode)
            cur_stags[s] = stag
            cur_cross[s] = cross
            cur_cdev[s] = cdev
            cur_shcnt[s] = shcnt
            cur_dtspd[s] = dtspd
            cur_rrdev[s] = rrdev
            cur_srdev[s] = srdev
            cur_scores[s] = _fast_set_score(stag, cdev, shcnt, dtspd, rrdev, srdev,
                                             target_stagger, w_cross, w_shift, w_date, w_rr, w_sr)
        cur_total = cur_scores.sum()

        loc_best = cur_total
        loc_best_lp = lp.copy()
        loc_best_rp = rp.copy()
        loc_best_stags = cur_stags.copy()
        loc_best_cross = cur_cross.copy()
        loc_best_cdev = cur_cdev.copy()
        loc_best_shcnt = cur_shcnt.copy()
        loc_best_dtspd = cur_dtspd.copy()
        loc_best_rrdev = cur_rrdev.copy()
        loc_best_srdev = cur_srdev.copy()

        for it in range(n_iterations):
            temp = T_START * (1.0 - it / n_iterations) + T_END

            # Choose move type
            if rng.random() < INTER_SET_PROB:
                perm = lp if rng.random() < 0.5 else rp
                a = int(rng.integers(0, ns2))
                b = int(rng.integers(0, ns2))
                while a // 2 == b // 2:
                    b = int(rng.integers(0, ns2))
            else:
                perm = lp if rng.random() < 0.5 else rp
                s_idx = int(rng.integers(0, n_sets))
                a = 2 * s_idx
                b = 2 * s_idx + 1

            set_a = a // 2
            set_b = b // 2

            # Perform swap
            perm[a], perm[b] = perm[b], perm[a]

            # Recompute affected sets
            stag_a, cross_a, cdev_a, shcnt_a, dtspd_a, rrdev_a, srdev_a = _fast_set_metrics(
                lp, rp, set_a, l_sz, l_sr, l_sh, l_dt, r_sz, r_sr, r_sh, r_dt, mode_rr=rr_mode)
            new_sc_a = _fast_set_score(stag_a, cdev_a, shcnt_a, dtspd_a, rrdev_a, srdev_a,
                                        target_stagger, w_cross, w_shift, w_date, w_rr, w_sr)
            delta = new_sc_a - cur_scores[set_a]

            if set_a != set_b:
                stag_b, cross_b, cdev_b, shcnt_b, dtspd_b, rrdev_b, srdev_b = _fast_set_metrics(
                    lp, rp, set_b, l_sz, l_sr, l_sh, l_dt, r_sz, r_sr, r_sh, r_dt, mode_rr=rr_mode)
                new_sc_b = _fast_set_score(stag_b, cdev_b, shcnt_b, dtspd_b, rrdev_b, srdev_b,
                                            target_stagger, w_cross, w_shift, w_date, w_rr, w_sr)
                delta += new_sc_b - cur_scores[set_b]

            # Accept or reject
            if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
                cur_total += delta
                cur_scores[set_a] = new_sc_a
                cur_stags[set_a] = stag_a
                cur_cross[set_a] = cross_a
                cur_cdev[set_a] = cdev_a
                cur_shcnt[set_a] = shcnt_a
                cur_dtspd[set_a] = dtspd_a
                cur_rrdev[set_a] = rrdev_a
                cur_srdev[set_a] = srdev_a
                if set_a != set_b:
                    cur_scores[set_b] = new_sc_b
                    cur_stags[set_b] = stag_b
                    cur_cross[set_b] = cross_b
                    cur_cdev[set_b] = cdev_b
                    cur_shcnt[set_b] = shcnt_b
                    cur_dtspd[set_b] = dtspd_b
                    cur_rrdev[set_b] = rrdev_b
                    cur_srdev[set_b] = srdev_b

                if cur_total < loc_best:
                    loc_best = cur_total
                    loc_best_lp = lp.copy()
                    loc_best_rp = rp.copy()
                    loc_best_stags = cur_stags.copy()
                    loc_best_cross = cur_cross.copy()
                    loc_best_cdev = cur_cdev.copy()
                    loc_best_shcnt = cur_shcnt.copy()
                    loc_best_dtspd = cur_dtspd.copy()
                    loc_best_rrdev = cur_rrdev.copy()
                    loc_best_srdev = cur_srdev.copy()
            else:
                perm[a], perm[b] = perm[b], perm[a]

        if loc_best < best_score:
            best_score = loc_best
            best_lp = loc_best_lp.copy()
            best_rp = loc_best_rp.copy()
            best_metrics = []
            for s in range(n_sets):
                lf_i = loc_best_lp[2 * s]
                lr_i = loc_best_lp[2 * s + 1]
                rf_i = loc_best_rp[2 * s]
                rr_i = loc_best_rp[2 * s + 1]
                avg_sr = (l_sr[lf_i] + r_sr[rf_i] + l_sr[lr_i] + r_sr[rr_i]) / 4.0
                best_metrics.append({
                    "stagger": float(loc_best_stags[s]),
                    "cross_weight": float(loc_best_cross[s]),
                    "cross_dev": float(loc_best_cdev[s]),
                    "shift_count": int(loc_best_shcnt[s]),
                    "date_spread": int(loc_best_dtspd[s]),
                    "rr_dev": float(loc_best_rrdev[s]),
                    "rr_size": float(r_sz[loc_best_rp[2 * s + 1]]),
                    "soft_rear_dev": float(loc_best_srdev[s]),
                    "avg_sr": float(avg_sr),
                })

        if progress_callback:
            progress_callback((restart + 1) / n_restarts)

    return best_lp, best_rp, best_score, best_metrics, rr_mode


# ============================================================
# MANUAL SWAP
# ============================================================

def perform_swap(set_a: int, pos_a: str, set_b: int, pos_b: str):
    """Swap two tires in session_state solution. Recomputes affected metrics."""
    lp = st.session_state.left_perm
    rp = st.session_state.right_perm
    lefts = st.session_state.lefts
    rights = st.session_state.rights
    n_sets = st.session_state.n_sets
    target = st.session_state.target_stagger
    weights = build_weights(st.session_state.priority_order)

    pos_to_idx = {"LF": 0, "LR": 1, "RF": 0, "RR": 1}

    if pos_a in ("LF", "LR"):
        perm = lp
    else:
        perm = rp

    idx_a = 2 * set_a + pos_to_idx[pos_a]
    idx_b = 2 * set_b + pos_to_idx[pos_b]

    perm[idx_a], perm[idx_b] = perm[idx_b], perm[idx_a]

    # Recompute affected sets
    mode_rr = st.session_state.mode_rr
    metrics = st.session_state.set_metrics
    metrics[set_a] = _compute_from_perms(lp, rp, lefts, rights, set_a, mode_rr=mode_rr)
    if set_a != set_b:
        metrics[set_b] = _compute_from_perms(lp, rp, lefts, rights, set_b, mode_rr=mode_rr)

    st.session_state.set_metrics = metrics
    st.session_state.left_perm = lp
    st.session_state.right_perm = rp

    # Recompute total score
    total = sum(_set_score(m, target, weights) for m in metrics)
    st.session_state.solution_score = total


def perform_set_swap(set_a: int, set_b: int):
    """Swap all 4 tires between two sets (renumber them). Metrics just swap too."""
    lp = st.session_state.left_perm
    rp = st.session_state.right_perm
    metrics = st.session_state.set_metrics

    # Swap both left positions (LF, LR)
    for offset in (0, 1):
        ia = 2 * set_a + offset
        ib = 2 * set_b + offset
        lp[ia], lp[ib] = lp[ib], lp[ia]

    # Swap both right positions (RF, RR)
    for offset in (0, 1):
        ia = 2 * set_a + offset
        ib = 2 * set_b + offset
        rp[ia], rp[ib] = rp[ib], rp[ia]

    # Swap metrics
    metrics[set_a], metrics[set_b] = metrics[set_b], metrics[set_a]

    st.session_state.left_perm = lp
    st.session_state.right_perm = rp
    st.session_state.set_metrics = metrics


# ============================================================
# EXPORT
# ============================================================

def build_export_csv() -> str:
    """Build CSV string of the current solution."""
    lp = st.session_state.left_perm
    rp = st.session_state.right_perm
    lefts = st.session_state.lefts
    rights = st.session_state.rights
    n_sets = st.session_state.n_sets
    metrics = st.session_state.set_metrics

    rows = []
    for s in range(n_sets):
        lf = lefts.iloc[lp[2 * s]]
        lr = lefts.iloc[lp[2 * s + 1]]
        rf = rights.iloc[rp[2 * s]]
        rr = rights.iloc[rp[2 * s + 1]]
        m = metrics[s]

        for pos, tire in [("LF", lf), ("RF", rf), ("LR", lr), ("RR", rr)]:
            rows.append({
                "Set": s + 1,
                "Position": pos,
                "Tire #": tire["Tire_ID"],
                "D-Code": tire["D-Code"],
                "Size": tire["Size"],
                "Spring Rate": tire["Spring Rate"],
                "Shift": tire.get("Shift", ""),
                "Date": tire.get("Date", ""),
                "Stagger": round(m["stagger"], 1),
                "Cross Weight %": round(m["cross_weight"], 2),
            })

    return pd.DataFrame(rows).to_csv(index=False)


# ============================================================
# UI RENDERING HELPERS
# ============================================================

def _tire_html(tire, position: str) -> str:
    """HTML for one tire cell in the car grid."""
    css_class = "left-side" if position[0] == "L" else "right-side"
    tid = tire["Tire_ID"]
    size = tire["Size"]
    sr = tire["Spring Rate"]
    shift = tire.get("Shift", "")
    date = tire.get("Date", "")

    shift_val = shift if shift and shift != "nan" else ""
    date_val = date if date else ""

    return f'''<div class="tire-cell {css_class}">
        <div class="tire-num">#{tid}</div>
        <div class="tire-stats">
            <div class="tire-stat"><span class="label">Rollout</span><span class="val">{size:.0f}</span></div>
            <div class="tire-stat"><span class="label">SR</span><span class="val">{sr:.0f}</span></div>
            <div class="tire-stat"><span class="label">Shift</span><span class="val">{shift_val}</span></div>
            <div class="tire-stat"><span class="label">Date</span><span class="val">{date_val}</span></div>
        </div>
    </div>'''


def render_set_card(set_idx: int, metrics: dict, lefts, rights, lp, rp,
                    target_stagger: float) -> str:
    """Generate HTML for one set card."""
    lf = lefts.iloc[lp[2 * set_idx]]
    lr = lefts.iloc[lp[2 * set_idx + 1]]
    rf = rights.iloc[rp[2 * set_idx]]
    rr = rights.iloc[rp[2 * set_idx + 1]]

    front_stagger = rf["Size"] - lf["Size"]
    rear_stagger = rr["Size"] - lr["Size"]

    # Quality class
    stag_ok = abs(metrics["stagger"] - target_stagger) <= 1.0
    cross_ok = metrics["cross_dev"] <= 0.5
    if stag_ok and cross_ok:
        quality = "good"
    elif stag_ok or cross_ok:
        quality = "ok"
    else:
        quality = "bad"

    return f"""<div class="set-card">
        <div class="set-header">Set {set_idx + 1}</div>
        <div class="metrics-bar {quality}">
            <span>Front Stag: {front_stagger:.1f}</span>
            <span>X: {metrics['cross_weight']:.2f}%</span>
            <span>Rear Stag: {rear_stagger:.1f}</span>
        </div>
        <div class="car-grid">
            {_tire_html(lf, "LF")}
            {_tire_html(rf, "RF")}
            {_tire_html(lr, "LR")}
            {_tire_html(rr, "RR")}
        </div>
    </div>"""


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    # ---- Sidebar: File Upload & Configuration ----
    with st.sidebar:
        st.header("Data Import")
        uploaded = st.file_uploader(
            "Upload tire scan workbook",
            type=["xlsm", "xlsx", "xls"],
            help="Excel workbook with a 'Scan Data' sheet"
        )

        if uploaded:
            token = (uploaded.name, uploaded.size)
            if token != st.session_state._upload_token:
                with st.spinner("Reading scan data..."):
                    tires, ls_code, rs_code = load_scan_data(uploaded)
                if tires is not None and len(tires) >= 4:
                    st.session_state.tire_df = tires
                    st.session_state._upload_token = token

                    # Auto-assign D-code sides
                    dcodes = tires["D-Code"].unique().tolist()
                    if ls_code and ls_code in dcodes:
                        st.session_state.ls_dcode = ls_code
                        rs_codes = [d for d in dcodes if d != ls_code]
                        st.session_state.rs_dcode = rs_codes[0] if rs_codes else None
                    elif rs_code and rs_code in dcodes:
                        st.session_state.rs_dcode = rs_code
                        ls_codes = [d for d in dcodes if d != rs_code]
                        st.session_state.ls_dcode = ls_codes[0] if ls_codes else None
                    else:
                        # Auto-detect by average size (smaller = left)
                        means = tires.groupby("D-Code")["Size"].mean()
                        sorted_codes = means.sort_values().index.tolist()
                        st.session_state.ls_dcode = sorted_codes[0]
                        st.session_state.rs_dcode = sorted_codes[-1] if len(sorted_codes) > 1 else None

                    # Split pools
                    ls_dc = st.session_state.ls_dcode
                    st.session_state.lefts = tires[tires["D-Code"] == ls_dc].reset_index(drop=True)
                    st.session_state.rights = tires[tires["D-Code"] != ls_dc].reset_index(drop=True)

                    # Stagger analysis
                    info = analyze_stagger(st.session_state.lefts, st.session_state.rights)
                    st.session_state.stagger_info = info
                    st.session_state.n_sets = info["n_sets"]
                    st.session_state.target_stagger = info["max_achievable"]
                    st.session_state.data_loaded = True

                    # Pre-compute mode RR rollout for common RR rollout priority
                    r_sizes = st.session_state.rights["Size"].values
                    st.session_state.mode_rr = float(Counter(r_sizes).most_common(1)[0][0])

                    # Clear previous solution
                    st.session_state.left_perm = None
                    st.session_state.right_perm = None
                    st.session_state.set_metrics = None
                    st.session_state.solution_score = None
                    st.session_state.variant_solutions = {}
                    st.session_state.active_variant = "Original Sort"
                elif tires is not None:
                    st.error(f"Only {len(tires)} tires found — need at least 4.")

    if not st.session_state.data_loaded:
        st.info("Upload an Excel tire scan workbook to get started.")
        return

    # ---- Tabs ----
    tab_config, tab_results = st.tabs(["Configure", "Results"])

    # ================================================================
    # CONFIGURE TAB
    # ================================================================
    with tab_config:
        tires = st.session_state.tire_df
        lefts = st.session_state.lefts
        rights = st.session_state.rights
        info = st.session_state.stagger_info

        # Data summary
        st.subheader("Tire Inventory")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Tires", len(tires))
        c2.metric("Left Side", f"{len(lefts)}  (D-Code {st.session_state.ls_dcode})")
        c3.metric("Right Side", f"{len(rights)}  (D-Code {st.session_state.rs_dcode})")
        c4.metric("Sets", info["n_sets"])

        if info["unused_lefts"] > 0 or info["unused_rights"] > 0:
            st.caption(
                f"Unused tires: {info['unused_lefts']} left, {info['unused_rights']} right"
            )

        # D-Code override
        with st.expander("D-Code Side Assignment", expanded=False):
            dcodes = tires["D-Code"].unique().tolist()
            new_ls = st.selectbox(
                "Left-side D-Code",
                dcodes,
                index=dcodes.index(st.session_state.ls_dcode)
                if st.session_state.ls_dcode in dcodes else 0,
            )
            if new_ls != st.session_state.ls_dcode:
                st.session_state.ls_dcode = new_ls
                rs_codes = [d for d in dcodes if d != new_ls]
                st.session_state.rs_dcode = rs_codes[0] if rs_codes else None
                st.session_state.lefts = tires[tires["D-Code"] == new_ls].reset_index(drop=True)
                st.session_state.rights = tires[tires["D-Code"] != new_ls].reset_index(drop=True)
                new_info = analyze_stagger(st.session_state.lefts, st.session_state.rights)
                st.session_state.stagger_info = new_info
                st.session_state.n_sets = new_info["n_sets"]
                st.session_state.target_stagger = new_info["max_achievable"]
                r_sizes = st.session_state.rights["Size"].values
                st.session_state.mode_rr = float(Counter(r_sizes).most_common(1)[0][0])
                st.rerun()

        st.divider()

        # Stagger configuration
        st.subheader("Stagger Target")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Min Achievable", f"{info['min_achievable']:.1f} mm")
        sc2.metric("Max Achievable", f"{info['max_achievable']:.1f} mm")

        target = sc3.number_input(
            "Target Stagger (mm)",
            min_value=0.0,
            max_value=float(info["max_achievable"] + 20),
            value=float(st.session_state.target_stagger),
            step=0.5,
            help="Stagger = RR size - LR size. Defaults to max achievable across all sets."
        )
        st.session_state.target_stagger = target

        st.divider()

        # Priority ranking
        st.subheader("Optimization Priorities")
        st.markdown(
            '<div class="priority-fixed">#1  STAGGER (always primary)</div>',
            unsafe_allow_html=True,
        )
        st.caption("Drag to reorder secondary priorities:")

        if HAS_SORTABLES:
            new_order = sort_items(st.session_state.priority_order)
            if new_order != st.session_state.priority_order:
                st.session_state.priority_order = new_order
        else:
            # Fallback: up/down buttons
            prio = st.session_state.priority_order
            for i, p in enumerate(prio):
                pc1, pc2, pc3 = st.columns([4, 1, 1])
                pc1.write(f"**#{i + 2}** — {p}")
                if pc2.button("\u25B2", key=f"up_{i}", disabled=(i == 0)):
                    prio[i], prio[i - 1] = prio[i - 1], prio[i]
                    st.session_state.priority_order = prio
                    st.rerun()
                if pc3.button("\u25BC", key=f"dn_{i}", disabled=(i == len(prio) - 1)):
                    prio[i], prio[i + 1] = prio[i + 1], prio[i]
                    st.session_state.priority_order = prio
                    st.rerun()

        st.divider()

        # Run optimizer
        if st.button("Run Optimizer", type="primary", use_container_width=True):
            _lefts = st.session_state.lefts
            _rights = st.session_state.rights
            _n_sets = st.session_state.n_sets
            _target = st.session_state.target_stagger
            _prio = st.session_state.priority_order

            # Pre-extract arrays once, share across all runs
            _arrays = _extract_arrays(_lefts, _rights)

            progress = st.progress(0, text="Optimizing (Original Sort)...")
            total_steps = N_RESTARTS + QUICK_RESTARTS * len(PRIORITY_OPTIONS)
            done_steps = [0]

            def _update_main(frac):
                done_steps[0] = int(frac * N_RESTARTS)
                progress.progress(done_steps[0] / total_steps,
                                  text=f"Original Sort — restart {done_steps[0]}/{N_RESTARTS}")

            # 1) Full optimization with user's priority order
            lp, rp, score, metrics, rr_mode = run_optimizer(
                _lefts, _rights, _n_sets, _target, _prio,
                progress_callback=_update_main, _precomputed_arrays=_arrays,
            )
            st.session_state.mode_rr = float(rr_mode)
            variants = {}
            # Reorder original sort by RR rollout grouping
            lp, rp, metrics = _reorder_sets_by_rr(lp, rp, metrics, _n_sets)
            variants["Original Sort"] = {
                "left_perm": lp.copy(), "right_perm": rp.copy(),
                "score": score, "metrics": [m.copy() for m in metrics],
            }

            # 2) Quick variants — each priority promoted to #1 secondary
            base_done = N_RESTARTS
            for vi, pname in enumerate(PRIORITY_OPTIONS):
                var_label = f"Optimal {_BTN_LABELS.get(pname, pname)}"
                var_prio = [pname] + [p for p in PRIORITY_OPTIONS if p != pname]

                def _update_var(frac, _vi=vi, _label=var_label):
                    step = base_done + _vi * QUICK_RESTARTS + int(frac * QUICK_RESTARTS)
                    progress.progress(step / total_steps,
                                      text=f"{_label} — restart {int(frac * QUICK_RESTARTS)}/{QUICK_RESTARTS}")

                vlp, vrp, vscore, vmetrics, _ = run_optimizer(
                    _lefts, _rights, _n_sets, _target, var_prio,
                    progress_callback=_update_var,
                    n_restarts=QUICK_RESTARTS, n_iterations=QUICK_ITERATIONS,
                    _precomputed_arrays=_arrays,
                )
                # Reorder all variants: group by RR rollout, least common first
                vlp, vrp, vmetrics = _reorder_sets_by_rr(
                    vlp, vrp, vmetrics, _n_sets)
                variants[var_label] = {
                    "left_perm": vlp.copy(), "right_perm": vrp.copy(),
                    "score": vscore, "metrics": [m.copy() for m in vmetrics],
                }

            # Store all variants and set active to original
            st.session_state.variant_solutions = variants
            st.session_state.active_variant = "Original Sort"
            st.session_state.left_perm = lp
            st.session_state.right_perm = rp
            st.session_state.solution_score = score
            st.session_state.set_metrics = metrics
            progress.empty()
            st.success("Optimization complete!")
            st.rerun()

        # Quick preview of data
        with st.expander("Raw Tire Data", expanded=False):
            # Render as HTML table to bypass pyarrow compatibility issues
            def _fmt_float(x):
                try:
                    return f"{x:.0f}" if x == int(x) else f"{x:.2f}"
                except (ValueError, TypeError):
                    return str(x)
            html = tires.to_html(index=False, classes="tire-data-table", border=0,
                                  float_format=_fmt_float)
            st.markdown(html, unsafe_allow_html=True)

    # ================================================================
    # RESULTS TAB
    # ================================================================
    with tab_results:
        if st.session_state.set_metrics is None:
            st.info("Run the optimizer on the Configure tab first.")
            return

        # ---- Variant + ordering buttons (single row) ----
        variants = st.session_state.variant_solutions
        if variants:
            active = st.session_state.active_variant
            btn_names = list(variants.keys())
            all_cols = st.columns(len(btn_names) + 2)
            for col, vname in zip(all_cols, btn_names):
                with col:
                    is_active = (vname == active)
                    btn_type = "primary" if is_active else "secondary"
                    if st.button(vname, key=f"var_{vname}", type=btn_type,
                                 use_container_width=True):
                        if vname != active:
                            sol = variants[vname]
                            st.session_state.left_perm = sol["left_perm"].copy()
                            st.session_state.right_perm = sol["right_perm"].copy()
                            st.session_state.solution_score = sol["score"]
                            st.session_state.set_metrics = [m.copy() for m in sol["metrics"]]
                            st.session_state.active_variant = vname
                            st.rerun()
            with all_cols[len(btn_names)]:
                if st.button("Softest First", key="order_softest", use_container_width=True):
                    _lp = st.session_state.left_perm
                    _rp = st.session_state.right_perm
                    _m = st.session_state.set_metrics
                    _ns = st.session_state.n_sets
                    _lp, _rp, _m = _reorder_sets_by_sr(_lp, _rp, _m, _ns)
                    st.session_state.left_perm = _lp
                    st.session_state.right_perm = _rp
                    st.session_state.set_metrics = _m
                    st.rerun()
            with all_cols[len(btn_names) + 1]:
                if st.button("Group by RR", key="order_rr", use_container_width=True):
                    _lp = st.session_state.left_perm
                    _rp = st.session_state.right_perm
                    _m = st.session_state.set_metrics
                    _ns = st.session_state.n_sets
                    _lp, _rp, _m = _reorder_sets_by_rr(_lp, _rp, _m, _ns)
                    st.session_state.left_perm = _lp
                    st.session_state.right_perm = _rp
                    st.session_state.set_metrics = _m
                    st.rerun()

        lefts = st.session_state.lefts
        rights = st.session_state.rights
        lp = st.session_state.left_perm
        rp = st.session_state.right_perm
        n_sets = st.session_state.n_sets
        metrics = st.session_state.set_metrics
        target = st.session_state.target_stagger

        # ---- Set cards in grid — auto-fit to up to 3 rows, consistent widths ----
        if n_sets <= 2:
            n_rows = 1
        elif n_sets <= 6:
            n_rows = 2
        else:
            n_rows = 3
        cards_per_row = -(-n_sets // n_rows)  # ceil division

        for row_start in range(0, n_sets, cards_per_row):
            row_end = min(row_start + cards_per_row, n_sets)
            cols = st.columns(cards_per_row)
            for col, si in zip(cols, range(row_start, row_end)):
                with col:
                    html = render_set_card(si, metrics[si], lefts, rights, lp, rp, target)
                    st.markdown(html, unsafe_allow_html=True)

        # ---- Manual Swap Panel ----

        # Build lookup: tire ID → (side, set_idx, position)
        tire_lookup = {}
        for s in range(n_sets):
            for pos, pool, perm in [("LF", lefts, lp), ("LR", lefts, lp),
                                     ("RF", rights, rp), ("RR", rights, rp)]:
                slot = 2 * s + (0 if pos in ("LF", "RF") else 1)
                tire = pool.iloc[perm[slot]]
                tire_lookup[str(tire["Tire_ID"])] = (s, pos)

        sc1, sc2, sc3, sc4 = st.columns([2, 2, 1.5, 1.5])
        with sc1:
            id_a = st.text_input("Tire A ID", key="swap_id_a", placeholder="e.g. 5")
        with sc2:
            id_b = st.text_input("Tire B ID", key="swap_id_b", placeholder="e.g. 12")
        with sc3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            swap_clicked = st.button("Swap", type="primary", use_container_width=True)
        with sc4:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            csv_data = build_export_csv()
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="tire_sort_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if swap_clicked and id_a and id_b:
            id_a = id_a.strip().lstrip("#")
            id_b = id_b.strip().lstrip("#")
            if id_a == id_b:
                st.warning("Cannot swap a tire with itself.")
            elif id_a not in tire_lookup:
                st.error(f"Tire #{id_a} not found in current sets.")
            elif id_b not in tire_lookup:
                st.error(f"Tire #{id_b} not found in current sets.")
            else:
                set_a, pos_a = tire_lookup[id_a]
                set_b, pos_b = tire_lookup[id_b]
                side_a = "L" if pos_a[0] == "L" else "R"
                side_b = "L" if pos_b[0] == "L" else "R"
                if side_a != side_b:
                    st.error("Cannot swap left-side and right-side tires.")
                else:
                    perform_swap(set_a, pos_a, set_b, pos_b)
                    st.rerun()

        # ---- Set Swap + Copy IDs ----
        ss1, ss2, ss3, ss4 = st.columns([2, 2, 1.5, 1.5])
        with ss1:
            set_a_input = st.text_input("Set A", key="swap_set_a", placeholder="e.g. 1")
        with ss2:
            set_b_input = st.text_input("Set B", key="swap_set_b", placeholder="e.g. 5")
        with ss3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            set_swap_clicked = st.button("Swap Sets", type="primary", use_container_width=True)
        with ss4:
            # Build 2x2 clipboard text (tab-separated for Excel paste)
            clip_lines = []
            for s in range(n_sets):
                lf_id = lefts.iloc[lp[2 * s]]["Tire_ID"]
                lr_id = lefts.iloc[lp[2 * s + 1]]["Tire_ID"]
                rf_id = rights.iloc[rp[2 * s]]["Tire_ID"]
                rr_id = rights.iloc[rp[2 * s + 1]]["Tire_ID"]
                clip_lines.append(f"{lf_id}\\t{rf_id}")
                clip_lines.append(f"{lr_id}\\t{rr_id}")
                if s < n_sets - 1:
                    clip_lines.append("")
            clip_js = "\\n".join(clip_lines)

            components.html(f"""
<button id="cpBtn" onclick="
    var text = '{clip_js}';
    if (navigator.clipboard) {{
        navigator.clipboard.writeText(text).then(function() {{
            document.getElementById('cpBtn').innerText = 'Copied!';
            setTimeout(function() {{ document.getElementById('cpBtn').innerText = 'Copy IDs'; }}, 2000);
        }}).catch(function() {{ fallbackCopy(text); }});
    }} else {{ fallbackCopy(text); }}"
    style="padding:0.5rem 1rem;background:#5C6BC0;color:white;border:none;border-radius:0.5rem;cursor:pointer;font-weight:600;font-size:14px;width:100%">
    Copy IDs
</button>
<script>
function fallbackCopy(text) {{
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    document.getElementById('cpBtn').innerText = 'Copied!';
    setTimeout(function() {{ document.getElementById('cpBtn').innerText = 'Copy IDs'; }}, 2000);
}}
</script>
""", height=42)

        if set_swap_clicked and set_a_input and set_b_input:
            try:
                sa = int(set_a_input.strip()) - 1
                sb = int(set_b_input.strip()) - 1
            except ValueError:
                st.error("Enter valid set numbers.")
                sa, sb = -1, -1
            if sa == sb and sa >= 0:
                st.warning("Cannot swap a set with itself.")
            elif sa < 0 or sa >= n_sets:
                st.error(f"Set {sa + 1} not found. Valid range: 1–{n_sets}.")
            elif sb < 0 or sb >= n_sets:
                st.error(f"Set {sb + 1} not found. Valid range: 1–{n_sets}.")
            else:
                perform_set_swap(sa, sb)
                st.rerun()



# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
