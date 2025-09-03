import os
import time
import threading
import warnings
from math import erf, sqrt
from datetime import datetime

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, whoami, hf_hub_download
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# =========================
# SPEED + SAFETY + TRAIN TAB
# =========================

warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples",
    category=UserWarning,
)

PRIMARY_REPO = "aniktanims/dice-data"
FILE_PRIMARY = "Dice_updated.xlsx"
FILE_DIGITAL = "digitaldice.xlsx"
FILE_LIVE    = "livedice.xlsx"

# ---------- Utils: triples / classes ----------
def triple_to_class(d1, d2, d3):
    return (int(d1) - 1) * 36 + (int(d2) - 1) * 6 + (int(d3) - 1)

def class_to_triple(cls):
    cls = int(cls)
    d1 = cls // 36
    rem = cls % 36
    d2 = rem // 6
    d3 = rem % 6
    return d1 + 1, d2 + 1, d3 + 1

def clean_and_standardize(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_raw = df_raw.copy()
    df_raw.columns = [str(c).strip().lower() for c in df_raw.columns]
    col_map = {}
    for c in df_raw.columns:
        if "dice 1" in c or c in ("dice1","die1"): col_map["dice1"] = c
        elif "dice 2" in c or c in ("dice2","die2"): col_map["dice2"] = c
        elif "dice 3" in c or c in ("dice3","die3"): col_map["dice3"] = c
        elif "total" in c: col_map["total"] = c
    req = ["dice1","dice2","dice3","total"]
    miss = [r for r in req if r not in col_map]
    if miss: raise ValueError(f"Missing columns: {miss}. Found: {df_raw.columns.tolist()}")
    df = df_raw[[col_map["dice1"], col_map["dice2"], col_map["dice3"], col_map["total"]]].copy()
    df.columns = ["dice1","dice2","dice3","total"]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().astype(int)
    df[["dice1","dice2","dice3"]] = df[["dice1","dice2","dice3"]].clip(1,6)
    bad = df["total"] != (df["dice1"] + df["dice2"] + df["dice3"])
    if bad.any():
        df.loc[bad, "total"] = df.loc[bad, ["dice1","dice2","dice3"]].sum(axis=1)
    return df.reset_index(drop=True)

def make_supervision(df: pd.DataFrame):
    if len(df) < 2: return None, None
    X_curr = df.iloc[:-1].reset_index(drop=True)
    Y_next = df.iloc[1:].reset_index(drop=True)
    X = X_curr[["dice1","dice2","dice3"]].astype(int).values
    y = np.array([triple_to_class(d1,d2,d3) for d1,d2,d3 in
                  zip(Y_next["dice1"], Y_next["dice2"], Y_next["dice3"])], dtype=int)
    return X, y

# ---------- Fast model (no retrain on predict) ----------
def train_triple_model(df: pd.DataFrame, n_estimators=180, random_state=42):
    X, y = make_supervision(df)
    if X is None or len(y) < 10:
        return None, None
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    clf.fit(X, y)
    uniq = len(np.unique(y))
    acc = None
    if len(y) >= 80 and uniq <= 0.5 * len(y):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        clf_tmp = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        clf_tmp.fit(Xtr, ytr)
        acc = accuracy_score(yte, clf_tmp.predict(Xte))
    return clf, float(acc) if acc is not None else None

# ---------- Behavior (recent window & factors) ----------
def recent_window_df(df: pd.DataFrame, n:int=300):
    if not isinstance(df, pd.DataFrame) or len(df)==0:
        return df
    n = int(max(20, n))
    return df.iloc[-n:].reset_index(drop=True) if len(df) > n else df.copy()

def fit_total_markov(df: pd.DataFrame, alpha: float = 1.0):
    A = np.ones((16,16), dtype=float) * alpha
    if isinstance(df, pd.DataFrame) and len(df) >= 2:
        T = df["total"].astype(int).values
        for i in range(len(T)-1):
            a = T[i]-3; b = T[i+1]-3
            if 0 <= a < 16 and 0 <= b < 16: A[a,b] += 1.0
    A /= A.sum(axis=1, keepdims=True)
    return A

def fit_die_transitions(df: pd.DataFrame, alpha: float = 1.0):
    mats = [np.ones((6,6), dtype=float)*alpha for _ in range(3)]
    if isinstance(df, pd.DataFrame) and len(df) >= 2:
        a = df[["dice1","dice2","dice3"]].astype(int).values
        b = df[["dice1","dice2","dice3"]].astype(int).shift(-1).dropna().values
        a = a[:-1]
        for i in range(len(a)):
            for d in range(3):
                cur = int(a[i,d])-1; nxt = int(b[i,d])-1
                if 0 <= cur < 6 and 0 <= nxt < 6: mats[d][cur,nxt] += 1.0
    for d in range(3): mats[d] /= mats[d].sum(axis=1, keepdims=True)
    return mats

def fit_die_bias(df: pd.DataFrame, alpha: float = 1.0):
    biases = []
    for col in ["dice1","dice2","dice3"]:
        counts = np.ones(6, dtype=float)*alpha
        if isinstance(df, pd.DataFrame) and len(df):
            vals, cnt = np.unique(df[col].astype(int).values, return_counts=True)
            for v,c in zip(vals,cnt):
                if 1 <= v <= 6: counts[v-1] += float(c)
        biases.append(counts / counts.sum())
    return biases

def softmax_from_log(logp, temperature: float = 1.0):
    z = (logp / max(temperature, 1e-6))
    z = z - np.max(z)
    p = np.exp(z); s = p.sum()
    return p / s if s > 0 else np.ones_like(p)/len(p)

def ensemble_probs(clf, prev_triple, df_hist: pd.DataFrame,
                   w_rf=0.7, w_tot=0.2, w_die=0.07, w_bias=0.03,
                   alpha=1.0, temperature=1.0):
    d1,d2,d3 = map(int, prev_triple)
    row = np.array([[d1,d2,d3]], dtype=int)

    rf_probs = clf.predict_proba(row)[0]
    classes = clf.classes_
    K = len(classes)

    A_tot = fit_total_markov(df_hist, alpha=alpha)
    A_die = fit_die_transitions(df_hist, alpha=alpha)
    B_die = fit_die_bias(df_hist, alpha=alpha)

    eps = 1e-12
    tot_row = A_tot[(d1+d2+d3)-3]

    tot_factor = np.empty(K, dtype=float)
    die_factor = np.empty(K, dtype=float)
    bias_factor = np.empty(K, dtype=float)

    for i, cls in enumerate(classes):
        a,b,c = class_to_triple(int(cls))
        Tnext = a+b+c
        tot_factor[i] = tot_row[Tnext-3]
        d1f = A_die[0][d1-1,a-1]; d2f = A_die[1][d2-1,b-1]; d3f = A_die[2][d3-1,c-1]
        die_factor[i] = d1f*d2f*d3f
        bias_factor[i] = B_die[0][a-1]*B_die[1][b-1]*B_die[2][c-1]

    def _nz(x):
        s = x.sum()
        return x/s if s>0 else np.ones_like(x)/len(x)

    rf_probs = _nz(rf_probs + eps)
    tot_probs = _nz(tot_factor + eps)
    die_probs = _nz(die_factor + eps)
    bias_probs= _nz(bias_factor+ eps)

    weights = np.array([max(w_rf,0), max(w_tot,0), max(w_die,0), max(w_bias,0)], dtype=float)
    weights = weights / max(weights.sum(), 1e-12)
    wrf, wtot, wdie, wbias = weights.tolist()

    logp = (wrf*np.log(rf_probs) + wtot*np.log(tot_probs) + wdie*np.log(die_probs) + wbias*np.log(bias_probs))
    probs = softmax_from_log(logp, temperature=temperature)
    return classes, probs

def predict_ensemble_for_input_full(clf, d1:int, d2:int, d3:int, df_hist,
                                    w_rf=0.7, w_tot=0.2, w_die=0.07, w_bias=0.03,
                                    alpha=1.0, temperature=1.0):
    classes, probs = ensemble_probs(
        clf, (d1,d2,d3), df_hist,
        w_rf=w_rf, w_tot=w_tot, w_die=w_die, w_bias=w_bias,
        alpha=alpha, temperature=temperature
    )
    idx = int(np.argmax(probs))
    best_cls = int(classes[idx])
    p = float(probs[idx])
    nd1, nd2, nd3 = class_to_triple(best_cls)
    tot = nd1 + nd2 + nd3

    order = np.argsort(probs)[::-1][:5]
    top5 = []
    last_topk = []
    for i in order:
        cls = int(classes[i])
        t1,t2,t3 = class_to_triple(cls)
        last_topk.append((t1,t2,t3))
        top5.append((t1,t2,t3,t1+t2+t3, round(float(probs[i])*100,2)))

    totals = {}
    for i, cls in enumerate(classes):
        t1,t2,t3 = class_to_triple(int(cls))
        T = t1+t2+t3
        totals[T] = totals.get(T, 0.0) + float(probs[i])
    totals = {k: round(v*100,2) for k,v in sorted(totals.items())}
    return (nd1, nd2, nd3, tot, p), top5, totals, last_topk, classes, probs

# ---------- Fairness / diagnostics ----------
def dirichlet_bias_stats(df_win: pd.DataFrame, alpha=1.0):
    out = []
    kls = []
    uniform = np.ones(6)/6
    for col in ["dice1","dice2","dice3"]:
        counts = np.ones(6)*alpha
        if isinstance(df_win, pd.DataFrame) and len(df_win)>0:
            vals, cnt = np.unique(df_win[col].astype(int).values, return_counts=True)
            for v,c in zip(vals,cnt):
                if 1<=v<=6: counts[v-1] += float(c)
        p = counts / counts.sum()
        out.append(p)
        kls.append( float(np.sum(p * np.log(np.maximum(p,1e-12)/uniform))) )
    return out, float(np.mean(kls))

def runs_test_parity(totals: np.ndarray):
    if len(totals) < 40:
        return 0.0, 1.0
    x = (totals % 2).astype(int)
    n1 = int(x.sum())
    n2 = len(x) - n1
    if n1==0 or n2==0:
        return 0.0, 1.0
    runs = 1 + np.sum(x[1:] != x[:-1])
    mu = 1 + 2*n1*n2/(n1+n2)
    var = (2*n1*n2*(2*n1*n2 - n1 - n2)) / (((n1+n2)**2) * (n1+n2-1))
    if var <= 0: 
        return 0.0, 1.0
    z = (runs - mu) / np.sqrt(var)
    p = 2*(1 - 0.5*(1+erf(abs(z)/sqrt(2))))
    return float(z), float(p)

def page_hinkley_totals(totals: np.ndarray, delta=0.05, lam=50.0, alpha=0.999):
    if len(totals) < 50:
        return False, 0.0
    x = (totals - 3) / 15.0
    mean = 0.0
    m_t = 0.0
    ph = 0.0
    for xi in x:
        mean = alpha*mean + (1-alpha)*xi
        m_t = min(0.0, m_t + (xi - mean - delta))
        ph = max(ph, -m_t)
    return (ph > lam), float(ph)

def fairness_indicator(df_win: pd.DataFrame, alpha=1.0, runs_alpha=0.01, ph_lam=50.0):
    if not isinstance(df_win, pd.DataFrame) or len(df_win)==0:
        return {"label":"—","kl":0.0,"runs_p":1.0,"ph":0.0,"ph_flag":False,"color":"#64748b"}
    _, kl_avg = dirichlet_bias_stats(df_win, alpha=alpha)
    z, p = runs_test_parity(df_win["total"].astype(int).values)
    ph_flag, ph_val = page_hinkley_totals(df_win["total"].astype(int).values, lam=ph_lam)
    suspect = (kl_avg > 0.12) or (p < runs_alpha) or ph_flag
    return {
        "label": "Suspect" if suspect else "Fair",
        "kl": kl_avg,
        "runs_p": p,
        "ph": ph_val,
        "ph_flag": ph_flag,
        "color": "#ef4444" if suspect else "#16a34a"
    }

def sicbo_category_probs(classes, probs, triples_lose=True):
    def cls_to_sum(c):
        a,b,c2 = class_to_triple(int(c))
        return a+b+c2
    totals = np.array([cls_to_sum(c) for c in classes], dtype=int)
    p = np.array(probs, dtype=float)
    is_triple = np.array([1 if (class_to_triple(int(c))[0]==class_to_triple(int(c))[1]==class_to_triple(int(c))[2]) else 0
                          for c in classes], dtype=bool)
    big = (totals >= 11) & (totals <= 17)
    small = (totals >= 4) & (totals <= 10)
    odd = (totals % 2 == 1)
    even = ~odd
    if triples_lose:
        big = big & (~is_triple)
        small = small & (~is_triple)
        odd = odd & (~is_triple)
        even = even & (~is_triple)
    out = {
        "Small(4–10)": float(p[small].sum()*100),
        "Big(11–17)": float(p[big].sum()*100),
        "Odd": float(p[odd].sum()*100),
        "Even": float(p[even].sum()*100),
        "Any Triple": float(p[is_triple].sum()*100),
    }
    for T in range(3,19):
        out[f"Total {T}"] = float(p[totals==T].sum()*100)
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

# ---------- HF dataset I/O ----------
def _api():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("Missing HF_TOKEN (add it in Settings → Variables and secrets).")
    return HfApi(token=token)

def ensure_file_exists(filename: str):
    """Make sure a dataset file exists; if missing, create blank and upload."""
    try:
        hf_hub_download(repo_id=PRIMARY_REPO, repo_type="dataset", filename=filename,
                        revision="main", force_download=False, local_files_only=False)
        return
    except Exception:
        # create blank
        tmp = filename
        pd.DataFrame(columns=["dice1","dice2","dice3","total"]).to_excel(tmp, index=False)
        api = _api()
        try:
            api.create_repo(repo_id=PRIMARY_REPO, repo_type="dataset", exist_ok=True)
        except Exception:
            pass
        api.upload_file(path_or_fileobj=tmp, path_in_repo=filename,
                        repo_id=PRIMARY_REPO, repo_type="dataset",
                        commit_message=f"Init blank {filename}")

def ensure_dataset_initialized():
    ok_msg = []
    try:
        api = _api()
        try:
            api.create_repo(repo_id=PRIMARY_REPO, repo_type="dataset", exist_ok=True)
        except Exception:
            pass
        for fn in [FILE_PRIMARY, FILE_DIGITAL, FILE_LIVE]:
            ensure_file_exists(fn)
        ok_msg.append("Dataset present.")
        return True, " ".join(ok_msg)
    except Exception as e:
        return False, f"Failed to initialize dataset: {e}"

def load_dataset_df_file(filename: str):
    ensure_file_exists(filename)
    path = hf_hub_download(repo_id=PRIMARY_REPO, repo_type="dataset", filename=filename,
                           revision="main", force_download=True, local_files_only=False)
    df = pd.read_excel(path)
    return clean_and_standardize(df)

def push_df_file(df: pd.DataFrame, filename: str, msg="Persist dataset"):
    tmp = filename
    df.to_excel(tmp, index=False)
    api = _api()
    api.upload_file(path_or_fileobj=tmp, path_in_repo=filename,
                    repo_id=PRIMARY_REPO, repo_type="dataset",
                    commit_message=msg)

def append_triple_to_file(filename: str, triple):
    """Blocking append → instant sync (no data loss)."""
    d1, d2, d3 = map(int, triple)
    df = load_dataset_df_file(filename)
    add = pd.DataFrame([{"dice1": d1, "dice2": d2, "dice3": d3, "total": d1+d2+d3}])
    df_new = pd.concat([df, add], ignore_index=True)
    df_new = clean_and_standardize(df_new)
    push_df_file(df_new, filename, msg=f"Train append to {filename}")
    return df_new

# ---------- Optional background sync/retrain for Predict tab ----------
class Store:
    def __init__(self):
        self.df = None
        self.active_file = FILE_PRIMARY
        self.version = 0
        self.model = None
        self.acc = None
        self.model_version = -1
        self.lock = threading.Lock()
        self.sync_running = False
        self.bg_retrain_running = False

STORE = Store()

def schedule_sync_to_hf():
    """Upload current active df in background (Predict tab)."""
    def worker():
        with STORE.lock:
            if STORE.sync_running or STORE.df is None:
                return
            STORE.sync_running = True
            df_copy = STORE.df.copy()
            filename = STORE.active_file
        try:
            push_df_file(df_copy, filename, msg=f"Auto sync {filename}")
        except Exception:
            pass
        finally:
            with STORE.lock:
                STORE.sync_running = False
    t = threading.Thread(target=worker, daemon=True)
    t.start()

def schedule_background_retrain(n_estimators=180):
    def worker():
        with STORE.lock:
            if STORE.bg_retrain_running or STORE.df is None:
                return
            STORE.bg_retrain_running = True
            df_copy = STORE.df.copy()
            v_before = STORE.version
        try:
            clf, acc = train_triple_model(df_copy, n_estimators=n_estimators)
            with STORE.lock:
                if STORE.version == v_before:
                    STORE.model, STORE.acc = clf, acc
                    STORE.model_version = STORE.version
        except Exception:
            pass
        finally:
            with STORE.lock:
                STORE.bg_retrain_running = False
    t = threading.Thread(target=worker, daemon=True)
    t.start()

# ---------- Boot / Reload ----------
def boot_load(active_file=FILE_PRIMARY):
    ok, msg0 = ensure_dataset_initialized()
    if not ok:
        return f"❌ {msg0}", None, None, None
    try:
        df = load_dataset_df_file(active_file)
        with STORE.lock:
            STORE.active_file = active_file
            STORE.df = df
            STORE.version += 1
        clf, acc = (train_triple_model(df) if len(df) >= 10 else (None, None))
        with STORE.lock:
            STORE.model, STORE.acc = clf, acc
            STORE.model_version = STORE.version
        return f"✅ {msg0} (active: {active_file})", df, clf, acc
    except Exception as e:
        return f"❌ Load error: {e}", None, None, None

# ---------- Stats / visuals ----------
def stats_line(total:int, correct:int, wrong:int, top5:int):
    total = int(total or 0); correct = int(correct or 0); wrong = int(wrong or 0); top5 = int(top5 or 0)
    acc = (correct/total*100) if total>0 else 0.0
    top5r = (top5/total*100) if total>0 else 0.0
    wrongp = (wrong/total*100) if total>0 else 0.0
    return f"**Session**  T:{total} • ✅:{correct} ({acc:.0f}%) • 🎯Top5:{top5} ({top5r:.0f}%) • ❌:{wrong} ({wrongp:.0f}%)"

def data_stats_line(df: pd.DataFrame|None, fname:str):
    n = len(df) if isinstance(df, pd.DataFrame) else 0
    return f"**Data** ({fname}) rows: **{n}**"

def make_totals_sparkline(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4.5, 0.9), dpi=150)
    ax.plot(df["total"].values if isinstance(df, pd.DataFrame) and len(df)>0 else [], linewidth=1.5)
    ax.set_ylim(2.5, 18.5)
    ax.axis("off")
    return fig

# ================== Predict Callbacks ==================
def _format_tables(top5, totals):
    def table_top5(rows):
        if not rows: return ""
        html = ["<div class='tblwrap'><table class='tbl'><thead><tr><th>Triple</th><th>Total</th><th>%</th></tr></thead><tbody>"]
        for a,b,c,t,pp in rows:
            html.append(f"<tr><td>({a},{b},{c})</td><td>{t}</td><td>{pp:.2f}</td></tr>")
        html.append("</tbody></table></div>")
        return "".join(html)
    def table_totals(dct):
        if not dct: return ""
        html = ["<div class='tblwrap'><table class='tbl'><thead><tr><th>Total</th><th>%</th></tr></thead><tbody>"]
        for k,v in dct.items():
            html.append(f"<tr><td>{k}</td><td>{v:.2f}</td></tr>")
        html.append("</tbody></table></div>")
        return "".join(html)
    return table_top5(top5), table_totals(totals)

def fairness_indicator(df_win: pd.DataFrame, alpha=1.0, runs_alpha=0.01, ph_lam=50.0):
    # (redeclared to keep file self-contained for clarity)
    if not isinstance(df_win, pd.DataFrame) or len(df_win)==0:
        return {"label":"—","kl":0.0,"runs_p":1.0,"ph":0.0,"ph_flag":False,"color":"#64748b"}
    _, kl_avg = dirichlet_bias_stats(df_win, alpha=alpha)
    z, p = runs_test_parity(df_win["total"].astype(int).values)
    ph_flag, ph_val = page_hinkley_totals(df_win["total"].astype(int).values, lam=ph_lam)
    suspect = (kl_avg > 0.12) or (p < runs_alpha) or ph_flag
    return {
        "label": "Suspect" if suspect else "Fair",
        "kl": kl_avg,
        "runs_p": p,
        "ph": ph_val,
        "ph_flag": ph_flag,
        "color": "#ef4444" if suspect else "#16a34a"
    }

def on_reload_active(active_file):
    msg, df, clf, acc = boot_load(active_file=active_file)
    return f"🔄 {msg}", df, clf, acc, data_stats_line(df, active_file), make_totals_sparkline(df if df is not None else pd.DataFrame())

def on_predict(d1, d2, d3,
               last_input_state, last_pred_state, pending_state,
               sess_total, sess_correct, sess_wrong, sess_top5, log_md,
               w_rf, w_tot, w_die, w_bias, temp, alpha_smooth,
               window_n, ph_lambda, triples_lose):
    # 12 outputs
    with STORE.lock:
        df = STORE.df
        clf = STORE.model
        fname = STORE.active_file

    if not isinstance(df, pd.DataFrame) or len(df)==0 or clf is None:
        return ("Load/train first.", "", "", "", last_input_state, last_pred_state, False,
                stats_line(sess_total, sess_correct, sess_wrong, sess_top5), data_stats_line(df, fname), log_md, None, "")

    try:
        d1 = int(d1); d2 = int(d2); d3 = int(d3)
    except Exception:
        return ("Pick all dice (1–6).", "", "", "", last_input_state, last_pred_state, False,
                stats_line(sess_total, sess_correct, sess_wrong, sess_top5), data_stats_line(df, fname), log_md, None, "")

    if not all(1 <= x <= 6 for x in [d1,d2,d3]):
        return ("Each die must be 1..6.", "", "", "", last_input_state, last_pred_state, False,
                stats_line(sess_total, sess_correct, sess_wrong, sess_top5), data_stats_line(df, fname), log_md, None, "")

    df_recent = recent_window_df(df, n=int(window_n))
    (nd1, nd2, nd3, tot, p), top5, totals, last_topk, classes, probs = predict_ensemble_for_input_full(
        clf, d1, d2, d3, df_recent,
        w_rf=w_rf, w_tot=w_tot, w_die=w_die, w_bias=w_bias,
        alpha=alpha_smooth, temperature=float(temp)
    )
    line = f"**Next:** {nd1},{nd2},{nd3} = {tot}  ({p*100:.0f}%)"
    html_top5, html_totals = _format_tables(top5, totals)

    fi = fairness_indicator(df_recent, alpha=alpha_smooth, ph_lam=float(ph_lambda))
    fair_html = f"<div style='font-weight:600;color:{fi['color']};'>Regime: {fi['label']}</div>" \
                f"<div class='muted'>KL(avg)={fi['kl']:.3f} • Runs p={fi['runs_p']:.3f} • PH={fi['ph']:.1f}</div>"

    cats = sicbo_category_probs(classes, probs, triples_lose=bool(triples_lose))
    cats_tbl = ['<div class="tblwrap"><table class="tbl"><thead><tr><th>Category</th><th>%</th></tr></thead><tbody>']
    for k,v in list(cats.items())[:15]:
        cats_tbl.append(f"<tr><td>{k}</td><td>{v:.2f}</td></tr>")
    cats_tbl.append("</tbody></table></div>")
    cats_html = "".join(cats_tbl)

    return (
        line, html_top5, html_totals, fair_html,
        (d1, d2, d3), (nd1, nd2, nd3, tot, float(p)), True,
        stats_line(sess_total, sess_correct, sess_wrong, sess_top5), data_stats_line(df, fname),
        log_md, last_topk, cats_html
    )

def _update_metrics_on_truth(triple_truth, last_topk, sess_total, sess_correct, sess_wrong, sess_top5, correct_hit):
    sess_total = int(sess_total or 0) + 1
    if correct_hit:
        sess_correct = int(sess_correct or 0) + 1
        sess_top5 = int(sess_top5 or 0) + 1
    else:
        sess_wrong = int(sess_wrong or 0) + 1
        if isinstance(last_topk, list) and tuple(map(int, triple_truth)) in [tuple(x) for x in last_topk]:
            sess_top5 = int(sess_top5 or 0) + 1
    return sess_total, sess_correct, sess_wrong, sess_top5

def _append_pair_in_memory(input_triple, next_triple):
    d1, d2, d3 = map(int, input_triple)
    nd1, nd2, nd3 = map(int, next_triple)
    with STORE.lock:
        df = STORE.df if STORE.df is not None else pd.DataFrame(columns=["dice1","dice2","dice3","total"])
        add = pd.DataFrame([
            {"dice1": d1, "dice2": d2, "dice3": d3, "total": d1+d2+d3},
            {"dice1": nd1, "dice2": nd2, "dice3": nd3, "total": nd1+nd2+nd3},
        ])
        df_new = pd.concat([df, add], ignore_index=True)
        df_new = clean_and_standardize(df_new)
        STORE.df = df_new
        STORE.version += 1
        return df_new, STORE.version

def on_accept(last_input_state, last_pred_state, last_topk_state, pending_state,
              sess_total, sess_correct, sess_wrong, sess_top5,
              auto_sync, bg_retrain, trees, log_md):
    if not pending_state or last_input_state is None or last_pred_state is None:
        with STORE.lock:
            df = STORE.df; fname = STORE.active_file
        return ("Predict first.", df, STORE.model, STORE.acc,
                sess_total, sess_correct, sess_wrong, sess_top5, False,
                stats_line(sess_total, sess_correct, sess_wrong, sess_top5),
                data_stats_line(df, fname), log_md,
                gr.update(), gr.update(), gr.update(), last_input_state, make_totals_sparkline(df if df is not None else pd.DataFrame()))

    df_new, _ = _append_pair_in_memory(last_input_state, last_pred_state[:3])
    sess_total, sess_correct, sess_wrong, sess_top5 = _update_metrics_on_truth(
        last_pred_state[:3], last_topk_state, sess_total, sess_correct, sess_wrong, sess_top5, correct_hit=True
    )
    if bg_retrain: schedule_background_retrain(n_estimators=int(trees))
    if auto_sync: schedule_sync_to_hf()

    nd1, nd2, nd3 = map(int, last_pred_state[:3])
    next_d1, next_d2, next_d3 = str(nd1), str(nd2), str(nd3)
    next_last_input = (nd1, nd2, nd3)

    with STORE.lock:
        clf, acc = STORE.model, STORE.acc
        fname = STORE.active_file
    return ("Correct ✅ (saved in memory)",
            df_new, clf, acc,
            sess_total, sess_correct, sess_wrong, sess_top5, False,
            stats_line(sess_total, sess_correct, sess_wrong, sess_top5),
            data_stats_line(df_new, fname), ("Auto-sync queued" if auto_sync else "Manual sync"),
            next_d1, next_d2, next_d3, next_last_input, make_totals_sparkline(df_new))

def on_learn(nd1, nd2, nd3,
             last_input_state, last_topk_state, pending_state,
             sess_total, sess_correct, sess_wrong, sess_top5,
             auto_sync, bg_retrain, trees, log_md):
    if not pending_state or last_input_state is None:
        with STORE.lock:
            df = STORE.df; fname = STORE.active_file
        return ("Predict first.", df, STORE.model, STORE.acc,
                sess_total, sess_correct, sess_wrong, sess_top5, True,
                stats_line(sess_total, sess_correct, sess_wrong, sess_top5),
                data_stats_line(df, fname), log_md,
                gr.update(), gr.update(), gr.update(), last_input_state, make_totals_sparkline(df if df is not None else pd.DataFrame()))
    try:
        nd1 = int(nd1); nd2 = int(nd2); nd3 = int(nd3)
    except Exception:
        with STORE.lock:
            df = STORE.df; fname = STORE.active_file
        return ("Pick corrected triple.", df, STORE.model, STORE.acc,
                sess_total, sess_correct, sess_wrong, sess_top5, True,
                stats_line(sess_total, sess_correct, sess_wrong, sess_top5),
                data_stats_line(df, fname), log_md,
                gr.update(), gr.update(), gr.update(), last_input_state, make_totals_sparkline(df if df is not None else pd.DataFrame()))

    df_new, _ = _append_pair_in_memory(last_input_state, (nd1, nd2, nd3))
    sess_total, sess_correct, sess_wrong, sess_top5 = _update_metrics_on_truth(
        (nd1, nd2, nd3), last_topk_state, sess_total, sess_correct, sess_wrong, sess_top5, correct_hit=False
    )
    if bg_retrain: schedule_background_retrain(n_estimators=int(trees))
    if auto_sync: schedule_sync_to_hf()

    next_d1, next_d2, next_d3 = str(nd1), str(nd2), str(nd3)
    next_last_input = (nd1, nd2, nd3)

    with STORE.lock:
        clf, acc = STORE.model, STORE.acc
        fname = STORE.active_file
    return ("Learned ✅ (saved in memory)",
            df_new, clf, acc,
            sess_total, sess_correct, sess_wrong, sess_top5, False,
            stats_line(sess_total, sess_correct, sess_wrong, sess_top5),
            data_stats_line(df_new, fname), ("Auto-sync queued" if auto_sync else "Manual sync"),
            next_d1, next_d2, next_d3, next_last_input, make_totals_sparkline(df_new))

def on_sync_now():
    schedule_sync_to_hf()
    with STORE.lock:
        df = STORE.df; fname = STORE.active_file
    return "🔁 Syncing to HF in background...", data_stats_line(df, fname)

def on_retrain_now(trees):
    schedule_background_retrain(n_estimators=int(trees))
    with STORE.lock:
        df = STORE.df; fname = STORE.active_file
    return "🧠 Retrain started in background...", data_stats_line(df, fname)

def on_upload_merge(file):
    with STORE.lock:
        fname = STORE.active_file
    if file is None:
        msg, df, clf, acc = on_reload_active(fname)
        return f"Upload a .xlsx to merge. {msg}", df, data_stats_line(df, fname), make_totals_sparkline(df if df is not None else pd.DataFrame())
    try:
        df_raw = pd.read_excel(file.name, sheet_name=0)
        up = clean_and_standardize(df_raw)
        with STORE.lock:
            base = STORE.df if STORE.df is not None else pd.DataFrame(columns=["dice1","dice2","dice3","total"])
            merged = pd.concat([base, up], ignore_index=True)
            merged = clean_and_standardize(merged).drop_duplicates()
            STORE.df = merged
            STORE.version += 1
            fname = STORE.active_file
        # background sync since auto-sync is ON by default in UI
        schedule_sync_to_hf()
        return f"✅ Merged upload into {fname} ({len(merged)} rows).", merged, data_stats_line(merged, fname), make_totals_sparkline(merged)
    except Exception as e:
        with STORE.lock:
            df = STORE.df; fname = STORE.active_file
        return f"Upload error: {e}", df, data_stats_line(df, fname), make_totals_sparkline(df if df is not None else pd.DataFrame())

def on_save_local():
    with STORE.lock:
        df = STORE.df; fname = STORE.active_file
    if not isinstance(df, pd.DataFrame) or len(df)==0:
        return gr.File.update(value=None, visible=False), "Nothing to save.", data_stats_line(df, fname)
    out = fname
    df.to_excel(out, index=False)
    return gr.File.update(value=out, visible=True), f"💾 Saved locally: {len(df)} rows.", data_stats_line(df, fname)

def on_check_token_repo():
    token_present = bool(os.getenv("HF_TOKEN"))
    try:
        info = whoami(token=os.getenv("HF_TOKEN")) if token_present else None
        user = info.get("name") if info else None
    except Exception as e:
        user = f"whoami failed: {e}"
    with STORE.lock:
        fname = STORE.active_file
    return f"Token: {'✅ present' if token_present else '❌ missing'} | User: {user or '—'} | Dataset: {PRIMARY_REPO} | Active file: {fname}"

# ================== Train Tab (instant sync) ==================
def on_train_submit(mode, td1, td2, td3):
    filename = FILE_DIGITAL if mode == "Digital Game" else FILE_LIVE
    try:
        d1 = int(td1); d2 = int(td2); d3 = int(td3)
    except Exception:
        return f"Pick all dice 1–6 for {mode}.", ""
    if not all(1 <= x <= 6 for x in [d1,d2,d3]):
        return "Each die must be 1..6.", ""
    # Append → blocking upload to avoid loss
    df_new = append_triple_to_file(filename, (d1,d2,d3))
    # If the Predict tab is currently using this file, refresh memory too
    with STORE.lock:
        if STORE.active_file == filename:
            STORE.df = df_new
            STORE.version += 1
    return f"✅ Saved to {filename} (rows: {len(df_new)})", f"({d1},{d2},{d3}) total={d1+d2+d3}"

# ================== Danger Zone (prevent accidental delete) ==================
def on_blank_file(which_file, confirm_text):
    filename = { "Primary": FILE_PRIMARY, "Digital": FILE_DIGITAL, "Live": FILE_LIVE }.get(which_file, FILE_PRIMARY)
    required = f"BLANK {filename}"
    if (confirm_text or "").strip() != required:
        return f"Type exactly: {required}", ""
    # safety: create a backup copy first
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_key = f"backup/{stamp}_{filename}"
    try:
        # download current file
        cur = load_dataset_df_file(filename)
        # upload backup
        tmp_bak = f"_tmp_{stamp}_{filename}"
        cur.to_excel(tmp_bak, index=False)
        api = _api()
        api.upload_file(path_or_fileobj=tmp_bak, path_in_repo=backup_key,
                        repo_id=PRIMARY_REPO, repo_type="dataset",
                        commit_message=f"Backup before blanking {filename}")
        # now blank
        empty = pd.DataFrame(columns=["dice1","dice2","dice3","total"])
        push_df_file(empty, filename, msg=f"Blank {filename}")
        # update memory if active
        with STORE.lock:
            if STORE.active_file == filename:
                STORE.df = empty
                STORE.version += 1
        return f"✅ Blanked {filename}. Backup saved at {backup_key}", "Done."
    except Exception as e:
        return f"❌ Failed: {e}", ""

# ================== UI ==================
CSS = """
.gradio-container {max-width: 520px !important; margin:auto; padding: 8px;}
button {min-height:48px;}
.smallbtn {min-height:40px;}
.muted{color:#64748b; font-size:12px;}
.tblwrap {overflow-x:auto;}
.tbl{width:100%; border-collapse:collapse; font-size:12px;}
.tbl th,.tbl td{padding:6px 8px; border-bottom:1px solid #e5e7eb; text-align:left;}
.dice-grid .wrap .gr-radio {display:flex; flex-wrap:wrap; gap:6px;}
.dice-grid .wrap .gr-radio .item {flex:1 1 calc(33.33% - 6px); text-align:center;}
#btn-predict > button, #btn-predict button,
#btn-correct > button, #btn-correct button { background:#16a34a !important; border:1px solid #15803d !important; color:#fff !important; }
#btn-wrong > button, #btn-wrong button { background:#ef4444 !important; border:1px solid #dc2626 !important; color:#fff !important; }
#btn-predict button:hover, #btn-correct button:hover, #btn-wrong button:hover { filter: brightness(.95); }
#btn-predict button:active, #btn-correct button:active, #btn-wrong button:active { transform: translateY(1px); }
"""

boot_msg, df_init, clf_init, acc_init = boot_load(active_file=FILE_PRIMARY)

with STORE.lock:
    active_name = STORE.active_file
data_init = data_stats_line(df_init, active_name)
spark_init = make_totals_sparkline(df_init if df_init is not None else pd.DataFrame())

with gr.Blocks(theme=gr.themes.Soft(),
               title="Dice Predictor — FAST + TRAIN",
               css=CSS,
               analytics_enabled=False) as demo:

    with gr.Tabs():

        # -------------------------
        # PREDICT TAB (fast & compact)
        # -------------------------
        with gr.Tab("Predict", id="tab-predict"):
            # Session states (UI only)
            last_input_state  = gr.State(None)
            last_pred_state   = gr.State(None)
            pending_state     = gr.State(False)
            last_topk_state   = gr.State(None)

            sess_total_state   = gr.State(0)
            sess_correct_state = gr.State(0)
            sess_wrong_state   = gr.State(0)
            sess_top5_state    = gr.State(0)

            # Top info
            top_line = gr.Markdown(boot_msg, elem_classes=["muted"])
            stats_md = gr.Markdown(stats_line(0,0,0,0))
            data_md  = gr.Markdown(data_init)
            log_md   = gr.Markdown("", elem_classes=["muted"])
            spark    = gr.Plot(value=spark_init, show_label=False)

            with gr.Accordion("Data source", open=False):
                active_file = gr.Radio(
                    choices=[FILE_PRIMARY, FILE_DIGITAL, FILE_LIVE],
                    value=FILE_PRIMARY, label="Active file for Predict"
                )
                reload_btn = gr.Button("Reload this file", elem_classes=["smallbtn"])

            with gr.Row(elem_classes=["dice-grid"]):
                d1 = gr.Radio(choices=["1","2","3","4","5","6"], value="3", label="Dice 1", interactive=True)
                d2 = gr.Radio(choices=["1","2","3","4","5","6"], value="2", label="Dice 2", interactive=True)
                d3 = gr.Radio(choices=["1","2","3","4","5","6"], value="5", label="Dice 3", interactive=True)

            with gr.Row():
                predict_btn = gr.Button("Predict", elem_id="btn-predict")

            pred_line   = gr.Markdown("")
            top5_html   = gr.HTML("")
            totals_html = gr.HTML("")
            fair_md     = gr.Markdown("")
            cats_html   = gr.HTML("")

            with gr.Row():
                accept_btn = gr.Button("Correct ✅", elem_id="btn-correct", elem_classes=["smallbtn"])
                learn_btn  = gr.Button("Wrong → Learn", elem_id="btn-wrong",   elem_classes=["smallbtn"])

            with gr.Row(elem_classes=["dice-grid"]):
                nd1 = gr.Radio(choices=["1","2","3","4","5","6"], label="correct nd1", interactive=True)
                nd2 = gr.Radio(choices=["1","2","3","4","5","6"], label="correct nd2", interactive=True)
                nd3 = gr.Radio(choices=["1","2","3","4","5","6"], label="correct nd3", interactive=True)

            with gr.Accordion("Speed & Behavior (defaults OK)", open=False):
                with gr.Row():
                    trees = gr.Slider(50, 400, value=180, step=10, label="RF trees (retrain)")
                    window_n = gr.Slider(50, 1000, value=300, step=10, label="Recent window size")
                with gr.Row():
                    w_rf   = gr.Slider(0, 1, value=0.70, step=0.05, label="RF weight")
                    w_tot  = gr.Slider(0, 1, value=0.20, step=0.05, label="Total Markov weight")
                with gr.Row():
                    w_die  = gr.Slider(0, 1, value=0.07, step=0.01, label="Per-die transition weight")
                    w_bias = gr.Slider(0, 1, value=0.03, step=0.01, label="Per-die bias weight")
                with gr.Row():
                    temp = gr.Slider(0.3, 2.0, value=1.0, step=0.05, label="Temperature")
                    alpha_smooth = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Laplace α")
                with gr.Row():
                    ph_lambda = gr.Slider(10, 200, value=50, step=5, label="PH drift threshold (λ)")
                    triples_lose = gr.Checkbox(value=True, label="Triples lose on Big/Small & Odd/Even")
                with gr.Row():
                    auto_predict = gr.Checkbox(value=True,  label="Auto-predict after Correct/Learn")
                    bg_retrain   = gr.Checkbox(value=True,  label="Background retrain")
                    auto_sync    = gr.Checkbox(value=True,  label="Auto-sync to HF in background")  # DEFAULT ON ✅

            with gr.Accordion("Dataset / Tools", open=False):
                with gr.Row():
                    file_in = gr.File(label="Upload .xlsx (merge into ACTIVE file)", file_types=[".xlsx"])
                    upload_btn = gr.Button("Upload & MERGE", elem_classes=["smallbtn"])
                with gr.Row():
                    sync_btn = gr.Button("Sync NOW to HF", elem_classes=["smallbtn"])
                    retrain_btn = gr.Button("Retrain NOW", elem_classes=["smallbtn"])
                with gr.Row():
                    save_btn = gr.Button("Save & Download (local copy)", elem_classes=["smallbtn"])
                    saved_file = gr.File(label="", visible=False)
                with gr.Row():
                    check_btn  = gr.Button("Token/Repo Check", elem_classes=["smallbtn"])
                check_out = gr.Markdown("", elem_classes=["muted"])

            # Wiring — source reload
            def _on_reload_button(active_file_value):
                msg, df, clf, acc, data_line, spark_fig = on_reload_active(active_file_value)
                with STORE.lock:
                    STORE.active_file = active_file_value
                    STORE.df = df
                    STORE.version += 1
                    STORE.model, STORE.acc = clf, acc
                    STORE.model_version = STORE.version
                return msg, data_line, spark_fig

            reload_btn.click(_on_reload_button, inputs=[active_file],
                             outputs=[top_line, data_md, spark])

            # Wiring — predict
            predict_btn.click(
                on_predict,
                inputs=[d1, d2, d3,
                        last_input_state, last_pred_state, pending_state,
                        sess_total_state, sess_correct_state, sess_wrong_state, sess_top5_state, log_md,
                        w_rf, w_tot, w_die, w_bias, temp, alpha_smooth,
                        window_n, ph_lambda, triples_lose],
                outputs=[pred_line, top5_html, totals_html, fair_md,
                         last_input_state, last_pred_state, pending_state,
                         stats_md, data_md, log_md, last_topk_state, cats_html]
            )

            # ACCEPT → maybe auto-predict
            def maybe_auto_predict(auto_flag, d1, d2, d3, *rest):
                if not auto_flag:
                    return (gr.update(), gr.update(), gr.update(), gr.update(),
                            rest[4], rest[5], rest[6], gr.update(), gr.update(), rest[9], rest[10], gr.update())
                return on_predict(d1, d2, d3, *rest)

            accept_btn.click(
                on_accept,
                inputs=[last_input_state, last_pred_state, last_topk_state, pending_state,
                        sess_total_state, sess_correct_state, sess_wrong_state, sess_top5_state,
                        auto_sync, bg_retrain, trees, log_md],
                outputs=[log_md,              # message
                         gr.State(), gr.State(), gr.State(),  # df/model/acc are internal (not displayed)
                         sess_total_state, sess_correct_state, sess_wrong_state, sess_top5_state,
                         pending_state, stats_md, data_md, log_md,
                         d1, d2, d3, last_input_state, spark]
            ).then(
                maybe_auto_predict,
                inputs=[auto_predict,
                        d1, d2, d3,
                        last_input_state, last_pred_state, pending_state,
                        sess_total_state, sess_correct_state, sess_wrong_state, sess_top5_state, log_md,
                        w_rf, w_tot, w_die, w_bias, temp, alpha_smooth,
                        window_n, ph_lambda, triples_lose],
                outputs=[pred_line, top5_html, totals_html, fair_md,
                         last_input_state, last_pred_state, pending_state,
                         stats_md, data_md, log_md, last_topk_state, cats_html]
            )

            # LEARN → maybe auto-predict
            learn_btn.click(
                on_learn,
                inputs=[nd1, nd2, nd3,
                        last_input_state, last_topk_state, pending_state,
                        sess_total_state, sess_correct_state, sess_wrong_state, sess_top5_state,
                        auto_sync, bg_retrain, trees, log_md],
                outputs=[log_md,
                         gr.State(), gr.State(), gr.State(),
                         sess_total_state, sess_correct_state, sess_wrong_state, sess_top5_state,
                         pending_state, stats_md, data_md, log_md,
                         d1, d2, d3, last_input_state, spark]
            ).then(
                maybe_auto_predict,
                inputs=[auto_predict,
                        d1, d2, d3,
                        last_input_state, last_pred_state, pending_state,
                        sess_total_state, sess_correct_state, sess_wrong_state, sess_top5_state, log_md,
                        w_rf, w_tot, w_die, w_bias, temp, alpha_smooth,
                        window_n, ph_lambda, triples_lose],
                outputs=[pred_line, top5_html, totals_html, fair_md,
                         last_input_state, last_pred_state, pending_state,
                         stats_md, data_md, log_md, last_topk_state, cats_html]
            )

            # Upload & merge into ACTIVE file (auto-sync runs in background)
            upload_btn.click(
                on_upload_merge,
                inputs=[file_in],
                outputs=[top_line, gr.State(), data_md, spark]
            )

            # Manual sync & retrain (background)
            sync_btn.click(on_sync_now, outputs=[log_md, data_md])
            retrain_btn.click(on_retrain_now, inputs=[trees], outputs=[log_md, data_md])

            # Save local
            save_btn.click(on_save_local, outputs=[saved_file, top_line, data_md])

            # Token check
            check_btn.click(on_check_token_repo, outputs=[check_out])

            # ---------- Danger Zone (blank with backup + confirmation) ----------
            with gr.Accordion("Danger Zone (blank a file) — protected", open=False):
                which_file = gr.Radio(choices=["Primary","Digital","Live"], value="Primary", label="Which file to BLANK")
                confirm_text = gr.Textbox(label="Type confirmation", placeholder=f"Type: BLANK {FILE_PRIMARY}")
                blank_btn = gr.Button("Blank file (creates backup first)", elem_classes=["smallbtn"])
                danger_out = gr.Markdown("", elem_classes=["muted"])
                blank_btn.click(on_blank_file, inputs=[which_file, confirm_text], outputs=[danger_out, top_line])

        # -------------------------
        # TRAIN TAB (instant sync)
        # -------------------------
        with gr.Tab("Train (Digital / Live)", id="tab-train"):
            train_mode = gr.Radio(choices=["Digital Game", "Live"], value="Digital Game", label="Database")
            with gr.Row(elem_classes=["dice-grid"]):
                td1 = gr.Radio(choices=["1","2","3","4","5","6"], value="3", label="Dice 1", interactive=True)
                td2 = gr.Radio(choices=["1","2","3","4","5","6"], value="2", label="Dice 2", interactive=True)
                td3 = gr.Radio(choices=["1","2","3","4","5","6"], value="5", label="Dice 3", interactive=True)
            train_btn = gr.Button("Submit (Instant Sync)")
            train_status = gr.Markdown("", elem_classes=["muted"])
            last_saved = gr.Markdown("", elem_classes=["muted"])

            def after_train_refresh():
                # refresh Predict tab's data line if active file == submitted file
                with STORE.lock:
                    df = STORE.df; fname = STORE.active_file
                return data_stats_line(df, fname)

            train_btn.click(on_train_submit, inputs=[train_mode, td1, td2, td3], outputs=[train_status, last_saved])\
                     .then(after_train_refresh, outputs=[data_md])

# Launch
if __name__ == "__main__":
    PORT = int(os.getenv("PORT", "7860"))
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        ssr_mode=False,
        show_api=False,
        show_error=True,
        share=False,
        prevent_thread_lock=False
    )
