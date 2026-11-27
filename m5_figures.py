"""Milestone 5 figures and numbers.

Refits the Milestone 4 final model (tuned Logistic Regression + isotonic
calibration), then writes every figure used in the report and slides plus a
JSON file with every number quoted in them.

Run from a folder that holds the three TTC xlsx files:
    python m5_figures.py <output_dir>
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

warnings.filterwarnings("ignore")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "m5_out")
(OUT / "report").mkdir(parents=True, exist_ok=True)
(OUT / "slides").mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 313
THRESHOLD = 0.088            # F2-optimal threshold chosen in Milestone 4 (out-of-fold, training only)
DAYS_COVERED = 731           # 1 Jan 2023 to 31 Dec 2024

# --------------------------------------------------------------------------
# Data: identical cleaning and split to Milestones 3 and 4
# --------------------------------------------------------------------------
LINE_MAP = {
    "YU": "YU", "YUS": "YU", "LINE 1": "YU", "YONGE UNIVERSITY": "YU",
    "YONGE-UNIVERSITY": "YU", "YU LINE 1": "YU",
    "BD": "BD", "LINE 2": "BD", "BD LINE 2": "BD", "BLOOR DANFORTH": "BD",
    "BLOOR-DANFORTH": "BD",
    "SHP": "SHP", "SHEP": "SHP", "LINE 4": "SHP", "SHEPPARD": "SHP",
    "SRT": "SRT", "LINE 3": "SRT", "SCARBOROUGH RT": "SRT",
}
MULTI = ["YU/BD", "YU / BD", "YU/ BD", "YUS/BD", "BD/YU", "YU & BD", "YU-BD"]


def clean_line(v):
    if pd.isna(v):
        return "Unknown"
    t = str(v).strip().upper()
    if t in MULTI or ("YU" in t and "BD" in t):
        return "Multiple lines"
    return LINE_MAP.get(t, "Other")


def load():
    df = pd.concat([pd.read_excel("ttc-subway-delay-data-2023.xlsx"),
                    pd.read_excel("ttc-subway-delay-data-2024.xlsx")], ignore_index=True)
    raw_rows = len(df)
    df["Prolonged_Disruption"] = (df["Min Delay"] > 15).astype(int)
    df["Month"] = df["Date"].dt.month
    df["Season"] = df["Month"].map({1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                                    6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall",
                                    11: "Fall", 12: "Winter"})
    df["Is_Weekend"] = df["Day"].isin(["Saturday", "Sunday"]).astype(int)
    df["Line_Clean"] = df["Line"].apply(clean_line)
    df["Station_Clean"] = df["Station"].str.upper().str.strip().str.replace(r"\s+", " ", regex=True)
    df = df.drop_duplicates(subset=["Date", "Time", "Station", "Code", "Min Delay", "Min Gap", "Line"], keep="first")
    df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M").dt.hour
    df["Is_Rush_Hour"] = ((df["Hour"].between(6, 9) | df["Hour"].between(15, 19)) & (df["Is_Weekend"] == 0)).astype(int)
    codes = pd.read_excel("ttc-subway-delay-codes.xlsx", header=None)
    sub = codes.iloc[2:, [2, 3]].dropna(); sub.columns = ["Code", "Code_Description"]
    srt = codes.iloc[2:, [6, 7]].dropna(); srt.columns = ["Code", "Code_Description"]
    lookup = (pd.concat([sub, srt], ignore_index=True)
              .assign(Code=lambda d: d["Code"].str.strip()).drop_duplicates("Code"))
    df = df.merge(lookup, on="Code", how="left")
    df["Code_Description"] = df["Code_Description"].fillna("No code available / undocumented")
    df["stratify_group"] = (df["Prolonged_Disruption"].astype(str) + "_" + df["Season"]
                            + "_" + df["Is_Weekend"].astype(str))
    tr, te = train_test_split(df, test_size=0.20, random_state=RANDOM_STATE, stratify=df["stratify_group"])
    return raw_rows, tr.drop(columns="stratify_group").reset_index(drop=True), te.drop(columns="stratify_group").reset_index(drop=True)


raw_rows, train_df, test_df = load()
all_df = pd.concat([train_df.assign(Split="train"), test_df.assign(Split="test")], ignore_index=True)

low_card = ["Line_Clean", "Bound"]
high_card = ["Code", "Station_Clean"]
numeric = ["Hour", "Is_Rush_Hour"]
FEATURES = low_card + high_card + numeric


def target_encoder(smooth):
    if tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 9):
        return TargetEncoder(smooth=smooth, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE))
    return TargetEncoder(smooth=smooth, random_state=RANDOM_STATE)


def tuned_lr():
    pre = ColumnTransformer([
        ("low_card", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                               ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), low_card),
        ("high_card", target_encoder(5), high_card),
        ("num", StandardScaler(), numeric)])
    model = LogisticRegression(C=1.1450033889997997, class_weight="balanced", max_iter=5000,
                               random_state=RANDOM_STATE)
    return Pipeline([("preprocessor", pre), ("model", model)])


def prep(d):
    X = d[FEATURES].copy()
    X["Bound"] = X["Bound"].fillna("Unknown")
    return X


X_train, y_train = prep(train_df), train_df["Prolonged_Disruption"].values
X_test, y_test = prep(test_df), test_df["Prolonged_Disruption"].values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

final = CalibratedClassifierCV(tuned_lr(), method="isotonic", cv=5).fit(X_train, y_train)
p_test = final.predict_proba(X_test)[:, 1]
p_oof = cross_val_predict(CalibratedClassifierCV(tuned_lr(), method="isotonic", cv=5),
                          X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

code_baseline = Pipeline([("enc", ColumnTransformer([("code", target_encoder("auto"), ["Code"])])),
                          ("m", LogisticRegression(max_iter=1000))]).fit(X_train, y_train)
p_code_test = code_baseline.predict_proba(X_test)[:, 1]

# Out-of-sample prediction for every incident: OOF for training rows, final model for test rows
all_df["p"] = np.concatenate([p_oof, p_test])
all_df["flag"] = all_df["p"] >= THRESHOLD
all_df["y"] = all_df["Prolonged_Disruption"]

N = {}  # every number quoted in the deliverables
SHORT = {"SUDP": "Disorderly patron", "SUO": "Passenger other", "MUIS": "Injured/ill, no aid",
         "SUUT": "Person at track level", "MUI": "Injured/ill, transported",
         "MUIR": "Injured/ill, refused aid", "SUAP": "Assault, patron involved", "PUOPO": "OPTO signal issue",
         "SUEAS": "Station emergency alarm", "PUTWZ": "Track work-zone problem", "SUPOL": "Police investigation",
         "MUPAA": "Passenger alarm, no trouble", "TUSC": "Crew change"}


def put(k, v):
    N[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v


# --------------------------------------------------------------------------
# Style (reference palette from the dataviz skill)
# --------------------------------------------------------------------------
BLUE, ORANGE, AQUA, RED = "#257de0", "#f47230", "#00b894", "#dc3c3d"
GRAY, LIGHT_GRAY = "#a3a3a3", "#d9d9d9"
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"


def style(base):
    plt.rcParams.update({
        "font.family": "Inter", "font.size": base, "axes.titlesize": base + 1,
        "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.labelsize": base, "axes.labelcolor": TEXT2, "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "axes.edgecolor": LIGHT_GRAY, "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": base - 1,
        "text.color": TEXT, "figure.dpi": 100, "savefig.dpi": 220, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06, "lines.linewidth": 2,
    })


def save(fig, name, kind):
    if kind == "slides" and name in ("f1_imbalance", "f2_codes"):
        fig.axes[0].set_title("")
    fig.savefig(OUT / kind / f"{name}.png", facecolor="white")
    plt.close(fig)


def per_day(count, n_rows):
    """Convert a count in a sample of n_rows incidents to incidents per day."""
    return count / (DAYS_COVERED * n_rows / len(all_df))


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------
put("raw_rows", raw_rows); put("rows", len(all_df)); put("train_rows", len(train_df)); put("test_rows", len(test_df))
put("incidents_per_day", len(all_df) / DAYS_COVERED)
put("prolonged_rate", all_df["y"].mean()); put("prolonged_n", all_df["y"].sum())
put("prolonged_per_day", all_df["y"].sum() / DAYS_COVERED)
put("zero_delay_share", (all_df["Min Delay"] == 0).mean())
put("median_nonzero_delay", all_df.loc[all_df["Min Delay"] > 0, "Min Delay"].median())
put("delay_over_60_n", (all_df["Min Delay"] > 60).sum())
put("max_delay", all_df["Min Delay"].max())
put("bound_missing_share", all_df["Bound"].isna().mean())
put("prolonged_rate_bound_missing", all_df.loc[all_df["Bound"].isna(), "y"].mean())
put("prolonged_rate_bound_present", all_df.loc[all_df["Bound"].notna(), "y"].mean())
put("n_codes", all_df["Code"].nunique()); put("n_stations", all_df["Station_Clean"].nunique())

flag_t = p_test >= THRESHOLD
tn, fp, fn, tp = confusion_matrix(y_test, flag_t).ravel()
for k, v in dict(tn=tn, fp=fp, fn=fn, tp=tp).items():
    put(f"test_{k}", v)
put("test_pr_auc", average_precision_score(y_test, p_test)); put("test_roc_auc", roc_auc_score(y_test, p_test))
put("test_brier", brier_score_loss(y_test, p_test)); put("test_prevalence", y_test.mean())
put("test_recall", tp / (tp + fn)); put("test_precision", tp / (tp + fp)); put("test_flag_rate", flag_t.mean())
put("code_pr_auc", average_precision_score(y_test, p_code_test)); put("code_roc_auc", roc_auc_score(y_test, p_code_test))
put("oof_pr_auc", average_precision_score(y_train, p_oof)); put("oof_roc_auc", roc_auc_score(y_train, p_oof))
put("share_p_over_50", (all_df["p"] >= 0.5).mean())
put("share_pos_p_5_to_30", all_df.loc[all_df["y"] == 1, "p"].between(0.05, 0.30).mean())
put("alerts_per_day", per_day(tp + fp, len(y_test))); put("caught_per_day", per_day(tp, len(y_test)))
put("missed_per_day", per_day(fn, len(y_test))); put("false_alarms_per_day", per_day(fp, len(y_test)))

rng = np.random.default_rng(RANDOM_STATE)
boot = []
for _ in range(1000):
    i = rng.integers(0, len(y_test), len(y_test))
    boot.append((average_precision_score(y_test[i], p_test[i]),
                 average_precision_score(y_test[i], p_test[i]) - average_precision_score(y_test[i], p_code_test[i])))
boot = np.array(boot)
put("test_pr_auc_ci", list(np.percentile(boot[:, 0], [2.5, 97.5])))
put("gain_over_code_ci", list(np.percentile(boot[:, 1], [2.5, 97.5])))

ops = []
for t in [0.03, 0.05, THRESHOLD, 0.15, 0.20, 0.30]:
    f = p_test >= t
    tp_ = int((f & (y_test == 1)).sum()); fp_ = int((f & (y_test == 0)).sum())
    ops.append({"threshold": t, "alerts_per_day": per_day(tp_ + fp_, len(y_test)),
                "caught_per_day": per_day(tp_, len(y_test)), "recall": tp_ / y_test.sum(),
                "precision": tp_ / max(tp_ + fp_, 1), "false_alarms_per_hit": fp_ / max(tp_, 1)})
put("operating_points", ops)

# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def fig_imbalance(kind, w, h, base):
    style(base)
    bins = [-1, 0, 5, 10, 15, 30, 60, 10_000]
    labels = ["0", "1-5", "6-10", "11-15", "16-30", "31-60", ">60"]
    counts = pd.cut(all_df["Min Delay"], bins, labels=labels).value_counts().reindex(labels)
    share = counts / counts.sum()
    fig, ax = plt.subplots(figsize=(w, h))
    colors = [LIGHT_GRAY] * 4 + [BLUE] * 3
    bars = ax.bar(labels, share * 100, color=colors, width=0.72)
    for b, s, n in zip(bars, share, counts):
        ax.text(b.get_x() + b.get_width() / 2, s * 100 + 1.2, f"{s:.1%}\n({n:,})", ha="center", va="bottom",
                fontsize=base - 2, color=TEXT2, linespacing=1.1)
    ax.axvline(3.5, color=TEXT, lw=1)
    ax.text(3.58, 66, "15-minute line:\nshuttle decision", fontsize=base - 1, va="top")
    ax.text(5, 30, f"Prolonged: {all_df['y'].mean():.1%} of incidents\n(about {N['prolonged_per_day']:.1f} per day)",
            ha="center", fontsize=base - 1, color=BLUE, fontweight="semibold")
    ax.set_ylim(0, 75); ax.set_ylabel("Share of incidents (%)"); ax.set_xlabel("Recorded delay (minutes)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Most logged incidents cause no delay; 3.2% exceed 15 minutes")
    save(fig, "f1_imbalance", kind)


def fig_codes(kind, w, h, base):
    style(base)
    g = (all_df.groupby("Code").agg(n=("y", "size"), pos=("y", "sum"), rate=("y", "mean"),
                                   desc=("Code_Description", "first")).query("n >= 150"))
    fig, ax = plt.subplots(figsize=(w, h))
    ax.scatter(g["n"], g["rate"] * 100, s=12 + g["pos"] * 1.6, color=BLUE, alpha=0.55,
               edgecolor="white", linewidth=1)
    near = {"MUI": (10, -4), "SUUT": (12, -4), "SUEAS": (10, -4), "PUTWZ": (10, -6)}
    far = {"MUIR": (300, 10.5), "SUAP": (300, 8.3), "SUO": (2000, 12.0), "SUDP": (2000, 9.0)}
    for code, r in g.iterrows():
        label = f"{code} {SHORT.get(code, '')}: {r['rate']:.1%} of {int(r['n']):,}"
        if code in near:
            ax.annotate(label, (r["n"], r["rate"] * 100), xytext=near[code], textcoords="offset points",
                        fontsize=base - 2.2, color=TEXT, va="center")
        elif code in far:
            ax.annotate(label, (r["n"], r["rate"] * 100), xytext=far[code], textcoords="data",
                        fontsize=base - 2.2, color=TEXT, va="center",
                        arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.8, "shrinkA": 2, "shrinkB": 4})
    ax.axhline(all_df["y"].mean() * 100, color=GRAY, lw=1)
    ax.text(520, all_df["y"].mean() * 100 + 0.45, "network average 3.2%", fontsize=base - 2, color=TEXT2, ha="left")
    ax.set_xscale("log"); ax.set_xlabel("Incidents with this code, 2023-2024 (log scale)")
    ax.set_xticks([200, 500, 1000, 2000, 5000]); ax.set_xticklabels(["200", "500", "1,000", "2,000", "5,000"])
    ax.minorticks_off(); ax.set_ylim(-1, 23); ax.set_xlim(130, 9500)
    ax.set_ylabel("Share that became prolonged (%)")
    ax.set_title("The incident code sets the risk: rare serious codes vs. common minor ones")
    ax.text(0.0, -0.2, "Each bubble is one incident code with at least 150 incidents; bubble area = number of prolonged incidents.",
            transform=ax.transAxes, ha="left", va="top", fontsize=base - 2.5, color=TEXT2)
    save(fig, "f2_codes", kind)
    top = g.sort_values("pos", ascending=False).head(8)
    put("codes_table", [{"code": c, "desc": r["desc"], "n": int(r["n"]), "pos": int(r["pos"]), "rate": float(r["rate"])}
                        for c, r in top.iterrows()])


def fig_time(kind, w, h, base):
    style(base)
    m = all_df.assign(ym=all_df["Date"].dt.to_period("M")).groupby("ym").agg(
        n=("y", "size"), rate=("y", "mean"), pred=("p", "mean"))
    x = m.index.to_timestamp()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(w, h), sharex=True, gridspec_kw={"height_ratios": [1, 1.25]})
    a1.bar(x, m["n"], width=24, color=LIGHT_GRAY)
    a1.set_ylabel("Incidents"); a1.set_title("Monthly incidents (top) and prolonged-disruption rate (bottom)")
    a1.grid(axis="x", visible=False)
    a2.plot(x, m["rate"] * 100, color=BLUE, marker="o", ms=4, label="Actual rate")
    a2.plot(x, m["pred"] * 100, color=ORANGE, marker="o", ms=4, label="Model's average prediction")
    srt_end = pd.Timestamp("2023-07-24")
    for a in (a1, a2):
        a.axvline(srt_end, color=TEXT2, lw=1)
    a2.text(srt_end + pd.Timedelta(days=8), 0.4, "Line 3 (SRT) closes, Jul 2023", fontsize=base - 2, color=TEXT2, va="bottom")
    a2.set_ylabel("Prolonged (%)"); a2.set_ylim(0, max(m["rate"].max(), m["pred"].max()) * 100 * 1.25)
    a2.legend(loc="upper right", ncol=2)
    a2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    a2.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
    save(fig, "f3_time", kind)
    put("monthly_rate_min", m["rate"].min()); put("monthly_rate_max", m["rate"].max())
    put("monthly_abs_gap_max", (m["rate"] - m["pred"]).abs().max())
    put("monthly_n_min", m["n"].min()); put("monthly_n_max", m["n"].max())
    by_year = all_df.groupby(all_df["Date"].dt.year)["y"].mean()
    put("rate_2023", by_year[2023]); put("rate_2024", by_year[2024])
    hr = all_df.groupby("Hour")["y"].mean()
    put("hour_rate_max_hour", int(hr.idxmax())); put("hour_rate_max", hr.max())
    put("hour_rate_min_hour", int(hr.idxmin())); put("hour_rate_min", hr.min())


def fig_importance(kind, w, h, base):
    style(base)
    perm = permutation_importance(final, X_test, y_test, scoring="average_precision", n_repeats=20,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    names = {"Code": "Incident code", "Station_Clean": "Station", "Bound": "Direction (Bound)",
             "Line_Clean": "Line", "Hour": "Hour of day", "Is_Rush_Hour": "Rush hour flag"}
    imp = pd.DataFrame({"f": FEATURES, "m": perm.importances_mean, "s": perm.importances_std}).sort_values("m")
    put("perm_importance", {r.f: [float(r.m), float(r.s)] for r in imp.itertuples()})

    # Exact SHAP values for a linear model (independent-feature form): coef * (x - mean), in log-odds
    lr = tuned_lr().fit(X_train, y_train)
    pre, coef = lr.named_steps["preprocessor"], lr.named_steps["model"].coef_[0]
    cols = pre.get_feature_names_out()
    Xt_test = pre.transform(X_test); Xt_mean = pre.transform(X_train).mean(axis=0)
    contrib = (Xt_test - Xt_mean) * coef
    raw = [c.split("__", 1)[1] for c in cols]
    group = [next(f for f in FEATURES if r == f or r.startswith(f + "_")) for r in raw]
    shap = pd.DataFrame(contrib, columns=cols).T.groupby(group).sum().T
    mean_abs = shap.abs().mean().sort_values()
    put("shap_mean_abs", mean_abs.to_dict())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(w, h), gridspec_kw={"wspace": 0.55})
    a1.barh([names[f] for f in imp["f"]], imp["m"], xerr=imp["s"], color=BLUE, height=0.6,
            error_kw={"ecolor": TEXT2, "elinewidth": 1, "capsize": 2})
    a1.set_xlabel("Drop in test PR-AUC when shuffled")
    a1.set_title("Permutation importance (test set)"); a1.grid(axis="y", visible=False)
    order = mean_abs.index.tolist()
    data = [shap[f].values for f in order]
    parts = a2.violinplot(data, vert=False, showextrema=False, widths=0.8)
    for b in parts["bodies"]:
        b.set_facecolor(BLUE); b.set_alpha(0.45); b.set_edgecolor("none")
    a2.scatter(mean_abs.values, range(1, len(order) + 1), color=ORANGE, s=22, zorder=3, label="Mean |SHAP|")
    a2.set_yticks(range(1, len(order) + 1)); a2.set_yticklabels([names[f] for f in order])
    a2.axvline(0, color=TEXT2, lw=0.8); a2.set_xlim(-3, 4.5)
    a2.set_xlabel("SHAP value (log-odds of a prolonged disruption)")
    a2.set_title("SHAP values per incident (test set)"); a2.grid(axis="y", visible=False)
    a2.legend(loc="lower right")
    a2.text(1.0, -0.3, "Tuned LR before calibration; axis cut at 4.5 (a few rare codes reach 11).",
            transform=a2.transAxes, ha="right", va="top", fontsize=base - 2.5, color=TEXT2)
    save(fig, "f4_importance", kind)


def fig_curves(kind, w, h, base):
    style(base)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(w, h), gridspec_kw={"wspace": 0.3})
    for p, lab, col in [(p_test, "Final model", BLUE), (p_code_test, "Incident code only", GRAY)]:
        pr, rc, _ = precision_recall_curve(y_test, p)
        a1.plot(rc, pr, color=col, lw=2 if col == BLUE else 1.5,
                label=f"{lab} (PR-AUC {average_precision_score(y_test, p):.3f})")
        fpr, tpr, _ = roc_curve(y_test, p)
        a2.plot(fpr, tpr, color=col, lw=2 if col == BLUE else 1.5,
                label=f"{lab} (ROC-AUC {roc_auc_score(y_test, p):.3f})")
    a1.axhline(y_test.mean(), color=TEXT2, lw=1)
    a1.text(0.98, y_test.mean() + 0.03, f"no skill ({y_test.mean():.3f})", ha="right", fontsize=base - 2, color=TEXT2)
    a1.scatter([N["test_recall"]], [N["test_precision"]], color=ORANGE, s=45, zorder=4,
               label=f"Alert threshold {THRESHOLD}")
    a2.plot([0, 1], [0, 1], color=TEXT2, lw=1)
    fpr_op = fp / (fp + tn)
    a2.scatter([fpr_op], [N["test_recall"]], color=ORANGE, s=45, zorder=4)
    a1.set(xlabel="Recall (share of prolonged incidents caught)", ylabel="Precision (share of alerts that were right)",
           xlim=(0, 1), ylim=(0, 1))
    a2.set(xlabel="False-positive rate", ylabel="True-positive rate (recall)", xlim=(0, 1), ylim=(0, 1))
    a1.set_title("Precision-recall curve"); a2.set_title("ROC curve")
    a1.legend(loc="upper right"); a2.legend(loc="lower right")
    save(fig, "f5_curves", kind)
    put("test_fpr", fpr_op)



def fig_performance(kind, w, h, base):
    """PR curve, ROC curve and calibration in one row (report)."""
    style(base)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(w, h), gridspec_kw={"wspace": 0.38})
    for p, lab, col in [(p_test, "Final model", BLUE), (p_code_test, "Code only", GRAY)]:
        pr, rc, _ = precision_recall_curve(y_test, p)
        a1.plot(rc, pr, color=col, lw=1.8 if col == BLUE else 1.3, label=f"{lab} ({average_precision_score(y_test, p):.3f})")
        fpr, tpr, _ = roc_curve(y_test, p)
        a2.plot(fpr, tpr, color=col, lw=1.8 if col == BLUE else 1.3, label=f"{lab} ({roc_auc_score(y_test, p):.3f})")
    a1.axhline(y_test.mean(), color=TEXT2, lw=0.9)
    a1.text(0.02, y_test.mean() + 0.025, "no skill (0.032)", ha="left", fontsize=base - 2, color=TEXT2)
    a1.scatter([N["test_recall"]], [N["test_precision"]], color=ORANGE, s=30, zorder=4, label=f"Threshold {THRESHOLD}")
    for a_ in (a1, a2):
        a_.set_xticks([0, 0.25, 0.5, 0.75, 1]); a_.set_xticklabels(["0", "", "0.5", "", "1"])
    a2.plot([0, 1], [0, 1], color=TEXT2, lw=0.9)
    a2.scatter([fp / (fp + tn)], [N["test_recall"]], color=ORANGE, s=30, zorder=4)
    a1.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1)); a1.set_title("Precision-recall (PR-AUC)")
    a2.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1)); a2.set_title("ROC (ROC-AUC)")
    a1.legend(loc="upper right", fontsize=base - 1.5, handlelength=1.2); a2.legend(loc="lower right", fontsize=base - 1.5, handlelength=1.2)
    fp_, mp_ = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
    a3.plot([0, 20], [0, 20], color=TEXT2, lw=0.9, label="Perfect")
    a3.plot(mp_ * 100, fp_ * 100, color=BLUE, marker="o", ms=3.5, lw=1.8, label="Final model")
    a3.set(xlim=(0, 20), ylim=(0, 20), xlabel="Predicted (%)", ylabel="Observed prolonged (%)")
    a3.set_title("Calibration (deciles)"); a3.legend(loc="upper left", fontsize=base - 1.5, handlelength=1.2)
    save(fig, "f5_performance", kind)

def fig_decision(kind, w, h, base):
    style(base)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(w, h), gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.45})
    cm = np.array([[tn, fp], [fn, tp]])
    shade = np.array([["#eef4fc", "#fde3d6"], ["#fde3d6", "#cfe2f9"]])
    names = [["Correctly left alone", "False alarm"], ["Missed", "Caught"]]
    for i in range(2):
        for j in range(2):
            a1.add_patch(plt.Rectangle((j, 1 - i), 0.98, 0.98, color=shade[i][j]))
            a1.text(j + 0.49, 1 - i + 0.62, f"{cm[i, j]:,}", ha="center", fontsize=base + 4, fontweight="semibold")
            a1.text(j + 0.49, 1 - i + 0.40, names[i][j], ha="center", fontsize=base - 1)
            a1.text(j + 0.49, 1 - i + 0.22, f"~{per_day(cm[i, j], len(y_test)):.1f} per day", ha="center",
                    fontsize=base - 2, color=TEXT2)
    a1.set_xlim(0, 2); a1.set_ylim(0, 2); a1.set_aspect("equal"); a1.grid(False)
    a1.set_xticks([0.49, 1.49]); a1.set_xticklabels(["No alert", "Alert"])
    a1.set_yticks([0.49, 1.49]); a1.set_yticklabels(["Prolonged", "Short"])
    a1.tick_params(length=0)
    for s in a1.spines.values():
        s.set_visible(False)
    a1.set_title(f"Confusion matrix (test set)")
    a1.set_xlabel(f"Model at threshold {THRESHOLD}"); a1.set_ylabel("Actual outcome")

    ts = np.linspace(0.005, 0.6, 300)
    alerts = [per_day((p_test >= t).sum(), len(y_test)) for t in ts]
    rec = [((p_test >= t) & (y_test == 1)).sum() / y_test.sum() for t in ts]
    a2.plot(alerts, np.array(rec) * 100, color=BLUE)
    for o in ops:
        a2.scatter(o["alerts_per_day"], o["recall"] * 100, color=ORANGE if o["threshold"] == THRESHOLD else TEXT,
                   s=36 if o["threshold"] == THRESHOLD else 16, zorder=3)
        a2.annotate(f"{o['threshold']:g}", (o["alerts_per_day"], o["recall"] * 100), xytext=(5, -11),
                    textcoords="offset points", fontsize=base - 2, color=TEXT2)
    a2.set_xlabel("Shuttle alerts per day"); a2.set_ylabel("Prolonged incidents caught (%)")
    a2.set_xlim(0, 25); a2.set_ylim(0, 100)
    a2.set_title("Alert volume vs. disruptions caught")
    a2.text(24.5, 6, "Labels = probability threshold", ha="right", fontsize=base - 2, color=TEXT2)
    save(fig, "f6_decision", kind)


def fig_calibration(kind, w, h, base):
    style(base)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(w, h), gridspec_kw={"wspace": 0.3})
    fp_, mp_ = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
    a1.plot([0, 20], [0, 20], color=TEXT2, lw=1, label="Perfect calibration")
    a1.plot(mp_ * 100, fp_ * 100, color=BLUE, marker="o", ms=5, label="Final model")
    a1.set(xlim=(0, 20), ylim=(0, 20), xlabel="Predicted probability (%)", ylabel="Observed share prolonged (%)")
    a1.set_title("Calibration (test set, deciles)"); a1.legend(loc="upper left")
    bins = np.arange(0, 0.42, 0.02)
    for lab, mask, col in [("Short incidents", all_df["y"] == 0, GRAY), ("Prolonged incidents", all_df["y"] == 1, BLUE)]:
        v = all_df.loc[mask, "p"].clip(upper=0.399)
        a2.hist(v, bins=bins, weights=np.full(len(v), 100 / len(v)), histtype="step", lw=2, color=col, label=lab)
    a2.axvline(THRESHOLD, color=ORANGE, lw=1.5)
    a2.text(THRESHOLD + 0.008, 40, f"alert at {THRESHOLD}", color=TEXT, fontsize=base - 2)
    a2.set_ylim(0, 75); a2.set_xlabel("Predicted probability (last bin includes 0.4+)"); a2.set_ylabel("Share of the group (%)")
    a2.set_title("Predicted probabilities by outcome"); a2.legend(loc="upper right")
    save(fig, "f7_calibration", kind)


def fig_residuals(kind, w, h, base):
    style(base)
    d = all_df.copy()
    d["Time of day"] = pd.cut(d["Hour"], [-1, 5, 9, 14, 19, 23],
                              labels=["Night 0-5h", "AM peak 6-9h", "Midday 10-14h", "PM peak 15-19h", "Evening 20-23h"])
    d["Line"] = d["Line_Clean"].replace({"YU": "Line 1 YU", "BD": "Line 2 BD", "SRT": "Line 3 SRT", "SHP": "Line 4 SHP"})
    d["Year"] = d["Date"].dt.year.astype(str)
    rows = []
    rng = np.random.default_rng(1)
    for dim, keep in [("Line", ["Line 1 YU", "Line 2 BD", "Line 3 SRT", "Line 4 SHP"]),
                      ("Time of day", None), ("Year", ["2023", "2024"])]:
        for val, g in d.groupby(dim, observed=True):
            if keep is not None and val not in keep:
                continue
            res = g["y"].values - g["p"].values
            bs = [res[rng.integers(0, len(res), len(res))].mean() for _ in range(400)]
            pos = g["y"] == 1
            rows.append({"dim": dim, "val": val, "n": len(g), "res": res.mean() * 100,
                         "lo": np.percentile(bs, 2.5) * 100, "hi": np.percentile(bs, 97.5) * 100,
                         "flag": g["flag"].mean(), "recall": g.loc[pos, "flag"].mean(), "actual": g["y"].mean(),
                         "pred": g["p"].mean(), "pr_auc": average_precision_score(g["y"], g["p"])})
    r = pd.DataFrame(rows)
    put("subgroups", r.drop(columns=["lo", "hi"]).to_dict("records"))
    fig, ax = plt.subplots(figsize=(w, h))
    y = np.arange(len(r))[::-1]
    ax.axvline(0, color=TEXT2, lw=1)
    col = [RED if (lo > 0 or hi < 0) else BLUE for lo, hi in zip(r["lo"], r["hi"])]
    ax.errorbar(r["res"], y, xerr=[r["res"] - r["lo"], r["hi"] - r["res"]], fmt="none", ecolor=GRAY, elinewidth=1.4, capsize=2)
    ax.scatter(r["res"], y, color=col, s=34, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([f"{v}  (n={n:,})" for v, n in zip(r["val"], r["n"])])
    for k, (dim, g) in enumerate(r.groupby("dim", sort=False)):
        if k:
            ax.axhline(y[g.index.min()] + 0.5, color=LIGHT_GRAY, lw=0.8)
    ax.set_xlabel("Residual: actual minus predicted prolonged rate (percentage points)")
    ax.set_title("Calibration residuals by subgroup (95% bootstrap intervals)")
    ax.text(0.01, 0.98, "Red = interval excludes zero.  Right of 0 = model under-predicts; left = over-predicts.",
            transform=ax.transAxes, ha="left", va="top", fontsize=base - 2, color=TEXT2)
    ax.set_ylim(y.min() - 0.6, y.max() + 1.2)
    ax.grid(axis="y", visible=False)
    save(fig, "f8_residuals", kind)


def fig_errors(kind, w, h, base):
    style(base)
    pos = all_df[all_df["y"] == 1]
    bands = pd.cut(pos["Min Delay"], [15, 20, 30, 60, 10_000], labels=["16-20", "21-30", "31-60", ">60"])
    rec = pos.groupby(bands, observed=True)["flag"].agg(["mean", "size"])
    put("recall_by_band", {str(k): [float(v["mean"]), int(v["size"])] for k, v in rec.iterrows()})
    fn_ = all_df[(all_df["y"] == 1) & ~all_df["flag"]]
    fp_ = all_df[(all_df["y"] == 0) & all_df["flag"]]
    put("pooled_fn", len(fn_)); put("pooled_fp", len(fp_)); put("pooled_tp", int(((all_df["y"] == 1) & all_df["flag"]).sum()))
    fnc = fn_["Code"].value_counts().head(5); fpc = fp_["Code"].value_counts().head(5)
    put("fn_top", {c: [int(v), float(v / len(fn_)), float(all_df.loc[all_df.Code == c, "y"].mean())] for c, v in fnc.items()})
    put("fp_top", {c: [int(v), float(v / len(fp_)), float(all_df.loc[all_df.Code == c, "y"].mean())] for c, v in fpc.items()})
    put("fp_zero_delay_share", (fp_["Min Delay"] == 0).mean())
    fig = plt.figure(figsize=(w, h))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.55], hspace=0.9, wspace=0.95)
    a1 = fig.add_subplot(gs[:, 0]); a2 = fig.add_subplot(gs[0, 1]); a3 = fig.add_subplot(gs[1, 1])
    a1.bar([s_.replace("-", "–") for s_ in rec.index.astype(str)], rec["mean"] * 100, color=BLUE, width=0.65)
    for i, m_ in enumerate(rec["mean"]):
        a1.text(i, m_ * 100 + 2, f"{m_:.0%}", ha="center", fontsize=base - 2)
    a1.set(ylim=(0, 100), xlabel="Actual delay (min)", ylabel="Prolonged incidents caught (%)")
    a1.grid(axis="x", visible=False); a1.set_title("Recall by delay length")
    for ax, s_, col, title, total in [(a2, fnc.head(4), BLUE, "Missed disruptions: top codes", len(fn_)),
                                      (a3, fpc.head(4), ORANGE, "False alarms: top codes", len(fp_))]:
        s_ = s_[::-1]
        ax.barh([f"{SHORT.get(c, c)} ({c})" for c in s_.index], s_.values / total * 100, color=col, height=0.62)
        for i, v in enumerate(s_.values):
            ax.text(v / total * 100 + 1, i, f"{v / total:.0%}", va="center", fontsize=base - 2)
        ax.set_xlim(0, 42); ax.grid(axis="y", visible=False); ax.set_title(title, fontsize=base)
        ax.tick_params(axis="y", labelsize=base - 1.5)
    a3.set_xlabel("Share of all errors of that kind (%)")
    save(fig, "f9_errors", kind)


SPECS = {  # name: (function, report size in inches, slide size in inches)
    "imbalance": (fig_imbalance, (5.4, 2.1), (8.6, 4.2)),
    "codes": (fig_codes, (6.5, 2.45), (9.4, 4.7)),
    "time": (fig_time, (6.2, 2.25), (11.5, 4.6)),
    "importance": (fig_importance, (6.8, 2.2), (11.8, 4.4)),
    "performance": (fig_performance, (6.8, 2.3), (11.8, 4.4)),
    "curves": (fig_curves, (6.4, 2.6), (11.8, 4.9)),
    "decision": (fig_decision, (6.4, 2.35), (11.8, 4.8)),
    "calibration": (fig_calibration, (6.4, 2.35), (11.8, 4.6)),
    "residuals": (fig_residuals, (6.0, 2.3), (9.2, 4.9)),
    "errors": (fig_errors, (6.6, 2.35), (11.8, 4.8)),
}
for name, (fn_, rsize, ssize) in SPECS.items():
    fn_("report", *rsize, 8)
    fn_("slides", *ssize, 13)
    print("done", name, flush=True)

json.dump(N, open(OUT / "numbers.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in N.items() if not isinstance(v, (list, dict))}, indent=1, default=float))
