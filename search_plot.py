import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

SAVE_DIR = "save/figures"
METHOD = "supervised"
TARGET = "cit_36mo"
REPS = 5
CSV_PATH = f"save/figures/summary_results_{METHOD}_{TARGET}.csv" 

INDIVIDUAL_ZONES = ["frontal", "central", "occipital", "parietal", "all"]
BANDS = {
    "theta": [3.0, 6.0],
    "alpha": [6.0, 9.0],
    "beta": [9.0, 20.0],
    "gamma": [20.0, 48.0],
    "all": [0.2, 48.0]
}
os.makedirs(SAVE_DIR, exist_ok=True)

results_df = pd.read_csv(CSV_PATH)

results_df["Frequency Band"] = results_df["frequency"]
results_df["nRMSE"] = results_df["nrmse"]

# === 2) (Opcional) Guardar agregado ===
summary_df = (
    results_df.groupby(["zone", "frequency"])["nrmse"]
    .agg(["mean", "std"]).reset_index()
    .rename(columns={"mean": "nrmse_mean", "std": "nrmse_std"})
)
csv_path = os.path.join(SAVE_DIR, f"agg_summary_results_{METHOD}_{TARGET}.csv")
summary_df.to_csv(csv_path, index=False)
print(f"Aggregated Results saved to: {csv_path}")

plt.figure(figsize=(16, 9))
sns.set_theme(style="whitegrid")

zone_order = INDIVIDUAL_ZONES
band_order = list(BANDS.keys())

BAND_RANGES_HZ = {
    "theta": "3–6 Hz",
    "alpha": "6–9 Hz",
    "beta":  "9–20 Hz",
    "gamma": "20–48 Hz",
    "all": "0.2–48 Hz"
}

band_labels = []
for b in band_order:
    r = BAND_RANGES_HZ.get(b, "")
    band_labels.append(f"{b}\n{r}" if r else b)

ax = sns.barplot(
    data=results_df,
    x="Frequency Band",
    y="nRMSE",
    hue="zone",
    order=band_order,
    hue_order=zone_order,
    errorbar="sd"
)

ax.set_title('Supervised nRMSE Performance on predicting Cognitive age', fontsize=16)
ax.set_xlabel("Frequency Band", fontsize=12)
ax.set_ylabel("Normalized RMSE (nRMSE)", fontsize=12)

ax.set_xticks(range(len(band_order)))
ax.set_xticklabels(band_labels)

leg = ax.legend(
    title="Zone",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True
)
leg.get_frame().set_alpha(0.85)

plt.tight_layout()

plot_path = os.path.join(SAVE_DIR, f"summary_plot_{METHOD}_{TARGET}.png")
plt.savefig(plot_path, dpi=300)
print(f"Summary plot saved to: {plot_path}")
plt.close()

# Test ANOVA

ALPHA = 0.05

# Asegurar tipos
results_df["zone"] = results_df["zone"].astype("category")
results_df["frequency"] = results_df["frequency"].astype("category")

# (Optional) fix category order (aesthetic only / output ordering)
results_df["zone"] = results_df["zone"].cat.set_categories(INDIVIDUAL_ZONES, ordered=True)
results_df["frequency"] = results_df["frequency"].cat.set_categories(list(BANDS.keys()), ordered=True)

# 1) Replicaciones por celda
cell_counts = results_df.groupby(["zone", "frequency"]).size().unstack(fill_value=0)

print("\n" + "="*72)
print("2-way ANOVA (zone × frequency) on nRMSE")
print("="*72)

print("\n[1/4] Replications per combination (zone, frequency)")
print(cell_counts.to_string())

min_rep = int(cell_counts.to_numpy().min()) if cell_counts.size else 0
max_rep = int(cell_counts.to_numpy().max()) if cell_counts.size else 0
n_total = len(results_df)

print(f"\nReplication summary: n_total={n_total} | min_rep={min_rep} | max_rep={max_rep}")
if min_rep < 2:
    print("\n⚠️  ANOVA NOT applicable: there are cells with <2 observations.")
    print("   Solution: create replications (repeated CV, multiple seeds, bootstrap, etc.).")
else:
    # 2) OLS fit with interaction (Type III recommended with Sum contrasts)
    print("\n[2/4] Fitting model (Sum contrasts): nrmse ~ C(zone, Sum) * C(frequency, Sum)")
    model = ols("nrmse ~ C(zone, Sum) * C(frequency, Sum)", data=results_df).fit()
    print("OK ✅ Modelo ajustado")

    # 3) ANOVA Type III
    anova_tbl = anova_lm(model, typ=3)  # Type III
    anova_pretty = anova_tbl.copy()

    # 4) Effect sizes
    ss_total = anova_tbl["sum_sq"].sum()
    ss_resid = anova_tbl.loc["Residual", "sum_sq"]

    anova_pretty["eta_sq"] = anova_tbl["sum_sq"] / ss_total
    anova_pretty["partial_eta_sq"] = anova_tbl["sum_sq"] / (anova_tbl["sum_sq"] + ss_resid)

    # Round for printing
    cols = ["sum_sq", "df", "F", "PR(>F)", "eta_sq", "partial_eta_sq"]
    anova_pretty = anova_pretty[cols]
    anova_pretty["sum_sq"] = anova_pretty["sum_sq"].round(6)
    anova_pretty["F"] = anova_pretty["F"].round(4)
    anova_pretty["PR(>F)"] = anova_pretty["PR(>F)"].map(lambda x: float(x) if pd.notnull(x) else x)
    anova_pretty["PR(>F)"] = anova_pretty["PR(>F)"].map(lambda x: f"{x:.4g}" if pd.notnull(x) else "")
    anova_pretty["eta_sq"] = anova_pretty["eta_sq"].round(4)
    anova_pretty["partial_eta_sq"] = anova_pretty["partial_eta_sq"].round(4)

    print("\n[3/4] ANOVA table (Type III) + effect sizes")
    print(anova_pretty.to_string())

    # Significance decision
    print(f"\nSignificance decision (alpha={ALPHA}):")
    for term in anova_tbl.index:
        if term == "Residual":
            continue
        p = anova_tbl.loc[term, "PR(>F)"]
        verdict = "✅ SIGNIFICATIVO" if p < ALPHA else "❌ no significativo"
        print(f" - {term:<24} p={p:.4g}  -> {verdict}")

    # 5) Assumption checks (indicative)
    resid = model.resid
    shapiro_p = stats.shapiro(resid).pvalue
    groups = [g["nrmse"].values for _, g in results_df.groupby(["zone", "frequency"])]
    levene_p = stats.levene(*groups, center="median").pvalue

    def interpret_p(p):
        return "OK (no strong evidence of violation)" if p >= ALPHA else "⚠️ possible violation"

    print("\n[4/4] Quick assumption checks (indicative)")
    print(f" - Shapiro (normality of residuals): p={shapiro_p:.4g} -> {interpret_p(shapiro_p)}")
    print(f" - Levene  (homogeneity of variances): p={levene_p:.4g} -> {interpret_p(levene_p)}")