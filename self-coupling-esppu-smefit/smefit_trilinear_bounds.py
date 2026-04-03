import numpy as np
import arviz as az
import pathlib
import json
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.lines import Line2D
import pandas as pd

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 20})
rc('text', usetex=True)

compute_bounds = False

result_dir = pathlib.Path(
    "/data/theorie/jthoeve/smefit_release/results/esppu_paper")

runs_ind_lin = ["260324_jth_HLLHC_250GEV_LIN_IND_aggressive",
                "260324_jth_LEP3_250GEV_LIN_IND_aggressive",
                "260324_jth_FCCee_250GEV_LIN_IND_aggressive",
                "260330_jth_LCF500_250GEV_LIN_IND_aggressive",
                "260330_jth_LCF1000_250GEV_LIN_IND_aggressive"]

runs_ind_quad = ["260324_jth_HLLHC_250GEV_QUAD_IND_aggressive",
                "260324_jth_LEP3_250GEV_QUAD_IND_aggressive",
                "260324_jth_FCCee_250GEV_QUAD_IND_aggressive",
                "260330_jth_LCF500_250GEV_QUAD_IND_aggressive",
                "260330_jth_LCF1000_250GEV_QUAD_IND_aggressive"]

runs_glob_lin = ["260326_jth_HLLHC_250GEV_LIN_GLOB_aggressive",
                 "260326_jth_LEP3_250GEV_LIN_GLOB_aggressive",
                 "260326_jth_FCCee_250GEV_LIN_GLOB_aggressive",
                 "260330_jth_LCF500_250GEV_LIN_GLOB_aggressive",
                 "260330_jth_LCF1000_250GEV_LIN_GLOB_aggressive", ]

runs_glob_quad = [ "260325_jth_HLLHC_250GEV_QUAD_GLOB_aggressive_no_whitening",
                   "260325_jth_LEP3_250GEV_QUAD_GLOB_aggressive",
                   "260325_jth_FCCee_250GEV_QUAD_GLOB_aggressive",
                   "260330_jth_LCF500_250GEV_QUAD_GLOB_aggressive",
                   "260330_jth_LCF1000_250GEV_QUAD_GLOB_aggressive"]

runs_all = runs_ind_lin + runs_ind_quad + runs_glob_lin + runs_glob_quad

vSM = 0.24622
mH = 0.125
corr_factor_Op = -2.0 * (vSM ** 4) / (mH ** 2)
corr_factor_OpBox = 3.0 * vSM ** 2
corr_factor_OpD = -(3.0 / 4.0) * vSM ** 2


def find_bounds(runs, hdi=True, th_scenario="aggressive"):

    q_low = 16.
    q_high = 84.

    bounds_dict = {}
    for run in runs:

        collider = run.split("_")[2]
        eft_order = run.split("_")[4]
        ind_or_glob = run.split("_")[5]
        idx = (collider, eft_order, ind_or_glob)


        with open(result_dir / run.replace("aggressive", th_scenario) / "fit_results.json", "r") as f:
            fit_results = json.load(f)

        samples_op = np.array(fit_results["samples"]["Op"])
        individual = True if "IND" in run else False

        samples_OpBox = np.array(fit_results["samples"]["OpBox"])
        samples_OpD = np.array(fit_results["samples"]["OpD"])

        if not individual:
            samples_kappa = corr_factor_Op * samples_op + corr_factor_OpBox * samples_OpBox + corr_factor_OpD * samples_OpD
        else:
            samples_kappa = corr_factor_Op * samples_op

        if not hdi:
            low_Op, high_Op = np.percentile(samples_op, [q_low, q_high])
            low_kappa, high_kappa = np.percentile(samples_kappa, [q_low, q_high])
        else:
            low_Op, high_Op = az.hdi(samples_op, hdi_prob=.68)
            low_kappa, high_kappa = az.hdi(samples_kappa, hdi_prob=.68)

            low_Op = float(low_Op)
            high_Op = float(high_Op)
            low_kappa = float(low_kappa)
            high_kappa = float(high_kappa)



        # symmetrise linear bounds
        if not ("HLLHC" in run and "QUAD" in run):
            half_width_Op = (high_Op - low_Op) / 2
            low_Op = - half_width_Op
            high_Op = half_width_Op

            half_width_kappa = (high_kappa - low_kappa) / 2
            low_kappa = - half_width_kappa
            high_kappa = half_width_kappa


        # round everything to 2 decimals using string formatting

        bounds_dict[idx] = [low_Op, high_Op, low_kappa, high_kappa]

    bounds_df = pd.DataFrame(bounds_dict)
    bounds_df = bounds_df.T
    bounds_df.columns = ["low_Op", "high_Op", "low_kappa", "high_kappa"]
    return bounds_df


def build_latex_bounds_table(df_aggressive, df_conservative, output_path="trilinear_bounds_table.tex"):
    """Write a grouped LaTeX table of kappa bounds for aggressive/conservative scenarios."""
    collider_order = ["HLLHC", "LEP3", "FCCee", "LCF500", "LCF1000"]
    eft_order = ["LIN", "QUAD"]
    fit_order = ["IND", "GLOB"]

    collider_labels = {
        "HLLHC": r"${\rm HL}\textnormal{-}{\rm LHC}$",
        "LEP3": r"${\rm LEP3}$",
        "FCCee": r"${\rm FCC}\textnormal{-}{\rm ee}$",
        "LCF500": r"${\rm LCF550}$",
        "LCF1000": r"${\rm LCF1000}$",
    }
    eft_labels = {
        "LIN": r"$\mathcal{O}(\Lambda^{-2})$",
        "QUAD": r"$\mathcal{O}(\Lambda^{-4})$",
    }
    fit_labels = {
        "IND": r"${\rm Individual}$",
        "GLOB": r"${\rm Marginalised}$",
    }

    # Ensure a consistent 3-level index layout.
    if not isinstance(df_aggressive.index, pd.MultiIndex):
        df_aggressive.index = pd.MultiIndex.from_tuples(df_aggressive.index, names=["collider", "order", "fit"])
    if not isinstance(df_conservative.index, pd.MultiIndex):
        df_conservative.index = pd.MultiIndex.from_tuples(df_conservative.index, names=["collider", "order", "fit"])

    rows = []
    idx_rows = []
    for collider in collider_order:
        for order in eft_order:
            for fit in fit_order:
                idx = (collider, order, fit)
                if idx not in df_aggressive.index or idx not in df_conservative.index:
                    continue

                row_aggr = df_aggressive.loc[idx]
                row_cons = df_conservative.loc[idx]
                aggr_interval = rf"$[{row_aggr['low_kappa']:.3f},\,{row_aggr['high_kappa']:.3f}]$"
                cons_interval = rf"$[{row_cons['low_kappa']:.3f},\,{row_cons['high_kappa']:.3f}]$"

                idx_rows.append((collider_labels.get(collider, collider), eft_labels[order], fit_labels[fit]))
                rows.append({
                    r"Aggressive": aggr_interval,
                    r"Conservative": cons_interval,
                })

    table_df = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(idx_rows, names=["Collider", "EFT order", "Fit"]),
    )

    latex_table = table_df.to_latex(
        escape=False,
        multirow=True,
        column_format="lllcc",
        caption=r"Bounds on $\delta\kappa_3$ grouped by collider, EFT order, and fit strategy.",
        label="tab:trilinear-bounds",
    )
    pathlib.Path(output_path).write_text(latex_table)
    return latex_table


if compute_bounds:
    bounds_trilinear_aggressive = find_bounds(runs_all, hdi=True, th_scenario="aggressive")
    bounds_trilinear_conservative = find_bounds(runs_all, hdi=True, th_scenario="conservative")

    # save to csv
    bounds_trilinear_aggressive.to_csv("trilinear_bounds_aggressive.csv")
    bounds_trilinear_conservative.to_csv("trilinear_bounds_conservative.csv")

else:
    bounds_trilinear_aggressive = pd.read_csv(f"trilinear_bounds_aggressive.csv", index_col=[0, 1, 2])
    bounds_trilinear_conservative = pd.read_csv(f"trilinear_bounds_conservative.csv", index_col=[0, 1, 2])


latex_table = build_latex_bounds_table(
    bounds_trilinear_aggressive,
    bounds_trilinear_conservative,
    output_path="trilinear_bounds_table.tex",
)




# Styling
colors = {
    ("LIN", "IND"): "#e41a1c",      # red
    ("LIN", "GLOB"): "#4daf4a",   # green
    ("QUAD", "IND"): "#ff7f00",  # orange
    ("QUAD", "GLOB"): "#377eb8" # blue
}

labels = {
    ("LIN", "IND"): r"${\rm Ind.}, \mathcal{O}(\Lambda^{-2})$",
    ("LIN", "GLOB"): r"${\rm Marg.}, \mathcal{O}(\Lambda^{-2})$",
    ("QUAD", "IND"): r"${\rm Ind.}, \mathcal{O}(\Lambda^{-4})$",
    ("QUAD", "GLOB"): r"${\rm Marg.}, \mathcal{O}(\Lambda^{-4})$",
}

colliders = bounds_trilinear_aggressive.index.get_level_values(0).unique()

fig, ax = plt.subplots(figsize=(12, 10))

# vertical spacing
y_base = np.arange(len(colliders))[::-1]  # top to bottom
offsets = {
    ("LIN", "IND"): 0.25,
    ("LIN", "GLOB"): 0.08,
    ("QUAD", "IND"): -0.08,
    ("QUAD", "GLOB"): -0.25,
}

bar_height = 0.07

for i, collider in enumerate(colliders):
    for key in offsets:
        order, fit_type = key

        try:
            row_aggressive = bounds_trilinear_aggressive.loc[(collider, order, fit_type)]
            row_conservative = bounds_trilinear_conservative.loc[(collider, order, fit_type)]
        except KeyError:
            continue

        low_conservative = row_conservative["low_kappa"]
        high_conservative = row_conservative["high_kappa"]
        low_aggressive = row_aggressive["low_kappa"]
        high_aggressive = row_aggressive["high_kappa"]

        width_aggressive = high_aggressive - low_aggressive
        width_conservative = high_conservative - low_conservative

        y = y_base[i] + offsets[key]

        ax.barh(
            y,
            width_aggressive,
            left=low_aggressive,
            height=bar_height,
            color=colors[key],
            alpha=0.85,
            label=labels[key] if i == 0 else None  # avoid duplicate legend
        )

        ax.barh(
            y,
            width_conservative,
            left=low_conservative,
            height=bar_height,
            color=colors[key],
            alpha=0.3,
            label=None  # avoid duplicate legend
        )


# Formatting
ax.set_yticks(y_base)
ax.set_xticks(np.arange(-0.5, 0.51, 0.1))
ax.set_yticklabels([r"${\rm HL}\textnormal{-}{\rm LHC}$",
                    r"${\rm LEP3}$",
                    r"${\rm FCC}\textnormal{-}{\rm ee}$",
                    r"${\rm LCF550}$",
                    r"${\rm LCF1000}$",])
# ax.set_xticklabels([f"{x:.1f}" for x in ax.get_xticks()])

ax.axvline(0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel(r"$\delta \kappa_3$")

ax.grid(axis="x", which="major", linestyle=":", linewidth=0.8, alpha=0.6)
ax.xaxis.minorticks_on()
ax.grid(axis="x", which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
ax.tick_params(
    axis='both',
    which='both',
    direction='in',
    top=True,
    right=True
)

# Clean legend (unique entries only)
handles, legend_labels = ax.get_legend_handles_labels()
unique = dict(zip(legend_labels, handles))
leg1 = ax.legend(unique.values(), unique.keys(), frameon=False, loc='lower left')
ax.set_xlim(-0.5, 0.5)



# Custom legend elements
ci_handles = [
    Line2D([0], [0], color='lightgray', lw=3, label=r"${\rm Conservative}$"),
    Line2D([0], [0], color='black', lw=3, label=r"${\rm Aggressive}$")
]

# Add legend (position as needed)
leg2 = ax.legend(handles=ci_handles, loc='center right', bbox_to_anchor=(0.98, 0.17), frameon=False)
ax.add_artist(leg1)
# add smefit logo to bottom right
logo = plt.imread("logo.png")
new_ax = fig.add_axes([0.68, 0.12, 0.2, 0.2], anchor='SE', zorder=10)
new_ax.imshow(logo)
new_ax.axis('off')

ax.set_title(r"${\rm 68\%\,C.I., \;}\mu_0=250\,{\rm GeV}$")

# plt.tight_layout()
plt.savefig("smefit_trilinear_bounds_agg_cons.pdf")
