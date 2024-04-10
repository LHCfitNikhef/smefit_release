# -*- coding: utf-8 -*-
import importlib
import json
import pathlib
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from latex_dicts import mod_dict
from latex_dicts import uv_param_dict
from latex_dicts import inv_param_dict
#import arviz as az
import math
from sigfig import round
import pandas as pd
import matplotlib.patches as patches

import itertools

import numpy as np
import matplotlib

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc, use
from histogram_tools import find_xrange

from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 22})
rc('text', usetex=True)


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):

            super().__init__(*args, aspect='equal', **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels, fontsize=12):
            self.set_thetagrids(np.degrees(theta), labels, fontsize=fontsize)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def plot_spider(
        df,
        labels,
        title,
        ymax=100,
        log_scale=True,
        fontsize=12,
        figsize=(9, 9),
        legend_loc="best",
):
    """
    Plot error bars at given confidence level

    Parameters
    ----------
        error: dict
           confidence level bounds per fit and coefficient
    """

    def log_transform(x, delta):
        return np.log10(x) + delta

    # check if more than one fit is loaded
    if df.shape[1] < 2:
        print("At least two fits are required for the spider plot atm")

    theta = radar_factory(len(df), frame="circle")

    # normalise to first fit
    data = df.values

    delta = np.abs(np.log10(min(data.flatten()))) + 0.1

    if log_scale:
        data = log_transform(data, delta)



    spoke_labels = [inv_param_dict[op[0]][op[1]] for op in df.index]

    fig = plt.figure(figsize=figsize)
    # outer_ax_width = 0.7
    outer_ax_width = 0.8
    left_outer_ax = (1 - outer_ax_width) / 2
    rect = [left_outer_ax, left_outer_ax, outer_ax_width, outer_ax_width]
    n_axis = 3
    axes = [fig.add_axes(rect, projection="radar") for i in range(n_axis)]

    y_log_min = math.floor(np.log10(df.values.min()))
    y_log_max = math.ceil(np.log10(df.values.max()))



    radial_lines = [i + np.log10(j) for i in range(y_log_max + 1) for j in range(1, 10)]
    radial_labels = []
    for rad_line in radial_lines:
        if rad_line % 1 == 0:
            radial_labels.append(rf"$\mathbf{{10^{{{int(rad_line)}}}}}$")
        else:
            radial_labels.append("")
    import pdb; pdb.set_trace()




    # for radial_line in np.arange(y_log_min, y_log_max + 1, n_lines):
    #     radial
    # radial_labels = [rf"$\mathbf{{10^{{{i}}}}}$" for i in np.arange(y_log_min, y_log_max + 1)]




    # y_log_min = math.floor(df.values.min())
    # y_log_max = math.ceil(df.values.max())
    # radial_labels = [f"{i}" for i in np.arange(y_log_min, y_log_max + 1)]


    # take first axis as main, the rest only serve to show the remaining percentage axes
    ax = axes[0]

    start_angle = 360 - 45

    angles = np.arange(
        start_angle, start_angle + 360, 360.0 / n_axis
    )  # zero degrees is 12 o'clock

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    markers = itertools.cycle(['*', 'o', 'P'])

    for i, data_fit_i in enumerate(data.T):

        ax.plot(theta, data_fit_i, color=colors[i], zorder=1)
        ax.scatter(
            theta, data_fit_i, marker=next(markers), s=50, color=colors[i], zorder=1
        )
        ax.fill(
            theta,
            data_fit_i,
            alpha=0.25,
            label="_nolegend_",
            color=colors[i],
            zorder=1,
        )
        

    for i, axis in enumerate(axes):
        if i > 0:
            axis.patch.set_visible(False)
            #axis.rgrid("off")
            axis.xaxis.set_visible(False)

        angle = angles[i]
        text_alignment = "right" if angle % 360 > 180 else "left"

        axis.yaxis.set_tick_params(labelsize=11, zorder=100)

        # if i == 0:
        #     axis.set_rgrids(
        #         [i for i in range(10)],
        #         angle=angle,
        #         labels=radial_labels,
        #         horizontalalignment=text_alignment,
        #         zorder=0,
        #     )
        # else:
        #     axis.set_rgrids([], angle=angle)  # Hide radial lines here

        # if i == 0:
        #     axis.set_rgrids(
        #         np.arange(y_log_min, y_log_max + 1),
        #         angle=angle,
        #         labels=radial_labels,
        #         horizontalalignment=text_alignment,
        #         zorder=0,
        #     )
        # else:
        #
        #     axis.set_rgrids(
        #         np.arange(y_log_min, y_log_max + 1)[1:],
        #         labels=radial_labels[1:],
        #         angle=angle,
        #         horizontalalignment=text_alignment,
        #         zorder=0,
        #     )

        if i == 0:
            axis.set_rgrids(
                radial_lines,
                angle=angle,
                labels=radial_labels,
                horizontalalignment=text_alignment,
                zorder=0,
            )
        else:

            axis.set_rgrids(
                radial_lines[1:],
                labels=radial_labels[1:],
                angle=angle,
                horizontalalignment=text_alignment,
                zorder=0,
            )


        axis.set_ylim(0, 1.1 * y_log_max)

    ax.set_varlabels(spoke_labels, fontsize=fontsize)
    ax.tick_params(axis="x", pad=35)

    ax2 = fig.add_axes(rect=[0, 0, 1, 1])
    width_disk = 0.01
    ax2.patch.set_visible(False)
    ax2.grid("off")
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    delta_disk = 0
    radius = outer_ax_width / 2 + (1 + delta_disk) * width_disk

    #ax2.set_title(title, fontsize=18)

    class_names = df.index.get_level_values(0)
    angle_sweep = 360 * class_names.value_counts(sort=False) / len(class_names)

    # determine angles of the colored arcs
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    filled_start_angle = 0 # 12'o clock


    for i, (idx, angle) in enumerate(angle_sweep.items()):

        filled_end_angle = angle + filled_start_angle  #  End angle in degrees

        center = (0.5, 0.5)  # Coordinates relative to the figure



        alpha = 0.3

        ax2.axis("off")

        # Create the filled portion of the circular patch
        filled_wedge = patches.Wedge(
            center,
            radius,
            filled_start_angle + 90 - 0.5 * angle_sweep.iloc[0],  # start at 12'o clock
            filled_end_angle + 90 - 0.5 * angle_sweep.iloc[0],
            facecolor=colors[i],
            alpha=alpha,
            ec=None,
            width=width_disk,
            transform=ax2.transAxes,
        )
        ax2.add_patch(filled_wedge)

        mid_angle = filled_start_angle - 0.5 * angle_sweep.iloc[0] + 0.1 * (filled_end_angle - filled_start_angle)
        print(i, mid_angle)
        ax.text(mid_angle * (np.pi / 180), 1.2 * y_log_max, mod_dict[idx], color='black', fontsize=12, ha='center', va='bottom',
                bbox=dict(facecolor='none', edgecolor=colors[i], boxstyle='round'))

        filled_start_angle = filled_end_angle

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[i],
            linewidth=3,
            marker=next(markers),
            markersize=10,
        )
        for i in range(len(labels))
    ]

    ax2.legend(
        handles,
        labels,
        frameon=False,
        fontsize=15,
        loc=legend_loc,
        ncol=len(labels),
        bbox_to_anchor=(0.0, -0.05, 1.0, 0.05),
        bbox_transform=fig.transFigure,
    )

    #self._plot_logo(ax2, [0.75, 0.95, 0.001, 0.07])


    plt.savefig("/data/theorie/jthoeve/smefit_release/smefit_uv/results_uv_param/spider_plot_uv_v4.png", bbox_inches="tight")


collections = ["Granada"]


here = pathlib.Path(__file__).parent

# result dir
# result_dir = here / "results_fcc"
# pathlib.Path.mkdir(result_dir, parents=True, exist_ok=True)

# mod_list = []
# for col in collections:
#     base_path = pathlib.Path(f"{here.parent}/runcards/uv_models/UV_scan/{col}/")
#     sys.path = [str(base_path)] + sys.path
#     for p in base_path.iterdir():
#         if '21' in p.name:
#             continue
#         if p.name.startswith("InvarsFit") and p.suffix == ".py":
#             mod_list.append(importlib.import_module(f"{p.stem}"))


use("PDF")
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amssymb}"})
#
# # compute the invariants
pQCD = ['NLO']
EFT = ['NHO']

# for model in mod_list:
#     for pQCD in ['NLO']:
#         for EFT in ['HO']:
#             model.MODEL_SPECS['pto'] = pQCD
#             model.MODEL_SPECS['eft'] = EFT
#             invariants = []
#             for k, attr in model.__dict__.items():
#                 if k.startswith('inv'):
#                     invariants.append(attr)
#             try:
#                 model.inspect_model(model.MODEL_SPECS, invariants)
#             except FileNotFoundError:
#                 print("File not found", model)
#                 continue
# #
#sys.exit()
# Specify the path to the JSON file
posterior_path = f"{here.parent}/results/smefit_fcc_uv_spider/{{}}_{{}}_UV_{{}}_{{}}_{{}}_NS/inv_posterior.json"


def get_bounds(collection, mod_nrs):
    
    lhc_bounds = {}
    hllhc_bounds = {}
    fcc_bounds = {}
    x_labels = []

    for col, mod in zip(collection, mod_nrs):


        posterior_path_mod_1 = pathlib.Path(posterior_path.format(col, "lhc", mod, "NLO", "HO"))
        posterior_path_mod_2 = pathlib.Path(posterior_path.format(col, "hllhc", mod, "NLO", "HO"))
        posterior_path_mod_3 = pathlib.Path(posterior_path.format(col, "fcc", mod, "NLO", "HO"))

        if posterior_path_mod_1.exists():
            # Open the JSON file and load its contents
            try:
                with open(posterior_path_mod_1) as f:
                    posterior_1 = json.load(f)

                with open(posterior_path_mod_2) as f:
                    posterior_2 = json.load(f)

                with open(posterior_path_mod_3) as f:
                    posterior_3 = json.load(f)
            except FileNotFoundError:
                continue

            n_invariants = 0
            for key in posterior_1.keys():
                if key.startswith('inv'):
                    n_invariants += 1

            for (key, samples_1_list), (_, samples_2_list), (_, samples_3_list) in zip(
                    posterior_1.items(), posterior_2.items(), posterior_3.items()
            ):
                if not key.startswith("inv"):
                    continue
                else:
                    samples_1 = np.array(samples_1_list)
                    samples_2 = np.array(samples_2_list)
                    samples_3 = np.array(samples_3_list)

                    lhc_width = np.percentile(samples_1, 97.5) - np.percentile(samples_1, 2.5)
                    hllhc_width = np.percentile(samples_2, 97.5) - np.percentile(samples_2, 2.5)
                    fcc_width = np.percentile(samples_3, 97.5) - np.percentile(samples_3, 2.5)

                    lhc_bounds[(mod, key)] = [lhc_width]
                    hllhc_bounds[(mod, key)] = [hllhc_width]
                    fcc_bounds[(mod, key)] = [fcc_width]

                    x_labels.append(mod_dict[mod])


    lhc_bounds = pd.DataFrame(lhc_bounds, index=[r"${\rm LHC}$"]).T
    hllhc_bounds = pd.DataFrame(hllhc_bounds, index=[r"${\rm HL-LHC}$"]).T
    fcc_bounds = pd.DataFrame(fcc_bounds, index=[r"${\rm FCC}$"]).T

    bounds = pd.concat([lhc_bounds, hllhc_bounds, fcc_bounds], axis=1)


    bounds.drop( ("5_10", 'inv1'), inplace=True)

    plot_spider(bounds, title=r'${\rm UV\:couplings}$',
                labels=[  '$\mathrm{LHC}$', '$\mathrm{HL}\,\\textnormal{-}\,\mathrm{LHC}$',
  '$\mathrm{HL}\,\\textnormal{-}\,\mathrm{LHC}+\mathrm{FCC}\,\\textnormal{-}\,\mathrm{ee}$',],
                legend_loc='upper center', log_scale=True, figsize=[10, 10])


get_bounds(["Granada", "Granada", "OneLoop", "OneLoop", "Granada", "OneLoop"],
           ['48_10', '49_10', "T1_10", "T2_10", '5_10', "Varphi_10"])
