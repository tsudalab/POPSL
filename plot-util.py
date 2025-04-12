from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
# from pymoo.util.ref_dirs import get_reference_directions
from pymoo.factory import get_visualization #, get_reference_directions
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.problems import get_problem
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize
from operator import itemgetter, attrgetter
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from pymoo.core.problem import ElementwiseProblem
from pymoo.decomposition.weighted_sum import WeightedSum
from pymoo.decomposition.aasf import AASF
import matplotlib.pyplot as plt
from pymoo.problems.multi.zdt import ZDT2
from pymoo.problems.many.dtlz import DTLZ2
from pymoo.problems.many.wfg import WFG1
from pymoo.indicators.hv import Hypervolume as HV
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from scipy.stats import pareto
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from net.problem import DTLZ5
# from paretoset import paretoset
import pickle
from os.path import splitext, join, basename, abspath
from glob import glob
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import MinMaxScaler
import os

def evaluate_pf_metrics(
    problem_name,
    dim,
    ref_point=None,
    data_dir=None,
    save_path=None,
    num_iter=10,
    num_seeds=5,
    use_pymoo_pf=True,
    pf_file=None,
    sample_limit=None,
    verbose=False
):
    """
    General evaluation function for multi-objective optimization results.

    This function computes HV, GD, and IGD against a known or generated Pareto front.

    Parameters:
    -----------
    problem_name : str
        Name of the problem (e.g., 'dtlz2', 'zdt3')
    dim : int
        Number of objectives
    ref_point : np.ndarray or None
        Reference point for HV computation. If None, defaults to 1.1 * ones(dim)
    data_dir : str
        Directory where predicted solution files are stored (e.g., 'results/nsga/dtlz2_4')
    save_path : str or None
        Path to save the evaluation results as CSV (optional)
    num_iter : int
        Number of outer-loop iterations (indexed by i)
    num_seeds : int
        Number of runs per iteration (indexed by j)
    use_pymoo_pf : bool
        Whether to use pymoo's built-in Pareto front generation
    pf_file : str or None
        If provided, load ground-truth Pareto front from this file
    sample_limit : int or None
        If specified, subsample the predicted front to this many points
    verbose : bool
        Whether to print intermediate HV, GD, IGD values

    Returns:
    --------
    res_arr : np.ndarray
        Array of computed metrics with shape (num_iter, num_seeds * 3)
    """

    # Load or generate Pareto front
    if use_pymoo_pf:
        ref_dirs = get_reference_directions("das-dennis", dim, n_partitions=100 if dim == 2 else 12)
        pf = get_problem(problem_name).pareto_front(ref_dirs)
    else:
        pf_path = pf_file or abspath(f'data/ground_truths/{problem_name}.csv')
        pf = np.genfromtxt(pf_path, dtype='float', delimiter=',')

    # Define reference point
    if ref_point is None:
        ref_point = 1.1 * np.ones(pf.shape[1])

    # Initialize indicators
    ind_gd = GD(pf)
    ind_igd = IGD(pf)
    hv = HV(ref_point=ref_point)
    hv_gt = hv._do(pf)

    if verbose:
        print(f"[Ground Truth HV] {hv_gt:.6f} | Ref point: {ref_point}")

    # Data cache for cumulative evaluation
    data_list = {}
    res_list = []

    # Iterate over outer iterations
    for i in range(num_iter):
        tmp_list = []

        for j in range(num_seeds):
            file_path = abspath(os.path.join(data_dir, f'y-{i}-{j}.csv'))

            if not os.path.exists(file_path):
                if verbose:
                    print(f"[Warning] Missing file: {file_path}")
                continue

            data_y = np.genfromtxt(file_path, dtype='float', delimiter=',')

            # Accumulate data across iterations
            if i == 0:
                data_list.setdefault(j, []).extend(data_y)
            else:
                data_list[j].extend(data_y)
                data_y = np.array(data_list[j])

            # Apply subsampling if requested
            if sample_limit and len(data_y) > sample_limit:
                idx = np.random.choice(len(data_y), sample_limit, replace=False)
                data_y = data_y[idx]

            # Compute indicators
            hv_pred = hv._do(data_y)
            log_hv_diff = np.abs(hv_pred - hv_gt)
            gd_score = ind_gd(data_y)
            igd_score = ind_igd(data_y)

            tmp_list.extend([log_hv_diff, gd_score, igd_score])

            if verbose:
                print(f"[{i}-{j}] HV diff={log_hv_diff:.6f}, GD={gd_score:.6f}, IGD={igd_score:.6f}")

        res_list.append(tmp_list)

    # Convert to array
    res_arr = np.array(res_list)

    # Save result
    if save_path:
        save_path = abspath(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savetxt(save_path, res_arr, fmt='%.8f', delimiter=',', newline='\n')

    return res_arr

def plot_benchmark_summary(result_name, dataset_name, x_step=5, x_start=10, save_as="hvd-igd.pdf"):
    """
    Plot HVD and IGD comparison charts for a benchmark dataset.

    Parameters:
        result_name (str): folder and file prefix, e.g., "re5", "zdt3", etc.
        dataset_name (str): used for display (not actively used in the function).
        x_step (int): step size for x-axis (e.g., 5 or 10).
        x_start (int): start of x-axis.
        save_as (str): filename to save the final plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    c_list = ['#073E7F', '#39a432', '#7a52a6', '#be0e23']  # Colors for PO-PSL, PSL-MOBO, DGEMO, etc.

    def load_csv_metrics(filename):
        path = abspath(f"res-com/{result_name}/{filename}")
        data = np.genfromtxt(path, dtype='float', delimiter=',')
        return data[:, [0, 3, 6, 9, 12]], data[:, [1, 4, 7, 10, 13]], data[:, [2, 5, 8, 11, 14]]

    hv_epsl, gd_epsl, igd_epsl = load_csv_metrics(f"result-{result_name}-epsl.csv")
    hv_psl, gd_psl, igd_psl = load_csv_metrics(f"result-{result_name}-psl.csv")
    hv_dg, gd_dg, igd_dg = load_csv_metrics(f"result-{result_name}-dgemo.csv")

    x_index = hv_epsl.shape[0]
    x_axis = [int(x_step * i + x_start) for i in range(x_index)]

    def plot_metric(ax, avg_std_list, labels, ylabel, title):
        for idx, (avg, std) in enumerate(avg_std_list):
            r1 = avg - std
            r2 = avg + std
            ax.plot(x_axis, avg, linewidth=1, label=labels[idx], color=c_list[idx])
            ax.fill_between(x_axis, r1, r2, alpha=0.2, color=c_list[idx])
        ax.set_xticks(x_axis)
        ax.tick_params(labelsize=12)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.set_xlabel('number of evaluation', fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(title)
        ax.legend(loc='upper right', prop={'weight': 'bold', 'size': 14})

    # Compute means and stds
    avg_std_hv = [(np.mean(hv_epsl, axis=1), np.std(hv_epsl, axis=1)),
                  (np.mean(hv_psl, axis=1), np.std(hv_psl, axis=1)),
                  (np.mean(hv_dg, axis=1), np.std(hv_dg, axis=1))]
    avg_std_igd = [(np.mean(igd_epsl, axis=1), np.std(igd_epsl, axis=1)),
                   (np.mean(igd_psl, axis=1), np.std(igd_psl, axis=1)),
                   (np.mean(igd_dg, axis=1), np.std(igd_dg, axis=1))]

    plot_metric(axes[0], avg_std_hv, ['PO-PSL', 'PSL-MOBO', 'DGEMO'], 'HVD', '(a)')
    plot_metric(axes[1], avg_std_igd, ['PO-PSL', 'PSL-MOBO', 'DGEMO'], 'IGD', '(b)')

    plt.tight_layout()
    plt.savefig(save_as)
    plt.close()
    print(f"Saved figure to {save_as}")

# ---------- 3D Plot Utilities ----------
def plot_3d_surface_with_solutions(gt_file, sol_file_list, labels, output_file, view=(30, 60), cmap='viridis'):
    data_gt = np.genfromtxt(abspath(gt_file), dtype='float', delimiter=',')
    x, y, z = data_gt[:, 0], data_gt[:, 1], data_gt[:, 2]
    xi, yi = np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='linear')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xi, yi, zi, cmap=cmap, edgecolor='none', alpha=0.5)

    for file, label in zip(sol_file_list, labels):
        data = np.genfromtxt(abspath(file), dtype='float', delimiter=',')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], label=label)

    ax.view_init(*view)
    ax.set_xlabel('obj_1', fontsize=14, fontweight='bold')
    ax.set_ylabel('obj_2', fontsize=14, fontweight='bold')
    ax.set_zlabel('obj_3', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', prop={'weight': 'bold', 'size': 14})
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    plt.savefig(output_file)
    plt.close()

# ---------- 2D Plot Utilities ----------
def plot_2d_pf(gt_file, sol_file_list, labels, output_file, interval=10000):
    gt_data = np.genfromtxt(abspath(gt_file), dtype='float', delimiter=',')
    x_gt, y_gt = gt_data[:, 0], gt_data[:, 1]

    fig, ax = plt.subplots(1, figsize=(7, 5))
    ax.plot(x_gt, y_gt, 'o', linewidth=3, ls='--', color='darkorange', alpha=0.8, label='Ground truth')

    for file, label in zip(sol_file_list, labels):
        data = np.genfromtxt(abspath(file), dtype='float', delimiter=',')
        x, y = data[:, 0], data[:, 1]
        sorted_idx = np.argsort(x)
        x_sorted, y_sorted = x[sorted_idx], y[sorted_idx]
        xi = np.linspace(min(x_sorted), max(x_sorted), interval)
        spline_interp = UnivariateSpline(x_sorted, y_sorted, s=1)
        yi = spline_interp(xi)
        ax.plot(xi, yi, linewidth=6, alpha=0.4, label=f'{label} (smoothed)')
        ax.scatter(x, y, s=40, alpha=0.6, label=f'{label} (points)')

    ax.set_xlabel('obj_1', fontsize=14, fontweight='bold')
    ax.set_ylabel('obj_2', fontsize=14, fontweight='bold')
    ax.grid(color='darkgray', linestyle='--', linewidth=1)
    ax.legend(loc='upper right', prop={'weight': 'bold', 'size': 14})
    plt.tick_params(labelsize=12)
    plt.savefig(output_file)
    plt.close()
    
if __name__ == '__main__':
    evaluate_pf_metrics(
        problem_name='zdt3',
        dim=4,
        data_dir='your dir/zdt3',
        save_path='res-com/zdt3/result-zdt3-popsl.csv',
        num_iter=10,
        num_seeds=5,
        ref_point=np.ones(4),
        verbose=True
    )
    plot_benchmark_summary(result_name='zdt3', dataset_name='zdt3', x_step=10, x_start=20, save_as='hvd-igd-zdt3.pdf')
    plot_2d_pf(
        gt_file='data/gt_zdt3.csv',
        sol_file_list=['fig/pf-zdt3/zdt3-popsl.csv', 'fig/pf-zdt3/zdt3-psl.csv'],
        labels=['POPSL', 'PSL-MOBO'],
        output_file='fig/pf-zdt3/pf-zdt3-combined.tiff'
    )
