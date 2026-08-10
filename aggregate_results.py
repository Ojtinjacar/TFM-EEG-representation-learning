"""
Script to aggregate results from multiple parallel runs of run_downstream.py.
(Parallelisation by folds)

Combines multiple raw CSVs containing partial results (different fold_ranges)
and computes the global R² and nRMSE metrics correctly from all subjects.

Usage:
    python aggregate_results.py --input_pattern "save/downstream_results/downstream_raw_results_leave_one_out_*.csv" \
                                --output_dir save/downstream_results \
                                --cv_strategy leave_one_out
                                
    python aggregate_results.py --input_pattern "save/downstream_results/downstream_raw_results_kfold_folds*.csv" \
                                --output_dir save/downstream_results \
                                --cv_strategy kfold
"""

import argparse
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score


def parse_subject_avgs(subject_avgs_str):
    """
    Parse subject_avgs from CSV string format back to list of tuples.

    Args:
        subject_avgs_str: String in format "y_true1,y_pred1;y_true2,y_pred2;..."

    Returns:
        List of tuples [(y_true, y_pred), ...]
    """
    if pd.isna(subject_avgs_str) or subject_avgs_str == '':
        return []

    pairs = subject_avgs_str.split(';')
    result = []
    for pair in pairs:
        if pair.strip():
            y_true, y_pred = pair.split(',')
            result.append((float(y_true), float(y_pred)))
    return result


def plot_results(df, save_dir, cv_strategy):
    """
    Generates and saves bar plots for the results.
    Creates separate plots for each target, with subplots for each eval_mode.
    - nRMSE_global and R2_global: without error bars (global metrics)
    - RMSE: with error bars (mean ± std across folds)
    - Supervised is grouped with fine_tuning methods
    """
    os.makedirs(save_dir, exist_ok=True)

    targets = df['target'].unique()

    # Pastel colour palette
    pastel_colors = ['#AEC6CF', '#FFB347', '#B39EB5', '#FF6961', '#77DD77', '#FDFD96', '#84B6F4', '#FDCAE1']

    for target in targets:
        df_target = df[df['target'] == target].copy()

        # Get unique eval_modes (excluding 'none' for supervised)
        eval_modes = df_target['eval_mode'].unique()
        eval_modes = [em for em in eval_modes if em != 'none']

        n_eval_modes = len(eval_modes)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(3, n_eval_modes, figsize=(6 * n_eval_modes, 14))

        # If there is only one column, reshape to 2D array
        if n_eval_modes == 1:
            axes = axes.reshape(-1, 1)

        # --- Columns for each eval_mode ---
        for col_idx, eval_mode in enumerate(eval_modes):
            # Get SSL methods for this eval_mode
            df_eval = df_target[(df_target['eval_mode'] == eval_mode) & (df_target['method'] != 'supervised')].copy()

            # If fine_tuning, add supervised
            if eval_mode == 'fine_tuning':
                df_supervised = df_target[df_target['method'] == 'supervised'].copy()
                df_eval = pd.concat([df_supervised, df_eval], ignore_index=True)

            if df_eval.empty:
                for row in range(3):
                    axes[row, col_idx].axis('off')
                continue

            methods = df_eval['method'].tolist()
            n_methods = len(methods)
            x_pos = range(n_methods)
            colors = pastel_colors[:n_methods]

            # nRMSE global (no error bars)
            axes[0, col_idx].bar(
                x_pos,
                df_eval['nRMSE_global'],
                color=colors,
                edgecolor='gray',
                linewidth=1.2
            )
            axes[0, col_idx].set_xticks(x_pos)
            axes[0, col_idx].set_xticklabels(methods)
            axes[0, col_idx].set_title(f'{eval_mode.replace("_", " ").title()}\nnRMSE (global)', fontsize=14, fontweight='bold')
            axes[0, col_idx].set_ylabel('nRMSE', fontsize=12)
            axes[0, col_idx].tick_params(axis='x', rotation=45)

            # R2 global (no error bars)
            axes[1, col_idx].bar(
                x_pos,
                df_eval['R2_global'],
                color=colors,
                edgecolor='gray',
                linewidth=1.2
            )
            axes[1, col_idx].set_xticks(x_pos)
            axes[1, col_idx].set_xticklabels(methods)
            axes[1, col_idx].set_title(f'{eval_mode.replace("_", " ").title()}\nR² (global)', fontsize=14, fontweight='bold')
            axes[1, col_idx].set_ylabel('R²', fontsize=12)
            axes[1, col_idx].tick_params(axis='x', rotation=45)

            # RMSE with error bars
            axes[2, col_idx].bar(
                x_pos,
                df_eval['RMSE_mean'],
                color=colors,
                edgecolor='gray',
                linewidth=1.2
            )
            axes[2, col_idx].errorbar(
                x=x_pos,
                y=df_eval['RMSE_mean'],
                yerr=df_eval['RMSE_std'],
                fmt='none',
                c='black',
                capsize=5,
                capthick=1.5
            )
            axes[2, col_idx].set_xticks(x_pos)
            axes[2, col_idx].set_xticklabels(methods)
            axes[2, col_idx].set_title(f'{eval_mode.replace("_", " ").title()}\nRMSE (mean ± std)', fontsize=14, fontweight='bold')
            axes[2, col_idx].set_ylabel('RMSE', fontsize=12)
            axes[2, col_idx].set_xlabel('Method', fontsize=12)
            axes[2, col_idx].tick_params(axis='x', rotation=45)

        # Overall title
        fig.suptitle(f'Downstream Results for {target} Prediction', fontsize=18, fontweight='bold', y=1.02)

        plt.tight_layout()

        # Save the figure
        plot_path = os.path.join(save_dir, f'downstream_results_{target}_{cv_strategy}_aggregated.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Results plot saved to: {plot_path}")
        plt.close()


def main(args):
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS FROM MULTIPLE RUNS")
    print(f"{'='*80}")

    # Find all CSV files matching the pattern
    csv_files = glob.glob(args.input_pattern)

    if not csv_files:
        print(f"[ERROR] No CSV files found matching pattern: {args.input_pattern}")
        return

    print(f"\nFound {len(csv_files)} CSV files to aggregate:")
    for csv_file in sorted(csv_files):
        print(f"  - {csv_file}")

    # Load and combine all CSVs
    print("\n[INFO] Loading and combining CSV files...")
    df_list = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df_list.append(df)
        print(f"  Loaded {len(df)} rows from {os.path.basename(csv_file)}")

    df_combined = pd.concat(df_list, ignore_index=True)
    print(f"\n[INFO] Combined dataset has {len(df_combined)} rows")

    # Check for duplicate folds
    duplicates = df_combined.groupby(['fold', 'method', 'eval_mode', 'target']).size()
    duplicates = duplicates[duplicates > 1]
    if len(duplicates) > 0:
        print("\n[WARNING] Found duplicate fold/method/eval_mode/target combinations:")
        print(duplicates)
        print("  Keeping first occurrence of each duplicate.")
        df_combined = df_combined.drop_duplicates(subset=['fold', 'method', 'eval_mode', 'target'], keep='first')

    # Parse subject_avgs and aggregate
    print("\n[INFO] Parsing subject_avgs from CSV...")
    subject_avgs_by_experiment = {}

    for _, row in df_combined.iterrows():
        key = (row['method'], row['eval_mode'], row['target'])
        if key not in subject_avgs_by_experiment:
            subject_avgs_by_experiment[key] = []

        subject_avgs = parse_subject_avgs(row['subject_avgs'])
        subject_avgs_by_experiment[key].extend(subject_avgs)

    print(f"[INFO] Parsed subject averages for {len(subject_avgs_by_experiment)} unique experiments")

    # Aggregate results by method, eval_mode, and target
    print("\n[INFO] Calculating aggregated statistics...")
    df_agg = df_combined.groupby(['method', 'eval_mode', 'target']).agg(
        nRMSE_mean=('nRMSE', 'mean'),
        nRMSE_std=('nRMSE', 'std'),
        R2_mean=('R2', 'mean'),
        R2_std=('R2', 'std'),
        RMSE_mean=('RMSE', 'mean'),
        RMSE_std=('RMSE', 'std'),
        n_folds=('fold', 'count')
    ).reset_index()

    # Calculate global R² and nRMSE from all subject averages
    print("\n[INFO] Calculating global R² and nRMSE from subject-level averages...")
    r2_global_list = []
    nrmse_global_list = []

    for _, row in df_agg.iterrows():
        key = (row['method'], row['eval_mode'], row['target'])
        if key in subject_avgs_by_experiment and len(subject_avgs_by_experiment[key]) > 1:
            avgs = subject_avgs_by_experiment[key]
            y_true = np.array([a[0] for a in avgs])
            y_pred = np.array([a[1] for a in avgs])

            # R² global
            r2_global = r2_score(y_true, y_pred)
            r2_global_list.append(r2_global)

            # nRMSE global (normalised by std of the actual values across all subjects)
            rmse_global = np.sqrt(np.mean((y_true - y_pred) ** 2))
            std_global = np.std(y_true)
            nrmse_global = np.nan if std_global < 1e-8 else rmse_global / std_global
            nrmse_global_list.append(nrmse_global)

            print(f"  {row['method']} | {row['eval_mode']} | {row['target']}: "
                  f"R²_global={r2_global:.4f}, nRMSE_global={nrmse_global:.4f} "
                  f"(from {len(avgs)} subjects)")
        else:
            # If <= 1 subject or no data, use original metrics
            r2_global_list.append(row['R2_mean'])
            nrmse_global_list.append(row['nRMSE_mean'])

    df_agg['R2_global'] = r2_global_list
    df_agg['nRMSE_global'] = nrmse_global_list

    # Save aggregated results
    os.makedirs(args.output_dir, exist_ok=True)

    raw_csv_path = os.path.join(args.output_dir, f'downstream_raw_results_{args.cv_strategy}_aggregated.csv')
    df_combined.to_csv(raw_csv_path, index=False)
    print(f"\n[INFO] Combined raw results saved to: {raw_csv_path}")

    agg_csv_path = os.path.join(args.output_dir, f'downstream_agg_results_{args.cv_strategy}_aggregated.csv')
    df_agg.to_csv(agg_csv_path, index=False)
    print(f"[INFO] Aggregated results saved to: {agg_csv_path}")

    # Display results
    print("\n" + "="*80)
    print("AGGREGATED RESULTS")
    print("="*80)
    print(df_agg.to_string())

    # Plot results
    print(f"\n[INFO] Generating plots...")
    plot_results(df_agg, args.output_dir, args.cv_strategy)

    print(f"\n{'='*80}")
    print("AGGREGATION COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate results from multiple parallel runs of run_downstream.py"
    )
    parser.add_argument(
        "--input_pattern",
        type=str,
        required=True,
        help="Glob pattern for input CSV files (e.g., 'save/results/downstream_raw_results_leave_one_out_*.csv')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="save/downstream_results",
        help="Directory to save aggregated results and plots"
    )
    parser.add_argument(
        "--cv_strategy",
        type=str,
        required=True,
        choices=["kfold", "leave_one_out"],
        help="Cross-validation strategy used in the original runs"
    )

    args = parser.parse_args()
    main(args)