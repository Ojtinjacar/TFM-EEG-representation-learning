import os
import subprocess
import argparse
import re
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

def find_latest_model(model_dir, pattern):
    """Finds the latest model file matching a pattern."""
    try:
        list_of_files = glob.glob(os.path.join(model_dir, pattern))
        if not list_of_files:
            return None
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    except Exception as e:
        print(f"Error finding model: {e}")
        return None

def main(args):
    # --- Iteration Setup ---
    BANDS = {
        "theta": [3.0, 6.0],
        "alpha": [6.0, 9.0],
        "beta": [9.0, 20.0],
        "gamma": [20.0, 48.0],
        "all": [0.2, 48.0]
    }
    INDIVIDUAL_ZONES = ["frontal", "central", "occipital", "parietal"]
    ZONES = INDIVIDUAL_ZONES + ["all"]
    
    # --- Base Paths ---
    BASE_DATA_PATH = "data/raw/all_epochs_raw.npy"
    BASE_META_PATH = "data/raw/all_epochs_metadata.csv"
    CHANNELS_TXT = "data/raw/channel_names.txt"
    SOCIO_PATH = "data/raw/metadata.csv" # Assuming this is the socioeconomic file
    
    MODEL_SAVE_DIR = "save/models"
    FIGURE_SAVE_DIR = "save/figures"

    # Ensure output directories exist (supervised downstream relies on defaults)
    for d in (MODEL_SAVE_DIR, FIGURE_SAVE_DIR,
              os.path.join(FIGURE_SAVE_DIR, "pretrain"),
              os.path.join(FIGURE_SAVE_DIR, "prediction")):
        os.makedirs(d, exist_ok=True)

    # Subject pool for the held-out test split (subject-based evaluation).
    all_subjects = sorted(pd.read_csv(BASE_META_PATH)["subject"].unique().tolist())
    TEST_K = min(args.test_k, len(all_subjects) - 1)

    # List to store the results of each run
    results = []

    print(f"=== STARTING PIPELINE FOR METHOD: {args.method} OVER {args.repetitions} REPETITIONS ===")

    for i in range(1, args.repetitions + 1):
        # Repeated random subject holdout, seeded per repetition: the SAME test
        # subjects are used across all zones/bands within a repetition (fair zone
        # comparison), and CHANGE between repetitions (genuine replication for the ANOVA).
        rep_rng = random.Random(1000 + i)
        test_subjects = rep_rng.sample(all_subjects, k=TEST_K)
        print(f"\n========== REPETITION {i}/{args.repetitions} ==========")
        print(f"Held-out test subjects ({len(test_subjects)}): {test_subjects}")
        for zone in ZONES:
            for band_name, (l_freq, h_freq) in BANDS.items():
                
                run_id = f"{zone}_{band_name}"
                print(f"\n--- Running: {run_id} ---")

                processed_data_dir = f"data/processed/{run_id}"
                os.makedirs(processed_data_dir, exist_ok=True)

                print(f"  [1/4] Already preprocessing not done for {run_id}? : {args.preprocessing}")
                if args.preprocessing:
                    print(f"  [1/4] Preprocessing data for {run_id}...")

                    zones_for_postprocessing = INDIVIDUAL_ZONES if zone == "all" else [zone]
                    post_cmd = [
                        "python", "src/postprocessing.py",
                        "--data_path", BASE_DATA_PATH,
                        "--meta_path", BASE_META_PATH,
                        "--socio_path", SOCIO_PATH,
                        "--channels_txt", CHANNELS_TXT,
                        "--l_freq", str(l_freq),
                        "--h_freq", str(h_freq),
                        "--zones", *zones_for_postprocessing,
                        "--output_path", processed_data_dir,
                        "--norm_mode", args.norm_mode,
                    ]
                    subprocess.run(post_cmd, check=True, capture_output=True)

                if args.method in ["SimCLR", "AE"]:

                    # --- 2. Backbone Training ---
                    print(f"  [2/4] Training {args.method} model for {run_id}...")
                    if args.method == "SimCLR":
                        train_script = "src/train_simclr.py"
                        train_cmd = [
                            "python", train_script,
                            "--data_path", processed_data_dir,
                            "--zone", zone,
                            "--frequency", band_name,
                            "--save_dir", MODEL_SAVE_DIR,
                            "--plot_dir", os.path.join(FIGURE_SAVE_DIR, "pretrain"),
                            "--batch_size", "512",
                            "--num_epochs", str(args.num_epochs),
                            "--fold_id", f"rep{i}",
                            "--aug_mode", args.aug_mode,
                            "--positives", args.positives,
                            "--neighbor_metric", args.neighbor_metric,
                            "--neighbor_index_dir", args.neighbor_index_dir,
                            # Prevent leakage: the SSL backbone must NOT see test subjects.
                            "--exclude_subjects", *[str(s) for s in test_subjects],
                        ]
                    elif args.method == "AE":
                        train_script = "src/train_auto.py"
                        train_cmd = [
                            "python", train_script,
                            "--data-path", os.path.join(processed_data_dir, "processed_windows.npy"),
                            "--meta-path", os.path.join(processed_data_dir, "processed_metadata.csv"),
                            "--zone", zone,
                            "--frequency", band_name,
                            "--save-model-dir", MODEL_SAVE_DIR,
                            "--save-fig-dir", os.path.join(FIGURE_SAVE_DIR, "pretrain"),
                            "--epochs", str(args.num_epochs),
                            "--fold_id", f"rep{i}",
                            # Prevent leakage: the SSL backbone must NOT see test subjects.
                            "--exclude_subjects", *[str(s) for s in test_subjects],
                        ]
                    
                    subprocess.run(train_cmd, check=True, capture_output=True)

                    # --- 3. Model Search and Downstream Evaluation ---
                    print(f"  [3/4] Evaluating on downstream task for {run_id}...")
                    
                    # The model name is variable, we search for it by pattern
                    model_pattern = f"{args.method}_{zone}_{band_name}*.pth"
                    backbone_model_path = find_latest_model(MODEL_SAVE_DIR, model_pattern)

                    if not backbone_model_path:
                        print(f"  [ERROR] Model not found for {run_id}. Skipping evaluation.")
                        continue

                    print(f"        Model found: {backbone_model_path}")

                    downstream_cmd = [
                        "python", "src/downstream.py",
                        "--data_path", os.path.join(processed_data_dir, "processed_windows.npy"),
                        "--meta_path", os.path.join(processed_data_dir, "processed_metadata.csv"),
                        "--model_path", backbone_model_path,
                        "--method", args.method,
                        "--zone", zone,
                        "--frequency", band_name,
                        "--target", args.target,
                        "--test_subjects", *[str(s) for s in test_subjects],
                        "--num_epochs", str(args.num_epochs),
                    ]

                elif args.method == "supervised":

                    print(f"  [2/4] Skipping pre-training for supervised method.")
                    print(f"  [3/4] Running end-to-end supervised training for {run_id}...")
                    downstream_cmd = [
                        "python", "src/downstream.py",
                        "--data_path", os.path.join(processed_data_dir, "processed_windows.npy"),
                        "--meta_path", os.path.join(processed_data_dir, "processed_metadata.csv"),
                        "--method", "supervised",
                        "--zone", zone,
                        "--frequency", band_name,
                        "--target", args.target,
                        "--test_subjects", *[str(s) for s in test_subjects],
                        "--num_epochs", str(args.num_epochs),
                    ]
                # We run and capture the output to parse the nRMSE
                result = subprocess.run(downstream_cmd, check=True, capture_output=True, text=True)
                
                # --- 4. Collecting Results ---
                print(f"  [4/4] Collecting results for {run_id}...")
                output = result.stdout
                match = re.search(r"Test nRMSE \(Subject-Avg\)=([\d.]+)", output)
                
                if match:
                    nrmse = float(match.group(1))
                    print(f"        nRMSE found: {nrmse:.4f}")
                    results.append({
                        "repetition": i,
                        "zone": zone,
                        "frequency": band_name,
                        "nrmse": nrmse
                    })
                else:
                    print("        WARNING: Could not find nRMSE in the output.")

    # --- 5. Final Visualization ---
    if not results:
        print("\nNo results collected. Cannot generate plot.")
        return

    print("\n=== GENERATING RESULTS PLOT ===")
    results_df = pd.DataFrame(results)

    # Save numerical results to a CSV
    raw_csv_path = os.path.join(FIGURE_SAVE_DIR, f"summary_results_{args.method}_{args.target}.csv")
    results_df.to_csv(raw_csv_path, index=False)
    print(f"Results saved to: {raw_csv_path}")

    # Aggregate for CSV
    summary_df = results_df.groupby(['zone', 'frequency'])['nrmse'].agg(['mean', 'std']).reset_index()
    summary_df = summary_df.rename(columns={'mean': 'nrmse_mean', 'std': 'nrmse_std'})
    
    # Save numerical results to a CSV
    csv_path = os.path.join(FIGURE_SAVE_DIR, f"agg_summary_results_{args.method}_{args.target}.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"Aggregated Results saved to: {csv_path}")

    # Generate plot from raw (unaggregated) data
    plt.figure(figsize=(16, 9))
    sns.set_theme(style="whitegrid")

    # Define order for consistency
    zone_order = INDIVIDUAL_ZONES + ["all"]
    band_order = list(BANDS.keys())

    ax = sns.barplot(
        data=results_df,
        x="frequency",
        y="nrmse",
        hue="zone",
        order=band_order,
        hue_order=zone_order,
        errorbar="sd"  # Use standard deviation for error bars
    )

    ax.set_title(f'nRMSE Performance for Method "{args.method}" on Target "{args.target}" (Reps={args.repetitions})', fontsize=16)
    ax.set_xlabel("Frequency Band", fontsize=12)
    ax.set_ylabel("Normalized RMSE (nRMSE)", fontsize=12)
    plt.legend(title="Zone", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make room for legend

    # Save the plot
    plot_path = os.path.join(FIGURE_SAVE_DIR, f"summary_plot_{args.method}_{args.target}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Summary plot saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete experimentation pipeline for methods"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["SimCLR", "AE", "supervised"],
        required=True,
        help="Pre-training method to use"
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["age", "cit_36mo"],
        required=True,
        help="Prediction target"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of times to run each experiment for statistical analysis."
    )
    parser.add_argument(
        "--preprocessing",
        action="store_true",
        default=False,
        help="Needed to preprocess?"
    )
    parser.add_argument(
        "--test_k",
        type=int,
        default=9,
        help="Number of held-out test subjects per repetition (subject-based split)."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Training epochs per supervised cell (forwarded to downstream.py)."
    )
    parser.add_argument(
        "--norm_mode",
        type=str,
        default="per_channel",
        choices=["per_channel", "global", "none"],
        help="Amplitude normalization forwarded to postprocessing.py (per_channel|global|none)."
    )
    parser.add_argument(
        "--aug_mode",
        type=str,
        default="legacy",
        choices=["legacy", "no_swap", "legacy_plus_psd", "zone_preserving"],
        help="Augmentation strategy forwarded to train_simclr.py (legacy|zone_preserving)."
    )
    parser.add_argument(
        "--positives",
        type=str,
        default="augment",
        choices=["augment", "neighbor"],
        help="Positive pair source forwarded to train_simclr.py (augment|neighbor)."
    )
    parser.add_argument(
        "--neighbor_metric",
        type=str,
        default="cosine",
        choices=["cosine", "wasserstein", "riemann"],
        help="Neighbor distance forwarded to train_simclr.py (cosine|wasserstein|riemann)."
    )
    parser.add_argument(
        "--neighbor_index_dir",
        type=str,
        default="data/processed/neighbor_index",
        help="Directory with neighbor_index_<metric>.npy forwarded to train_simclr.py."
    )
    args = parser.parse_args()
    main(args)