import argparse
import csv
import json
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def load_tag_stats(stats_file: Path) -> dict:
    if not stats_file.exists():
        raise FileNotFoundError(
            f"Stats file not found: {stats_file}. Pass --stats-file explicitly if your file is elsewhere."
        )

    with stats_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "category_to_prediction_stats" not in data:
        raise ValueError(f"Missing 'category_to_prediction_stats' key in {stats_file}")

    return data["category_to_prediction_stats"]


def aligned_arrays(tag_stats: dict):
    precision = np.asarray(tag_stats["precision"], dtype=float)
    recall = np.asarray(tag_stats["recall"], dtype=float)
    thresholds = np.asarray(tag_stats["thresholds"], dtype=float)

    if len(precision) == len(thresholds) + 1:
        precision, recall = precision[:-1], recall[:-1]

    if not (len(precision) == len(recall) == len(thresholds)):
        raise ValueError("Mismatched precision/recall/threshold lengths after alignment")

    return precision, recall, thresholds


def pick_threshold_for_target_precision(
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
    target_precision: float,
):
    finite_mask = np.isfinite(precision) & np.isfinite(recall) & np.isfinite(thresholds)
    precision, recall, thresholds = precision[finite_mask], recall[finite_mask], thresholds[finite_mask]

    if len(precision) == 0:
        return None

    meets_target = np.where(precision >= target_precision)[0]

    if len(meets_target) == 0:
        max_idx = int(np.argmax(precision))
        return {
            "reached_target": False,
            "threshold": float(thresholds[max_idx]),
            "precision": float(precision[max_idx]),
            "recall": float(recall[max_idx]),
            "max_precision": float(precision[max_idx]),
            "max_precision_threshold": float(thresholds[max_idx]),
        }

    best_idx = int(meets_target[np.argmax(recall[meets_target] - (thresholds[meets_target] * 1e-12))])
    max_idx = int(np.argmax(precision))

    return {
        "reached_target": True,
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "max_precision": float(precision[max_idx]),
        "max_precision_threshold": float(thresholds[max_idx]),
    }


def plot_precision_vs_threshold(tag_stats_map: dict, target_precision: float, min_instances: int, output_plot: Path):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate the graph. Install it with: pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=(14, 9))
    rows = []

    for tag_name, tag_stats in tag_stats_map.items():
        n_instances = int(tag_stats.get("n_instances", 0))
        if n_instances < min_instances:
            continue

        precision, recall, thresholds = aligned_arrays(tag_stats)
        result = pick_threshold_for_target_precision(precision, recall, thresholds, target_precision)
        if result is None:
            continue

        ax.plot(thresholds, precision, linewidth=2, label=f"{tag_name} (n={n_instances})")

        marker_color = "green" if result["reached_target"] else "red"
        ax.scatter([result["threshold"]], [result["precision"]], color=marker_color, s=42, zorder=3)

        rows.append({
            "tag": tag_name,
            "n_instances": n_instances,
            "target_precision": target_precision,
            "reached_target": result["reached_target"],
            "selected_threshold": result["threshold"],
            "selected_precision": result["precision"],
            "selected_recall": result["recall"],
            "max_precision": result["max_precision"],
            "max_precision_threshold": result["max_precision_threshold"],
        })

    ax.axhline(y=target_precision, color="black", linestyle="--", linewidth=1.2,
               label=f"Target precision = {target_precision:.2f}")

    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Confidence Threshold (Validated DINOv2 Crosswalk)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

    fig.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)

    return rows


def save_summary_csv(rows, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "tag",
            "n_instances",
            "target_precision",
            "reached_target",
            "selected_threshold",
            "selected_precision",
            "selected_recall",
            "max_precision",
            "max_precision_threshold",
        ])
        writer.writeheader()

        for row in sorted(rows, key=lambda r: r["n_instances"], reverse=True):
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Plot precision vs confidence threshold from test.py stats JSON and identify threshold points."
    )

    parser.add_argument("--stats-file", type=Path,
        default=REPO_ROOT / Path("datasets/crops-crosswalk-tags/validated-dino-inference-stats-test.json"),
        help="Path to JSON output from notebooks/test.py")

    parser.add_argument("--target-precision", type=float, default=0.92,
                        help="Target precision (default: 0.92)")

    parser.add_argument("--min-instances", type=int, default=10,
                        help="Only include tags with at least this many GT positives")

    parser.add_argument("--output-plot", type=Path,
        default=REPO_ROOT / Path("datasets/crops-crosswalk-tags/validated-dino-precision-vs-threshold-test.png"),
        help="Output image path")

    parser.add_argument("--output-csv", type=Path,
        default=REPO_ROOT / Path("datasets/crops-crosswalk-tags/validated-dino-thresholds-at-target-precision-test.csv"),
        help="Output CSV path")

    parser.add_argument("--skip-plot", action="store_true",
                        help="Skip graph generation and only write CSV summary")

    args = parser.parse_args()

    if not (0.0 <= args.target_precision <= 1.0):
        raise ValueError("--target-precision must be between 0 and 1")

    tag_stats_map = load_tag_stats(args.stats_file)

    if args.skip_plot:
        rows = []

        for tag_name, tag_stats in tag_stats_map.items():
            n_instances = int(tag_stats.get("n_instances", 0))
            if n_instances < args.min_instances:
                continue

            precision, recall, thresholds = aligned_arrays(tag_stats)
            result = pick_threshold_for_target_precision(precision, recall, thresholds, args.target_precision)

            if result is None:
                continue

            rows.append({
                "tag": tag_name,
                "n_instances": n_instances,
                "target_precision": args.target_precision,
                "reached_target": result["reached_target"],
                "selected_threshold": result["threshold"],
                "selected_precision": result["precision"],
                "selected_recall": result["recall"],
                "max_precision": result["max_precision"],
                "max_precision_threshold": result["max_precision_threshold"],
            })

    else:
        try:
            rows = plot_precision_vs_threshold(
                tag_stats_map=tag_stats_map,
                target_precision=args.target_precision,
                min_instances=args.min_instances,
                output_plot=args.output_plot,
            )
        except ModuleNotFoundError as exc:
            raise SystemExit(str(exc))

    save_summary_csv(rows, args.output_csv)

    if not rows:
        print("No tags were plotted. Try lowering --min-instances.")
        return

    reached = [r for r in rows if r["reached_target"]]
    not_reached = [r for r in rows if not r["reached_target"]]

    print("Plot generation skipped (--skip-plot)." if args.skip_plot else f"Saved plot: {args.output_plot}")
    print(f"Saved summary CSV: {args.output_csv}")
    print(f"Tags processed: {len(rows)}")
    print(f"Reached target precision ({args.target_precision:.2f}): {len(reached)}")
    print(f"Did not reach target precision: {len(not_reached)}\n")

    print("Selected thresholds by tag:")
    for row in sorted(rows, key=lambda r: r["n_instances"], reverse=True):
        marker = "OK" if row["reached_target"] else "MAX_ONLY"
        print(
            f"- {row['tag']}: threshold={row['selected_threshold']:.4f}, "
            f"precision={row['selected_precision']:.4f}, "
            f"recall={row['selected_recall']:.4f} [{marker}]"
        )


if __name__ == "__main__":
    main()