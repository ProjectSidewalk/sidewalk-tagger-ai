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


def _build_rows(tag_stats_map: dict, min_instances: int, target_precision: float) -> list:
    rows = []
    for tag_name, tag_stats in tag_stats_map.items():
        n_instances = int(tag_stats.get("n_instances", 0))
        if n_instances < min_instances:
            continue
        precision, recall, thresholds = aligned_arrays(tag_stats)
        result = pick_threshold_for_target_precision(precision, recall, thresholds, target_precision)
        if result is None:
            continue
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
    return rows


def plot_precision_vs_threshold(tag_stats_map: dict, target_precision: float, min_instances: int, output_plot: Path, direction: str = "positive"):
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

    if direction == "negative":
        ax.set_xlabel("(1 − confidence) threshold")
        ax.set_ylabel("Precision (negative class: tag NOT applied)")
        ax.set_title("Precision vs (1 − Confidence) Threshold — negative class")
    else:
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


def _negative_path(path: Path) -> Path:
    """Insert '-negative' before the last hyphen-separated segment of the stem.

    e.g. validated-dino-inference-stats.json -> validated-dino-inference-stats-negative.json
    """
    parts = path.stem.rsplit("-", 1)
    if len(parts) == 2:
        return path.parent / (parts[0] + "-negative-" + parts[1] + path.suffix)
    return path.parent / (path.stem + "-negative" + path.suffix)


def _run_direction(stats_file, target_precision, min_instances, output_plot, output_csv, skip_plot, direction):
    tag_stats_map = load_tag_stats(stats_file)

    if skip_plot:
        rows = _build_rows(tag_stats_map, min_instances, target_precision)
    else:
        try:
            rows = plot_precision_vs_threshold(tag_stats_map, target_precision, min_instances, output_plot, direction)
        except ModuleNotFoundError as exc:
            raise SystemExit(str(exc))

    save_summary_csv(rows, output_csv)

    direction_label = "positive" if direction == "positive" else "negative (tag NOT applied)"
    print(f"\n=== {direction_label} ===")

    if not rows:
        print("No tags were processed. Try lowering --min-instances.")
        return

    reached = [r for r in rows if r["reached_target"]]
    not_reached = [r for r in rows if not r["reached_target"]]

    print("Plot generation skipped (--skip-plot)." if skip_plot else f"Saved plot: {output_plot}")
    print(f"Saved summary CSV: {output_csv}")
    print(f"Tags processed: {len(rows)}")
    print(f"Reached target precision ({target_precision:.2f}): {len(reached)}")
    print(f"Did not reach target precision: {len(not_reached)}\n")

    print("Selected thresholds by tag:")
    for row in sorted(rows, key=lambda r: r["n_instances"], reverse=True):
        marker = "OK" if row["reached_target"] else "MAX_ONLY"
        print(
            f"- {row['tag']}: threshold={row['selected_threshold']:.4f}, "
            f"precision={row['selected_precision']:.4f}, "
            f"recall={row['selected_recall']:.4f} [{marker}]"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot precision vs confidence threshold from test.py stats JSON and identify threshold points."
    )

    parser.add_argument("--stats-file", type=Path,
        default=REPO_ROOT / Path("results/crosswalk/validated-dino-inference-stats.json"),
        help="Path to positive-direction JSON output from notebooks/test.py")

    parser.add_argument("--target-precision", type=float, default=0.92,
                        help="Target precision (default: 0.92)")

    parser.add_argument("--min-instances", type=int, default=10,
                        help="Only include tags with at least this many GT positives")

    parser.add_argument("--output-plot", type=Path,
        default=REPO_ROOT / Path("results/crosswalk/validated-dino-precision-vs-threshold.png"),
        help="Output image path for positive direction")

    parser.add_argument("--output-csv", type=Path,
        default=REPO_ROOT / Path("results/crosswalk/validated-dino-thresholds-at-target-precision.csv"),
        help="Output CSV path for positive direction")

    parser.add_argument("--skip-plot", action="store_true",
                        help="Skip graph generation and only write CSV summaries")

    parser.add_argument("--stats-file-negative", type=Path, default=None,
                        help="Path to negative-direction JSON (default: auto-derived from --stats-file)")

    parser.add_argument("--output-plot-negative", type=Path, default=None,
                        help="Output image path for negative direction (default: auto-derived from --output-plot)")

    parser.add_argument("--output-csv-negative", type=Path, default=None,
                        help="Output CSV path for negative direction (default: auto-derived from --output-csv)")

    args = parser.parse_args()

    if not (0.0 <= args.target_precision <= 1.0):
        raise ValueError("--target-precision must be between 0 and 1")

    _run_direction(
        stats_file=args.stats_file,
        target_precision=args.target_precision,
        min_instances=args.min_instances,
        output_plot=args.output_plot,
        output_csv=args.output_csv,
        skip_plot=args.skip_plot,
        direction="positive",
    )

    neg_stats_file = args.stats_file_negative or _negative_path(args.stats_file)
    if not neg_stats_file.exists():
        print(f"\nNegative stats file not found ({neg_stats_file}), skipping negative direction.")
        print("Re-run test.py to generate it.")
        return

    _run_direction(
        stats_file=neg_stats_file,
        target_precision=args.target_precision,
        min_instances=args.min_instances,
        output_plot=args.output_plot_negative or _negative_path(args.output_plot),
        output_csv=args.output_csv_negative or _negative_path(args.output_csv),
        skip_plot=args.skip_plot,
        direction="negative",
    )


if __name__ == "__main__":
    main()
