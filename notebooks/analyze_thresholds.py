import argparse
import csv
import json
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Z-score for the 95% Wilson interval. Hard-coded so we don't take a scipy dependency.
WILSON_Z = 1.96


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

    # sklearn's precision_recall_curve appends a sentinel point (precision=1, recall=0) with no corresponding threshold,
    # so precision/recall are one element longer than thresholds.
    if len(precision) == len(thresholds) + 1:
        precision, recall = precision[:-1], recall[:-1]

    if not (len(precision) == len(recall) == len(thresholds)):
        raise ValueError("Mismatched precision/recall/threshold lengths after alignment")

    return precision, recall, thresholds


def wilson_lower_bound(p_hat: np.ndarray, n: np.ndarray, z: float = WILSON_Z) -> np.ndarray:
    """Lower bound of the Wilson score interval at confidence level implied by z (default 95%)."""
    p_hat = np.asarray(p_hat, dtype=float)
    n = np.asarray(n, dtype=float)
    safe_n = np.where(n > 0, n, 1.0)
    denom = 1.0 + z**2 / safe_n
    centre = p_hat + z**2 / (2.0 * safe_n)
    radicand = np.maximum(p_hat * (1.0 - p_hat) / safe_n + z**2 / (4.0 * safe_n**2), 0.0)
    margin = z * np.sqrt(radicand)
    lb = (centre - margin) / denom
    return np.where(n > 0, lb, 0.0)


def derive_counts(precision: np.ndarray, recall: np.ndarray, n_instances: int):
    """Recover (TP, FP, support) at each PR-curve point. precision = TP/(TP+FP); recall = TP/n_instances."""
    tp = np.rint(recall * n_instances).astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        support = np.where(precision > 0, tp / np.where(precision > 0, precision, 1.0), 0.0)
    support = np.rint(support).astype(int)
    fp = np.maximum(support - tp, 0)
    return tp, fp, support


def base_rate_from_curve(precision: np.ndarray) -> float:
    """At the smallest threshold every sample is predicted positive, so precision[0] = base rate of analyzed class."""
    if len(precision) == 0:
        return 0.0
    return float(precision[0])


def pick_qualifying_threshold(
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
    n_instances: int,
    target_precision: float,
    target_lower_bound: float,
    min_lift_over_prior: float,
):
    finite_mask = np.isfinite(precision) & np.isfinite(recall) & np.isfinite(thresholds)
    precision = precision[finite_mask]
    recall = recall[finite_mask]
    thresholds = thresholds[finite_mask]

    if len(precision) == 0 or n_instances <= 0:
        return None

    base_rate = base_rate_from_curve(precision)
    effective_target_lower_bound = max(target_lower_bound, base_rate + min_lift_over_prior)

    tp, fp, support = derive_counts(precision, recall, n_instances)
    lower_bounds = wilson_lower_bound(precision, support)

    # Total test items in this direction's evaluation. base_rate = n_instances / total at the lowest-threshold point.
    total_test = int(round(n_instances / base_rate)) if base_rate > 0 else int(n_instances)
    n_opposite = max(total_test - n_instances, 0)

    def _row(idx: int, reached: bool, max_lb_idx: int) -> dict:
        tp_i = int(tp[idx])
        fp_i = int(fp[idx])
        return {
            "reached_target": reached,
            "threshold": float(thresholds[idx]),
            "precision": float(precision[idx]),
            "precision_lower_bound": float(lower_bounds[idx]),
            "recall": float(recall[idx]),
            "tp": tp_i,
            "fp": fp_i,
            "fn": max(n_instances - tp_i, 0),
            "tn": max(n_opposite - fp_i, 0),
            "base_rate": base_rate,
            "effective_target_lower_bound": effective_target_lower_bound,
            "max_lower_bound": float(lower_bounds[max_lb_idx]),
            "max_lower_bound_threshold": float(thresholds[max_lb_idx]),
        }

    max_lb_idx = int(np.argmax(lower_bounds))

    meets_target = np.where((precision >= target_precision) & (lower_bounds >= effective_target_lower_bound))[0]
    if len(meets_target) == 0:
        return _row(max_lb_idx, reached=False, max_lb_idx=max_lb_idx)

    # Among qualifying thresholds, maximise recall. The 1e-12 tie-break prefers the lower (less aggressive) threshold.
    best_idx = int(meets_target[np.argmax(recall[meets_target] - (thresholds[meets_target] * 1e-12))])
    return _row(best_idx, reached=True, max_lb_idx=max_lb_idx)


def _user_facing_threshold(raw_threshold: float, direction: str) -> float:
    """Negative-direction PR curves use a (1 − confidence) scale internally; flip it for user-facing reports."""
    if direction == "negative":
        return 1.0 - raw_threshold
    return raw_threshold


def _row_for_csv(
    tag_name: str, target_precision: float, target_lower_bound: float, direction: str, result: dict,
) -> dict:
    return {
        "tag": tag_name,
        "target_precision": target_precision,
        "target_precision_lower_bound": target_lower_bound,
        "base_rate": result["base_rate"],
        "effective_target_lower_bound": result["effective_target_lower_bound"],
        "reached_target": result["reached_target"],
        "selected_threshold": _user_facing_threshold(result["threshold"], direction),
        "selected_precision": result["precision"],
        "selected_precision_lower_bound": result["precision_lower_bound"],
        "selected_recall": result["recall"],
        "tp": result["tp"],
        "fp": result["fp"],
        "fn": result["fn"],
        "tn": result["tn"],
        "max_lower_bound": result["max_lower_bound"],
        "max_lower_bound_threshold": _user_facing_threshold(result["max_lower_bound_threshold"], direction),
    }


def max_possible_lower_bound(n_instances: int) -> float:
    """Maximum Wilson lower bound any classifier could achieve for this tag's sample size (perfect-precision case)."""
    if n_instances <= 0:
        return 0.0
    return float(wilson_lower_bound(1.0, n_instances))


def find_degenerate_tags(pos_stats_map: dict, neg_stats_map: dict | None) -> list:
    """Return [(tag, n_pos, n_neg)] for tags whose test set has 0 examples in one of the two classes.

    Such tags can't be meaningfully evaluated in either direction: the empty side has nothing to score, and the
    saturated side is just the trivial-baseline classifier.
    """
    degenerate: list[tuple[str, int, int]] = []
    all_tags = set(pos_stats_map) | set(neg_stats_map or {})
    for tag in sorted(all_tags):
        n_pos = int(pos_stats_map.get(tag, {}).get("n_instances", 0))
        if neg_stats_map is not None:
            n_neg = int(neg_stats_map.get(tag, {}).get("n_instances", 0))
        else:
            # Derive n_neg from the positive PR curve when the negative file is missing.
            pos_curve = pos_stats_map.get(tag, {}).get("precision", [])
            base_rate = float(pos_curve[0]) if pos_curve else 0.0
            n_neg = int(round(n_pos / base_rate - n_pos)) if base_rate > 0 else 0
        if n_pos == 0 or n_neg == 0:
            degenerate.append((tag, n_pos, n_neg))
    return degenerate


def _build_rows(
    tag_stats_map: dict,
    target_precision: float,
    target_lower_bound: float,
    min_lift_over_prior: float,
    direction: str,
    skip_tags: set,
) -> list:
    rows = []
    for tag_name, tag_stats in tag_stats_map.items():
        if tag_name in skip_tags:
            continue
        n_instances = int(tag_stats.get("n_instances", 0))
        if n_instances <= 0:
            continue
        precision, recall, thresholds = aligned_arrays(tag_stats)
        result = pick_qualifying_threshold(
            precision, recall, thresholds, n_instances,
            target_precision, target_lower_bound, min_lift_over_prior,
        )
        if result is None:
            continue
        rows.append(_row_for_csv(tag_name, target_precision, target_lower_bound, direction, result))
    return rows


def plot_lower_bound_vs_threshold(
    tag_stats_map: dict,
    target_precision: float,
    target_lower_bound: float,
    min_lift_over_prior: float,
    output_plot: Path,
    direction: str = "positive",
    title: str = "",
    skip_tags: set | None = None,
):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate the graph. Install it with: pip install matplotlib"
        ) from exc

    skip_tags = skip_tags or set()
    fig, ax = plt.subplots(figsize=(14, 9))
    rows = []
    excluded_tags = []  # (tag_name, n_instances, effective_target) — included in CSV, omitted from plot only.

    for tag_name, tag_stats in tag_stats_map.items():
        if tag_name in skip_tags:
            continue
        n_instances = int(tag_stats.get("n_instances", 0))
        if n_instances <= 0:
            continue

        precision, recall, thresholds = aligned_arrays(tag_stats)
        result = pick_qualifying_threshold(
            precision, recall, thresholds, n_instances,
            target_precision, target_lower_bound, min_lift_over_prior,
        )
        if result is None:
            continue

        rows.append(_row_for_csv(tag_name, target_precision, target_lower_bound, direction, result))

        # Plot exclusion: even with a perfect classifier, this tag's structural ceiling on the Wilson lower bound
        # cannot clear its effective target. Catches both small-sample tags (Wilson cap from n_instances) and
        # high-prior tags (effective target > 1.0 from base_rate + min_lift_over_prior). Still listed in CSV.
        effective_target = result["effective_target_lower_bound"]
        if max_possible_lower_bound(n_instances) < effective_target:
            excluded_tags.append((tag_name, n_instances, effective_target))
            continue

        _, _, support = derive_counts(precision, recall, n_instances)
        lower_bounds = wilson_lower_bound(precision, support)

        ax.plot(thresholds, lower_bounds, linewidth=2, label=f"{tag_name} (n={n_instances})")

        # Only mark a selected point when the tag met both gates (precision target and effective lower bound).
        # An unmarked curve is the visual signal that no threshold qualifies.
        if result["reached_target"]:
            ax.scatter([result["threshold"]], [result["precision_lower_bound"]], color="green", s=42, zorder=3)

    ax.axhline(
        y=target_lower_bound,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Target lower bound = {target_lower_bound:.2f} (per-tag\neffective floor may be higher; see CSV)",
    )

    if excluded_tags:
        excluded_str = ", ".join(
            f"{name} (n={n}, eff_target={eff:.2f})" for name, n, eff in excluded_tags
        )
        note = (
            "Excluded from plot — sample size and/or class prior put the effective target lower bound "
            f"out of structural reach: {excluded_str}. (Still listed in CSV.)"
        )
        # Place the note below the axes so it doesn't collide with the curves.
        fig.text(0.01, 0.005, note, fontsize=8, style="italic", wrap=True)

    title_suffix = f" — {title}" if title else ""
    if direction == "negative":
        ax.set_xlabel("(1 − confidence) threshold")
        ax.set_ylabel("95% Wilson lower bound on precision (negative class: tag NOT applied)")
        ax.set_title(f"Wilson Lower Bound vs (1 − Confidence) Threshold — negative class{title_suffix}")
    else:
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel("95% Wilson lower bound on precision")
        ax.set_title(f"Wilson Lower Bound vs Confidence Threshold{title_suffix}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

    # Reserve a small bottom strip for the excluded-tags note when present.
    bottom_margin = 0.08 if excluded_tags else 0.05
    fig.tight_layout(rect=(0, bottom_margin, 1, 1))
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot)
    plt.close(fig)

    return rows


def save_summary_csv(rows, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "tag",
            "target_precision",
            "target_precision_lower_bound",
            "base_rate",
            "effective_target_lower_bound",
            "reached_target",
            "selected_threshold",
            "selected_precision",
            "selected_precision_lower_bound",
            "selected_recall",
            "tp",
            "fp",
            "fn",
            "tn",
            "max_lower_bound",
            "max_lower_bound_threshold",
        ])
        writer.writeheader()

        for row in sorted(rows, key=lambda r: r["tp"] + r["fn"], reverse=True):
            writer.writerow(row)


def _negative_path(path: Path) -> Path:
    return path.parent / (path.stem + "-negative" + path.suffix)


def _run_direction(
    tag_stats_map, stats_file, target_precision, target_lower_bound, min_lift_over_prior,
    output_plot, output_csv, skip_plot, direction, skip_tags,
):
    # Derive a human-readable title from the file path, e.g., "validated-dino / crosswalk".
    title = f"{stats_file.stem.split('-inference')[0]} / {stats_file.parent.name}"

    if skip_plot:
        rows = _build_rows(
            tag_stats_map, target_precision, target_lower_bound, min_lift_over_prior, direction, skip_tags,
        )
    else:
        try:
            rows = plot_lower_bound_vs_threshold(
                tag_stats_map, target_precision, target_lower_bound, min_lift_over_prior,
                output_plot, direction, title, skip_tags,
            )
        except ModuleNotFoundError as exc:
            raise SystemExit(str(exc))

    save_summary_csv(rows, output_csv)

    direction_label = "positive" if direction == "positive" else "negative (tag NOT applied)"
    print(f"\n=== {direction_label} ===")

    if not rows:
        print("No tags were processed (no tag in this direction has any ground-truth instances).")
        return rows

    reached = [r for r in rows if r["reached_target"]]
    not_reached = [r for r in rows if not r["reached_target"]]

    print("Plot generation skipped (--skip-plot)." if skip_plot else f"Saved plot: {output_plot}")
    print(f"Saved summary CSV: {output_csv}")
    print(f"Tags processed: {len(rows)}")
    print(f"Reached both targets (precision ≥ {target_precision:.2f} and lower bound ≥ effective): {len(reached)}")
    print(f"Did not reach targets: {len(not_reached)}\n")

    print("Selected thresholds by tag:")
    for row in sorted(rows, key=lambda r: r["tp"] + r["fn"], reverse=True):
        marker = "OK" if row["reached_target"] else "MAX_ONLY"
        print(
            f"- {row['tag']}: threshold={row['selected_threshold']:.4f}, "
            f"lower_bound={row['selected_precision_lower_bound']:.4f} "
            f"(eff_target={row['effective_target_lower_bound']:.4f}, base_rate={row['base_rate']:.4f}), "
            f"precision={row['selected_precision']:.4f}, "
            f"recall={row['selected_recall']:.4f}, "
            f"tp={row['tp']}, fp={row['fp']}, fn={row['fn']}, tn={row['tn']} [{marker}]"
        )

    return rows


def build_deployment_rows(pos_rows: list, neg_rows: list) -> list:
    """Combine per-direction rows into per-tag deployment thresholds, using the tightest validated boundary
    available for each side.

    The per-direction analyses validate two distinct claims:
      • pos.reached_target → "predict PRESENT at conf ≥ T_pos has ≥ target precision."  Validates ADD.
      • neg.reached_target → "predict ABSENT at conf ≤ T_neg has ≥ target precision."   Validates REMOVE.

    What we deploy depends on which directions qualified:

      ┌────────────┬────────────┬──────────────┬───────────────────────────────────────────────────────────┐
      │ pos        │ neg        │ status       │ deployment scheme                                         │
      ├────────────┼────────────┼──────────────┼───────────────────────────────────────────────────────────┤
      │ qualified  │ qualified  │ ready        │ If T_neg ≥ T_pos: disjoint scheme — ADD when conf > T_neg,│
      │            │            │              │ REMOVE when conf < T_pos, silent zone in between.         │
      │            │            │              │ If T_neg < T_pos (thresholds overlap): direct scheme —    │
      │            │            │              │ ADD when conf ≥ T_pos, REMOVE when conf ≤ T_neg.          │
      ├────────────┼────────────┼──────────────┼───────────────────────────────────────────────────────────┤
      │ qualified  │ failed     │ add_only     │ ADD when conf ≥ T_pos. No REMOVE deployed.                │
      ├────────────┼────────────┼──────────────┼───────────────────────────────────────────────────────────┤
      │ failed     │ qualified  │ remove_only  │ REMOVE when conf ≤ T_neg. No ADD deployed.                │
      ├────────────┼────────────┼──────────────┼───────────────────────────────────────────────────────────┤
      │ failed     │ failed     │ not_ready    │ Nothing deployed.                                         │
      └────────────┴────────────┴──────────────┴───────────────────────────────────────────────────────────┘

    Important consequence: `add_threshold` and `remove_threshold` do NOT have a uniform interpretation across rows.
    In the `ready` disjoint case they are set to the OPPOSITE direction's threshold (the tighter boundary); in the
    `ready` direct case and partial cases they are set to the SAME direction's threshold. The `deployment_status`
    column is the disambiguator. The threshold's *usage* in production code is constant — `conf > add_threshold` for
    ADD, `conf < remove_threshold` for REMOVE — but the underlying validation chain differs.

    Going from `add_only` to `ready` raises (tightens) the ADD bar when the disjoint scheme applies; same for REMOVE /
    `remove_only` → `ready`. When the direct scheme applies the bars are the same as the partial cases.
    """
    pos_by_tag = {r["tag"]: r for r in pos_rows}
    neg_by_tag = {r["tag"]: r for r in neg_rows}

    deployment_rows = []
    for tag in sorted(pos_by_tag.keys() & neg_by_tag.keys()):
        pos = pos_by_tag[tag]
        neg = neg_by_tag[tag]

        # Test-set class counts are direction-independent; pulling from the positive row is arbitrary.
        n_test_pos = pos["tp"] + pos["fn"]
        n_test_neg = pos["fp"] + pos["tn"]

        pos_ok = bool(pos["reached_target"])
        neg_ok = bool(neg["reached_target"])

        if pos_ok and neg_ok:
            status = "ready"
        elif pos_ok:
            status = "add_only"      # pos validates ADD (the natural direction → recommendation mapping).
        elif neg_ok:
            status = "remove_only"   # neg validates REMOVE.
        else:
            status = "not_ready"

        row = {
            "tag": tag,
            "n_test_pos": n_test_pos,
            "n_test_neg": n_test_neg,
            "deployment_status": status,
            "add_threshold": None,
            "add_tp": None,
            "add_fp": None,
            "add_precision": None,
            "add_recall": None,
            "remove_threshold": None,
            "remove_tp": None,
            "remove_fp": None,
            "remove_precision": None,
            "remove_recall": None,
            "silent_n_pos": None,
            "silent_n_neg": None,
        }

        if status == "ready":
            # User-facing thresholds: ADD when conf > t_neg, REMOVE when conf < t_pos.
            t_neg = neg["selected_threshold"]  # 1 − neg_raw; tag absent predicted below this
            t_pos = pos["selected_threshold"]  # pos_raw; tag present predicted above this

            if t_neg >= t_pos:
                # Disjoint scheme: ADD zone (conf > T_neg) and REMOVE zone (conf < T_pos) have a silent zone
                # in between. Counts come from the OPPOSITE direction's confusion matrix:
                #   • neg["tn"] = positives at conf > T_neg → true ADD.
                #   • neg["fn"] = negatives at conf > T_neg → false ADD.
                #   • pos["tn"] = negatives at conf < T_pos → true REMOVE.
                #   • pos["fn"] = positives at conf < T_pos → false REMOVE.
                add_tp = neg["tn"]
                add_fp = neg["fn"]
                add_n = add_tp + add_fp
                row["add_threshold"] = t_neg
                row["add_tp"] = add_tp
                row["add_fp"] = add_fp
                row["add_precision"] = add_tp / add_n if add_n > 0 else 0.0
                row["add_recall"] = add_tp / n_test_pos if n_test_pos > 0 else 0.0

                remove_tp = pos["tn"]
                remove_fp = pos["fn"]
                remove_n = remove_tp + remove_fp
                row["remove_threshold"] = t_pos
                row["remove_tp"] = remove_tp
                row["remove_fp"] = remove_fp
                row["remove_precision"] = remove_tp / remove_n if remove_n > 0 else 0.0
                row["remove_recall"] = remove_tp / n_test_neg if n_test_neg > 0 else 0.0
            else:
                # Thresholds overlap (T_neg < T_pos): the disjoint scheme would create a zone where both
                # ADD and REMOVE fire simultaneously. Fall back to each direction's own validated threshold:
                # ADD when conf >= T_pos, REMOVE when conf <= T_neg, silent zone T_neg <= conf <= T_pos.
                add_tp = pos["tp"]
                add_fp = pos["fp"]
                add_n = add_tp + add_fp
                row["add_threshold"] = t_pos
                row["add_tp"] = add_tp
                row["add_fp"] = add_fp
                row["add_precision"] = add_tp / add_n if add_n > 0 else 0.0
                row["add_recall"] = add_tp / n_test_pos if n_test_pos > 0 else 0.0

                remove_tp = neg["tp"]
                remove_fp = neg["fp"]
                remove_n = remove_tp + remove_fp
                row["remove_threshold"] = t_neg
                row["remove_tp"] = remove_tp
                row["remove_fp"] = remove_fp
                row["remove_precision"] = remove_tp / remove_n if remove_n > 0 else 0.0
                row["remove_recall"] = remove_tp / n_test_neg if n_test_neg > 0 else 0.0

            # Silent zone = test set minus the two deployment zones (formula is the same for both schemes).
            row["silent_n_pos"] = max(n_test_pos - row["add_tp"] - row["remove_fp"], 0)
            row["silent_n_neg"] = max(n_test_neg - row["add_fp"] - row["remove_tp"], 0)

        elif status == "add_only":
            # ADD zone is the positive direction's predicted-positive zone (conf ≥ T_pos). The TP/FP/precision/
            # recall come straight from the positive row — no recomputation needed.
            row["add_threshold"] = pos["selected_threshold"]
            row["add_tp"] = pos["tp"]
            row["add_fp"] = pos["fp"]
            row["add_precision"] = pos["selected_precision"]
            row["add_recall"] = pos["selected_recall"]

        elif status == "remove_only":
            # REMOVE zone is the negative direction's predicted-absent zone (conf ≤ T_neg). Numbers come
            # straight from the negative row.
            row["remove_threshold"] = neg["selected_threshold"]
            row["remove_tp"] = neg["tp"]
            row["remove_fp"] = neg["fp"]
            row["remove_precision"] = neg["selected_precision"]
            row["remove_recall"] = neg["selected_recall"]

        # status == "not_ready" → all NA (handled by the default row above).

        deployment_rows.append(row)
    return deployment_rows


def save_deployment_csv(rows: list, output_csv: Path):
    # Sort: ready tags first (most actionable), then add_only / remove_only, not_ready last; ties broken by sample size.
    status_order = {"ready": 0, "add_only": 1, "remove_only": 2, "not_ready": 3}
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "tag",
            "n_test_pos",
            "n_test_neg",
            "deployment_status",
            "add_threshold",
            "add_tp",
            "add_fp",
            "add_precision",
            "add_recall",
            "remove_threshold",
            "remove_tp",
            "remove_fp",
            "remove_precision",
            "remove_recall",
            "silent_n_pos",
            "silent_n_neg",
        ])
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda r: (status_order.get(r["deployment_status"], 99), -(r["n_test_pos"] + r["n_test_neg"])),
        ):
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Plot the 95% Wilson lower bound on precision vs threshold from evaluate.py stats JSON, "
                    "and select the best threshold per tag whose lower bound clears a target."
    )

    parser.add_argument(
        "--label-type", type=str, default="crosswalk",
        choices=["crosswalk", "curbramp", "surfaceproblem", "obstacle"],
        help="Label type; used to derive default --stats-file / --output-* paths",
    )

    parser.add_argument("--stats-file", type=Path, default=None,
        help="Path to positive-direction JSON output from notebooks/evaluate.py "
             "(default: results/<label-type>/validated-dino-inference-stats.json)")

    parser.add_argument("--target-precision", type=float, default=0.92,
                        help="Required observed precision at the selected threshold (default: 0.92)")

    parser.add_argument("--target-precision-lower-bound", type=float, default=0.90,
                        help="Required 95%% Wilson lower bound on precision at the selected threshold (default: 0.90)")

    parser.add_argument("--min-lift-over-prior", type=float, default=0.05,
                        help="Required absolute lift of the lower bound over the trivial-baseline precision "
                             "(base rate). Effective target = max(target_precision_lower_bound, "
                             "base_rate + min_lift_over_prior). Default: 0.05.")

    parser.add_argument("--output-plot", type=Path, default=None,
        help="Output image path for positive direction "
             "(default: results/<label-type>/precision-vs-threshold.svg)")

    parser.add_argument("--output-csv", type=Path, default=None,
        help="Output CSV path for positive direction "
             "(default: results/<label-type>/thresholds-at-target-precision.csv)")

    parser.add_argument("--skip-plot", action="store_true",
                        help="Skip graph generation and only write CSV summaries")

    parser.add_argument("--stats-file-negative", type=Path, default=None,
                        help="Path to negative-direction JSON (default: auto-derived from --stats-file)")

    parser.add_argument("--output-plot-negative", type=Path, default=None,
                        help="Output image path for negative direction (default: auto-derived from --output-plot)")

    parser.add_argument("--output-csv-negative", type=Path, default=None,
                        help="Output CSV path for negative direction (default: auto-derived from --output-csv)")

    parser.add_argument("--output-csv-deployment", type=Path, default=None,
                        help="Output CSV path for combined per-tag ADD/REMOVE/silent-zone deployment thresholds "
                             "(default: results/<label-type>/deployment-thresholds.csv)")

    args = parser.parse_args()

    # Resolve path defaults from --label-type when not provided explicitly.
    label_type = args.label_type
    if args.stats_file is None:
        args.stats_file = REPO_ROOT / f"results/{label_type}/validated-dino-inference-stats.json"
    if args.output_plot is None:
        args.output_plot = REPO_ROOT / f"results/{label_type}/precision-vs-threshold.svg"
    if args.output_csv is None:
        args.output_csv = REPO_ROOT / f"results/{label_type}/thresholds-at-target-precision.csv"
    if args.output_csv_deployment is None:
        args.output_csv_deployment = REPO_ROOT / f"results/{label_type}/deployment-thresholds.csv"

    if not (0.0 <= args.target_precision <= 1.0):
        raise ValueError("--target-precision must be between 0 and 1")
    if not (0.0 <= args.target_precision_lower_bound <= 1.0):
        raise ValueError("--target-precision-lower-bound must be between 0 and 1")
    if args.min_lift_over_prior < 0.0:
        raise ValueError("--min-lift-over-prior must be non-negative")

    # Load both directions' stats up-front so we can flag tags that are degenerate (no examples in one
    # of the two classes) consistently across positive and negative outputs.
    pos_stats_map = load_tag_stats(args.stats_file)
    neg_stats_file = args.stats_file_negative or _negative_path(args.stats_file)
    neg_stats_map = load_tag_stats(neg_stats_file) if neg_stats_file.exists() else None

    degenerate = find_degenerate_tags(pos_stats_map, neg_stats_map)
    skip_tags = {tag for tag, _, _ in degenerate}
    if degenerate:
        print(
            f"\nSkipped {len(degenerate)} tag(s) with no test-set examples in one of the two classes "
            f"(omitted from both CSVs and plots):"
        )
        for tag, n_pos, n_neg in degenerate:
            print(f"  - {tag} ({n_pos} pos / {n_neg} neg)")

    pos_rows = _run_direction(
        tag_stats_map=pos_stats_map,
        stats_file=args.stats_file,
        target_precision=args.target_precision,
        target_lower_bound=args.target_precision_lower_bound,
        min_lift_over_prior=args.min_lift_over_prior,
        output_plot=args.output_plot,
        output_csv=args.output_csv,
        skip_plot=args.skip_plot,
        direction="positive",
        skip_tags=skip_tags,
    ) or []

    if neg_stats_map is None:
        print(f"\nNegative stats file not found ({neg_stats_file}), skipping negative direction.")
        print("Re-run evaluate.py to generate it.")
        return

    neg_rows = _run_direction(
        tag_stats_map=neg_stats_map,
        stats_file=neg_stats_file,
        target_precision=args.target_precision,
        target_lower_bound=args.target_precision_lower_bound,
        min_lift_over_prior=args.min_lift_over_prior,
        output_plot=args.output_plot_negative or _negative_path(args.output_plot),
        output_csv=args.output_csv_negative or _negative_path(args.output_csv),
        skip_plot=args.skip_plot,
        direction="negative",
        skip_tags=skip_tags,
    ) or []

    deployment_rows = build_deployment_rows(pos_rows, neg_rows)
    if deployment_rows:
        save_deployment_csv(deployment_rows, args.output_csv_deployment)
        counts = {"ready": 0, "add_only": 0, "remove_only": 0, "not_ready": 0}
        for r in deployment_rows:
            counts[r["deployment_status"]] = counts.get(r["deployment_status"], 0) + 1
        print(f"\n=== deployment thresholds ===")
        print(f"Saved deployment CSV: {args.output_csv_deployment}")
        print(f"Tags ready (ADD + REMOVE): {counts['ready']}/{len(deployment_rows)}")
        print(f"Tags ready for ADD only: {counts['add_only']}/{len(deployment_rows)}")
        print(f"Tags ready for REMOVE only: {counts['remove_only']}/{len(deployment_rows)}")
        print(f"Tags not deployable: {counts['not_ready']}/{len(deployment_rows)}")


if __name__ == "__main__":
    main()
