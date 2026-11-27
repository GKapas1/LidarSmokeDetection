"""Render before/after views of the full-recording surface-label pilot."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from smoke_labeler.object_surfaces import floor_coordinates as coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = {0: "#15917d", 1: "#e66a26", 255: "#b6bec7"}
    cases = [("tree_smoke1", 123, "Lower trunk", [-2.12, -1.63, .08, 1.05]),
             ("corridor_smoke1", 98, "Corridor drive 1", [-5.65, -.4, .05, 1.9]),
             ("corridor_smoke2", 57, "Corridor drive 2", [-5.65, -.4, .05, 1.9])]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for column, (drive, frame, title, bounds) in enumerate(cases):
        scene = drive.split("_")[0]
        folder = args.prepared / "diagnostics/other_scenes_v3" / scene
        floor = json.loads((folder / "reference_alignment_results.json").read_text())
        with np.load(folder / drive / "scans" / f"frame_{frame:06d}.npz") as file:
            ids, world = file["ids"], file["world"]
        xyz = coordinates(world, floor)
        point_labels = []
        for dataset in [args.base, args.dataset]:
            with np.load(dataset / "chunks" / (drive + ".npz")) as file:
                point_labels.append(file["label"][file["frame_ptr"][frame] + ids])
        show = ((xyz[:, 0] >= bounds[0]) & (xyz[:, 0] <= bounds[1])
                & (xyz[:, 2] >= bounds[2]) & (xyz[:, 2] <= bounds[3]))
        if scene == "tree":
            show &= np.abs(xyz[:, 1] + 1.4) < .25
        for row, labels in enumerate(point_labels):
            ax = axes[row, column]
            for label in [255, 0, 1]:
                keep = show & (labels == label)
                ax.scatter(xyz[keep, 0], xyz[keep, 2], s=7 if label == 255 else 10,
                           c=colors[label], linewidths=0, alpha=.7)
            ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel="Floor-aligned X (m)",
                   ylabel="Height above floor (m)")
            ax.set_title(f"{title} · scan {frame}" + (" · v4" if row == 0 else " · surface pilot"),
                         loc="left", fontsize=10)
    handles = [Line2D([], [], linestyle="", marker="o", color=colors[k], label=label)
               for k, label in [(0, "Background"), (1, "Smoke"), (255, "Ignored")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Full-recording surface refinement", fontsize=19, fontweight="bold", x=.07, ha="left")
    fig.subplots_adjust(left=.07, right=.98, top=.89, bottom=.12, wspace=.28, hspace=.35)
    fig.savefig(args.output / "before-after.png", dpi=160)
    plt.close(fig)

    with (args.dataset / "audit/summary.json").open() as file:
        summary = json.load(file)
    records = [summary["records"][name] for name in ["corridor_smoke1", "corridor_smoke2", "tree_smoke1"]]
    labels = ["Corridor 1", "Corridor 2", "Tree"]
    transitions = [("255->0", "Added background", "#15917d"),
                   ("1->0", "Smoke corrected to background", "#297ba4"),
                   ("1->255", "Smoke set aside as ambiguous", "#909ca8"),
                   ("255->1", "Added smoke", "#e66a26")]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    left = np.zeros(3)
    for key, title, color in transitions:
        counts = np.array([row["transitions"][key] for row in records])
        if not counts.any():
            continue
        ax.barh(labels, counts, left=left, color=color, label=title)
        left += counts
    for index, count in enumerate(left):
        ax.annotate(f"{int(count):,}", (count, index), xytext=(5, 0), textcoords="offset points", va="center")
    ax.set(xlabel="Changed point labels across each full recording", xlim=(0, max(left) * 1.18))
    ax.invert_yaxis()
    ax.set_title("What changed in the pilot labels", loc="left", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.savefig(args.output / "label-changes.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
