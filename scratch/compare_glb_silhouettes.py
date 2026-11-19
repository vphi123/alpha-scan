"""Compare a Y-up, +Z-front Alpha Scan GLB against four fixed binary reference masks."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image


def compare_view(vertices, faces, reference_path, horizontal_axis, output_path):
    with Image.open(reference_path) as image:
        ref = np.asarray(image.convert("L"))
    if not np.isin(ref, [0, 1, 255]).all() or not ref.any():
        raise ValueError(f"Expected a nonempty binary foreground mask: {reference_path}")
    y, x = np.nonzero(ref)
    ref = ref[y.min():y.max() + 1, x.min():x.max() + 1]
    height = 550
    width = max(1, round(ref.shape[1] * height / ref.shape[0]))
    ref = cv2.resize((ref > 0).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

    xy = np.column_stack((vertices @ horizontal_axis, -vertices[:, 1]))
    xy -= xy.min(axis=0)
    xy *= (height - 1) / xy[:, 1].max()
    pixels = np.rint(xy).astype(np.int32)
    model = np.zeros((height, pixels[:, 0].max() + 1), dtype=np.uint8)
    for triangle in pixels[faces]:
        cv2.fillConvexPoly(model, triangle, 1)

    canvas_width = max(ref.shape[1], model.shape[1])
    masks = []
    for mask in [ref, model]:
        pad = canvas_width - mask.shape[1]
        masks.append(np.pad(mask.astype(bool), ((0, 0), (pad // 2, pad - pad // 2))))
    photo, mesh = masks
    overlap = photo & mesh
    if output_path:
        overlay = np.full((height, canvas_width, 3), 30, dtype=np.uint8)
        overlay[photo] = [245, 150, 30]
        overlay[mesh] = [40, 155, 245]
        overlay[overlap] = 240
        Image.fromarray(overlay).save(output_path)
    return float(overlap.sum() / (photo | mesh).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("glb", type=Path)
    # Side names follow the capture slots: in the right image the subject faces screen-right.
    axes = {"front": [1, 0, 0], "right": [0, 0, 1], "back": [-1, 0, 0], "left": [0, 0, -1]}
    for label in axes:
        parser.add_argument("--" + label, required=True, type=Path, help="Binary reference mask")
    parser.add_argument("--output-dir", type=Path, help="Save silhouette overlays and scores.json")
    args = parser.parse_args()
    mesh = trimesh.load(args.glb, force="mesh", process=False)
    if mesh.is_empty or np.ptp(mesh.vertices[:, 1]) <= 0:
        parser.error("Expected a body mesh with positive height")
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = {}
    for label, axis in axes.items():
        output = args.output_dir / f"{label}.png" if args.output_dir else None
        scores[label] = compare_view(mesh.vertices, mesh.faces, getattr(args, label), axis, output)
    report = {"orthographic_iou": scores, "mean": float(np.mean(list(scores.values()))),
              "minimum": min(scores.values()), "all_views_at_least_90pct": min(scores.values()) >= 0.9}
    result = json.dumps(report, indent=2)
    if args.output_dir:
        (args.output_dir / "scores.json").write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
