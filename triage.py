"""Image triage with MegaDetector (CPU).

Runs MegaDetector v5 (via the official `megadetector` Python package) on the
~200 sample images in data/sample_images/, reports the share classified empty,
and — using the dataset's own labels as ground truth — the precision and
recall of "empty vs non-empty" filtering. Results are saved to
data/triage_results.json. A few images get bounding boxes drawn for the UI.

If the megadetector package or its model cannot load, we print a clear
message and write a results file with an "error" field instead of crashing
or fabricating numbers.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import config


def load_ground_truth() -> dict:
    """Ground truth for the sample images: flattened filename -> empty/animal/human.

    Prefers data/sample_manifest.json (written when the sample was
    downloaded); falls back to parsing the full COCO JSON and matching on
    the basename of file_name.
    """
    manifest = config.DATA_DIR / "sample_manifest.json"
    if manifest.exists():
        return {m["file_name"].replace("/", "_"): m["label"]
                for m in json.loads(manifest.read_text())}
    if not config.COCO_JSON_PATH.exists():
        return {}
    coco = json.loads(config.COCO_JSON_PATH.read_text(encoding="utf-8"))
    cat = {c["id"]: (c.get("common_name") or c.get("name") or "").lower()
           for c in coco.get("categories", [])}
    gt, id_to_file = {}, {}
    for img in coco.get("images", []):
        id_to_file[img["id"]] = Path(str(img.get("file_name", ""))).name
        gt[id_to_file[img["id"]]] = "empty"
    for a in coco.get("annotations", []):
        name = cat.get(a.get("category_id"), "")
        f = id_to_file.get(a["image_id"])
        if f is None:
            continue
        if name in config.HUMAN_CATEGORIES:
            gt[f] = "human"
        elif name not in config.EMPTY_CATEGORIES and gt[f] == "empty":
            gt[f] = "animal"
    return gt


def run_megadetector(image_paths):
    """Run MegaDetector v5 on CPU. Returns list of result dicts.
    Raises with a clear message if the package is unavailable."""
    try:
        from megadetector.detection.run_detector import load_detector
    except Exception as e:
        raise RuntimeError(
            "The 'megadetector' package could not be imported.\n"
            "Install it with:  pip install megadetector\n"
            "and ensure PyTorch CPU wheels are installed. "
            f"(Original error: {e})"
        ) from e
    try:
        detector = load_detector(config.MEGADETECTOR_MODEL, force_cpu=True)
    except Exception as e:
        raise RuntimeError(
            "MegaDetector model failed to load (download blocked or corrupt).\n"
            "Check your network and that MDv5 weights are available to the "
            f"megadetector package. (Original error: {e})"
        ) from e
    results = []
    for p in image_paths:
        # NB: megadetector 10.x preprocessing fails on raw path strings for
        # some JPEGs ("tuple index out of range"); passing a PIL image avoids it.
        img = Image.open(p).convert("RGB")
        res = detector.generate_detections_one_image(
            img, image_id=str(p),
            detection_threshold=config.MEGADETECTOR_THRESHOLD)
        res["image_path"] = str(p)
        results.append(res)
    return results


def classify(result) -> tuple:
    """(label, detections) from a MegaDetector result.
    label is 'empty' or 'non-empty' at the configured threshold."""
    dets = [d for d in (result.get("detections") or [])
            if d.get("conf", 0) >= config.MEGADETECTOR_THRESHOLD]
    return ("empty" if not dets else "non-empty"), dets


def precision_recall(pred_empty: dict) -> dict:
    """Precision/recall of the empty-vs-non-empty filter vs ground truth."""
    gt = load_ground_truth()
    tp = fp = fn = 0
    for fname, pred in pred_empty.items():
        g = gt.get(fname)
        if g is None:
            continue
        true_empty = (g == "empty")
        if pred == "empty" and true_empty:
            tp += 1
        elif pred == "empty" and not true_empty:
            fp += 1
        elif pred != "empty" and true_empty:
            fn += 1
    n = tp + fp + fn
    if n == 0:
        return {"n_evaluated": 0}
    return {
        "n_evaluated": n,
        "precision_empty": tp / (tp + fp) if tp + fp else None,
        "recall_empty": tp / (tp + fn) if tp + fn else None,
    }


def draw_boxes(result, out_path: Path):
    """Draw MegaDetector boxes on one image; save for the UI."""
    img = Image.open(result["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(img)
    for d in (result.get("detections") or []):
        if d.get("conf", 0) < config.MEGADETECTOR_THRESHOLD:
            continue
        x, y, w, h = d["bbox"]
        box = [x * img.width, y * img.height,
               (x + w) * img.width, (y + h) * img.height]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], max(box[1] - 12, 0)),
                  f"{d.get('category','?')} {d['conf']:.2f}", fill="red")
    img.thumbnail((800, 800))
    img.save(out_path)


def main():
    import pandas as pd
    out = {"dataset": config.DATASET_NAME,
           "threshold": config.MEGADETECTOR_THRESHOLD}
    image_paths = sorted(config.SAMPLE_IMAGE_DIR.glob("*.jpg")) + \
        sorted(config.SAMPLE_IMAGE_DIR.glob("*.jpeg")) + \
        sorted(config.SAMPLE_IMAGE_DIR.glob("*.png"))
    if not image_paths:
        out["error"] = (f"No sample images found in {config.SAMPLE_IMAGE_DIR}. "
                        "Place ~200 images there (mix of empty and non-empty).")
        config.TRIAGE_RESULTS_PATH.write_text(json.dumps(out, indent=2))
        print(out["error"])
        return
    print(f"Running MegaDetector v5 on {len(image_paths)} images (CPU)...")
    try:
        results = run_megadetector(image_paths)
    except RuntimeError as e:
        out["error"] = str(e)
        config.TRIAGE_RESULTS_PATH.write_text(json.dumps(out, indent=2))
        print(e)
        return

    pred_empty, boxes_dir = {}, config.DATA_DIR / "triage_boxes"
    boxes_dir.mkdir(exist_ok=True)
    for r in results:
        label, _ = classify(r)
        pred_empty[Path(r["image_path"]).name] = label
    n_empty = sum(1 for v in pred_empty.values() if v == "empty")
    for r in results[:6]:  # a few examples with boxes for the UI
        draw_boxes(r, boxes_dir / f"{Path(r['image_path']).stem}_boxed.jpg")

    pr = precision_recall(pred_empty)
    out.update({
        "n_images": len(image_paths),
        "n_predicted_empty": n_empty,
        "share_empty": n_empty / len(image_paths),
        **pr,
        "note": ("Ground truth = dataset's own labels. 'empty' precision/recall "
                 "measure how well MegaDetector filtering would remove empty "
                 "frames for human review."),
        "boxed_examples": sorted(str(p) for p in boxes_dir.glob("*_boxed.jpg")),
    })
    config.TRIAGE_RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"Share classified empty: {out['share_empty']:.1%}")
    if pr.get("n_evaluated"):
        print(f"Precision (empty): {pr['precision_empty']:.2f}  "
              f"Recall (empty): {pr['recall_empty']:.2f}")
    print(f"Results saved to {config.TRIAGE_RESULTS_PATH}")


if __name__ == "__main__":
    main()
