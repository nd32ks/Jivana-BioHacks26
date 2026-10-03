"""Live-demo components for JIVANA's Streamlit app.

Two honestly-labelled live demos:

1. Live MegaDetector classification — runs MegaDetector v5 on sample
   images one at a time, drawing boxes as it goes, with running
   precision/recall against the dataset's own labels.
2. Live occupancy fitting — re-fits the single-season occupancy model and
   animates the maximum-likelihood optimizer converging. This is model
   FITTING, not neural-network training, and the UI says so.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

import config
import occupancy


@st.cache_resource(show_spinner="Loading MegaDetector v5 (one-time)...")
def get_detector():
    """Load MegaDetector once per app session (cached in Streamlit)."""
    from megadetector.detection.run_detector import load_detector
    return load_detector(config.MEGADETECTOR_MODEL, force_cpu=True)


def sample_images(n: int, seed: int = 7) -> list[Path]:
    """Pick n sample images, preferring a mix of empty and non-empty."""
    manifest_p = config.DATA_DIR / "sample_manifest.json"
    if not manifest_p.exists():
        return []
    manifest = json.loads(manifest_p.read_text())
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(manifest))[:n]
    paths = []
    for i in idx:
        p = config.SAMPLE_IMAGE_DIR / manifest[i]["file_name"].replace("/", "_")
        if p.exists():
            paths.append(p)
    return paths


def draw_boxes_pil(img: Image.Image, detections: list) -> Image.Image:
    """Return a copy of img with MegaDetector boxes drawn (for the live feed)."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for d in detections or []:
        x, y, w, h = d["bbox"]
        box = [x * img.width, y * img.height,
               (x + w) * img.width, (y + h) * img.height]
        draw.rectangle(box, outline="red", width=4)
        draw.text((box[0] + 2, max(box[1] - 14, 0)),
                  f"{d.get('category', '?')} {d['conf']:.2f}", fill="red")
    return img


def ground_truth() -> dict:
    p = config.DATA_DIR / "sample_manifest.json"
    if not p.exists():
        return {}
    return {m["file_name"].replace("/", "_"): m["label"]
            for m in json.loads(p.read_text())}


def run_live_classification(image_paths: list[Path], placeholder, stats_ph,
                            chart_ph, delay: float = 0.0):
    """Classify images one at a time, updating the given Streamlit
    placeholders. Returns (n_empty, precision, recall)."""
    detector = get_detector()
    gt = ground_truth()
    n_empty = tp = fp = fn = 0
    nll_x, prec_y, rec_y = [], [], []
    for i, p in enumerate(image_paths, 1):
        img = Image.open(p).convert("RGB")
        res = detector.generate_detections_one_image(
            img, image_id=str(p),
            detection_threshold=config.MEGADETECTOR_THRESHOLD)
        dets = [d for d in (res.get("detections") or [])
                if d["conf"] >= config.MEGADETECTOR_THRESHOLD]
        label = "empty" if not dets else "non-empty"
        n_empty += label == "empty"
        g = gt.get(p.name)
        if g is not None:
            true_empty = g == "empty"
            tp += (label == "empty" and true_empty)
            fp += (label == "empty" and not true_empty)
            fn += (label != "empty" and true_empty)
        shown = img.copy(); shown.thumbnail((700, 700))
        placeholder.image(shown if not dets else
                          draw_boxes_pil(shown, dets),
                          caption=f"{p.name} — {label} "
                                  f"({len(dets)} detection(s) >= "
                                  f"{config.MEGADETECTOR_THRESHOLD})")
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        stats_ph.write(f"**{i}/{len(image_paths)}** processed · "
                       f"**{n_empty}** classified empty "
                       f"({n_empty / i:.0%}) · empty-class precision "
                       f"**{prec:.2f}** · recall **{rec:.2f}**")
        nll_x.append(i)
        prec_y.append(prec); rec_y.append(rec)
        if i >= 3:
            chart_ph.plotly_chart(
                go.Figure([go.Scatter(x=nll_x, y=prec_y, name="precision"),
                           go.Scatter(x=nll_x, y=rec_y, name="recall")])
                .update_layout(xaxis_title="images processed",
                               yaxis_title="value (vs dataset labels)",
                               yaxis_range=[0, 1.05],
                               height=260, margin=dict(t=20, b=30)),
                use_container_width=True)
        if delay:
            import time; time.sleep(delay)
    return n_empty, (tp / (tp + fp) if tp + fp else None), \
        (tp / (tp + fn) if tp + fn else None)


def animate_occupancy_fit(df, species: str, group: str, chart_ph, delay=0.15):
    """Fit occupancy live and animate the optimizer's convergence.
    Returns the fit dict (same as fit_occupancy)."""
    g = df[df["disturbance"] == group]
    histories = occupancy.build_detection_history(g, species)
    trace = []
    fit = occupancy.fit_occupancy(histories, trace=trace)
    if fit is None or not trace:
        return fit
    nll = [t[0] for t in trace]
    psi = [t[1] for t in trace]
    p = [t[2] for t in trace]
    steps = list(range(1, len(trace) + 1))
    # subsample long traces so the animation stays snappy
    if len(trace) > 60:
        keep = np.linspace(0, len(trace) - 1, 60).astype(int)
        nll = [nll[j] for j in keep]; psi = [psi[j] for j in keep]
        p = [p[j] for j in keep]; steps = [steps[j] for j in keep]
    fig = go.Figure()
    fig.add_scatter(x=steps, y=nll, name="negative log-likelihood",
                    line=dict(color="#d29922"))
    fig.add_scatter(x=steps, y=psi, name="psi (occupancy)",
                    yaxis="y2", line=dict(color="#3fb950"))
    fig.add_scatter(x=steps, y=p, name="p (detection)", yaxis="y2",
                    line=dict(color="#58a6ff"))
    fig.update_layout(
        xaxis_title="optimizer step", height=340,
        yaxis=dict(title="NLL"), yaxis2=dict(title="probability",
                   overlaying="y", range=[0, 1]),
        margin=dict(t=20, b=30), legend=dict(orientation="h"))
    for upto in range(2, len(steps) + 1):
        chart_ph.plotly_chart(
            fig.update_traces(selector=dict(name="negative log-likelihood"),
                              x=steps[:upto], y=nll[:upto])
            .update_traces(selector=dict(name="psi (occupancy)"),
                           x=steps[:upto], y=psi[:upto])
            .update_traces(selector=dict(name="p (detection)"),
                           x=steps[:upto], y=p[:upto]),
            use_container_width=True)
        import time; time.sleep(delay)
    return fit
