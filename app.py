"""JIVANA — Streamlit app. Run:  streamlit run app.py

Honesty commitments baked into this UI:
- Occupancy is "probability a site is used, accounting for imperfect
  detection" — never called abundance.
- Disturbance is a proxy: human detections per camera-day.
- No real-time claims: analyses update after each data collection (SD cards).
- GPS coordinates are withheld by the data provider; locations are shown as
  a ranked table, never a map with invented coordinates.
- Every model output carries a confidence interval.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import health_tags
import load_data
import occupancy
import priority
import theme
from activity import analyse_all_species, analyse_species
from disturbance import assign_groups, human_rate_per_location, split_locations

st.set_page_config(page_title="JIVANA", layout="wide")
theme.apply()

HONESTY = """
**What JIVANA claims**
- Species may shift activity toward night-time when human activity rises;
  we measure that shift from camera-trap timestamps.
- Disturbance is a **proxy**: independent human detections per camera-day.
- Occupancy is the probability a site is used, **accounting for imperfect
  detection** — it is **not** abundance.
- Camera-days are approximated from first to last image date per location.
- Nocturnality uses clock time (18:00–06:00), **not** sun time.
- Data arrive **after each data collection** (SD cards) — nothing here is
  real-time.
- GPS coordinates are **withheld by the data provider**; locations are shown
  as a ranked table, not a map.
- Health tagging is manual; JIVANA does **not** auto-diagnose health.
"""


@st.cache_data(show_spinner=False)
def get_data():
    df = load_data.load_coco()
    rates = human_rate_per_location(df)
    split = split_locations(rates)
    df = assign_groups(df, split)
    species = sorted(c for c in df["category"].unique()
                     if c not in ("HUMAN", "EMPTY"))
    return df, rates, split, species


@st.cache_data(show_spinner="Analysing activity shifts (bootstrap CIs)...")
def get_analysis():
    df, rates, split, species = get_data()
    table, insufficient = analyse_all_species(df, species)
    return df, rates, split, species, table, insufficient


def sidebar(df=None, species=None):
    with st.sidebar:
        st.header("JIVANA")
        st.caption(f"Dataset: {config.DATASET_NAME}")
        st.caption(f"Independence window: {config.INDEPENDENCE_WINDOW_MINUTES} min")
        st.caption(f"Disturbance split: top/bottom tercile of human rate")
        st.caption(f"Min sample: {config.MIN_DETECTIONS_PER_GROUP} events/group")
        st.caption(f"Bootstrap: {config.BOOTSTRAP_ITERATIONS} reps")
        with st.expander("About — please read"):
            st.markdown(HONESTY)


def load_triage_results():
    p = config.TRIAGE_RESULTS_PATH
    return json.loads(p.read_text()) if p.exists() else None


def try_get_data():
    try:
        return get_data()
    except load_data.DatasetMissingError as e:
        st.error(str(e))
        st.stop()


# ============================================================ pages =======
def page_overview():
    st.markdown("<h1>Night shift,<br>spotted early.</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:22px;color:#454745;max-width:900px'>JIVANA helps "
        "field biologists and park managers spot a <b>behavioural warning "
        "signal</b>: when human activity rises, wild species become more "
        "nocturnal — often <b>before</b> populations decline. Analyses update "
        "<b>after each data collection</b> (when SD cards come in).</p>",
        unsafe_allow_html=True)
    df, rates, split, species = try_get_data()
    s = load_data.dataset_summary(df)
    c = st.columns(6)
    c[0].metric("Independent events", f"{s['n_rows']:,}")
    c[1].metric("Locations", s["n_locations"])
    c[2].metric("Species", s["n_species"])
    c[3].metric("Date range", f"{s['date_min']:%Y-%m-%d} → {s['date_max']:%Y-%m-%d}")
    c[4].metric("Human events", s["n_human_events"])
    c[5].metric("High-disturbance sites", len(split["high"]))
    st.caption("An 'event' merges repeated detections of the same category at "
               f"the same location within {config.INDEPENDENCE_WINDOW_MINUTES} minutes.")
    st.divider()
    st.subheader("Image triage (MegaDetector v5, sample)")
    tri = load_triage_results()
    if tri is None:
        st.info("Run `python triage.py` first to generate the triage demo results.")
    elif "error" in tri:
        st.warning(tri["error"])
    else:
        cc = st.columns(3)
        cc[0].metric("Images scanned", tri["n_images"])
        cc[1].metric("Classified empty", f"{tri['share_empty']:.1%}")
        if tri.get("n_evaluated"):
            cc[2].metric("Empty precision / recall",
                         f"{tri['precision_empty']:.2f} / {tri['recall_empty']:.2f}")
        st.caption("Precision/recall use the dataset's own labels as ground truth.")
    st.divider()
    theme.dark_card("What JIVANA does — and does not — claim", HONESTY)


def page_triage():
    st.title("Triage — filtering empty frames")
    tri = load_triage_results()
    if tri is None:
        st.info("Run `python triage.py` first."); return
    if "error" in tri:
        st.error(tri["error"]); return
    st.write(f"MegaDetector v5 (threshold {tri['threshold']}) classified "
             f"**{tri['share_empty']:.1%}** of {tri['n_images']} sample images as empty.")
    if tri.get("n_evaluated"):
        st.write(f"Against the dataset's labels: **precision {tri['precision_empty']:.2f}**, "
                 f"**recall {tri['recall_empty']:.2f}** for the empty class.")
    st.caption("Triage only prioritises human review; it never deletes images "
               "and never changes the analysis data.")
    imgs = [Path(p) for p in tri.get("boxed_examples", [])]
    if imgs:
        st.subheader("Example detections")
        cols = st.columns(min(3, len(imgs)))
        for i, p in enumerate(imgs[: len(cols)]):
            if p.exists():
                cols[i].image(str(p), caption=p.name)


def page_activity():
    st.title("Activity shift — the core signal")
    st.caption("Clock time is used, not sun time. Disturbance groups are the "
               "top/bottom terciles of the human-detection-rate proxy.")
    df, rates, split, species, table, insufficient = get_analysis()
    if split.get("n_zero_rate"):
        st.info(f"Honest data note: {split['n_zero_rate']} of {split['n_locations']} "
                "locations had **zero** independent human detections, and the "
                "data provider removed the human images themselves for privacy "
                "(labels remain in the metadata). The high-disturbance group is "
                "therefore 'least-zero' rather than truly high — the split is a "
                "coarse proxy and small rate differences should not be "
                "over-interpreted.")

    col_sel, col_flag = st.columns([1, 2])
    ready = [s for s in table["species"]] if len(table) else []
    if not ready:
        st.warning("No species has enough data in both disturbance groups yet "
                   f"(min {config.MIN_DETECTIONS_PER_GROUP} events per group).")
        return
    sp = col_sel.selectbox("Species", ready)
    res = analyse_species(df, sp)

    fig = go.Figure()
    fig.add_scatter(x=res["grid_hours"], y=res["density_low"], name="Low disturbance")
    fig.add_scatter(x=res["grid_hours"], y=res["density_high"],
                    name="High disturbance", fill="tonexty", opacity=0.5)
    fig.update_layout(xaxis_title="Hour of day", yaxis_title="Activity density",
                      title=f"{sp}: 24-hour activity by disturbance group")
    st.plotly_chart(fig, use_container_width=True)

    dlo, dhi = res["delta_ci"]; nlo, nhi = res["noct_ci"]
    conf = res["confidence"]
    conf_color = theme.SPRUCE if conf >= 67 else "#b8860b" if conf >= 34 else theme.RED
    conf_label = "high" if conf >= 67 else "moderate" if conf >= 34 else "low"
    flag_kind = "lime" if res["flag"] == "shifting toward night" else "linen"
    c = st.columns(4)
    c[0].metric("Overlap Δ (high vs low)",
                f"{res['delta']:.2f}", f"95% CI {dlo:.2f}–{dhi:.2f}")
    c[1].metric("Nocturnality high / low",
                f"{res['noct_high']:.0%} / {res['noct_low']:.0%}",
                f"Δ {res['noct_diff']:+.1%} (95% CI {nlo:+.1%}–{nhi:+.1%})")
    c[2].markdown(f"**Flag**  \n{theme.badge(res['flag'].upper(), flag_kind)}",
                  unsafe_allow_html=True)
    c[3].markdown(f"**Confidence index**  \n"
                  f"<span style='color:{conf_color};font-size:1.8em;font-weight:900'>"
                  f"{conf}</span><span style='color:{conf_color}'>/100 ({conf_label})</span>",
                  unsafe_allow_html=True)
    st.caption("A species is flagged 'shifting toward night' only if the "
               "nocturnality difference is positive AND its 95% CI excludes zero. "
               "The confidence index (0–100) combines sample adequacy (smaller "
               "group's n vs the minimum) and estimate precision (bootstrap CI "
               "width) — it is confidence in the estimate, not a probability "
               "that the shift is real.")

    st.subheader("All species, sorted by shift size")
    show = table.copy()
    for col in ("noct_diff", "noct_ci_lo", "noct_ci_hi", "noct_high", "noct_low"):
        show[col] = show[col].map("{:.1%}".format)
    front = ["species", "n_high", "n_low", "flag", "confidence"]
    show = show[front + [c for c in show.columns if c not in front]]

    def _conf_style(v):
        color = theme.SPRUCE if v >= 67 else "#b8860b" if v >= 34 else theme.RED
        return f"color: {color}; font-weight: 700"

    show = show.style.map(_conf_style, subset=["confidence"])
    st.dataframe(show, use_container_width=True, hide_index=True)
    if insufficient:
        st.caption("Insufficient data: " + ", ".join(
            f"{r['species']} ({r['total_events']} events)" for r in insufficient))


def page_priority():
    st.title("Patrol priority & occupancy")
    df, rates, split, species, table, _ = get_analysis()
    with_data = list(table["species"]) if len(table) else []
    ranked = priority.patrol_priority(df, rates, with_data)
    st.subheader("Patrol-priority ranking")
    st.caption("Priority = standardised human detection rate + standardised "
               "local nocturnality shift. Both components shown for transparency.")
    disp = ranked.copy()
    disp["human_rate"] = disp["human_rate"].map("{:.2f}".format)
    disp["local_night_shift"] = disp["local_night_shift"].map("{:+.1%}".format)
    disp["priority"] = disp["priority"].map("{:.2f}".format)
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.download_button("Download ranking (CSV)", ranked.to_csv(index=False),
                       "patrol_priority.csv")

    st.subheader("Occupancy by disturbance group")
    st.caption("Occupancy probability (share of sites used, accounting for "
               "imperfect detection). NOT abundance. Locations where the "
               "species was never detected are kept in the fit — they are "
               "what makes occupancy identifiable.")
    sp = st.selectbox("Species", species, key="occ_sp")
    res = occupancy.occupancy_by_group(df, sp)
    rows = []
    for g in ("low", "high"):
        if res.get(g):
            r = res[g]
            rows.append({"group": g, "sites": r["n_sites"],
                         "psi": f"{r['psi']:.2f}",
                         "psi 95% CI": f"{r['psi_ci'][0]:.2f}–{r['psi_ci'][1]:.2f}",
                         "p": f"{r['p']:.2f}",
                         "p 95% CI": f"{r['p_ci'][0]:.2f}–{r['p_ci'][1]:.2f}"})
        else:
            rows.append({"group": g, "sites": 0, "psi": "n/a", "psi 95% CI": "n/a",
                         "p": "n/a", "p 95% CI": "n/a"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_health():
    st.title("Health review — manual tagging only")
    st.caption("JIVANA does NOT auto-diagnose health. Tag suspected issues to "
               "build a labelled dataset for a future model.")
    df, rates, split, species = try_get_data()
    tagged = set(health_tags.get_tags()["image_id"]) if len(health_tags.get_tags()) else set()
    pool = df[df["category"].isin(species)].drop_duplicates("image_id")
    pool = pool[~pool["image_id"].isin(tagged)]
    if pool.empty:
        st.info("Every candidate image has been reviewed (or none loaded).")
    else:
        row = pool.iloc[0]
        st.write(f"**Image {row['image_id']}** — {row['category']} at "
                 f"**{row['location']}**, {row['datetime']:%Y-%m-%d %H:%M}")
        notes = st.text_input("Notes (optional)")
        reviewer = st.text_input("Reviewer (optional)")
        cols = st.columns(len(health_tags.VALID_TAGS))
        for i, tag in enumerate(sorted(health_tags.VALID_TAGS)):
            if cols[i].button(tag):
                health_tags.add_tag(str(row["image_id"]), tag,
                                    row["location"], row["category"],
                                    notes, reviewer)
                st.rerun()
    tags = health_tags.get_tags()
    st.subheader(f"Tags so far ({len(tags)})")
    if len(tags):
        st.dataframe(tags.drop(columns=["id"]), use_container_width=True, hide_index=True)
        out = health_tags.export_csv()
        st.download_button("Export CSV (training data for a future health model)",
                           Path(out).read_bytes(), "health_tags.csv")
    else:
        st.caption("No tags yet.")


def page_live():
    import live_demo
    st.markdown("<h1>Live demo</h1>", unsafe_allow_html=True)
    st.caption("Two honest live demos: MegaDetector running real inference "
               "on sample images, and the occupancy model being fitted by "
               "maximum likelihood. Nothing here is pre-recorded; the "
               "occupancy panel is model FITTING, not neural-network training.")

    st.subheader("1 · Live MegaDetector classification")
    n_avail = len(live_demo.sample_images(10_000))
    st.caption(f"Confidence threshold {config.MEGADETECTOR_THRESHOLD}. "
               f"{n_avail} sample images on disk. Precision/recall use the "
               "dataset's own labels as ground truth.")
    n_img = st.slider("Images to classify live", 5, max(30, min(1000, n_avail)),
                      12)
    if st.button("Run live classification", type="primary"):
        paths = live_demo.sample_images(n_img)
        if not paths:
            st.warning("No sample images found — run triage.py / see README.")
        else:
            stats_ph = st.empty()
            gallery_ph = st.empty()
            chart_ph = st.empty()
            with st.spinner("MegaDetector v5 running on CPU..."):
                n_empty, prec, rec = live_demo.run_live_classification(
                    paths, stats_ph, chart_ph, gallery_ph)
            st.success(f"Done: {n_empty}/{len(paths)} classified empty "
                       f"({n_empty / len(paths):.0%}). Final empty-class "
                       f"precision {prec:.2f}, recall {rec:.2f}.")

    st.divider()
    st.subheader("2 · Live occupancy model fitting")
    st.caption("Watch the optimizer converge to the maximum-likelihood "
               "estimates of occupancy (psi) and detection probability (p). "
               "This is numerical model fitting — JIVANA does not train "
               "neural networks.")
    df, rates, split, species, table, _ = get_analysis()
    c1, c2, c3 = st.columns(3)
    sp = c1.selectbox("Species", species, key="live_occ_sp")
    grp = c2.selectbox("Disturbance group", ["low", "high"], key="live_grp")
    delay = c3.slider("Animation speed (s/frame)", 0.02, 0.5, 0.12)
    if st.button("Fit live", key="fit_live"):
        chart_ph = st.empty()
        result_ph = st.empty()
        fit = live_demo.animate_occupancy_fit(df, sp, grp, chart_ph,
                                              delay=delay)
        if fit is None:
            result_ph.warning("Not enough sites with data to fit this "
                              "species/group.")
        else:
            lo, hi = fit["psi_ci"]; plo, phi = fit["p_ci"]
            result_ph.success(
                f"Converged after fitting {fit['n_sites']} sites: "
                f"occupancy ψ = **{fit['psi']:.2f}** (95% CI {lo:.2f}–{hi:.2f}), "
                f"detection p = **{fit['p']:.2f}** (95% CI {plo:.2f}–{phi:.2f}). "
                "Occupancy = probability a site is used, accounting for "
                "imperfect detection — not abundance.")


PAGES = {"1. Overview": page_overview, "2. Triage": page_triage,
         "3. Activity shift": page_activity, "4. Patrol priority": page_priority,
         "5. Health review": page_health, "6. Live demo": page_live}

sidebar()
choice = st.sidebar.radio("Pages", list(PAGES))
PAGES[choice]()
