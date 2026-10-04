"""Wise design system for JIVANA — deep moss with lime voltage.

Injected into Streamlit as global CSS. Design tokens follow the Wise
style reference: Forest Ink surfaces, Lime Voltage accents, pill shapes,
Inter body type, Inter Black (900) for display headlines.
"""

import streamlit as st

# ---------------------------------------------------------------- tokens --
FOREST = "#163300"
LIME = "#9fe870"
SPRUCE = "#054d28"
LINEN = "#e2f6d5"
BLUE = "#0b4c72"
RED = "#cb272f"
CHARCOAL = "#454745"
OBSIDIAN = "#0e0f0c"
PEBBLE = "#868685"
SLATE = "#6a6c6a"
FOG = "#e8ebe6"
PAPER = "#ffffff"

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');

html, body, [class*="css"], .stApp, p, li, span, label, input, button, td, th {{
    font-family: 'Inter', ui-sans-serif, system-ui, sans-serif !important;
}}

.stApp {{
    background: {PAPER};
    color: {CHARCOAL};
}}

/* ---- display type: the brand's shout ----------------------------- */
h1 {{
    font-weight: 900 !important;
    font-size: 52px !important;
    letter-spacing: -1.6px !important;
    line-height: 0.95 !important;
    color: {OBSIDIAN} !important;
    text-transform: uppercase;
}}
h2, h3 {{
    font-weight: 900 !important;
    letter-spacing: -0.5px !important;
    color: {OBSIDIAN} !important;
}}

/* ---- sidebar: forest ink inverted surface ------------------------ */
section[data-testid="stSidebar"] {{
    background: {FOREST};
}}
section[data-testid="stSidebar"] * {{
    color: {LINEN} !important;
}}
section[data-testid="stSidebar"] .stRadio label {{
    padding: 8px 14px;
    border-radius: 9999px;
}}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {{
    background: {LIME};
}}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) * {{
    color: {FOREST} !important;
    font-weight: 700;
}}
section[data-testid="stSidebar"] hr {{
    border-color: {SPRUCE};
}}

/* ---- pills & buttons --------------------------------------------- */
.stButton > button, .stDownloadButton > button, button[kind] {{
    border-radius: 9999px !important;
    border: 1px solid {FOREST} !important;
    background: {PAPER} !important;
    color: {FOREST} !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
}}
.stButton > button[kind="primary"], button[kind="primaryFormSubmit"] {{
    background: {LIME} !important;
    border: none !important;
    color: {CHARCOAL} !important;
}}

/* ---- metric cards: fog surfaces ---------------------------------- */
[data-testid="stMetric"] {{
    background: {FOG};
    border-radius: 10px;
    padding: 16px 20px !important;
    box-shadow: rgba(14,15,12,0.12) 0px 0px 0px 1px;
}}
[data-testid="stMetricLabel"] * {{
    color: {SLATE} !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}}
[data-testid="stMetricValue"] {{
    color: {OBSIDIAN} !important;
    font-weight: 700 !important;
    font-size: 28px !important;
}}

/* ---- badges -------------------------------------------------------- */
.badge {{
    display: inline-block;
    border-radius: 9999px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.3px;
}}
.badge-lime {{ background: {LIME}; color: {FOREST}; }}
.badge-linen {{ background: {LINEN}; color: {FOREST}; }}
.badge-forest {{ background: {FOREST}; color: {LIME}; }}
.badge-red {{ background: {RED}; color: {PAPER}; }}
.badge-blue {{ background: {BLUE}; color: {PAPER}; }}

/* ---- dark section card --------------------------------------------- */
.dark-card {{
    background: {FOREST};
    border-radius: 28px;
    padding: 40px;
    color: {PAPER};
}}
.dark-card h1, .dark-card h2, .dark-card h3 {{
    color: {LIME} !important;
}}
.dark-card p, .dark-card li {{
    color: {PAPER} !important;
}}

/* ---- captions & helpers -------------------------------------------- */
.caption-quiet {{ color: {PEBBLE}; font-size: 14px; }}
.section-gap {{ margin-top: 40px; }}

/* ---- alerts: flat, on-brand ---------------------------------------- */
[data-testid="stNotification"], .stAlert {{
    border-radius: 10px !important;
}}
"""


def apply():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def badge(text: str, kind: str = "linen") -> str:
    return f"<span class='badge badge-{kind}'>{text}</span>"


def dark_card(title: str, body_md: str) -> None:
    import re
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", body_md)
    html = re.sub(r"^\- ", "· ", html, flags=re.M)
    html = html.replace("\n", "<br>")
    st.markdown(
        f"<div class='dark-card'><h2>{title}</h2><p>{html}</p></div>",
        unsafe_allow_html=True)
