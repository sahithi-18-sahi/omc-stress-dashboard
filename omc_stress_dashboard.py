"""
OMC Transaction Banking Stress Model
 India Oil Marketing Companies · FY25–26
Updated with simplified OFAC formula, detailed plain-English explanations
Run: streamlit run omc_stress_dashboard.py
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="OMC Transaction Banking Stress Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# EMBEDDED RESEARCH DATA (FY25 + FY26, 24 months)
# ─────────────────────────────────────────────
MONTHS = [
    "Apr-24","May-24","Jun-24","Jul-24","Aug-24","Sep-24",
    "Oct-24","Nov-24","Dec-24","Jan-25","Feb-25","Mar-25",
    "Apr-25","May-25","Jun-25","Jul-25","Aug-25","Sep-25",
    "Oct-25","Nov-25","Dec-25","Jan-26","Feb-26","Mar-26",
]
WTI_P    = [85.35,80.02,79.77,81.80,76.68,70.24,71.99,69.95,70.12,75.74,71.53,68.24,
            63.54,62.17,68.17,68.39,64.86,63.96,60.89,60.06,57.97,60.04,64.51,91.38]
IB_P     = [89.44,83.62,82.55,84.15,78.27,73.69,75.12,73.02,73.34,80.20,77.52,71.11,
            67.72,64.04,69.77,70.95,69.11,69.61,65.08,64.31,62.22,63.08,69.01,85.40]
BRENT_P  = [89.94,81.75,82.25,85.15,80.36,74.02,75.63,74.35,73.86,79.27,75.44,72.73,
            68.13,64.45,71.44,71.04,67.87,67.99,64.54,63.80,62.54,66.60,70.89,103.13]
DUBAI_P  = [89.39,83.53,82.17,83.94,77.95,73.43,74.65,72.79,73.31,80.14,74.97,71.71,
            66.89,63.01,68.50,69.23,67.87,67.75,64.30,63.83,61.98,63.91,68.36,91.88]
FX_P     = [83.406,83.393,83.471,83.595,83.896,83.880,84.072,84.342,84.773,85.696,86.736,86.742,
            85.574,85.172,85.824,86.299,86.748,87.072,88.294,88.826,90.091,90.799,90.731,92.761]
URALS_P  = [76,83,74,71,82,76,69,67,68,67,76,68,
            66,62,57,60,69,64,63,58,54,51,58,108]

def pct_ret(arr):
    a = np.array(arr, dtype=float)
    return list((a[1:] - a[:-1]) / a[:-1])

WTI_R   = pct_ret(WTI_P)
IB_R    = pct_ret(IB_P)
BRENT_R = pct_ret(BRENT_P)
DUBAI_R = pct_ret(DUBAI_P)
FX_R    = pct_ret(FX_P)
URALS_R = pct_ret(URALS_P)
RET_MONTHS = MONTHS[1:]

DISC = [b - u for b, u in zip(BRENT_P, URALS_P)]

_price_df = pd.DataFrame({
    "WTI": WTI_P, "Indian Basket": IB_P, "Brent": BRENT_P,
    "Dubai": DUBAI_P, "USD/INR": FX_P, "Urals": URALS_P
})
PRICE_CORR = _price_df.corr().round(4)

_ret_df = pd.DataFrame({
    "WTI": WTI_R, "Indian Basket": IB_R, "Brent": BRENT_R,
    "Dubai": DUBAI_R, "USD/INR": FX_R, "Urals": URALS_R
})
RET_CORR = _ret_df.corr().round(4)

_X = np.column_stack([np.ones(23), BRENT_R, FX_R, URALS_R])
_y = np.array(IB_R)
OLS_BETA, _, _, _ = np.linalg.lstsq(_X, _y, rcond=None)
_pred = _X @ OLS_BETA
OLS_R2 = 1 - np.sum((_y - _pred)**2) / np.sum((_y - _y.mean())**2)

_sb = abs(OLS_BETA[1]) * np.std(BRENT_R)
_sf = abs(OLS_BETA[2]) * np.std(FX_R)
_su = abs(OLS_BETA[3]) * np.std(URALS_R)
_tot = _sb + _sf + _su
VAR_CONTRIB = {"Brent": _sb/_tot, "FX": _sf/_tot, "Urals": _su/_tot}

_pos_disc = [d for d in DISC if d > 0]
DISC_STATS = {
    "mean": np.mean(DISC), "max": max(DISC), "min": min(DISC),
    "mean_pos": np.mean(_pos_disc), "max_pos": max(_pos_disc),
    "p90_pos": np.percentile(_pos_disc, 90), "p95_pos": np.percentile(_pos_disc, 95),
    "n_pos": len(_pos_disc), "n_neg": len([d for d in DISC if d < 0]),
}

# ─────────────────────────────────────────────
# STRESS MODEL CONSTANTS
# ─────────────────────────────────────────────
SCENARIOS = {
    "Base":     {"brent": 75,  "fx": 85.0, "urals": 5,  "ofac": 0},
    "Moderate": {"brent": 90,  "fx": 88.0, "urals": 10, "ofac": 1},
    "Severe":   {"brent": 110, "fx": 92.0, "urals": 20, "ofac": 2},
}
BASE = {"brent": 75, "fx": 85.0, "urals": 5}
OFAC_LABELS = ["Low", "Medium", "High"]
PALETTE = ["#378add", "#1d9e75", "#ef9f27", "#534ab7", "#d85a30"]

# ─── Risk weights (Overall Risk Score) ───────
W_OIL  = 0.25
W_FX   = 0.20
W_RU   = 0.20
W_OFAC = 0.35

# ─── OFAC score constants ─────────────────────
# Sanction-environment multiplier: how much the regulatory climate
# amplifies exposure on top of baseline structural exposure.
#   Low    = 1.0  → no current enforcement escalation
#   Medium = 1.5  → policy shift / advisory warnings (not enforcement)
#   High   = 3.0  → active enforcement; correspondent banks pulling back
OFAC_ENV = [1.0, 1.5, 3.0]

# Urals discount ceiling: $15 is the 90th-percentile of observed
# FY25–26 discounts. Beyond this level the discount signals acute
# sanctions pressure (i.e. Russia can only sell at deeply punitive
# prices because compliant counterparties are avoiding it).
URALS_CEIL = 15.0

# Base institutional exposure weight per OMC —
# reflects ownership structure and % of Russian crude routed
# through sanctioned/SDN-adjacent entities.
#   IOCL  = 1  (no SDN-adjacent owner; PSU but diversified supply)
#   BPCL  = 2  (PSU; ~36% Russian crude; government policy buffer)
#   HPCL  = 2  (similar to BPCL)
#   Nayara= 4  (Rosneft ~49% owner; Rosneft is on OFAC SDN list;
#               every Nayara transaction is OFAC-proximate)
BASE_WEIGHT = {"iocl": 1, "bpcl": 2, "hpcl": 2, "ril": 0, "nayara": 4}

OMC = [
    {"id":"iocl",   "name":"IOCL",     "throughput":71.56, "russianShare":0.385, "oilR":"M","fxR":"H","ofacW":1},
    {"id":"bpcl",   "name":"BPCL",     "throughput":43.50, "russianShare":0.365, "oilR":"H","fxR":"H","ofacW":2},
    {"id":"hpcl",   "name":"HPCL",     "throughput":26.04, "russianShare":0.350, "oilR":"H","fxR":"H","ofacW":2},
    {"id":"ril",    "name":"Reliance", "throughput":80.00, "russianShare":0.050, "oilR":"M","fxR":"M","ofacW":0},
    {"id":"nayara", "name":"Nayara",   "throughput":20.00, "russianShare":0.825, "oilR":"M","fxR":"M","ofacW":4},
]
PSU      = OMC[:3]
OFAC_SET = [OMC[0], OMC[1], OMC[2], OMC[4]]

# ─────────────────────────────────────────────
# CORE FORMULAE
# ─────────────────────────────────────────────
def import_bill(o, brent, fx, urals):
    mmt  = o["throughput"] / 12
    barr = mmt * 7.33e6
    r_b  = barr * o["russianShare"]
    n_b  = barr * (1 - o["russianShare"])
    r_p  = max(brent - urals, 20)
    return round((r_b * r_p + n_b * brent) * fx / 1e7)

def fx_income(o, brent, fx, urals):
    mmt  = o["throughput"] / 12
    barr = mmt * 7.33e6
    r_b  = barr * o["russianShare"]
    n_b  = barr * (1 - o["russianShare"])
    r_p  = max(brent - urals, 20)
    return round((r_b * r_p + n_b * brent) * 0.0005 * fx / 1e7)

def fee_income(bill):
    return round(bill * 0.0015)

# ─────────────────────────────────────────────
# SIMPLIFIED OFAC SCORE
# ─────────────────────────────────────────────
# Three components, multiplied together, capped at 100:
#
#   A = Base Institutional Exposure  (0–4 scale, fixed per OMC)
#       Reflects ownership and structural SDN proximity.
#
#   B = Russian Share Factor  (0–1 scale = the % itself)
#       The larger the share of Russian crude, the larger the
#       surface area of transactions that could breach sanctions.
#
#   C = Sanction Environment Multiplier  (1.0 / 1.5 / 3.0)
#       How aggressively OFAC is currently enforcing.
#       Low=1.0 (normal), Medium=1.5 (policy shift), High=3.0 (active enforcement)
#
#   D = Urals Pressure Amplifier  (1.0 → 1.5 as discount rises)
#       A large Urals discount (Russia selling cheap) signals that
#       compliant buyers are avoiding Russian crude. Those still
#       buying face higher scrutiny. Capped at the observed $15 ceiling.
#       D = 1 + (min(Urals, $15) / $15) × 0.50
#       At Urals=$0  → D = 1.00  (no extra pressure)
#       At Urals=$15 → D = 1.50  (50% higher scrutiny pressure)
#
#   Raw score = A × B × C × D × 25   (×25 maps to 0–100 scale)
#   Final      = min(Raw, 100)
#
# Scale calibration (×25):
#   Nayara worst case: A=4, B=0.825, C=3.0, D=1.5 → 4×0.825×3.0×1.5×25 = 371.25 → capped 100
#   IOCL base case:    A=1, B=0.385, C=1.0, D=1.0 → 1×0.385×1.0×1.0×25 = 9.6  (low, realistic)
#   BPCL moderate:     A=2, B=0.365, C=1.5, D=1.33→ 2×0.365×1.5×1.33×25 = 36.4 (medium)

def ofac_score(o, ofac_v, urals):
    A = o["ofacW"]                                        # Base institutional weight
    B = o["russianShare"]                                 # Russian share (0–1)
    C = OFAC_ENV[ofac_v]                                  # Sanction environment multiplier
    D = 1 + (min(urals, URALS_CEIL) / URALS_CEIL) * 0.50 # Urals pressure amplifier
    raw = A * B * C * D * 25
    return min(100, round(raw))

def overall_risk(o, ofac_v, urals):
    oil_s = {"M": 2, "H": 3, "L": 1}.get(o["oilR"], 2)
    fx_s  = {"M": 2, "H": 3, "L": 1}.get(o["fxR"], 2)
    ofc   = ofac_score(o, ofac_v, urals)
    r_s   = 4 if o["russianShare"] > 0.5 else (3 if o["russianShare"] > 0.3 else 2)
    t = oil_s * W_OIL + fx_s * W_FX + (ofc / 25) * W_OFAC + r_s * W_RU
    return "VH" if t >= 3.2 else ("H" if t >= 2.5 else ("M" if t >= 1.8 else "L"))

def risk_label(r):
    return {"VH":"Very High","H":"High","M":"Medium","L":"Low"}.get(r, r)

def risk_color(r):
    return {"VH":"#e24b4a","H":"#ef9f27","M":"#378add","L":"#639922"}.get(r, "#888")

def signal_color(level):
    return {"green":"#639922","amber":"#ef9f27","red":"#e24b4a"}.get(level, "#888")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### OMC Stress Model")
    st.caption("Indian OMCs Stress Test model · FY25–26")
    st.markdown("---")
    st.markdown("**Preset Scenarios**")
    scenario = st.radio("", list(SCENARIOS.keys()), horizontal=True, label_visibility="collapsed")
    sc = SCENARIOS[scenario]
    st.markdown("---")
    st.markdown("**Scenario Inputs**")
    brent  = st.slider("Brent crude ($/bbl)", 55, 130, sc["brent"], 1)
    fx     = st.slider("USD / INR", 78.0, 97.0, float(sc["fx"]), 0.5)
    urals  = st.slider("Urals discount ($/bbl)", 0.0, 20.0, float(sc["urals"]), 0.5)
    ofac_v = st.select_slider("OFAC escalation", options=[0,1,2],
                               value=sc["ofac"], format_func=lambda x: OFAC_LABELS[x])
    st.markdown("---")
    st.markdown("**OMC Reference**")
    st.dataframe(pd.DataFrame([
        {"Company": o["name"], "Ru%": f'{o["russianShare"]*100:.0f}%', "MMT": o["throughput"]}
        for o in OMC
    ]), hide_index=True, use_container_width=True)
    st.markdown("---")
    st.markdown("**Signal Thresholds**")
    st.markdown("🟢 Brent<$80 · FX<84\n\n🟡 Brent $80–100 · FX 84–90\n\n🔴 Brent>$100 · FX>90")

# ─────────────────────────────────────────────
# SIGNAL STRIP
# ─────────────────────────────────────────────
b_lv = "green" if brent < 80 else ("amber" if brent < 100 else "red")
f_lv = "green" if fx    < 84 else ("amber" if fx    < 90  else "red")
u_lv = "green" if urals < 8  else ("amber" if urals < 15  else "red")
o_lv = "green" if ofac_v==0  else ("amber" if ofac_v==1   else "red")

st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0;
            border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;margin-bottom:16px">
  <div style="padding:10px 14px;border-right:1px solid #e0e0e0">
    <div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.05em">Brent Crude</div>
    <div style="font-size:18px;font-weight:600;margin-top:2px">
      <span style="display:inline-block;width:9px;height:9px;border-radius:50%;
                   background:{signal_color(b_lv)};margin-right:5px;vertical-align:middle"></span>
      ${brent}/bbl</div>
  </div>
  <div style="padding:10px 14px;border-right:1px solid #e0e0e0">
    <div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.05em">USD / INR</div>
    <div style="font-size:18px;font-weight:600;margin-top:2px">
      <span style="display:inline-block;width:9px;height:9px;border-radius:50%;
                   background:{signal_color(f_lv)};margin-right:5px;vertical-align:middle"></span>
      {fx:.1f}</div>
  </div>
  <div style="padding:10px 14px;border-right:1px solid #e0e0e0">
    <div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.05em">Urals Discount</div>
    <div style="font-size:18px;font-weight:600;margin-top:2px">
      <span style="display:inline-block;width:9px;height:9px;border-radius:50%;
                   background:{signal_color(u_lv)};margin-right:5px;vertical-align:middle"></span>
      ${urals:.1f}/bbl</div>
  </div>
  <div style="padding:10px 14px">
    <div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.05em">OFAC Level</div>
    <div style="font-size:18px;font-weight:600;margin-top:2px">
      <span style="display:inline-block;width:9px;height:9px;border-radius:50%;
                   background:{signal_color(o_lv)};margin-right:5px;vertical-align:middle"></span>
      {OFAC_LABELS[ofac_v]}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_metrics, tab_charts, tab_corr, tab_formulas, tab_risk = st.tabs([
    "📊 Output Metrics", "📈 Charts",
    "🔗 Correlation Analysis",
    "🔢 Formulas", "⚠️ Risk Ranking"
])

# ── COMPUTE ──────────────────────────────────
bills   = {o["id"]: import_bill(o, brent, fx, urals)                              for o in OMC}
bills0  = {o["id"]: import_bill(o, BASE["brent"], BASE["fx"], BASE["urals"])       for o in OMC}
fx_inc  = {o["id"]: fx_income(o, brent, fx, urals)                                for o in OMC}
fx_inc0 = {o["id"]: fx_income(o, BASE["brent"], BASE["fx"], BASE["urals"])         for o in OMC}
fees    = {o["id"]: fee_income(bills[o["id"]])                                     for o in OMC}
fees0   = {o["id"]: fee_income(bills0[o["id"]])                                    for o in OMC}
ofacs   = {o["id"]: ofac_score(o, ofac_v, urals)                                  for o in OFAC_SET}
ofacs0  = {o["id"]: ofac_score(o, 0, BASE["urals"])                                for o in OFAC_SET}

# ─────────────────────────────────────────────
# TAB 1 · OUTPUT METRICS
# ─────────────────────────────────────────────
with tab_metrics:
    st.markdown("##### Import Bill — monthly financing need (₹ Cr)")
    cols = st.columns(3)
    for i, o in enumerate(PSU):
        d = bills[o["id"]] - bills0[o["id"]]
        cols[i].metric(o["name"], f'₹{bills[o["id"]]:,} Cr',
                       delta=f'{d:+,} Cr vs base', delta_color="inverse")

    st.markdown("##### Working Capital Stress vs Base (₹ Cr/month)")
    cols = st.columns(3)
    for i, o in enumerate(PSU):
        wc = bills[o["id"]] - bills0[o["id"]]
        cols[i].metric(o["name"], f'{wc:+,} Cr', delta_color="off")

    st.divider()

    st.markdown("##### FX Income — bank FX settlement revenue (₹ Cr/month)")
    cols = st.columns(3)
    for i, o in enumerate(PSU):
        d = fx_inc[o["id"]] - fx_inc0[o["id"]]
        cols[i].metric(o["name"], f'₹{fx_inc[o["id"]]:,} Cr', delta=f'{d:+,} Cr vs base')

    st.markdown("##### Fee Income — LC + trade finance fees (₹ Cr/month)")
    cols = st.columns(3)
    for i, o in enumerate(PSU):
        d = fees[o["id"]] - fees0[o["id"]]
        cols[i].metric(o["name"], f'₹{fees[o["id"]]:,} Cr', delta=f'{d:+,} Cr vs base')

    st.divider()

    st.markdown("##### OFAC Exposure Score (0–100)")
    cols = st.columns(4)
    for i, o in enumerate(OFAC_SET):
        d = ofacs[o["id"]] - ofacs0[o["id"]]
        cols[i].metric(o["name"], f'{ofacs[o["id"]]}/100',
                       delta=f'{d:+} vs base', delta_color="inverse")

# ─────────────────────────────────────────────
# TAB 2 · CHARTS
# ─────────────────────────────────────────────
with tab_charts:
    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure(go.Bar(x=[o["name"] for o in OMC],
            y=[bills[o["id"]] for o in OMC], marker_color=PALETTE,
            text=[f'₹{bills[o["id"]]:,}' for o in OMC], textposition="outside"))
        fig.update_layout(title="Import Bill (₹ Cr/month)", height=280,
                          showlegend=False, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure(go.Bar(x=[o["name"] for o in PSU],
            y=[fees[o["id"]] for o in PSU], marker_color=PALETTE[:3],
            text=[f'₹{fees[o["id"]]}' for o in PSU], textposition="outside"))
        fig.update_layout(title="Fee Income (₹ Cr/month)", height=280,
                          showlegend=False, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure(go.Bar(x=[o["name"] for o in OFAC_SET],
            y=[ofacs[o["id"]] for o in OFAC_SET],
            marker_color=[PALETTE[0],PALETTE[1],PALETTE[2],PALETTE[4]],
            text=[str(ofacs[o["id"]]) for o in OFAC_SET], textposition="outside"))
        fig.update_layout(title="OFAC Exposure Score (0–100)", height=280,
                          showlegend=False, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure(go.Bar(x=[o["name"] for o in PSU],
            y=[fx_inc[o["id"]] for o in PSU], marker_color=PALETTE[:3],
            text=[f'₹{fx_inc[o["id"]]}' for o in PSU], textposition="outside"))
        fig.update_layout(title="FX Income (₹ Cr/month)", height=280,
                          showlegend=False, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

        wc_vals = [bills[o["id"]] - bills0[o["id"]] for o in PSU]
        fig = go.Figure(go.Bar(x=[o["name"] for o in PSU], y=wc_vals,
            marker_color=["#e24b4a" if v>=0 else "#639922" for v in wc_vals],
            text=[f'{v:+,}' for v in wc_vals], textposition="outside"))
        fig.update_layout(title="WC Stress vs Base (₹ Cr)", height=280,
                          showlegend=False, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Scenario Comparison — Import Bill: Base vs Moderate vs Severe")
    sc4 = [OMC[0], OMC[1], OMC[2], OMC[4]]
    fig = go.Figure()
    for label, s, color in [
        ("Base ($75/₹85)", SCENARIOS["Base"], "#b5d4f4"),
        ("Moderate ($90/₹88)", SCENARIOS["Moderate"], "#fac775"),
        ("Severe ($110/₹92)", SCENARIOS["Severe"], "#f09595"),
    ]:
        fig.add_trace(go.Bar(name=label, x=[o["name"] for o in sc4],
            y=[import_bill(o, s["brent"], s["fx"], s["urals"]) for o in sc4],
            marker_color=color))
    fig.update_layout(barmode="group", height=320,
                      legend=dict(orientation="h", y=-0.2),
                      margin=dict(t=10,b=10,l=10,r=10), yaxis_title="₹ Cr")
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# TAB 3 · CORRELATION ANALYSIS
# ─────────────────────────────────────────────
with tab_corr:
    st.markdown("#### Correlation Analysis — FY25 + FY26 (24 months)")
    st.caption("Source: PPAC, EIA, RBI, Trading Economics · Compiled in IMPORTS.xlsx workbook")

    sub1, sub2 = st.tabs(["Price-Level Correlation", "Returns Correlation"])

    with sub1:
        st.markdown("##### Price-Level Correlation Matrix")
        st.info("⚠️ Price-level correlations are inflated by shared long-term trends. Useful for co-movement direction but can be misleading — use returns correlation for quantitative risk analysis.", icon="📌")

        labs = list(PRICE_CORR.columns)
        z    = PRICE_CORR.values.tolist()
        text = [[f"{v:.3f}" for v in row] for row in z]
        fig = go.Figure(go.Heatmap(z=z, x=labs, y=labs, text=text, texttemplate="%{text}",
            colorscale="RdYlGn", zmin=-1, zmax=1, colorbar=dict(title="r", thickness=12)))
        fig.update_layout(height=420, margin=dict(t=20,b=20,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
**Key findings:**
- **Brent ↔ WTI: r ≈ 0.98** — Near-perfect synchronisation
- **Brent ↔ Indian Basket: r ≈ 0.92** — India's benchmark tracks Brent closely
- **Brent ↔ Urals: r ≈ 0.86** — Russian crude moves with global benchmarks despite discount
""")
        with col_b:
            st.markdown("""
- **Brent ↔ USD/INR: r ≈ –0.14** — Negative at price level (spurious trend divergence)
- **Interpretation:** Price-level correlations carry spurious trend components
- Use monthly % returns for reliable risk weighting
""")

    with sub2:
        st.markdown("##### Returns Correlation Matrix (monthly % changes, 24 months of prices → 23 monthly returns)")
        st.success("✅ Returns-based correlations remove trend effects — this is the basis for risk weight derivation.", icon="📊")

        labs_r = list(RET_CORR.columns)
        zr     = RET_CORR.values.tolist()
        textr  = [[f"{v:.3f}" for v in row] for row in zr]
        fig = go.Figure(go.Heatmap(z=zr, x=labs_r, y=labs_r, text=textr, texttemplate="%{text}",
            colorscale="RdYlGn", zmin=-1, zmax=1, colorbar=dict(title="r", thickness=12)))
        fig.update_layout(height=420, margin=dict(t=20,b=20,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
**Key findings:**
- **Brent ↔ Indian Basket: r = {RET_CORR.loc['Brent','Indian Basket']:.3f}** → R² = {RET_CORR.loc['Brent','Indian Basket']**2*100:.1f}% — Brent virtually IS the Indian import benchmark
- **WTI ↔ Brent: r = {RET_CORR.loc['WTI','Brent']:.3f}** — Near-identical month-to-month
- **Dubai ↔ Brent: r = {RET_CORR.loc['Dubai','Brent']:.3f}** — Gulf benchmarks co-integrated
""")
        with col_b:
            st.markdown(f"""
- **USD/INR ↔ Brent: r = {RET_CORR.loc['USD/INR','Brent']:.3f}** → Oil shocks transmit to INR
- **Urals ↔ Indian Basket: r = {RET_CORR.loc['Urals','Indian Basket']:.3f}** → R² = {RET_CORR.loc['Urals','Indian Basket']**2*100:.1f}%
- **USD/INR ↔ Indian Basket: r = {RET_CORR.loc['USD/INR','Indian Basket']:.3f}** → Rupee weakens when oil spikes
""")

        st.markdown("---")
        st.markdown("##### OLS Regression: Indian Basket Returns ~ Brent + FX + Urals")
        st.code(f"""Indian Basket return = {OLS_BETA[0]:.4f}
  + {OLS_BETA[1]:.4f} × Brent_return
  + {OLS_BETA[2]:.4f} × FX_return
  + {OLS_BETA[3]:.4f} × Urals_return

R² = {OLS_R2:.4f}  ({OLS_R2*100:.1f}% of IB return variation explained, 24 months prices → 23 returns)""", language=None)

        st.markdown(f"""
| Variable | β | Variance contribution |
|---|---|---|
| Brent returns | **{OLS_BETA[1]:.4f}** | **{VAR_CONTRIB['Brent']*100:.1f}%** |
| FX (USD/INR) | **{OLS_BETA[2]:.4f}** | **{VAR_CONTRIB['FX']*100:.1f}%** |
| Urals returns | **{OLS_BETA[3]:.4f}** | **{VAR_CONTRIB['Urals']*100:.1f}%** |
""")

        st.markdown("---")
        st.markdown("##### How β and Variance Contribution are computed — step by step")

        with st.expander("Step 1 — What is β (beta)?", expanded=True):
            st.markdown(f"""
β is the OLS coefficient — it answers: *"If this variable rises by 1%, how much does the Indian Basket return change?"*

| Variable | β value | Plain English |
|---|---|---|
| β₀ (intercept) | **{OLS_BETA[0]:.4f}** | Baseline drift each month (negligible) |
| β₁ Brent | **{OLS_BETA[1]:+.4f}** | Brent rises 1% → Indian Basket rises **{OLS_BETA[1]*100:.2f}%** |
| β₂ FX (USD/INR) | **{OLS_BETA[2]:+.4f}** | Rupee depreciates 1% → Indian Basket rises **{abs(OLS_BETA[2])*100:.2f}%** |
| β₃ Urals | **{OLS_BETA[3]:+.4f}** | Urals rises 1% → Indian Basket falls **{abs(OLS_BETA[3])*100:.2f}%** (Urals rising = smaller discount = costlier Russian crude) |

R² = **{OLS_R2*100:.1f}%** — these three variables together explain {OLS_R2*100:.1f}% of all Indian Basket monthly return variation.
""")

        with st.expander("Step 2 — What is STD (standard deviation)?", expanded=True):
            _sb_val = np.std(BRENT_R)
            _sf_val = np.std(FX_R)
            _su_val = np.std(URALS_R)
            st.markdown(f"""
STD measures **how much each variable actually swings** month to month across 23 observations.

| Variable | STD | Meaning |
|---|---|---|
| std(Brent returns) | **{_sb_val*100:.4f}%** | Brent moves ±{_sb_val*100:.2f}% per month on average |
| std(FX returns) | **{_sf_val*100:.4f}%** | Rupee barely moves — only ±{_sf_val*100:.2f}% per month |
| std(Urals returns) | **{_su_val*100:.4f}%** | Urals is most volatile — swings ±{_su_val*100:.2f}% per month |

**Why STD matters:** A variable with β=0.43 but STD=0.71% barely moves the output. A variable with β=0.05 but STD=19.65% can still matter a lot. β alone is not enough.
""")
            fig_std = go.Figure(go.Bar(
                x=["Brent", "FX (USD/INR)", "Urals"],
                y=[_sb_val*100, _sf_val*100, _su_val*100],
                marker_color=["#378add", "#1d9e75", "#ef9f27"],
                text=[f"{v:.2f}%" for v in [_sb_val*100, _sf_val*100, _su_val*100]],
                textposition="outside",
            ))
            fig_std.update_layout(
                title="Monthly Return Volatility (STD) per Variable",
                height=260, showlegend=False,
                yaxis_title="STD of monthly returns (%)",
                margin=dict(t=40, b=10, l=10, r=10),
                yaxis=dict(range=[0, 25])
            )
            st.plotly_chart(fig_std, use_container_width=True)

        with st.expander("Step 3 — Variance Contribution = |β| × STD", expanded=True):
            _sb_val = np.std(BRENT_R)
            _sf_val = np.std(FX_R)
            _su_val = np.std(URALS_R)
            vb = abs(OLS_BETA[1]) * _sb_val
            vf = abs(OLS_BETA[2]) * _sf_val
            vu = abs(OLS_BETA[3]) * _su_val
            tot = vb + vf + vu
            st.markdown(f"""
Multiply β by how much the variable actually moves:

```
Brent:  |{OLS_BETA[1]:.4f}| × {_sb_val:.4f} = {vb:.6f}  →  {vb/tot*100:.1f}%
FX:     |{OLS_BETA[2]:.4f}| × {_sf_val:.4f} = {vf:.6f}  →  {vf/tot*100:.1f}%
Urals:  |{OLS_BETA[3]:.4f}| × {_su_val:.4f} = {vu:.6f}  →  {vu/tot*100:.1f}%
─────────────────────────────────────────────────────────
Total:                              {tot:.6f}  →  100.0%
```

**Why FX drops from β={abs(OLS_BETA[2]):.2f} to only {vf/tot*100:.1f}%:** The rupee STD is only {_sf_val*100:.2f}% — it barely moves month to month.

**Why Urals reaches {vu/tot*100:.1f}% despite β={abs(OLS_BETA[3]):.4f}:** Urals STD is {_su_val*100:.2f}% — nearly 2× more volatile than Brent.

**Why Brent dominates at {vb/tot*100:.1f}%:** It has BOTH the largest β ({OLS_BETA[1]:.4f}) AND large monthly swings ({_sb_val*100:.2f}% STD).
""")
            fig_vc = go.Figure()
            fig_vc.add_trace(go.Bar(
                name="|β| (sensitivity)",
                x=["Brent", "FX", "Urals"],
                y=[abs(OLS_BETA[1]), abs(OLS_BETA[2]), abs(OLS_BETA[3])],
                marker_color=["#b5d4f4", "#a8e6cf", "#fac775"],
                text=[f"{v:.4f}" for v in [abs(OLS_BETA[1]), abs(OLS_BETA[2]), abs(OLS_BETA[3])]],
                textposition="outside",
            ))
            fig_vc.add_trace(go.Bar(
                name="Variance contribution %",
                x=["Brent", "FX", "Urals"],
                y=[vb/tot*100, vf/tot*100, vu/tot*100],
                marker_color=["#378add", "#1d9e75", "#ef9f27"],
                text=[f"{v:.1f}%" for v in [vb/tot*100, vf/tot*100, vu/tot*100]],
                textposition="outside",
            ))
            fig_vc.update_layout(
                barmode="group", height=300,
                title="|β| alone vs actual variance contribution — why they differ",
                legend=dict(orientation="h", y=-0.25),
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_vc, use_container_width=True)

        with st.expander("Monthly Returns Table — raw data behind the regression", expanded=False):
            ret_months_labels = ["May-24","Jun-24","Jul-24","Aug-24","Sep-24","Oct-24",
                                 "Nov-24","Dec-24","Jan-25","Feb-25","Mar-25","Apr-25",
                                 "May-25","Jun-25","Jul-25","Aug-25","Sep-25","Oct-25",
                                 "Nov-25","Dec-25","Jan-26","Feb-26","Mar-26"]
            ret_table = pd.DataFrame({
                "Month":          ret_months_labels,
                "Brent_r (%)":    [round(v*100, 4) for v in BRENT_R],
                "IB_r (%)":       [round(v*100, 4) for v in IB_R],
                "FX_r (%)":       [round(v*100, 4) for v in FX_R],
                "Urals_r (%)":    [round(v*100, 4) for v in URALS_R],
            })
            st.dataframe(ret_table, hide_index=True, use_container_width=True)
            st.caption("Returns = (Current month price − Previous month price) / Previous month price × 100")

        st.markdown("---")
        st.markdown("##### Urals Discount Distribution (FY25–FY26, 24 months)")
        col_d1, col_d2 = st.columns([2,1])
        with col_d1:
            disc_colors = ["#e24b4a" if d > 10 else ("#ef9f27" if d > 5 else ("#378add" if d > 0 else "#888")) for d in DISC]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=MONTHS, y=DISC, marker_color=disc_colors, name="Brent–Urals Discount"))
            fig.add_hline(y=DISC_STATS["p90_pos"], line_dash="dash", line_color="#e24b4a",
                          annotation_text=f"90th pctile=${DISC_STATS['p90_pos']:.1f} → model ceiling=$15",
                          annotation_position="top right")
            fig.add_hline(y=0, line_color="#888", line_width=1)
            fig.update_layout(height=320, margin=dict(t=30,b=60,l=10,r=10),
                              yaxis_title="$/bbl", xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        with col_d2:
            st.markdown(f"""
**Discount Statistics**

| Metric | Value |
|---|---|
| Mean (all 24M) | ${DISC_STATS['mean']:.2f} |
| Max observed | ${DISC_STATS['max_pos']:.2f} |
| 90th pctile | ${DISC_STATS['p90_pos']:.2f} |
| 95th pctile | ${DISC_STATS['p95_pos']:.2f} |
| Positive months | {DISC_STATS['n_pos']}/24 |
| Inverted months | {DISC_STATS['n_neg']}/24 |

**Model ceiling: $15**
→ 90th pctile = ${DISC_STATS['p90_pos']:.1f}
→ Old ceiling $25 = FY22 crisis peak
→ $15 reflects post-normalisation regime
""")

# ─────────────────────────────────────────────
# TAB 4 · FORMULAS
# ─────────────────────────────────────────────
with tab_formulas:

    # ── Formula 1 ──────────────────────────────
    with st.expander("1 · Import Bill (₹ Cr / month)", expanded=True):
        st.code("""Monthly barrels    = (Throughput MMT ÷ 12) × 7,330,000
Russian barrels    = Monthly barrels × Russian share %
Non-Russian barrels= Monthly barrels × (1 − Russian share %)
Russian price      = max(Brent − Urals discount, $20)
Import USD         = (Russian barrels × Russian price)
                   + (Non-Russian barrels × Brent)
Import Bill (₹ Cr) = Import USD × USD/INR ÷ 10,000,000""", language=None)
        st.markdown(f"""
**Why each piece?**

- **1 MMT = 7.33 million barrels** — PPAC standard conversion factor for crude oil.
- **÷ 12** — converts annual throughput to a single month.
- **Russian share** — each OMC imports a different proportion of Russian Urals crude. Russian crude is priced at a discount to Brent; non-Russian crude is priced at Brent.
- **max(Brent − Urals, $20)** — the $20 floor prevents the formula producing an absurd negative price if Urals discount ever exceeds Brent (never observed but mathematically possible).
- **÷ 10,000,000** — converts raw USD to ₹ Crore (1 Cr = 10 million). The USD figure is multiplied by the FX rate first to get INR, then divided by 10 million.
- **OLS confirms** β_Brent = {OLS_BETA[1]:.3f} — near unit-elasticity, meaning a 1% rise in Brent raises the import bill by approximately 1%.
""")

    # ── Formula 2 ──────────────────────────────
    with st.expander("2 · Working Capital Stress (₹ Cr / month)", expanded=True):
        st.code("""WC Stress = Import Bill (current scenario)
          − Import Bill (base: Brent=$75 · FX=₹85 · Urals=$5)""", language=None)
        st.markdown("""
**Why this matters:**

- The base scenario ($75 Brent, ₹85 FX, $5 Urals discount) represents a "normal" operating environment derived from FY24 averages.
- A **positive** WC Stress means the OMC needs more working capital (larger LC lines, more cash collateral) than it would in normal times.
- A **negative** number means the scenario is actually cheaper than base — the OMC needs less LC headroom.
- Banks use this to pre-approve additional LC headroom before a stress event materialises, not after.
""")

    # ── Formula 3 ──────────────────────────────
    with st.expander("3 · FX Income (₹ Cr / month)", expanded=True):
        st.code("""FX Income = Total Import USD × 0.05% spread × USD/INR ÷ 10,000,000

Where Total Import USD = (Russian barrels × Russian price)
                       + (Non-Russian barrels × Brent)""", language=None)
        st.markdown(f"""
**Why each piece?**

- **0.05% (5 bps)** — this is the bank's FX settlement spread. Every dollar the OMC needs to pay for crude is converted through the bank at a margin. 5 bps is a standard wholesale FX spread for large PSU clients.
- **Total Import USD** — uses the same dollar value computed in the Import Bill formula. The bank earns its spread on every dollar that flows through it.
- **× USD/INR ÷ 10,000,000** — converts to ₹ Crore exactly as in the Import Bill formula.
- **Relationship to oil price:** When Brent is high, the dollar import value is large, so FX income rises with oil — this is a natural hedge for the bank. The returns correlation r(FX, Indian Basket) = {RET_CORR.loc['USD/INR','Indian Basket']:.3f} confirms this linkage.
""")

    # ── Formula 4 ──────────────────────────────
    with st.expander("4 · Fee Income (₹ Cr / month)", expanded=True):
        st.code("""Fee Income = Import Bill (₹ Cr) × 0.15%""", language=None)
        st.markdown("""
**Why 0.15%?**

- This 15 basis point rate represents the blended monthly income from three trade finance products the bank earns on each OMC relationship:
  - **LC issuance fee** — charged when the bank opens a Letter of Credit guaranteeing payment to the oil seller.
  - **Bank Guarantee commission** — charged for guaranteeing the OMC's performance obligations.
  - **Trade finance processing** — documentation, discrepancy handling, amendment fees.
- 0.15%/month equates to ~1.8% per annum, which is the typical all-in fee rate for PSU OMC trade finance mandates in India.
- **Directly Brent-driven:** Because fee income is a fixed % of the import bill, and the import bill rises with Brent, fee income is effectively an oil-price-linked revenue line for the bank.
""")

    # ── Formula 5 · OFAC ───────────────────────
    with st.expander("5 · OFAC Exposure Score (0–100) — Simplified", expanded=True):
        # Live worked example for currently selected scenario
        _ex_nayara = OMC[4]
        _ex_iocl   = OMC[0]
        _A_n = _ex_nayara["ofacW"];  _B_n = _ex_nayara["russianShare"]
        _A_i = _ex_iocl["ofacW"];    _B_i = _ex_iocl["russianShare"]
        _C   = OFAC_ENV[ofac_v]
        _D   = 1 + (min(urals, URALS_CEIL) / URALS_CEIL) * 0.50
        _raw_n = _A_n * _B_n * _C * _D * 25
        _raw_i = _A_i * _B_i * _C * _D * 25

        st.code("""Score = min( A × B × C × D × 25 , 100 )

A = Base Institutional Weight   (fixed per OMC; see table below)
B = Russian Share               (fraction of total crude that is Russian, 0–1)
C = Sanction Environment        (Low=1.0 · Medium=1.5 · High=3.0)
D = Urals Pressure Amplifier    (1.0 → 1.5 as Urals discount rises from $0 → $15)

D = 1 + ( min(Urals discount, $15) / $15 ) × 0.50

× 25 = scaling factor to map the result to a 0–100 range
min( …, 100 ) = hard cap so no entity exceeds 100""", language=None)

        st.markdown(f"""
---
**What each component measures :**

**A · Base Institutional Weight**

| OMC | A | 
|---|---|---|
| Reliance | **0** | 
| IOCL | **1** | 
| BPCL | **2** | 
| HPCL | **2** | 
| Nayara | **4** | 
**B · Russian Share** — straightforward percentage. The more Russian crude an OMC buys, the more transactions potentially touch sanctioned supply chains. Nayara at 82.5% has ~2× the surface area of IOCL at 38.5%.

**C · Sanction Environment Multiplier** — how aggressively OFAC is currently enforcing:
- **Low (1.0):** Normal environment; OFAC advisories in place but no active enforcement against Indian buyers.
- **Medium (1.5):** Policy shift — new advisories, secondary-sanction warnings, or pressure on correspondent banks. Not yet enforcement, but compliance cost rises materially.
- **High (3.0):** Active enforcement — designations of intermediaries, withdrawal of correspondent banks, blocked transactions. Risk is 3× base.

*(Note: Medium is 1.5, not 2.0. The step from Low→Medium is a policy signal; the step from Medium→High is an enforcement action — a much larger jump in real consequence.)*

**D · Urals Pressure Amplifier** — why does the Urals discount affect OFAC risk?

A large discount on Urals crude signals that **compliant buyers are avoiding Russian crude**. Russia can only sell at a steep discount because sanctioned entities or sanctions-adjacent intermediaries are the only willing buyers. An OMC that continues buying at a large discount is therefore:

The amplifier goes from **1.0 at $0 discount** (no signal) to **1.5 at the $15 ceiling** (50% additional scrutiny pressure). The $15 ceiling is the **90th percentile of observed FY25–26 discounts** — beyond this level the discount is in genuinely extreme territory.

```
D at Urals=$0  : 1 + (0/15) × 0.50  = 1.00   (no amplification)
D at Urals=$7.5: 1 + (7.5/15) × 0.50 = 1.25  (25% amplification)
D at Urals=$15 : 1 + (15/15) × 0.50 = 1.50   (50% amplification)
D at Urals>$15 : capped at $15 in formula → D stays at 1.50
```

**× 25 — scale calibration:**
The raw product A×B×C×D produces small decimals (e.g. 1 × 0.385 × 1.0 × 1.0 = 0.385). Multiplying by 25 maps IOCL's base case to ~9.6 (appropriately low) and allows Nayara's severe case to reach 100 (the cap). Without this factor the numbers would sit between 0 and 4, which is hard to interpret.

---
**Worked example — current scenario ({OFAC_LABELS[ofac_v]} OFAC · Urals=${urals:.1f}/bbl):**

| Step | Nayara | IOCL |
|---|---|---|
| A (institutional weight) | {_A_n} | {_A_i} |
| B (Russian share) | {_B_n:.3f} | {_B_i:.3f} |
| C (environment) | {_C:.1f} | {_C:.1f} |
| D (Urals amplifier) | {_D:.3f} | {_D:.3f} |
| A×B×C×D×25 (raw) | **{_raw_n:.1f}** | **{_raw_i:.1f}** |
| Final score (capped 100) | **{min(100,round(_raw_n))}** | **{min(100,round(_raw_i))}** |
""")

    # ── Formula 6 ──────────────────────────────
    with st.expander("6 · Overall Risk Score", expanded=True):
        st.code("""Overall Score = (Oil score    × 0.25)
              + (FX score     × 0.20)
              + (Russia score × 0.20)
              + (OFAC÷25      × 0.35)

Rating thresholds:
  Very High  ≥ 3.2
  High       ≥ 2.5
  Medium     ≥ 1.8
  Low        < 1.8""", language=None)
        st.markdown(f"""


# ─────────────────────────────────────────────
# TAB 5 · RISK RANKING
# ─────────────────────────────────────────────
with tab_risk:
    col_risk, col_actions = st.columns([3, 2])

    with col_risk:
        st.markdown("##### Risk Ranking")
        risk_data = sorted([{
            "omc": o,
            "risk": overall_risk(o, ofac_v, urals),
            "ofac_score": ofac_score(o, ofac_v, urals),
        } for o in OMC], key=lambda x: {"VH":4,"H":3,"M":2,"L":1}.get(x["risk"],0), reverse=True)

        for rd in risk_data:
            o   = rd["omc"]
            rv  = rd["risk"]
            osc = rd["ofac_score"]
            bar_w = {"VH":100,"H":75,"M":45,"L":20}.get(rv, 20)
            bar_c = risk_color(rv)
            st.markdown(f"""
<div style="border:1px solid #e0e0e0;border-radius:8px;padding:10px 14px;margin-bottom:8px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
    <span style="font-weight:600;font-size:13px">{o["name"]}</span>
    <span style="background:{bar_c}22;color:{bar_c};font-size:11px;font-weight:600;
                 padding:2px 8px;border-radius:20px;border:1px solid {bar_c}88">
      {risk_label(rv)}
    </span>
  </div>
  <div style="background:#eee;border-radius:3px;height:6px;margin-bottom:8px;overflow:hidden">
    <div style="width:{bar_w}%;height:6px;border-radius:3px;background:{bar_c}"></div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:11px;color:#666">
    <span>Russian share: <b>{o["russianShare"]*100:.0f}%</b></span>
    <span>OFAC score: <b>{osc}/100</b></span>
    <span>Oil risk: <b>{o["oilR"]}</b></span>
    <span>FX risk: <b>{o["fxR"]}</b></span>
  </div>
</div>""", unsafe_allow_html=True)

    with col_actions:
        st.markdown("##### Action Signals")
        actions = []
        if brent >= 100:
            actions.append(("🔴 Brent > $100", "Review LC limits & collateral for all PSU OMCs"))
        elif brent >= 80:
            actions.append(("🟡 Brent $80–100", "Monitor LC utilisation; pre-approve headroom"))
        else:
            actions.append(("🟢 Brent normal", "Standard LC monitoring in place"))
        if fx >= 90:
            actions.append(("🔴 FX > ₹90", "Review FX settlement terms; flag rupee liquidity risk"))
        elif fx >= 84:
            actions.append(("🟡 FX ₹84–90", "Track daily settlement; flag rupee liquidity"))
        if urals >= 15:
            actions.append(("🔴 Urals ≥ $15 ceiling", "Escalate KYC on DMCC intermediaries"))
        elif urals >= 8:
            actions.append(("🟡 Urals $8–15", "Monitor routing; check correspondent appetite"))
        if ofac_v >= 2:
            actions.append(("🔴 OFAC High", "Mandatory escalation; hold Nayara disbursements"))
        elif ofac_v == 1:
            actions.append(("🟡 OFAC Medium", "Enhanced sanctions screening required"))

        for trigger, text in actions:
            st.markdown(f"""
<div style="padding:8px 0;border-bottom:1px solid #f0f0f0">
  <div style="font-size:11px;color:#888;margin-bottom:2px">{trigger}</div>
  <div style="font-size:12px;line-height:1.5">{text}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### Summary Table")
        st.dataframe(pd.DataFrame([{
            "OMC":        o["name"],
            "Bill (₹Cr)": import_bill(o, brent, fx, urals),
            "WC Δ (₹Cr)": import_bill(o, brent, fx, urals) - import_bill(o, BASE["brent"], BASE["fx"], BASE["urals"]),
            "OFAC":       ofac_score(o, ofac_v, urals),
            "Risk":       risk_label(overall_risk(o, ofac_v, urals)),
        } for o in OMC]), hide_index=True, use_container_width=True)
