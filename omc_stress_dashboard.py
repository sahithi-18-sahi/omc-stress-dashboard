"""
OMC Transaction Banking Stress Model
ICICI Bank · India Oil Marketing Companies · FY25–26
Updated with data-derived weights, correlation analytics, Urals factor justification
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
OFAC_MULT   = [1.0, 1.8, 3.0]
PALETTE = ["#378add", "#1d9e75", "#ef9f27", "#534ab7", "#d85a30"]
 
W_OIL  = 0.25
W_FX   = 0.20
W_RU   = 0.20
W_OFAC = 0.35
 
OMC = [
    {"id":"iocl",   "name":"IOCL",     "throughput":71.56, "russianShare":0.385, "oilR":"M","fxR":"H","ofacW":1},
    {"id":"bpcl",   "name":"BPCL",     "throughput":43.50, "russianShare":0.365, "oilR":"H","fxR":"H","ofacW":2},
    {"id":"hpcl",   "name":"HPCL",     "throughput":26.04, "russianShare":0.350, "oilR":"H","fxR":"H","ofacW":2},
    {"id":"ril",    "name":"Reliance", "throughput":80.00, "russianShare":0.050, "oilR":"M","fxR":"M","ofacW":0},
    {"id":"nayara", "name":"Nayara",   "throughput":20.00, "russianShare":0.825, "oilR":"M","fxR":"M","ofacW":4},
]
PSU      = OMC[:3]
SNRR_SET = [OMC[0], OMC[1], OMC[2], OMC[4]]
OFAC_SET = [OMC[0], OMC[1], OMC[2], OMC[4]]
 
URALS_CEIL = 15.0
 
# Urals routing uplift for SNRR income
# Source: RBI bilateral payment data
#   FY22 INR share of India-Russia trade: ~5%
#   FY25 INR share of India-Russia trade: ~32%
#   Max uplift at ceiling discount = 32% - 5% = 27 ppts → rounded to 0.27
#   Interpretation: at $15 Urals discount, ~27% more Russian settlement
#   flows through INR/Vostro routes vs SWIFT baseline
INR_ROUTING_UPLIFT = 0.27
 
# OFAC routing sensitivity for OFAC score
# Each additional % of INR/Vostro routing creates disproportionate compliance
# burden vs income — every Vostro transaction requires individual OFAC screening
# and correspondent bank approval, unlike bulk SWIFT settlement.
# Set at 0.50: at $15 discount, OFAC risk amplifies by 50% above baseline.
# Higher than INR_ROUTING_UPLIFT (0.27) because compliance risk grows faster
# than income — a Vostro transaction earning 0.10% float can trigger a review
# costing multiples of that income if flagged by OFAC.
OFAC_ROUTING_SENSITIVITY = 0.50
 
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
 
def snrr_income(o, brent, fx, urals):
    mmt   = o["throughput"] / 12
    barr  = mmt * 7.33e6
    r_p   = max(brent - urals, 20)
    r_inr = barr * o["russianShare"] * r_p * fx / 1e7
    uf    = 1 + (min(urals, URALS_CEIL) / URALS_CEIL) * INR_ROUTING_UPLIFT
    return round(r_inr * 0.001 * uf)
 
def ofac_score(o, ofac_v, urals):
    mult = OFAC_MULT[ofac_v]
    uf   = 1 + (min(urals, URALS_CEIL) / URALS_CEIL) * OFAC_ROUTING_SENSITIVITY
    return min(100, round(o["ofacW"] * o["russianShare"] * mult * uf * 12))
 
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
    st.caption("ICICI Bank · FY25–26")
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
tab_metrics, tab_charts, tab_corr, tab_method, tab_formulas, tab_risk = st.tabs([
    "📊 Output Metrics", "📈 Charts",
    "🔗 Correlation Analysis", "⚖️ Methodology",
    "🔢 Formulas", "⚠️ Risk Ranking"
])
 
# ── COMPUTE ──────────────────────────────────
bills   = {o["id"]: import_bill(o, brent, fx, urals)                              for o in OMC}
bills0  = {o["id"]: import_bill(o, BASE["brent"], BASE["fx"], BASE["urals"])       for o in OMC}
fx_inc  = {o["id"]: fx_income(o, brent, fx, urals)                                for o in OMC}
fx_inc0 = {o["id"]: fx_income(o, BASE["brent"], BASE["fx"], BASE["urals"])         for o in OMC}
fees    = {o["id"]: fee_income(bills[o["id"]])                                     for o in OMC}
fees0   = {o["id"]: fee_income(bills0[o["id"]])                                    for o in OMC}
snrr    = {o["id"]: snrr_income(o, brent, fx, urals)                              for o in SNRR_SET}
snrr0   = {o["id"]: snrr_income(o, BASE["brent"], BASE["fx"], BASE["urals"])       for o in SNRR_SET}
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
 
    st.markdown("##### SNRR / Vostro Income — INR float on Russian settlement (₹ Cr/month)")
    cols = st.columns(4)
    for i, o in enumerate(SNRR_SET):
        d = snrr[o["id"]] - snrr0[o["id"]]
        cols[i].metric(o["name"], f'₹{snrr[o["id"]]:,} Cr', delta=f'{d:+,} Cr vs base')
 
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
 
        fig = go.Figure(go.Bar(x=[o["name"] for o in SNRR_SET],
            y=[snrr[o["id"]] for o in SNRR_SET],
            marker_color=[PALETTE[0],PALETTE[1],PALETTE[2],PALETTE[4]],
            text=[f'₹{snrr[o["id"]]}' for o in SNRR_SET], textposition="outside"))
        fig.update_layout(title="SNRR Income (₹ Cr/month)", height=280,
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
        st.markdown("##### Returns Correlation Matrix (monthly % changes, n=23)")
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
 
R² = {OLS_R2:.4f}  ({OLS_R2*100:.1f}% of IB return variation explained, n=23)""", language=None)
 
        st.markdown(f"""
| Variable | β | Variance contribution |
|---|---|---|
| Brent returns | **{OLS_BETA[1]:.4f}** | **{VAR_CONTRIB['Brent']*100:.1f}%** |
| FX (USD/INR) | **{OLS_BETA[2]:.4f}** | **{VAR_CONTRIB['FX']*100:.1f}%** |
| Urals returns | **{OLS_BETA[3]:.4f}** | **{VAR_CONTRIB['Urals']*100:.1f}%** |
""")
 
        # ── β and STD deep-dive ──────────────────────────
        st.markdown("---")
        st.markdown("##### How β and Variance Contribution are computed — step by step")
 
        with st.expander("Step 1 — What is β (beta)?", expanded=True):
            st.markdown(f"""
β is the OLS coefficient — it answers: *"If this variable rises by 1%, how much does the Indian Basket return change?"*
 
| Variable | β value | Plain English |
|---|---|---|
| β₀ (intercept) | **{OLS_BETA[0]:.4f}** | Baseline drift each month (negligible) |
| β₁ Brent | **{OLS_BETA[1]:+.4f}** | Brent rises 1% → Indian Basket rises **{OLS_BETA[1]*100:.2f}%** |
| β₂ FX (USD/INR) | **{OLS_BETA[2]:+.4f}** | Rupee depreciates 1% → Indian Basket rises **{abs(OLS_BETA[2])*100:.2f}%** (negative sign: FX_return is positive when rupee weakens, but that raises import cost) |
| β₃ Urals | **{OLS_BETA[3]:+.4f}** | Urals rises 1% → Indian Basket falls **{abs(OLS_BETA[3])*100:.2f}%** (Urals rising = smaller discount = costlier Russian crude) |
 
R² = **{OLS_R2*100:.1f}%** — these three variables together explain {OLS_R2*100:.1f}% of all Indian Basket monthly return variation (n=23).
""")
 
        with st.expander("Step 2 — What is STD (standard deviation)?", expanded=True):
            _sb_val = np.std(BRENT_R)
            _sf_val = np.std(FX_R)
            _su_val = np.std(URALS_R)
            st.markdown(f"""
STD measures **how much each variable actually swings** month to month across your 23 observations.
 
| Variable | STD | Meaning |
|---|---|---|
| std(Brent returns) | **{_sb_val*100:.4f}%** | Brent moves ±{_sb_val*100:.2f}% per month on average |
| std(FX returns) | **{_sf_val*100:.4f}%** | Rupee barely moves — only ±{_sf_val*100:.2f}% per month |
| std(Urals returns) | **{_su_val*100:.4f}%** | Urals is most volatile — swings ±{_su_val*100:.2f}% per month |
 
**Why STD matters:** A variable with β=0.43 but STD=0.71% barely moves the output. A variable with β=0.05 but STD=19.65% can still matter a lot. β alone is not enough.
""")
            # Mini bar chart of STDs
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
This is the key step. Multiply β by how much the variable actually moves:
 
```
Brent:  |{OLS_BETA[1]:.4f}| × {_sb_val:.4f} = {vb:.6f}  →  {vb/tot*100:.1f}%
FX:     |{OLS_BETA[2]:.4f}| × {_sf_val:.4f} = {vf:.6f}  →  {vf/tot*100:.1f}%
Urals:  |{OLS_BETA[3]:.4f}| × {_su_val:.4f} = {vu:.6f}  →  {vu/tot*100:.1f}%
─────────────────────────────────────────────────────────
Total:                              {tot:.6f}  →  100.0%
```
 
**Why FX drops from β={abs(OLS_BETA[2]):.2f} to only {vf/tot*100:.1f}% contribution:**
The rupee STD is only {_sf_val*100:.2f}% — it barely moves month to month. A large β on a variable that never moves still produces near-zero impact.
 
**Why Urals reaches {vu/tot*100:.1f}% despite β={abs(OLS_BETA[3]):.4f}:**
Urals STD is {_su_val*100:.2f}% — nearly 2× more volatile than Brent. The small β gets multiplied by a very large swing.
 
**Why Brent dominates at {vb/tot*100:.1f}%:**
It has BOTH the largest β ({OLS_BETA[1]:.4f}) AND large monthly swings ({_sb_val*100:.2f}% STD).
""")
            # Side-by-side bar: beta vs contribution
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
 
        with st.expander("Monthly Returns Table — raw data behind the regression (n=23)", expanded=False):
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
                          annotation_text=f"90th pctile=${DISC_STATS['p90_pos']:.1f} → model ceiling=15",
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
# TAB 5 · FORMULAS
# ─────────────────────────────────────────────
with tab_formulas:
    formulas = [
        ("1 · Import Bill (₹ Cr / month)",
         "Monthly barrels    = (Throughput MMT ÷ 12) × 7,330,000\n"
         "Russian barrels    = Monthly barrels × Russian share %\n"
         "Russian price      = max(Brent − Urals discount, $20)\n"
         "Import USD         = (Russian barrels × Russian price) + (Other barrels × Brent)\n"
         "Import Bill (₹ Cr) = Import USD × USD/INR ÷ 10,000,000",
         f"Floor of $20/bbl prevents negative prices. 1 MMT = 7.33 mn barrels (PPAC standard). "
         f"OLS confirms β_Brent = {OLS_BETA[1]:.3f} (near unit-elasticity)."),
 
        ("2 · Working Capital Stress (₹ Cr / month)",
         "WC Stress = Import Bill (current) − Import Bill (base: $75 Brent / ₹85 FX / $5 Urals)",
         "Positive = additional LC headroom required vs normal operations."),
 
        ("3 · FX Income (₹ Cr / month)",
         "FX Income = Total Import USD × 0.05% spread × USD/INR ÷ 10,000,000",
         f"5 bps on USD/INR settlement. Returns r(FX, Indian Basket) = {RET_CORR.loc['USD/INR','Indian Basket']:.3f}. "
         f"FX amplifies oil stress (β_FX = 0.778 on import bill)."),
 
        ("4 · Fee Income (₹ Cr / month)",
         "Fee Income = Import Bill (₹ Cr) × 0.15%",
         "LC issuance + BG commission + trade finance at 0.15%/month (~1.8% p.a.). "
         "Scales directly with import bill — effectively Brent-driven."),
 
        ("5 · SNRR / Vostro Income (₹ Cr / month)",
         "Russian import (₹ Cr)  = Russian barrels × Russian price × USD/INR ÷ 10,000,000\n\n"
         "INR_ROUTING_UPLIFT     = 0.27   ← RBI data: INR share grew from 5% (FY22) to 32% (FY25)\n"
         "                                   32% − 5% = 27 ppts incremental routing at peak discount\n\n"
         "URALS_CEIL             = $15    ← 90th pctile of FY25-26 observed discounts\n\n"
         "Urals routing factor   = 1 + min(Urals discount, URALS_CEIL) / URALS_CEIL × INR_ROUTING_UPLIFT\n"
         "                       = 1 + min(Urals, 15) / 15 × 0.27\n\n"
         "SNRR Income            = Russian import (₹ Cr) × 0.10% × Urals routing factor",
         "The +1 is the baseline — SNRR income exists even at zero Urals discount. "
         "INR_ROUTING_UPLIFT (0.27) is the maximum additional share of Russian settlement "
         "flowing via Vostro vs SWIFT, anchored to RBI bilateral payment data. "
         "Nayara (82.5% Russian share) generates highest SNRR income."),
 
        ("6 · OFAC Exposure Score (0–100)",
         "OFAC multiplier            : Low=1.0 | Medium=1.8 | High=3.0\n\n"
         "OFAC_ROUTING_SENSITIVITY   = 0.50   ← independently set; measures compliance risk\n"
         "                                       from Vostro routing, NOT volume of routing\n"
         "                                       (each Vostro txn needs individual OFAC screening;\n"
         "                                        compliance cost >> 0.10% float income if flagged)\n\n"
         "URALS_CEIL                 = $15    ← same data-derived ceiling as SNRR\n\n"
         "Urals routing factor       = 1 + min(Urals, URALS_CEIL) / URALS_CEIL × OFAC_ROUTING_SENSITIVITY\n"
         "                           = 1 + min(Urals, 15) / 15 × 0.50\n\n"
         "Score = min(Base weight × Russian share × OFAC mult × Urals factor × 12, 100)\n\n"
         "Base weights: IOCL=1 | BPCL=2 | HPCL=2 | Nayara=4\n"
         "Medium=1.8 (not 2.0): policy shift, not enforcement action",
         "OFAC_ROUTING_SENSITIVITY (0.50) > INR_ROUTING_UPLIFT (0.27) because compliance risk "
         "scales with regulatory consequence of routing, not just volume. "
         "Nayara weight=4 → Rosneft (~49%) is SDN-adjacent; any transaction is OFAC-proximate."),
 
        ("7 · Overall Risk Score — DATA-DERIVED WEIGHTS",
         "Score = Oil(0.25) + FX(0.20) + Russia(0.20) + (OFAC÷25)(0.35)\n\n"
         f"Regression-derived raw: Brent {VAR_CONTRIB['Brent']*100:.1f}% | "
         f"FX {VAR_CONTRIB['FX']*100:.1f}% | Urals {VAR_CONTRIB['Urals']*100:.1f}% | OFAC 0%\n"
         "Adjusted: Oil 25% (hedgeable↓) | FX 20% (β=0.778 amplifier↑)\n"
         "          Russia 20% (structural lock-in↑) | OFAC 35% (binary tail risk↑)\n\n"
         "Russia category: >50% → 4 | >30% → 3 | else → 2\n"
         "Rating thresholds: VH≥3.2 | H≥2.5 | M≥1.8 | L<1.8",
         f"R²={OLS_R2*100:.1f}% from OLS (n=23). OFAC gets highest weight despite zero R² — "
         "absent from normal-period data = largest tail risk."),
    ]
 
    for title, eq, note in formulas:
        with st.expander(title, expanded=True):
            st.code(eq, language=None)
            st.caption(note)
 
# ─────────────────────────────────────────────
# TAB 6 · RISK RANKING
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
            actions.append(("🔴 FX > ₹90", "Push FX hedging; review SNRR conversions"))
        elif fx >= 84:
            actions.append(("🟡 FX ₹84–90", "Track daily settlement; flag rupee liquidity"))
        if urals >= 15:
            actions.append(("🔴 Urals ≥ $15 ceiling", "Escalate KYC on DMCC intermediaries"))
        elif urals >= 8:
            actions.append(("🟡 Urals $8–15", "Monitor Vostro routing; check correspondent appetite"))
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
 
        st.markdown("---")
        st.markdown("##### Model Weights (data-derived)")
        fig = go.Figure(go.Bar(
            x=["Oil Price","FX / INR","Russian / Urals","OFAC / Compliance"],
            y=[W_OIL*100, W_FX*100, W_RU*100, W_OFAC*100],
            marker_color=["#378add","#1d9e75","#ef9f27","#e24b4a"],
            text=[f"{v:.0f}%" for v in [W_OIL*100, W_FX*100, W_RU*100, W_OFAC*100]],
            textposition="outside",
        ))
        fig.update_layout(height=220, showlegend=False,
                          margin=dict(t=10,b=10,l=10,r=10),
                          yaxis=dict(range=[0,45], title="%"))
        st.plotly_chart(fig, use_container_width=True)
