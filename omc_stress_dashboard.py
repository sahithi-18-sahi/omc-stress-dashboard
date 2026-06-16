"""
OMC Transaction Banking Stress Model
 India Oil Marketing Companies · Sahithi
Streamlit + Plotly implementation of omc_stress_dashboard_v4.html
Run: streamlit run omc_stress_dashboard.py
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="OMC Transaction Banking Stress Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CONSTANTS & DATA
# ─────────────────────────────────────────────
SCENARIOS = {
    "Base":     {"brent": 75,  "fx": 85.0, "urals": 5,  "ofac": 0},
    "Moderate": {"brent": 90,  "fx": 88.0, "urals": 10, "ofac": 1},
    "Severe":   {"brent": 110, "fx": 92.0, "urals": 20, "ofac": 2},
}
BASE = {"brent": 75, "fx": 85.0, "urals": 5, "ofac": 0}
OFAC_LABELS = ["Low", "Medium", "High"]
OFAC_MULT   = [1.0,   1.8,      3.0]
PALETTE = ["#378add", "#1d9e75", "#ef9f27", "#534ab7", "#d85a30"]

OMC = [
    {"id": "iocl",   "name": "IOCL",     "throughput": 71.56, "russianShare": 0.385, "oilR": "M", "fxR": "H", "ofacW": 1},
    {"id": "bpcl",   "name": "BPCL",     "throughput": 40.51, "russianShare": 0.365, "oilR": "H", "fxR": "H", "ofacW": 2},
    {"id": "hpcl",   "name": "HPCL",     "throughput": 21.98, "russianShare": 0.358, "oilR": "H", "fxR": "H", "ofacW": 2},
    {"id": "ril",    "name": "Reliance", "throughput": 80.00, "russianShare": 0.05,  "oilR": "M", "fxR": "M", "ofacW": 0},
    {"id": "nayara", "name": "Nayara",   "throughput": 20.00, "russianShare": 0.825, "oilR": "M", "fxR": "M", "ofacW": 4},
]
PSU       = OMC[:3]
SNRR_SET  = [OMC[0], OMC[1], OMC[2], OMC[4]]
OFAC_SET  = [OMC[0], OMC[1], OMC[2], OMC[4]]

# ─────────────────────────────────────────────
# CORE FORMULAE
# ─────────────────────────────────────────────
def import_bill(o, brent, fx, urals):
    """Import bill: monthly financing need (₹ Cr)"""
    mmt   = o["throughput"] / 12
    barr  = mmt * 7.33e6
    r_b   = barr * o["russianShare"]
    n_b   = barr * (1 - o["russianShare"])
    r_p   = max(brent - urals, 20)
    return round((r_b * r_p + n_b * brent) * fx / 1e7)

def fx_income(o, brent, fx, urals):
    """FX income: 5 bps spread on USD settlement (₹ Cr/month)"""
    mmt  = o["throughput"] / 12
    barr = mmt * 7.33e6
    r_b  = barr * o["russianShare"]
    n_b  = barr * (1 - o["russianShare"])
    r_p  = max(brent - urals, 20)
    return round((r_b * r_p + n_b * brent) * 0.0005 * fx / 1e7)

def fee_income(bill):
    """Fee income: LC + trade finance + BG commissions at 0.15% (₹ Cr/month)"""
    return round(bill * 0.0015)

def snrr_income(o, brent, fx, urals):
    """SNRR: INR float on Russian settlement (₹ Cr/month)"""
    mmt   = o["throughput"] / 12
    barr  = mmt * 7.33e6
    r_p   = max(brent - urals, 20)
    r_inr = barr * o["russianShare"] * r_p * fx / 1e7
    uf    = 1 + (min(urals, 25) / 25) * 0.3
    return round(r_inr * 0.001 * uf)

def ofac_score(o, ofac_v, urals):
    """OFAC exposure score (0–100)"""
    mult = OFAC_MULT[ofac_v]
    uf   = 1 + (min(urals, 25) / 25) * 0.5
    return min(100, round(o["ofacW"] * o["russianShare"] * mult * uf * 12))

def overall_risk(o, ofac_v, urals):
    """Overall risk: VH / H / M / L"""
    oil_s = {"M": 2, "H": 3, "L": 1}.get(o["oilR"], 2)
    fx_s  = {"M": 2, "H": 3, "L": 1}.get(o["fxR"], 2)
    ofc   = ofac_score(o, ofac_v, urals)
    r_s   = 4 if o["russianShare"] > 0.5 else (3 if o["russianShare"] > 0.3 else 2)
    t     = oil_s * 0.20 + fx_s * 0.20 + (ofc / 25) * 0.35 + r_s * 0.25
    return "VH" if t >= 3.2 else ("H" if t >= 2.5 else ("M" if t >= 1.8 else "L"))

def risk_label(r):
    return {"VH": "Very High", "H": "High", "M": "Medium", "L": "Low"}.get(r, r)

def risk_color(r):
    return {"VH": "#e24b4a", "H": "#ef9f27", "M": "#378add", "L": "#639922"}.get(r, "#888")

def signal_color(level):
    return {"green": "#639922", "amber": "#ef9f27", "red": "#e24b4a"}.get(level, "#888")

# ─────────────────────────────────────────────
# SIDEBAR — INPUTS
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
    brent = st.slider("Brent crude ($/bbl)", 55, 130, sc["brent"], 1)
    fx    = st.slider("USD / INR", 78.0, 97.0, float(sc["fx"]), 0.5)
    urals = st.slider("Urals discount ($/bbl)", 0.0, 25.0, float(sc["urals"]), 0.5)
    ofac_v = st.select_slider("OFAC escalation", options=[0, 1, 2],
                               value=sc["ofac"],
                               format_func=lambda x: OFAC_LABELS[x])

    st.markdown("---")
    st.markdown("**OMC Reference**")
    ref_df = pd.DataFrame([
        {"Company": o["name"], "Ru%": f'{o["russianShare"]*100:.1f}%', "MMT": o["throughput"]}
        for o in OMC
    ])
    st.dataframe(ref_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("**Threshold Guide**")
    st.markdown("""
| Color | Condition |
|-------|-----------|
| 🟢 Green | Brent < $80 · FX < 84 |
| 🟡 Amber | Brent $80–100 · FX 84–90 |
| 🔴 Red   | Brent > $100 · FX > 90 |
""")

# ─────────────────────────────────────────────
# SIGNAL STRIP
# ─────────────────────────────────────────────
b_lv = "green" if brent < 80  else ("amber" if brent < 100  else "red")
f_lv = "green" if fx    < 84  else ("amber" if fx    < 90   else "red")
u_lv = "green" if urals < 10  else ("amber" if urals < 20   else "red")
o_lv = "green" if ofac_v == 0 else ("amber" if ofac_v == 1  else "red")

st.markdown(
    f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0;
                border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;margin-bottom:16px">
      <div style="padding:10px 14px;border-right:1px solid #e0e0e0">
        <div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.05em">Brent Crude</div>
        <div style="font-size:18px;font-weight:600;margin-top:2px">
          <span style="display:inline-block;width:9px;height:9px;border-radius:50%;
                       background:{signal_color(b_lv)};margin-right:5px;vertical-align:middle"></span>
          ${brent}/bbl
        </div>
      </div>
      <div style="padding:10px 14px;border-right:1px solid #e0e0e0">
        <div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.05em">USD / INR</div>
        <div style="font-size:18px;font-weight:600;margin-top:2px">
          <span style="display:inline-block;width:9px;height:9px;border-radius:50%;
                       background:{signal_color(f_lv)};margin-right:5px;vertical-align:middle"></span>
          {fx:.1f}
        </div>
      </div>
      <div style="padding:10px 14px;border-right:1px solid #e0e0e0">
        <div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.05em">Urals Discount</div>
        <div style="font-size:18px;font-weight:600;margin-top:2px">
          <span style="display:inline-block;width:9px;height:9px;border-radius:50%;
                       background:{signal_color(u_lv)};margin-right:5px;vertical-align:middle"></span>
          ${urals:.1f}/bbl
        </div>
      </div>
      <div style="padding:10px 14px">
        <div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.05em">OFAC Level</div>
        <div style="font-size:18px;font-weight:600;margin-top:2px">
          <span style="display:inline-block;width:9px;height:9px;border-radius:50%;
                       background:{signal_color(o_lv)};margin-right:5px;vertical-align:middle"></span>
          {OFAC_LABELS[ofac_v]}
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab_metrics, tab_charts, tab_formulas, tab_risk = st.tabs(
    ["📊 Output Metrics", "📈 Charts", "🔢 Formulas", "⚠️ Risk Ranking"]
)

# ─── COMPUTE ALL VALUES ───────────────────────
bills   = {o["id"]: import_bill(o,  brent, fx, urals)                          for o in OMC}
bills0  = {o["id"]: import_bill(o,  BASE["brent"], BASE["fx"], BASE["urals"])  for o in OMC}
fx_inc  = {o["id"]: fx_income(o,    brent, fx, urals)                          for o in OMC}
fx_inc0 = {o["id"]: fx_income(o,    BASE["brent"], BASE["fx"], BASE["urals"])  for o in OMC}
fees    = {o["id"]: fee_income(bills[o["id"]])                                 for o in OMC}
fees0   = {o["id"]: fee_income(bills0[o["id"]])                                for o in OMC}
snrr    = {o["id"]: snrr_income(o,  brent, fx, urals)                          for o in SNRR_SET}
snrr0   = {o["id"]: snrr_income(o,  BASE["brent"], BASE["fx"], BASE["urals"]) for o in SNRR_SET}
ofacs   = {o["id"]: ofac_score(o,   ofac_v, urals)     for o in OFAC_SET}
ofacs0  = {o["id"]: ofac_score(o,   BASE["ofac"], BASE["urals"]) for o in OFAC_SET}

def delta_str(cur, base, invert=False):
    d = cur - base
    sign = "+" if d >= 0 else ""
    if invert:
        color = "#e24b4a" if d >= 0 else "#639922"
    else:
        color = "#639922" if d >= 0 else "#e24b4a"
    return f'<span style="color:{color};font-size:11px">{sign}{d:,} vs base</span>'

def fmt_cr(v):
    return f"₹{v:,} Cr"

# ─── TAB 1: METRICS ─────────────────────────
with tab_metrics:
    # Import bill
    st.markdown("##### Import Bill — monthly financing need (₹ Cr)")
    cols = st.columns(3)
    for i, o in enumerate(PSU):
        with cols[i]:
            d = bills[o["id"]] - bills0[o["id"]]
            sign = "+" if d >= 0 else ""
            color = "#e24b4a" if d >= 0 else "#639922"
            st.metric(
                o["name"],
                fmt_cr(bills[o["id"]]),
                delta=f"{sign}{d:,} Cr vs base",
                delta_color="inverse",
            )

    # Working capital stress
    st.markdown("##### Working Capital Stress vs Base (₹ Cr/month)")
    cols = st.columns(3)
    for i, o in enumerate(PSU):
        with cols[i]:
            wc = bills[o["id"]] - bills0[o["id"]]
            sign = "+" if wc >= 0 else ""
            st.metric(o["name"], f"{sign}{wc:,} Cr", delta_color="off")

    st.divider()

    # FX income
    st.markdown("##### FX Income — bank FX settlement revenue (₹ Cr/month)")
    cols = st.columns(3)
    for i, o in enumerate(PSU):
        with cols[i]:
            d = fx_inc[o["id"]] - fx_inc0[o["id"]]
            sign = "+" if d >= 0 else ""
            st.metric(o["name"], fmt_cr(fx_inc[o["id"]]), delta=f"{sign}{d:,} Cr vs base")

    # Fee income
    st.markdown("##### Fee Income — LC + trade finance fees (₹ Cr/month)")
    cols = st.columns(3)
    for i, o in enumerate(PSU):
        with cols[i]:
            d = fees[o["id"]] - fees0[o["id"]]
            sign = "+" if d >= 0 else ""
            st.metric(o["name"], fmt_cr(fees[o["id"]]), delta=f"{sign}{d:,} Cr vs base")

    st.divider()

    # SNRR income
    st.markdown("##### SNRR  — INR float on Russian settlement (₹ Cr/month)")
    cols = st.columns(4)
    for i, o in enumerate(SNRR_SET):
        with cols[i]:
            d = snrr[o["id"]] - snrr0[o["id"]]
            sign = "+" if d >= 0 else ""
            st.metric(o["name"], fmt_cr(snrr[o["id"]]), delta=f"{sign}{d:,} Cr vs base")

    # OFAC score
    st.markdown("##### OFAC Exposure Score (0–100)")
    cols = st.columns(4)
    for i, o in enumerate(OFAC_SET):
        with cols[i]:
            d = ofacs[o["id"]] - ofacs0[o["id"]]
            sign = "+" if d >= 0 else ""
            st.metric(o["name"], f'{ofacs[o["id"]]}/100', delta=f"{sign}{d} vs base", delta_color="inverse")

# ─── TAB 2: CHARTS ─────────────────────────
with tab_charts:
    col1, col2 = st.columns(2)

    with col1:
        # Import bill — all OMCs
        fig = go.Figure(go.Bar(
            x=[o["name"] for o in OMC],
            y=[bills[o["id"]] for o in OMC],
            marker_color=PALETTE,
            text=[f'₹{bills[o["id"]]:,}' for o in OMC],
            textposition="outside",
        ))
        fig.update_layout(title="Import Bill (₹ Cr/month)", height=280,
                          showlegend=False, margin=dict(t=40, b=10, l=10, r=10),
                          yaxis_title="₹ Cr")
        st.plotly_chart(fig, use_container_width=True)

        # Fee income
        fig = go.Figure(go.Bar(
            x=[o["name"] for o in PSU],
            y=[fees[o["id"]] for o in PSU],
            marker_color=PALETTE[:3],
            text=[f'₹{fees[o["id"]]}' for o in PSU],
            textposition="outside",
        ))
        fig.update_layout(title="Fee Income (₹ Cr/month)", height=280,
                          showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        # OFAC score
        fig = go.Figure(go.Bar(
            x=[o["name"] for o in OFAC_SET],
            y=[ofacs[o["id"]] for o in OFAC_SET],
            marker_color=[PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[4]],
            text=[str(ofacs[o["id"]]) for o in OFAC_SET],
            textposition="outside",
        ))
        fig.update_layout(title="OFAC Exposure Score (0–100)", height=280,
                          showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # FX income
        fig = go.Figure(go.Bar(
            x=[o["name"] for o in PSU],
            y=[fx_inc[o["id"]] for o in PSU],
            marker_color=PALETTE[:3],
            text=[f'₹{fx_inc[o["id"]]}' for o in PSU],
            textposition="outside",
        ))
        fig.update_layout(title="FX Income (₹ Cr/month)", height=280,
                          showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        # SNRR income
        fig = go.Figure(go.Bar(
            x=[o["name"] for o in SNRR_SET],
            y=[snrr[o["id"]] for o in SNRR_SET],
            marker_color=[PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[4]],
            text=[f'₹{snrr[o["id"]]}' for o in SNRR_SET],
            textposition="outside",
        ))
        fig.update_layout(title="SNRR Income (₹ Cr/month)", height=280,
                          showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        # WC stress
        sc4 = [OMC[0], OMC[1], OMC[2]]
        wc_vals = [bills[o["id"]] - bills0[o["id"]] for o in sc4]
        wc_colors = ["#e24b4a" if v >= 0 else "#639922" for v in wc_vals]
        fig = go.Figure(go.Bar(
            x=[o["name"] for o in sc4],
            y=wc_vals,
            marker_color=wc_colors,
            text=[f'{"+":}{v:,}' if v >= 0 else f'{v:,}' for v in wc_vals],
            textposition="outside",
        ))
        fig.update_layout(title="WC Stress vs Base (₹ Cr)", height=280,
                          showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    # Scenario comparison
    st.markdown("##### Scenario Comparison — Import Bill: Base vs Moderate vs Severe")
    sc4_omc = [OMC[0], OMC[1], OMC[2], OMC[4]]
    fig = go.Figure()
    scenario_styles = [
        ("Base ($75/85)",     SCENARIOS["Base"],     "#b5d4f4"),
        ("Moderate ($90/88)", SCENARIOS["Moderate"], "#fac775"),
        ("Severe ($110/92)",  SCENARIOS["Severe"],   "#f09595"),
    ]
    for label, s, color in scenario_styles:
        fig.add_trace(go.Bar(
            name=label,
            x=[o["name"] for o in sc4_omc],
            y=[import_bill(o, s["brent"], s["fx"], s["urals"]) for o in sc4_omc],
            marker_color=color,
        ))
    fig.update_layout(barmode="group", height=320, legend=dict(orientation="h", y=-0.2),
                      margin=dict(t=10, b=10, l=10, r=10), yaxis_title="₹ Cr")
    st.plotly_chart(fig, use_container_width=True)

# ─── TAB 3: FORMULAS ────────────────────────
with tab_formulas:
    formulas = [
        ("1 · Import bill (₹ Cr / month)",
         "Monthly barrels = (Throughput MMT ÷ 12) × 7,330,000\n"
         "Russian barrels = Monthly barrels × Russian share %\n"
         "Russian price = max(Brent − Urals discount, $20)\n"
         "Import USD = (Russian barrels × Russian price) + (Other barrels × Brent)\n"
         "Import bill (₹ Cr) = Import USD × USD/INR ÷ 10,000,000",
         "Floor of $20/bbl prevents negative values. 1 MMT = 7.33 mn barrels (PPAC standard)."),

        ("2 · Working capital stress",
         "WC stress = Import bill (current) − Import bill (base: $75 / 85 / $5)",
         "Positive = additional funding required on 30-day LC cycle."),

        ("3 · FX income (₹ Cr / month)",
         "FX income = Import USD × 0.05% spread × USD/INR ÷ 10,000,000",
         "5 bps spread on total USD settlement volume. Rises with oil price and rupee weakness."),

        ("4 · Fee income (₹ Cr / month)",
         "Fee income = Import bill (₹ Cr) × 0.15%",
         "LC issuance + trade finance + bank guarantee commissions at 0.15% per month."),

        ("5 · SNRR  (₹ Cr / month)",
         "Russian import (₹ Cr) = Russian barrels × Russian price × USD/INR ÷ 10,000,000\n"
         "Urals factor = 1 + (Urals discount ÷ 25) × 0.30\n"
         "SNRR income = Russian import (₹ Cr) × 0.10% × Urals factor",
         "Wider Urals discount → more INR/Vostro routing → higher SNRR balance → more float income."),

        ("6 · OFAC exposure score (0–100)",
         "OFAC multiplier: Low=1.0 | Medium=1.8 | High=3.0\n"
         "Urals routing factor = 1 + (Urals ÷ 25) × 0.50\n"
         "Score = min(Base weight × Russian share × OFAC mult × Urals factor × 12, 100)\n"
         "Base weights: IOCL=1 | BPCL=2 | HPCL=2 | Nayara=4",
         "Nayara weight=4 reflects Rosneft majority ownership. Urals factor captures alternative payment routing burden."),

        ("7 · Overall risk score",
         "Score = Oil(0.20) + FX(0.20) + OFAC÷25(0.35) + Russia category(0.25)\n"
         "Russia: >50%=4 | >30%=3 | else=2 · VH≥3.2 | H≥2.5 | M≥1.8 | L<1.8",
         "OFAC carries highest weight (35%) reflecting binary compliance consequences."),
    ]

    for title, eq, note in formulas:
        with st.expander(title, expanded=True):
            st.code(eq, language=None)
            st.caption(note)

# ─── TAB 4: RISK RANKING + ACTION SIGNALS ───
with tab_risk:
    col_risk, col_actions = st.columns([3, 2])

    with col_risk:
        st.markdown("##### Risk Ranking")
        risk_data = []
        for o in OMC:
            rv = overall_risk(o, ofac_v, urals)
            osc = ofac_score(o, ofac_v, urals)
            snrr_val = snrr_income(o, brent, fx, urals) if o in SNRR_SET else None
            risk_data.append({
                "omc": o,
                "risk": rv,
                "ofac_score": osc,
                "snrr": snrr_val,
            })
        risk_data.sort(key=lambda x: {"VH": 4, "H": 3, "M": 2, "L": 1}.get(x["risk"], 0), reverse=True)

        for rd in risk_data:
            o   = rd["omc"]
            rv  = rd["risk"]
            osc = rd["ofac_score"]
            bar_w = {"VH": 100, "H": 75, "M": 45, "L": 20}.get(rv, 20)
            bar_c = risk_color(rv)
            snrr_display = f"SNRR ₹{snrr_income(o, brent, fx, urals)} Cr" if o in SNRR_SET else "—"

            st.markdown(
                f"""
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
                </div>
                """,
                unsafe_allow_html=True,
            )

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
            actions.append(("🔴 FX > 90", "Push FX hedging; review SNRR conversions"))
        elif fx >= 84:
            actions.append(("🟡 FX 84–90", "Track daily settlement; flag rupee liquidity"))

        if urals >= 20:
            actions.append(("🔴 Urals > $20", "Escalate KYC on DMCC intermediaries"))
        elif urals >= 10:
            actions.append(("🟡 Urals $10–20", "Monitor Vostro routing; check correspondent appetite"))

        if ofac_v >= 2:
            actions.append(("🔴 OFAC High", "Mandatory escalation; hold Nayara disbursements"))
        elif ofac_v == 1:
            actions.append(("🟡 OFAC Medium", "Enhanced sanctions screening required"))

        for trigger, text in actions:
            st.markdown(
                f"""
                <div style="padding:8px 0;border-bottom:1px solid #f0f0f0">
                  <div style="font-size:11px;color:#888;margin-bottom:2px">{trigger}</div>
                  <div style="font-size:12px;line-height:1.5">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("##### Summary Table")
        summary = []
        for o in OMC:
            bill_v = import_bill(o, brent, fx, urals)
            summary.append({
                "OMC":         o["name"],
                "Bill (₹Cr)":  bill_v,
                "WC Δ (₹Cr)":  bill_v - import_bill(o, BASE["brent"], BASE["fx"], BASE["urals"]),
                "OFAC":        ofac_score(o, ofac_v, urals),
                "Risk":        risk_label(overall_risk(o, ofac_v, urals)),
            })
        st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)
