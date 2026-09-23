"""Русскоязычный штаб планирования тарифных кампаний.

Интерфейс запускает тот же Agent.act(env), что и локальная проверка. Все
показатели берутся из текущих синтетических данных, фактических пилотов,
прогноза агента или отдельного результата локального мок-скорера.
"""

from __future__ import annotations

import html
import logging
import re
from io import StringIO

import pandas as pd
import streamlit as st

from dashboard_service import get_overview, run_scenario


LOGGER = logging.getLogger(__name__)


st.set_page_config(
    page_title="AI-оптимизатор маркетинговых кампаний",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      html, body, [data-testid="stAppViewContainer"] {
        background: #f5f6f7; color: #202226; font-family: -apple-system, BlinkMacSystemFont,
        "Segoe UI", Arial, sans-serif;
      }
      [data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer,
      [data-testid="stStatusWidget"], .stDeployButton { display: none !important; }
      .block-container { max-width: 1300px; padding-top: 1.3rem; padding-bottom: 3rem; }
      h1, h2, h3 { color: #1d1f22; letter-spacing: -.025em; }
      h2 { font-size: 1.34rem !important; margin-top: .75rem !important; }
      [data-testid="stVerticalBlock"] { gap: .7rem; }
      [data-testid="stHorizontalBlock"] { align-items: stretch; }
      [data-testid="stVerticalBlockBorderWrapper"] > div {
        border-color: #e8eaed !important; border-radius: 18px !important;
        background: #fff; box-shadow: 0 4px 20px rgba(24,26,30,.035);
      }
      .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
        background: #f8d335; color: #1e2024; border: 1px solid #efca24;
        border-radius: 10px; font-weight: 700; min-height: 3rem;
      }
      .stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
        background: #eec62c; color: #1e2024; border-color: #dfb819;
      }
      .stButton > button { width: 100%; }
      [data-testid="stNumberInput"] input { font-weight: 650; }
      .hero {
        background: #fff; border: 1px solid #e8eaed; border-radius: 22px;
        padding: 25px 29px 24px; box-shadow: 0 6px 28px rgba(24,26,30,.04);
        border-top: 5px solid #f8d335; margin-bottom: 8px;
      }
      .hero-meta { display: flex; align-items: center; gap: 11px; color: #656970;
        font-size: .82rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
      .brand-dot { display: inline-block; height: 16px; width: 16px; border-radius: 50%;
        background: #202226; box-shadow: inset -5px 0 #f8d335; }
      .hero h1 { margin: 11px 0 4px; font-size: clamp(1.8rem, 3.1vw, 2.65rem); line-height: 1.08; }
      .hero-sub { margin: 0; color: #34373c; font-size: 1.05rem; font-weight: 600; }
      .hero-desc { margin: 12px 0 0; color: #666b72; font-size: .94rem; }
      .hero-note { display: inline-block; margin-top: 15px; padding: 5px 10px;
        background: #f5f6f7; border-radius: 999px; color: #656970; font-size: .77rem; }
      .hero.compact { padding:17px 24px; }
      .hero.compact h1 { font-size:clamp(1.55rem,2.2vw,2rem); margin:6px 0 3px; }
      .hero.compact .hero-desc, .hero.compact .hero-note { display:none; }
      .metrics { display: grid; grid-template-columns: repeat(4,minmax(0,1fr));
        gap: 12px; margin: 13px 0 11px; }
      .metrics.result { grid-template-columns: repeat(6,minmax(0,1fr)); }
      .metric { background:#fff; border:1px solid #e8eaed; border-radius:16px;
        padding: 15px 17px; min-height: 103px; box-shadow: 0 3px 16px rgba(24,26,30,.025); }
      .metric-label { color:#72777e; font-size:.78rem; line-height:1.35; min-height:2em; }
      .metric-value { font-size: clamp(1.16rem,2vw,1.7rem); line-height:1.2;
        font-weight:750; margin-top:8px; white-space:nowrap; }
      .metric-help { color:#878b90; font-size:.72rem; margin-top:4px; }
      .section-eyebrow { color:#a17b00; text-transform:uppercase; letter-spacing:.08em;
        font-size:.72rem; font-weight:750; margin-bottom:2px; }
      .section-intro { color:#687077; margin:0 0 11px; font-size:.9rem; }
      .goal-chip { display:inline-block; border:1px solid #eadf9f; background:#fff9db;
        border-radius:999px; color:#5f4b0b; padding:7px 11px; font-size:.83rem; font-weight:650; }
      .rule-chip { display:inline-block; background:#f1f2f3; color:#4e5359; border-radius:999px;
        padding:7px 11px; font-size:.81rem; margin:0 5px 5px 0; }
      .timeline { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:9px; margin:9px 0 12px; }
      .timeline-step { background:#fff; border:1px solid #e7e9eb; border-radius:12px; padding:11px 13px; }
      .timeline-step.active { border-color:#e9c635; background:#fffbea; }
      .timeline-step.done { border-color:#dae8db; background:#f6faf6; }
      .timeline-index { color:#8a8e93; font-size:.72rem; font-weight:700; }
      .timeline-name { color:#272a2e; font-weight:680; font-size:.86rem; margin-top:2px; }
      .timeline-status { color:#777d82; font-size:.72rem; margin-top:3px; }
      .timeline-step.active .timeline-status { color:#8b6900; }
      .timeline-step.done .timeline-status { color:#327344; }
      .portfolio-strip { display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
        gap:10px; margin:7px 0 11px; }
      .strip-item { background:#f7f8f8; border-radius:11px; padding:11px 12px; }
      .strip-label { color:#777d82; font-size:.75rem; }
      .strip-value { color:#22252a; font-size:1.09rem; font-weight:750; margin-top:3px; }
      .bar-shell { height:10px; border-radius:99px; background:#eceeef; overflow:hidden; }
      .bar-used { height:100%; background:#f2cf35; border-radius:99px; }
      .bar-pilot { height:100%; background:#34373b; float:left; }
      .bar-final { height:100%; background:#f2cf35; float:left; }
      .bar-meta { display:flex; justify-content:space-between; color:#6c7279; font-size:.78rem; margin-top:6px; }
      .campaign-heading { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:9px; }
      .campaign-name { font-size:1.07rem; font-weight:750; color:#22252a; }
      .campaign-tag { border-radius:10px; padding:6px 10px; background:#fff6cc;
        color:#6f5600; font-size:.72rem; font-weight:650; text-align:right; line-height:1.35; }
      .campaign-tag strong { font-size:.93rem; }
      .campaign-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:9px; margin:9px 0 11px; }
      .campaign-field { border-top:1px solid #eceeef; padding-top:8px; min-width:0; }
      .campaign-field-label { color:#858a90; font-size:.73rem; }
      .campaign-field-value { color:#292c30; font-size:.89rem; font-weight:670; overflow-wrap:anywhere; }
      .campaign-reason { background:#f7f8f8; border-radius:10px; padding:10px 12px;
        color:#525860; font-size:.84rem; line-height:1.5; }
      .business-copy { color:#4f565d; font-size:.92rem; line-height:1.58; margin:2px 0 8px; }
      .reject-card { background:#fff; border:1px solid #e8eaed; border-left:3px solid #b7bdc3;
        border-radius:12px; padding:10px 13px; margin:8px 0; }
      .reject-title { font-weight:680; font-size:.85rem; color:#34383d; }
      .reject-reason { color:#70767d; font-size:.8rem; margin-top:4px; }
      .evidence-note { color:#72787e; font-size:.79rem; line-height:1.5; }
      .audience-chart { background:#fff; border:1px solid #e8eaed; border-radius:16px;
        padding:16px 18px; margin:8px 0 12px; }
      .audience-row { display:grid; grid-template-columns:135px minmax(0,1fr) 72px;
        align-items:center; gap:12px; margin:11px 0; font-size:.82rem; color:#525860; }
      .audience-track { height:10px; background:#f0f1f2; border-radius:999px; overflow:hidden; }
      .audience-fill { height:100%; background:#f2d34c; border-radius:999px; }
      .audience-count { text-align:right; color:#303338; font-weight:700; }
      .pilot-chart { background:#fff; border:1px solid #e8eaed; border-radius:16px;
        padding:14px 18px; margin:7px 0 10px; }
      .pilot-row { display:grid; grid-template-columns:158px minmax(0,1fr) 72px;
        align-items:center; gap:10px; margin:9px 0; font-size:.8rem; }
      .pilot-label { color:#5c6268; white-space:nowrap; }
      .pilot-track { position:relative; height:10px; border-radius:999px; background:#f0f1f2; }
      .pilot-track:before { content:""; position:absolute; top:-3px; bottom:-3px; left:50%;
        width:1px; background:#adb2b7; z-index:1; }
      .pilot-bar { position:absolute; top:0; height:10px; border-radius:999px; }
      .pilot-bar.positive { background:#64a276; }
      .pilot-bar.negative { background:#d39186; }
      .pilot-value { text-align:right; font-weight:700; color:#33373b; white-space:nowrap; }
      .scenario-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:11px; }
      .scenario-box { background:#f7f8f8; border:1px solid #eceeef; border-radius:12px; padding:13px; }
      .scenario-title { font-size:.75rem; color:#747a80; text-transform:uppercase; letter-spacing:.04em; }
      .scenario-value { font-size:1rem; font-weight:720; color:#25282c; margin-top:6px; line-height:1.5; }
      @media (max-width: 1350px) {
        .metrics.result { grid-template-columns:repeat(3,minmax(0,1fr)); }
        .timeline { grid-template-columns:repeat(3,minmax(0,1fr)); }
      }
      @media (max-width: 920px) {
        .metrics, .metrics.result { grid-template-columns:repeat(2,minmax(0,1fr)); }
        .portfolio-strip { grid-template-columns:repeat(2,minmax(0,1fr)); }
      }
      @media (max-width: 620px) {
        .block-container { padding-left:1rem; padding-right:1rem; }
        .hero { padding:20px; }
        .timeline, .campaign-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
        .scenario-grid { grid-template-columns:1fr; }
        .audience-row, .pilot-row { grid-template-columns:108px minmax(0,1fr) 64px; gap:6px; }
      }
      /* Рабочий штаб: контраст, движение и иерархия без внешних ассетов. */
      html, body, [data-testid="stAppViewContainer"] { background:#f3f4f1; }
      [data-testid="stAppViewContainer"] {
        background-image:radial-gradient(circle at 5% 0%,rgba(247,210,54,.13),transparent 31%),
          linear-gradient(90deg,rgba(34,37,40,.018) 1px,transparent 1px);
        background-size:auto,32px 32px;
      }
      .block-container { max-width:1390px; padding-top:1.6rem; }
      .hero { position:relative; overflow:hidden; background:#17191b; color:#f9faf8;
        border:1px solid #2f3234; border-radius:25px; border-top:0;
        padding:37px 40px 0; box-shadow:0 24px 55px rgba(25,28,30,.16); }
      .hero:before { content:""; position:absolute; inset:0;
        background:radial-gradient(circle at 82% 46%,rgba(245,208,43,.18),transparent 31%),
          repeating-linear-gradient(90deg,transparent 0 65px,rgba(255,255,255,.025) 66px 67px);
        pointer-events:none; }
      .hero-grid { position:relative; display:grid; grid-template-columns:minmax(0,1.6fr) minmax(295px,.88fr);
        gap:25px; align-items:center; min-height:275px; }
      .hero-copy { position:relative; z-index:2; padding-bottom:28px; }
      .hero-meta { color:#f8d335; font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; gap:10px; }
      .brand-dot { width:17px; height:17px; background:#f8d335; box-shadow:inset -6px 0 #17191b;
        border:1px solid rgba(255,255,255,.5); }
      .hero-live { display:inline-flex; align-items:center; gap:7px; margin-left:12px;
        padding:5px 9px; border:1px solid rgba(255,255,255,.18); border-radius:99px;
        color:#e5e9e6; font-size:.68rem; letter-spacing:.02em; text-transform:none; }
      .hero-live:before { content:""; width:6px; height:6px; background:#f7d334; border-radius:50%;
        box-shadow:0 0 0 4px rgba(247,211,52,.14); }
      .hero h1 { color:#fff; font-size:clamp(2rem,3.5vw,3.35rem); line-height:1.04;
        letter-spacing:-.055em; margin:24px 0 13px; font-weight:800; }
      .hero h1 span { color:#f8d335; }
      .hero-sub { color:#f4f5f2; font-size:1rem; font-weight:650; }
      .hero-desc { color:#b7bcb8; max-width:590px; font-size:.93rem; line-height:1.55; }
      .hero-steps { display:flex; align-items:center; flex-wrap:wrap; gap:10px;
        color:#c9cdc9; font-size:.69rem; font-weight:750; letter-spacing:.08em;
        text-transform:uppercase; margin-top:23px; }
      .hero-steps i { height:1px; width:23px; background:#7a806f; }
      .hero-visual { position:relative; min-height:275px; display:grid; place-items:center; isolation:isolate; }
      .hero-orbit { position:absolute; border:1px solid rgba(248,211,53,.28); border-radius:50%; }
      .hero-orbit-outer { width:280px; height:280px; box-shadow:0 0 0 37px rgba(248,211,53,.025); }
      .hero-orbit-inner { width:218px; height:218px; border-style:dashed; border-color:rgba(255,255,255,.23); }
      .hero-visual:before,.hero-visual:after { content:""; position:absolute; width:10px; height:10px;
        border-radius:50%; background:#f8d335; box-shadow:0 0 20px rgba(248,211,53,.8); }
      .hero-visual:before { top:30px; right:57px; }
      .hero-visual:after { bottom:45px; left:45px; width:6px; height:6px; }
      .hero-pulse { position:absolute; width:166px; height:166px; border-radius:50%;
        background:radial-gradient(circle at 34% 25%,#373628,#242622 54%,#191b1c 75%);
        box-shadow:0 20px 55px rgba(0,0,0,.3), inset 0 0 0 1px rgba(248,211,53,.28); }
      .hero-number { position:relative; z-index:1; text-align:center; display:flex;
        flex-direction:column; align-items:center; max-width:285px; }
      .hero-number-label { color:#f8d335; font-size:.62rem; letter-spacing:.1em;
        font-weight:800; line-height:1.3; }
      .hero-number strong { display:block; color:#fff; font-size:clamp(1.45rem,2.7vw,2.45rem);
        letter-spacing:-.055em; margin:7px 0 4px; white-space:nowrap; }
      .hero-number span { color:#b9bdb9; font-size:.68rem; }
      .hero-footer { position:relative; display:flex; justify-content:space-between;
        padding:13px 0 16px; border-top:1px solid rgba(255,255,255,.1);
        color:#a6aba8; font-size:.7rem; letter-spacing:.05em; }
      .hero-footer span:first-child { color:#f8d335; font-weight:780; letter-spacing:.13em; }
      .metrics { gap:12px; margin:17px 0 17px; }
      .metrics.result { grid-template-columns:repeat(4,minmax(0,1fr)); }
      .metric { position:relative; overflow:hidden; min-height:115px; border:1px solid #e3e6e3;
        padding:17px 19px; box-shadow:0 8px 28px rgba(23,27,28,.035); }
      .metric:before { content:""; position:absolute; top:0; left:0; right:0;
        height:3px; background:#f0d453; }
      .metric-label { color:#687078; font-weight:600; }
      .metric-value { font-size:clamp(1.35rem,2.2vw,1.9rem); letter-spacing:-.035em; }
      .metric-help { color:#8a9194; }
      .section-eyebrow { color:#9c7800; font-size:.69rem; letter-spacing:.14em; }
      .timeline-step { position:relative; padding:13px 15px; border:1px solid #e4e7e4;
        box-shadow:0 5px 16px rgba(25,28,29,.025); }
      .timeline-index { color:#a17b00; letter-spacing:.12em; }
      .timeline-step.done:before { content:"✓"; position:absolute; right:10px; top:9px;
        width:19px; height:19px; border-radius:50%; display:grid; place-items:center;
        background:#e7f1e7; color:#3e8051; font-size:.7rem; font-weight:800; }
      .timeline-step.active { box-shadow:0 0 0 2px rgba(248,211,53,.2); }
      .decision-flow { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:11px; margin:12px 0 19px; }
      .flow-step { position:relative; min-height:103px; background:#1c1e20; color:#fff;
        border:1px solid #313537; border-radius:15px; padding:14px 17px; }
      .flow-step:last-child { background:#f8d335; border-color:#e4c12d; color:#202123; }
      .flow-step:not(:last-child):after { content:"→"; position:absolute; right:-17px; top:35px;
        z-index:2; color:#ad8d09; font-size:1.35rem; font-weight:800; }
      .flow-label { color:#b4bbb7; font-size:.73rem; }
      .flow-step:last-child .flow-label { color:#62500b; }
      .flow-value { font-size:1.65rem; font-weight:800; letter-spacing:-.04em; margin-top:5px; }
      .flow-note { font-size:.69rem; color:#9da49f; margin-top:1px; }
      .flow-step:last-child .flow-note { color:#665614; }
      .portfolio-board { display:grid; grid-template-columns:205px minmax(0,1fr);
        gap:22px; align-items:center; padding:12px 5px; }
      .budget-ring { width:175px; height:175px; border-radius:50%; display:grid; place-items:center;
        margin:auto; box-shadow:0 12px 30px rgba(30,32,34,.12); }
      .budget-ring-core { width:125px; height:125px; border-radius:50%; background:#fff;
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        box-shadow:inset 0 0 0 1px #eceeed; }
      .budget-ring-core strong { font-size:2rem; line-height:1; letter-spacing:-.06em; }
      .budget-ring-core span { color:#81888c; font-size:.7rem; margin-top:5px; }
      .portfolio-board-title { color:#25292b; font-size:1.1rem; font-weight:770; margin-bottom:12px; }
      .budget-breakdown { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
      .budget-part { border:1px solid #e9ebe9; border-radius:10px; padding:11px 12px; background:#fafbf9; }
      .budget-part:before { content:""; display:inline-block; width:8px; height:8px;
        border-radius:50%; background:var(--part-color); margin-right:7px; }
      .budget-part-label { color:#667078; font-size:.75rem; }
      .budget-part-value { color:#292c2e; font-size:1.05rem; font-weight:760; margin-top:4px; }
      .portfolio-contacts { margin-top:16px; }
      .portfolio-contacts .bar-shell { height:9px; }
      .portfolio-contacts .bar-meta { margin-top:7px; }
      .campaign-cards { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:15px; margin:15px 0 18px; }
      .campaign-card { background:#fff; border:1px solid #e3e6e3; border-radius:19px;
        overflow:hidden; box-shadow:0 14px 35px rgba(25,28,29,.055); }
      .campaign-card-top { border-top:5px solid #f8d335; padding:17px 20px 12px; }
      .campaign-card-kicker { color:#917500; font-size:.68rem; letter-spacing:.13em; font-weight:800; }
      .campaign-route { display:flex; align-items:center; gap:11px; flex-wrap:wrap; margin:12px 0 11px;
        color:#2a2c30; font-size:1.15rem; font-weight:710; letter-spacing:-.02em; }
      .campaign-route .arrow { width:29px; height:29px; display:grid; place-items:center;
        border-radius:8px; background:#f8d335; color:#292b2b; font-size:1.1rem; }
      .campaign-route strong { font-weight:820; }
      .campaign-segment { color:#686f73; font-size:.79rem; line-height:1.45; }
      .campaign-economics { display:flex; align-items:center; justify-content:space-between;
        gap:12px; padding:13px 20px; background:#f8f9f6; border-top:1px solid #eceeeb;
        border-bottom:1px solid #eceeeb; }
      .campaign-economics-label { color:#697075; font-size:.72rem; }
      .campaign-economics strong { color:#252728; display:block; font-size:1.37rem;
        letter-spacing:-.035em; margin-top:2px; }
      .campaign-channel { padding:6px 10px; border-radius:99px; color:#58480a; background:#fff0ad;
        font-size:.75rem; font-weight:760; white-space:nowrap; }
      .campaign-stats { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:0;
        padding:13px 20px 0; }
      .campaign-stat { padding:4px 10px 4px 0; }
      .campaign-stat-label { color:#858b8e; font-size:.69rem; line-height:1.35; }
      .campaign-stat-value { color:#292d2e; font-weight:750; font-size:.91rem; margin-top:5px; }
      .campaign-why { margin:16px 20px 12px; border-left:3px solid #f8d335;
        padding:7px 10px; background:#fffdf4; color:#4b5355; font-size:.78rem; line-height:1.52; }
      .campaign-foot { padding:0 20px 17px; color:#90969a; font-size:.68rem; line-height:1.4; }
      .audience-chart,.pilot-chart { box-shadow:0 8px 28px rgba(25,28,29,.035); }
      .audience-fill { background:linear-gradient(90deg,#f5d340,#e8b817); }
      .pilot-bar.positive { background:#4f9a6b; }
      .pilot-bar.negative { background:#d28578; }
      .scenario-grid { margin:14px 0; }
      .scenario-box { border:1px solid #e0e4e0; padding:17px; }
      .scenario-box:last-child { background:#fff9df; border-color:#ebd789; }
      .scenario-delta { display:flex; gap:10px; flex-wrap:wrap; margin:8px 0 6px; }
      .scenario-delta span { background:#f3f5f0; border:1px solid #e5e9e3; border-radius:99px;
        color:#424a4a; font-size:.75rem; font-weight:680; padding:7px 10px; }
      .insight-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr));
        gap:12px; margin:9px 0 15px; }
      .insight-card { background:#fff; border:1px solid #e2e6e2; border-radius:15px;
        padding:18px 19px; min-height:164px; box-shadow:0 8px 26px rgba(25,28,29,.035); }
      .insight-card:first-child { background:#232628; border-color:#232628; color:#fff; }
      .insight-card:last-child { background:#fff9dd; border-color:#ead889; }
      .insight-kicker { color:#9a7b10; font-size:.69rem; font-weight:800; letter-spacing:.11em; }
      .insight-card:first-child .insight-kicker { color:#f8d335; }
      .insight-card strong { display:block; font-size:1.21rem; line-height:1.25;
        letter-spacing:-.025em; margin:11px 0 8px; }
      .insight-card p { color:#697176; font-size:.79rem; line-height:1.5; margin:0; }
      .insight-card:first-child p { color:#c9cfca; }
      .insight-foot { color:#777e82; font-size:.78rem; line-height:1.5; margin-bottom:10px; }
      @media (max-width: 1090px) {
        .hero-grid { grid-template-columns:1fr; gap:0; }
        .hero-copy { padding-bottom:10px; }
        .hero-visual { min-height:185px; }
        .hero-orbit-outer { width:205px; height:205px; }
        .hero-orbit-inner { width:156px; height:156px; }
        .hero-pulse { width:130px; height:130px; }
        .hero-visual:before { right:calc(50% - 90px); top:0; }
        .hero-visual:after { left:calc(50% - 80px); bottom:3px; }
        .hero-number strong { font-size:1.85rem; }
        .hero { padding:28px 30px 0; }
      }
      @media (max-width: 920px) {
        .metrics.result { grid-template-columns:repeat(2,minmax(0,1fr)); }
        .campaign-cards { grid-template-columns:1fr; }
        .decision-flow { grid-template-columns:repeat(2,minmax(0,1fr)); }
        .flow-step:nth-child(2):after { display:none; }
        .insight-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
        .insight-card:last-child { grid-column:1/-1; min-height:120px; }
      }
      @media (max-width: 620px) {
        .hero { padding:22px 21px 0; }
        .hero h1 { font-size:2rem; }
        .hero-meta { flex-wrap:wrap; }
        .hero-live { margin-left:0; }
        .hero-steps { gap:6px; font-size:.63rem; }
        .hero-steps i { width:12px; }
        .hero-footer { gap:8px; }
        .metrics,.metrics.result { grid-template-columns:repeat(2,minmax(0,1fr)); }
        .metric { padding:13px; min-height:103px; }
        .metric-value { font-size:1.08rem; white-space:normal; overflow-wrap:anywhere; }
        .portfolio-board { grid-template-columns:1fr; }
        .budget-breakdown { grid-template-columns:1fr; }
        .campaign-stats { grid-template-columns:repeat(2,minmax(0,1fr)); }
        .decision-flow { gap:8px; }
        .flow-step { min-height:95px; }
        .insight-grid { grid-template-columns:1fr; }
        .insight-card:last-child { grid-column:auto; }
      }
      @media (prefers-reduced-motion: no-preference) {
        .hero-pulse { animation:orbit-glow 6s ease-in-out infinite; }
        .hero-visual:before { animation:node-glow 3.6s ease-in-out infinite; }
        .campaign-card { animation:card-enter .5s ease-out both; }
        .campaign-card:nth-child(2) { animation-delay:.08s; }
      }
      @keyframes orbit-glow { 50% { box-shadow:0 20px 65px rgba(248,211,53,.17),inset 0 0 0 1px rgba(248,211,53,.45); } }
      @keyframes node-glow { 50% { box-shadow:0 0 28px rgba(248,211,53,1); transform:scale(1.25); } }
      @keyframes card-enter { from { opacity:0; transform:translateY(11px); } to { opacity:1; transform:translateY(0); } }
    </style>
    """,
    unsafe_allow_html=True,
)


STAGES = [
    "Анализ абонентской базы",
    "Поиск сегментов и гипотез",
    "Проведение пилотов",
    "Оценка эффекта и вариантов",
    "Оптимизация портфеля",
    "Проверка ограничений",
]
STAGE_INDEX = {
    "inspect": 0,
    "hypotheses": 1,
    "explore": 2,
    "confirm": 2,
    "screen": 3,
    "portfolio": 4,
    "fallback": 4,
    "validate": 5,
}
CHANNEL_LABELS = {
    "push": "Пуш-уведомление",
    "sms": "СМС",
    "digital_ads": "Цифровая реклама",
    "call": "Звонок",
}
ARPU_LABELS = {"LOW": "Низкий", "MID": "Средний", "HIGH": "Высокий"}
DATA_LABELS = {"NON_USER": "Без интернета", "LITE": "Умеренный интернет", "HEAVY": "Активный интернет"}
CALL_LABELS = {"LOW": "Мало звонков", "MEDIUM": "Средняя активность звонков", "HIGH": "Много звонков"}
CAMPAIGN_COLUMNS = [
    "campaign_name",
    "filter_arpu_segment",
    "filter_data_segment",
    "filter_call_segment",
    "filter_current_tariff",
    "target_tariff",
    "channel",
]


def _text(value: object) -> str:
    return html.escape(str(value), quote=True)


def _number(value: float | int) -> str:
    return f"{value:,.0f}".replace(",", " ")


def _plural(value: int, forms: tuple[str, str, str]) -> str:
    tail = value % 100
    if 11 <= tail <= 14:
        return forms[2]
    last = value % 10
    return forms[0] if last == 1 else forms[1] if 2 <= last <= 4 else forms[2]


def _parse_budget(value: str, maximum: int) -> int | None:
    digits = re.sub(r"\s+", "", value)
    if not digits.isdigit():
        return None
    budget = int(digits)
    return budget if 0 <= budget <= maximum else None


def _money(value: float | int | None, *, signed: bool = False) -> str:
    if value is None:
        return "Нет оценки"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{_number(value)} у.е."


def _tariff(code: object) -> str:
    match = re.fullmatch(r"tariff_(\d+)", str(code or ""))
    return f"Тариф {match.group(1)}" if match else str(code or "Не указан")


def _channel(code: object) -> str:
    return CHANNEL_LABELS.get(str(code), "Канал не указан")


def _segment(campaign: dict) -> str:
    pieces = []
    if campaign.get("arpu_segment"):
        level = ARPU_LABELS.get(str(campaign["arpu_segment"]), str(campaign["arpu_segment"]))
        pieces.append(f"уровень выручки: {level.lower()}")
    if campaign.get("data_segment"):
        pieces.append(DATA_LABELS.get(str(campaign["data_segment"]), str(campaign["data_segment"])).lower())
    if campaign.get("call_segment"):
        pieces.append(CALL_LABELS.get(str(campaign["call_segment"]), str(campaign["call_segment"])).lower())
    return " · ".join(pieces) if pieces else "Без дополнительных фильтров"


def _metric(label: str, value: str, help_text: str = "") -> str:
    return (
        '<div class="metric">'
        f'<div class="metric-label">{_text(label)}</div>'
        f'<div class="metric-value">{_text(value)}</div>'
        f'<div class="metric-help">{_text(help_text)}</div>'
        "</div>"
    )


def _hero_html(overview: dict, result: dict | None) -> str:
    ready = result is not None
    if ready:
        main_value = _money(result["agent_estimated_final_net"], signed=True)
        main_label = "ПРОГНОЗ ФИНАЛЬНЫХ КАМПАНИЙ"
        art_note = "После затрат на финальные контакты · без пилотов"
        state = "План рассчитан"
    else:
        main_value = _number(overview["audience_count"])
        main_label = "АБОНЕНТОВ В ТЕКУЩЕЙ БАЗЕ"
        art_note = "Основа для проверки гипотез"
        state = "Готов к анализу"
    return (
        '<div class="hero">'
        '<div class="hero-grid"><div class="hero-copy">'
        '<div class="hero-meta"><span class="brand-dot"></span>ШТАБ ТАРИФНЫХ КАМПАНИЙ'
        f'<span class="hero-live">{_text(state)}</span></div>'
        '<h1>AI-оптимизатор<br><span>маркетинговых кампаний</span></h1>'
        '<p class="hero-sub">Интеллектуальный подбор и оптимизация тарифных кампаний</p>'
        '<p class="hero-desc">От абонентской базы и пилотных наблюдений — '
        'к проверенному плану действий с учётом стоимости и лимитов.</p>'
        '<div class="hero-steps"><span>01 Анализ</span><i></i><span>02 Пилоты</span>'
        '<i></i><span>03 Портфель</span></div>'
        '</div><div class="hero-visual" aria-label="Ключевой показатель сценария">'
        '<div class="hero-orbit hero-orbit-outer"></div>'
        '<div class="hero-orbit hero-orbit-inner"></div>'
        '<div class="hero-pulse"></div>'
        '<div class="hero-number">'
        f'<div class="hero-number-label">{_text(main_label)}</div>'
        f'<strong>{_text(main_value)}</strong>'
        f'<span>{_text(art_note)}</span>'
        '</div></div></div>'
        '<div class="hero-footer"><span>ЛОКАЛЬНАЯ СРЕДА</span>'
        '<span>Синтетические данные кейса</span></div>'
        '</div>'
    )


def _metrics_html(overview: dict, result: dict | None) -> str:
    if result is None:
        segmented = sum(int(item["audience_count"]) for item in overview["segment_map"])
        items = [
            ("Профилей с тарифом и сегментом", _number(segmented), f"Из {_number(overview['audience_count'])} в базе"),
            ("Доступный бюджет", _money(overview["max_budget"]), "Верхний предел по кейсу"),
            ("Тарифов для выбора", _number(overview["tariff_count"]), "Справочник тарифов"),
            ("Лимит контактов", _number(overview["max_contacts"]), "Пилоты и итоговый план вместе"),
        ]
    else:
        totals = result["totals"]
        target = sum(item["new_unique_customers"] for item in result["campaigns"])
        items = [
            ("Выбрано кампаний", _number(totals["campaign_count"]), "Не более 10"),
            ("Адресатов итогового плана", _number(target), "Без повторов внутри плана"),
            ("Использовано бюджета", _money(totals["budget_used"]), f"Из {_money(totals['budget'])}"),
            ("Всего контактов", _number(totals["contacts_used"]), f"Из {_number(totals['contacts_limit'])}, включая пилоты"),
        ]
    css = "metrics result" if result is not None else "metrics"
    return f'<div class="{css}">' + "".join(_metric(*item) for item in items) + "</div>"


def _timeline_html(
    active: int | None,
    *,
    finished: bool,
    pilot_count: int = 0,
    hypothesis_count: int = 0,
    rejected_count: int = 0,
) -> str:
    cells = []
    for index, label in enumerate(STAGES):
        if finished or (active is not None and index < active):
            css, status = "done", "Завершено"
        elif active == index:
            css, status = "active", "Выполняется"
        else:
            css, status = "", "Ожидает"
        if index == 1 and hypothesis_count:
            status += f" · гипотез: {_number(hypothesis_count)}"
        if index == 2 and pilot_count:
            status += f" · пилотов: {_number(pilot_count)}"
        if index == 3 and rejected_count:
            status += f" · отклонено: {_number(rejected_count)}"
        cells.append(
            f'<div class="timeline-step {css}"><div class="timeline-index">ЭТАП {index + 1}</div>'
            f'<div class="timeline-name">{_text(label)}</div>'
            f'<div class="timeline-status">{_text(status)}</div></div>'
        )
    return '<div class="timeline">' + "".join(cells) + "</div>"


def _portfolio_summary(result: dict) -> str:
    totals = result["totals"]
    used_pct = 0 if totals["budget"] == 0 else min(100, totals["budget_used"] / totals["budget"] * 100)
    pilot_pct = 0 if totals["budget"] == 0 else min(100, totals["pilot_cost"] / totals["budget"] * 100)
    final_pct = 0 if totals["budget"] == 0 else min(100 - pilot_pct, totals["final_cost"] / totals["budget"] * 100)
    contact_pct = totals["contacts_used"] / totals["contacts_limit"] * 100
    allocated_pct = pilot_pct + final_pct
    budget_gradient = (
        f"conic-gradient(#272a2d 0 {pilot_pct:.2f}%, "
        f"#f8d335 {pilot_pct:.2f}% {allocated_pct:.2f}%, "
        f"#e8ebe8 {allocated_pct:.2f}% 100%)"
    )
    return (
        '<div class="portfolio-board">'
        f'<div class="budget-ring" style="background:{budget_gradient}" '
        f'aria-label="Использовано {_number(used_pct)} процентов бюджета">'
        '<div class="budget-ring-core">'
        f'<strong>{_number(used_pct)}%</strong><span>бюджета использовано</span></div></div>'
        '<div><div class="portfolio-board-title">Куда распределён бюджет сценария</div>'
        '<div class="budget-breakdown">'
        '<div class="budget-part" style="--part-color:#272a2d">'
        '<span class="budget-part-label">Пилотные проверки</span>'
        f'<div class="budget-part-value">{_text(_money(totals["pilot_cost"]))}</div></div>'
        '<div class="budget-part" style="--part-color:#f8d335">'
        '<span class="budget-part-label">Финальные кампании</span>'
        f'<div class="budget-part-value">{_text(_money(totals["final_cost"]))}</div></div>'
        '<div class="budget-part" style="--part-color:#e8ebe8">'
        '<span class="budget-part-label">Остаток</span>'
        f'<div class="budget-part-value">{_text(_money(max(0, totals["budget_remaining"])))}</div></div>'
        '</div><div class="portfolio-contacts">'
        f'<div class="bar-shell" aria-label="Использовано контактов {contact_pct:.1f} процента">'
        f'<div class="bar-used" style="width:{contact_pct:.2f}%"></div></div>'
        f'<div class="bar-meta"><span>Контакты: пилоты {_number(totals["pilot_contacts"])} · '
        f'кампании {_number(totals["final_contacts"])}</span>'
        f'<span>{_number(totals["contacts_used"])} из {_number(totals["contacts_limit"])}</span>'
        '</div></div></div></div>'
    )


def _decision_flow_html(result: dict) -> str:
    hypotheses = next(
        (int(event["n_customers"]) for event in result["events"] if event.get("stage") == "hypotheses"),
        0,
    )
    totals = result["totals"]
    values = [
        ("Абонентская база", _number(totals["audience_count"]), "входные профили"),
        ("Гипотезы", _number(hypotheses), "сформировано агентом"),
        ("Пилоты", _number(totals["pilot_count"]), "проверено на текущей базе"),
        ("План действий", _number(totals["campaign_count"]), "финальные кампании"),
    ]
    return '<div class="decision-flow">' + ''.join(
        '<div class="flow-step">'
        f'<div class="flow-label">{_text(label)}</div>'
        f'<div class="flow-value">{_text(value)}</div>'
        f'<div class="flow-note">{_text(note)}</div></div>'
        for label, value, note in values
    ) + '</div>'


def _audience_chart_html(overview: dict) -> str:
    counts = {level: 0 for level in ("LOW", "MID", "HIGH")}
    for cell in overview["segment_map"]:
        level = cell["arpu_segment"]
        if level in counts:
            counts[level] += int(cell["audience_count"])
    labelled = sum(counts.values())
    rows = []
    for level in ("LOW", "MID", "HIGH"):
        count = counts[level]
        share = count / labelled * 100 if labelled else 0
        rows.append(
            '<div class="audience-row">'
            f'<span>{_text(ARPU_LABELS[level])} уровень</span>'
            f'<div class="audience-track"><div class="audience-fill" style="width:{share:.2f}%"></div></div>'
            f'<span class="audience-count">{_number(count)}</span></div>'
        )
    return (
        '<div class="audience-chart">'
        '<div class="section-eyebrow">Структура базы</div>'
        '<div style="font-weight:700;margin-bottom:4px">Абоненты по уровню выручки</div>'
        + "".join(rows)
        + f'<div class="evidence-note">Показано {_number(labelled)} профилей '
        f'из {_number(overview["audience_count"])} с заполненными тарифом и сегментом.</div>'
        "</div>"
    )


def _pilot_chart_html(pilots: list[dict]) -> str:
    if not pilots:
        return ""
    shown = pilots[:8]
    maximum = max(0.10, max(abs(float(item["observed_lift_ratio"])) for item in pilots))
    rows = []
    for index, pilot in enumerate(shown, start=1):
        ratio = float(pilot["observed_lift_ratio"])
        half_width = min(50.0, abs(ratio) / maximum * 47.0)
        positive = ratio >= 0
        left = "50%" if positive else f"calc(50% - {half_width:.2f}%)"
        css = "positive" if positive else "negative"
        sign = "+" if ratio > 0 else ""
        rows.append(
            '<div class="pilot-row">'
            f'<span class="pilot-label">№{index} · {_text(_tariff(pilot["target_tariff"]))}</span>'
            '<div class="pilot-track">'
            f'<div class="pilot-bar {css}" style="left:{left};width:{half_width:.2f}%"></div>'
            '</div>'
            f'<span class="pilot-value">{sign}{ratio * 100:.1f} %</span>'
            '</div>'
        )
    caption = (
        f"Показаны первые {_number(len(shown))} из {_number(len(pilots))} пилотов."
        if len(pilots) > len(shown)
        else f"Показаны все {_number(len(pilots))} пилотов."
    )
    return (
        '<div class="pilot-chart">'
        '<div style="font-weight:700;margin-bottom:5px">Наблюдаемый относительный эффект</div>'
        + "".join(rows)
        + f'<div class="evidence-note">{_text(caption)} '
        "Зелёный — положительное наблюдение, красный — отрицательное; шкала общая."
        "</div></div>"
    )


def _campaign_card(campaign: dict, index: int, pilots: list[dict]) -> str:
    forecast = campaign["expected_net"]
    matching_pilots = [
        pilot for pilot in pilots
        if pilot.get("current_tariff") == campaign.get("current_tariff")
        and pilot.get("arpu_segment") == campaign.get("arpu_segment")
        and pilot.get("target_tariff") == campaign.get("target_tariff")
    ]
    evidence = (
        f"Гипотеза проверена на текущей аудитории: {_number(len(matching_pilots))} "
        f"{_plural(len(matching_pilots), ('пилот', 'пилота', 'пилотов'))}. "
        if matching_pilots else ""
    )
    reason = (
        evidence
        + "После пилотов консервативная оценка эффекта для этого перехода положительна. "
        f"Модельный прирост {_money(campaign['expected_lift'])} превышает стоимость контактов "
        f"{_money(campaign['cost'])}"
        if forecast is not None
        else "Надёжно положительный вариант не найден. Агент выбрал небольшой сегмент и бесплатный "
             "канал, чтобы ограничить возможный ущерб при обязательном итоговом плане."
    )
    stats = [
        ("Размер сегмента", _number(campaign["matched"])),
        ("Контакты по плану", _number(campaign["contacts"])),
        ("Стоимость контактов", _money(campaign["cost"])),
    ]
    return (
        '<div class="campaign-card">'
        '<div class="campaign-card-top">'
        f'<div class="campaign-card-kicker">РЕКОМЕНДАЦИЯ {index:02d}</div>'
        '<div class="campaign-route">'
        f'<span>{_text(_tariff(campaign["current_tariff"]))}</span>'
        '<span class="arrow">→</span>'
        f'<strong>{_text(_tariff(campaign["target_tariff"]))}</strong></div>'
        f'<div class="campaign-segment">{_text(_segment(campaign))}</div></div>'
        '<div class="campaign-economics"><div>'
        '<div class="campaign-economics-label">Прогноз кампании после затрат</div>'
        f'<strong>{_text(_money(forecast, signed=True) if forecast is not None else "Нет надёжной оценки")}</strong>'
        '</div>'
        f'<span class="campaign-channel">{_text(_channel(campaign["channel"]))}</span></div>'
        '<div class="campaign-stats">'
        + ''.join(
            '<div class="campaign-stat">'
            f'<div class="campaign-stat-label">{_text(label)}</div>'
            f'<div class="campaign-stat-value">{_text(value)}</div></div>'
            for label, value in stats
        )
        + '</div>'
        f'<div class="campaign-why"><b>Почему выбрана:</b> {_text(reason)}</div>'
        f'<div class="campaign-foot">Повторных контактов внутри финального плана: '
        f'{_number(campaign["repeat_contacts"])}. Прогноз не включает затраты и эффект пилотов.</div>'
        '</div>'
    )


def _explanation_html(result: dict) -> str:
    totals = result["totals"]
    rejected_count = len(result["rejected"])
    if result["agent_estimated_final_net"] is None:
        return (
            '<div class="insight-grid">'
            '<div class="insight-card"><div class="insight-kicker">РЕЗУЛЬТАТ ПРОВЕРКИ</div>'
            '<strong>Надёжно прибыльный портфель не найден</strong>'
            f'<p>Агент провёл {_number(totals["pilot_count"])} '
            f'{_plural(totals["pilot_count"], ("пилот", "пилота", "пилотов"))} '
            'на текущей базе.</p></div>'
            '<div class="insight-card"><div class="insight-kicker">МАЛОЗАТРАТНЫЙ ПЛАН</div>'
            '<strong>Бесплатный канал</strong><p>Небольшая финальная кампания '
            'ограничивает дополнительные расходы и выполняет требование кейса о наличии плана.</p></div>'
            '<div class="insight-card"><div class="insight-kicker">ОТСЕВ ГИПОТЕЗ</div>'
            f'<strong>{_number(rejected_count)} отклонено</strong>'
            '<p>Консервативная оценка этих переходов после пилотов не стала положительной.</p>'
            '</div></div>'
        )
    ranked = [item for item in result["campaigns"] if item["expected_net"] is not None]
    if ranked:
        top = max(ranked, key=lambda item: float(item["expected_net"]))
        route = f"{_tariff(top['current_tariff'])} → {_tariff(top['target_tariff'])}"
        leader_note = (
            f"{_segment(top)} · {_channel(top['channel'])}. "
            f"Прогноз {_money(top['expected_net'], signed=True)} после стоимости контактов "
            f"{_money(top['cost'])}"
        )
    else:
        route = "Портфель после пилотов"
        leader_note = "Выбранные сегменты прошли консервативную оценку эффекта."
    if totals["campaign_count"] >= 10:
        limit_note = "Достигнут лимит в 10 финальных кампаний."
    elif totals["contacts_used"] >= totals["contacts_limit"]:
        limit_note = "Достигнут общий лимит контактов."
    else:
        limit_note = (
            f"Остаток бюджета {_money(max(0, totals['budget_remaining']))}: "
            "дополнительные гипотезы не прошли отбор."
        )
    return (
        '<div class="insight-grid">'
        '<div class="insight-card"><div class="insight-kicker">ПРОВЕРКА ГИПОТЕЗ</div>'
        f'<strong>{_number(totals["pilot_count"])} пилотов</strong>'
        f'<p>После проверки выбрано {_number(totals["campaign_count"])} '
        f'{_plural(totals["campaign_count"], ("кампания", "кампании", "кампаний"))} '
        'с положительной консервативной оценкой эффекта.</p></div>'
        '<div class="insight-card"><div class="insight-kicker">ЛИДЕР ПОРТФЕЛЯ</div>'
        f'<strong>{_text(route)}</strong><p>{_text(leader_note)}</p></div>'
        '<div class="insight-card"><div class="insight-kicker">КОНТРОЛЬ РИСКА</div>'
        f'<strong>{_number(rejected_count)} гипотез отклонено</strong>'
        '<p>Нижняя оценка их эффекта после пилотов не превышает нуля.</p></div>'
        '</div>'
        f'<div class="insight-foot">{_text(limit_note)}</div>'
    )


def _scenario_comparison_html(previous: dict, result: dict) -> str:
    old = previous["totals"]
    new = result["totals"]
    old_forecast = previous["agent_estimated_final_net"]
    new_forecast = result["agent_estimated_final_net"]
    deltas = [
        ("Лимит бюджета", _money(new["budget"] - old["budget"], signed=True)),
        ("Расходы", _money(new["budget_used"] - old["budget_used"], signed=True)),
        ("Контакты", f"{new['contacts_used'] - old['contacts_used']:+,}".replace(",", " ")),
    ]
    if old_forecast is not None and new_forecast is not None:
        deltas.insert(0, ("Прогноз финальных кампаний", _money(new_forecast - old_forecast, signed=True)))
    old_plan = [(item["filters"], item["target_tariff"], item["channel"]) for item in previous["campaigns"]]
    new_plan = [(item["filters"], item["target_tariff"], item["channel"]) for item in result["campaigns"]]
    composition = (
        f"Состав финального плана не изменился. Новый лимит покрывает его расходы "
        f"{_money(new['budget_used'])}"
        if old_plan == new_plan else
        "Состав финального плана изменился после повторного запуска пилотов и оптимизации."
    )
    return (
        '<div class="scenario-grid">'
        '<div class="scenario-box"><div class="scenario-title">Предыдущий сценарий</div>'
        f'<div class="scenario-value">Бюджет: {_text(_money(old["budget"]))}<br>'
        f'Кампаний: {_number(old["campaign_count"])}<br>'
        f'Прогноз финальных кампаний: {_text(_money(old_forecast, signed=True))}</div></div>'
        '<div class="scenario-box"><div class="scenario-title">Новый сценарий</div>'
        f'<div class="scenario-value">Бюджет: {_text(_money(new["budget"]))}<br>'
        f'Кампаний: {_number(new["campaign_count"])}<br>'
        f'Прогноз финальных кампаний: {_text(_money(new_forecast, signed=True))}</div></div>'
        '</div><div class="scenario-delta">'
        + ''.join(
            f'<span>{_text(label)}: {_text(value)}</span>'
            for label, value in deltas
        )
        + f'</div><div class="evidence-note">{_text(composition)} '
        'Прогнозы относятся только к финальным кампаниям; результаты пилотов могут отличаться между запусками.'
        '</div>'
    )


def _rejected_title(item: dict) -> str:
    segment = ARPU_LABELS.get(str(item.get("arpu_segment")), "Не указан")
    return (
        f"{_tariff(item.get('current_tariff'))} · уровень выручки: {segment.lower()} "
        f"→ {_tariff(item.get('target_tariff'))}"
    )


def _export_csv(result: dict) -> bytes:
    rows = []
    for campaign in result["campaigns"]:
        row = {"campaign_name": campaign["campaign_name"], **campaign["filters"]}
        row.update({"target_tariff": campaign["target_tariff"], "channel": campaign["channel"]})
        rows.append(row)
    buffer = StringIO()
    pd.DataFrame(rows, columns=CAMPAIGN_COLUMNS).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")


try:
    overview = get_overview()
except Exception:
    LOGGER.exception("Не удалось загрузить исходные данные")
    st.error("Не удалось загрузить исходные данные проекта. Проверьте файлы в корне проекта и папке с данными.")
    st.stop()
result = st.session_state.get("current_scenario")

hero_slot = st.empty()
hero_slot.markdown(_hero_html(overview, result), unsafe_allow_html=True)

kpi_slot = st.empty()
kpi_slot.markdown(_metrics_html(overview, result), unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<div class="section-eyebrow">Управление планом</div>', unsafe_allow_html=True)
    st.markdown("### Сформировать план кампаний")
    st.markdown(
        '<span class="goal-chip">Цель: максимизация чистого финансового эффекта</span>'
        + (
            '<div class="evidence-note" style="margin-top:8px">Это единственная цель, '
            'которую прямо поддерживает скоринг кейса.</div>'
            if result is None else ""
        ),
        unsafe_allow_html=True,
    )
    budget_col, rules_col = st.columns([1.0, 1.5], gap="large")
    with budget_col:
        budget_text = st.text_input(
            "Бюджет кампаний, у.е.",
            value=_number(overview["max_budget"]),
            key="scenario_budget_text",
        )
        budget = _parse_budget(budget_text, int(overview["max_budget"]))
        st.caption("Предел по ТЗ — 100 000 у.е. на пилоты и финальные кампании вместе.")
    with rules_col:
        st.markdown(
            '<div style="padding-top:9px"><span class="rule-chip">Максимум кампаний: 10</span>'
            '<span class="rule-chip">Пилотов: до 20</span>'
            '<span class="rule-chip">Контактов: до 15 000</span></div>',
            unsafe_allow_html=True,
        )
    button_label = (
        "Пересчитать план"
        if result is not None and budget is not None and budget != int(result["totals"]["budget"])
        else "Сформировать оптимальный план"
    )
    run_clicked = st.button(button_label, type="primary", disabled=budget is None)

if budget is None:
    st.error(f"Введите целый бюджет от 0 до {_number(overview['max_budget'])} у.е.")
elif result is not None and budget != int(result["totals"]["budget"]) and not run_clicked:
    st.info(
        f"Сейчас показан план для бюджета {_money(result['totals']['budget'])} "
        "Нажмите «Пересчитать план», чтобы применить новое значение."
    )

progress_slot = st.empty()
if run_clicked:
    if result is not None:
        st.session_state["previous_scenario"] = result
    else:
        st.session_state.pop("previous_scenario", None)

    progress_state = {
        "last_stage": -1,
        "pilots_seen": 0,
        "hypotheses_seen": 0,
        "rejected_seen": 0,
    }

    def _on_event(event: dict) -> None:
        stage = STAGE_INDEX.get(str(event.get("stage")), progress_state["last_stage"])
        if event.get("stage") == "hypotheses":
            progress_state["hypotheses_seen"] = int(event.get("n_customers", 0))
        if event.get("action") == "pilot and posterior update":
            progress_state["pilots_seen"] += 1
        if str(event.get("action", "")).startswith("rejected: "):
            progress_state["rejected_seen"] += 1
        if (
            stage != progress_state["last_stage"]
            or event.get("action") == "pilot and posterior update"
            or str(event.get("action", "")).startswith("rejected: ")
        ):
            progress_state["last_stage"] = stage
            progress_slot.markdown(
                '<div class="section-eyebrow">Работа агента</div>'
                '<h2 style="margin-bottom:4px">План формируется по текущим данным</h2>'
                + _timeline_html(
                    stage, finished=False,
                    pilot_count=progress_state["pilots_seen"],
                    hypothesis_count=progress_state["hypotheses_seen"],
                    rejected_count=progress_state["rejected_seen"],
                ),
                unsafe_allow_html=True,
            )

    try:
        result = run_scenario(budget, seed=42, on_event=_on_event)
    except Exception:
        LOGGER.exception("Не удалось сформировать план")
        st.error("Не удалось сформировать план. Проверьте бюджет и наличие исходных данных проекта.")
        st.stop()
    st.session_state["current_scenario"] = result
    hero_slot.markdown(_hero_html(overview, result), unsafe_allow_html=True)
    kpi_slot.markdown(_metrics_html(overview, result), unsafe_allow_html=True)

if result is None:
    st.markdown("### Как система решает задачу")
    st.markdown(
        '<p class="section-intro">Агент исследует историю как предварительное предположение, '
        'проверяет гипотезы на текущей аудитории и выбирает план с учётом стоимости каналов и лимитов.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(_audience_chart_html(overview), unsafe_allow_html=True)
    st.markdown(_timeline_html(None, finished=False), unsafe_allow_html=True)
    st.markdown(
        '<div class="evidence-note">Пилоты расходуют контакты и бюджет. '
        'Их наблюдения зашумлены; итоговые решения учитывают эту неопределённость.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

progress_slot.markdown(
    '<div class="section-eyebrow">Работа агента</div>'
    '<h2>Как сформирован план</h2>'
    '<p class="section-intro">Этапы отражают события настоящего запуска агента; '
    'пилоты и проверка ограничений уже выполнены.</p>'
    + _timeline_html(
        5,
        finished=True,
        pilot_count=result["totals"]["pilot_count"],
        hypothesis_count=next(
            (int(event["n_customers"]) for event in result["events"] if event.get("stage") == "hypotheses"),
            0,
        ),
        rejected_count=len(result["rejected"]),
    ),
    unsafe_allow_html=True,
)

st.markdown("### От данных к решению")
st.markdown(_decision_flow_html(result), unsafe_allow_html=True)

st.markdown("### Рекомендованный портфель кампаний")
st.markdown(
    '<p class="section-intro">Финальный план построен в пределах выбранного бюджета. '
    'Прогноз агента относится к финальным кампаниям и не включает эффект пилотов.</p>',
    unsafe_allow_html=True,
)
with st.container(border=True):
    st.markdown(_portfolio_summary(result), unsafe_allow_html=True)

previous = st.session_state.get("previous_scenario")
if previous is not None and previous["totals"]["budget"] != result["totals"]["budget"]:
    st.markdown("### Сценарный анализ")
    st.markdown(_scenario_comparison_html(previous, result), unsafe_allow_html=True)

st.markdown(
    '<div class="campaign-cards">'
    + ''.join(_campaign_card(campaign, number, result["pilots"])
              for number, campaign in enumerate(result["campaigns"], start=1))
    + '</div>',
    unsafe_allow_html=True,
)

download_col, note_col = st.columns([1, 2], gap="large")
with download_col:
    st.download_button(
        "Скачать текущий план",
        data=_export_csv(result),
        file_name="plan_campaigns.csv",
        mime="text/csv",
        type="primary",
    )
with note_col:
    st.caption(
        "Файл содержит исходные коды тарифов и каналов в формате проверки кейса. "
        "Для официальной сдачи используйте файл результата, создаваемый командой из инструкции проекта."
    )

st.markdown("### Почему выбран именно этот план")
st.markdown(_explanation_html(result), unsafe_allow_html=True)

st.markdown("### Что показали пилоты")
pilot_count = len(result["pilots"])
positive_count = sum(float(item["observed_lift_ratio"]) > 0 for item in result["pilots"])
st.markdown(
    f'<p class="section-intro">Проведено {_number(pilot_count)} '
    f'{_plural(pilot_count, ("пилот", "пилота", "пилотов"))}; '
    f'у {_number(positive_count)} наблюдаемый эффект положительный. '
    'Это зашумлённые измерения, а не гарантированный результат для всего сегмента.</p>',
    unsafe_allow_html=True,
)
st.markdown(_pilot_chart_html(result["pilots"]), unsafe_allow_html=True)
with st.expander("Показать наблюдения пилотов"):
    for index, pilot in enumerate(result["pilots"], start=1):
        ratio = float(pilot["observed_lift_ratio"]) * 100
        sign = "+" if ratio > 0 else ""
        st.markdown(
            f"**Пилот №{index}.** {_tariff(pilot.get('current_tariff'))} → "
            f"{_tariff(pilot['target_tariff'])}; {_channel(pilot['channel'])}; "
            f"{_number(pilot['n_customers'])} абонентов; "
            f"наблюдаемый эффект {sign}{ratio:.1f} %."
        )

with st.expander("Структура абонентской базы"):
    st.markdown(_audience_chart_html(overview), unsafe_allow_html=True)

st.markdown("### Отклонённые варианты")
if result["rejected"]:
    st.markdown(
        '<p class="section-intro">Эти гипотезы прошли пилотную проверку, '
        'но их консервативная оценка эффекта не стала положительной.</p>',
        unsafe_allow_html=True,
    )
    for item in result["rejected"][:5]:
        st.markdown(
            '<div class="reject-card">'
            f'<div class="reject-title">{_text(_rejected_title(item))}</div>'
            '<div class="reject-reason">Не включено: нижняя оценка эффекта после пилота '
            'не превышает нуля.</div></div>',
            unsafe_allow_html=True,
        )
    if len(result["rejected"]) > 5:
        st.caption(f"Показано 5 из {_number(len(result['rejected']))} отклонённых гипотез.")
else:
    st.info("В журнале этого запуска нет гипотез, отклонённых по консервативной оценке.")

st.markdown("### Локальная проверка результата")
mock = result["mock_result"]
with st.container(border=True):
    st.markdown(
        f'<div class="portfolio-strip">'
        f'<div class="strip-item"><div class="strip-label">Чистый прирост в тестовой среде</div>'
        f'<div class="strip-value">{_text(_money(mock["net_arpu_gain"], signed=True))}</div></div>'
        f'<div class="strip-item"><div class="strip-label">Контактов с учётом пилотов</div>'
        f'<div class="strip-value">{_number(mock["total_contacts"])}</div></div>'
        f'<div class="strip-item"><div class="strip-label">Затраты на все контакты</div>'
        f'<div class="strip-value">{_text(_money(mock["total_cost"]))}</div></div>'
        f'<div class="strip-item"><div class="strip-label">Уникальных адресатов</div>'
        f'<div class="strip-value">{_number(mock["unique_customers_targeted"])}</div></div>'
        "</div>"
        '<div class="evidence-note">Это результат официального скорера на локальной мок-модели '
        'после завершения работы агента. Скрытые эффекты судейской среды неизвестны; '
        'число выше не является прогнозом конкурсного результата.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="evidence-note" style="margin-top:20px">Все данные в этом интерфейсе синтетические. '
    'Сервис не отправляет кампании абонентам и не использует внешние AI-сервисы.</div>',
    unsafe_allow_html=True,
)
