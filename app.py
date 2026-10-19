"""Interactive dashboard for the Cynbit-Nexus proof of concept."""
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.retrieval import HybridRRFRetriever
from src.router import CalibratedSemanticRouter
from src.simulator import BetaPertMonteCarloSimulator, HallucinationAuditor


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Cynbit-Nexus | AI Systems & Delivery Engine", layout="wide")


@st.cache_resource
def init_engines():
    """Load local corpus and fit the deterministic demo models once per process."""
    with (ROOT / "data" / "corpus.json").open(encoding="utf-8") as file:
        data = json.load(file)
    retriever = HybridRRFRetriever(data["projects"])
    router = CalibratedSemanticRouter()
    cv_stats = router.train_and_evaluate_cv()
    simulator = BetaPertMonteCarloSimulator(n_trials=5000)
    return data, retriever, router, simulator, cv_stats


data, retriever, router, simulator, cv_stats = init_engines()
st.title("Cynbit-Nexus: Hybrid Retrieval, Query Routing & Monte Carlo Scoping")
st.caption("AI/ML engineering proof of concept. Metrics are synthetic demonstrations, not production guarantees.")

with st.sidebar:
    st.header("Engine Controls & Telemetry")
    retrieval_mode = st.selectbox("Retrieval Mode", ["hybrid_rrf", "bm25", "dense"], index=0)
    scale_multiplier = st.slider("System Scale Multiplier", 0.7, 1.8, 1.0, 0.1)
    strict_compliance = st.checkbox("Apply security/compliance effort buffer", value=True)
    st.divider()
    st.markdown("**Synthetic 5-fold router evaluation:**")
    st.json(cv_stats)

presets = {
    item["expected_id"] + ": " + item["query"][:65] + "...": item["query"]
    for item in data["eval_queries"]
}
choice = st.selectbox("Load sample client RFP / engineering query:", list(presets.keys()))
query_text = st.text_area("Client requirements / technical prompt:", value=presets[choice], height=95)

if st.button("Run retrieval + routing + 5,000-trial Monte Carlo", type="primary"):
    start_time = time.perf_counter()
    retrieved = retriever.search(query_text, top_k=3, mode=retrieval_mode)
    route_result = router.route(query_text)
    monte_carlo = simulator.simulate(
        retrieved, scale_factor=scale_multiplier, strict_compliance=strict_compliance
    )
    audit = HallucinationAuditor.verify_synthesis(retrieved, monte_carlo)
    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("P80 Scope", f"{monte_carlo['p80_hours']} hrs", f"+{monte_carlo['var_buffer_hours']}h buffer")
    col2.metric("P95 Scope", f"{monte_carlo['p95_hours']} hrs", f"{monte_carlo['sprints_2wk']} two-week sprints")
    col3.metric("P(complex | query)", str(route_result["p_complex"]), route_result["selected_tier"].split(":")[0])
    col4.metric("Lexical grounding", f"{audit['entity_attribution_precision'] * 100:.0f}%", audit["grounding_status"])
    col5.metric("Pipeline time", f"{elapsed_ms} ms", "cache hit" if route_result["cache_hit"] else "evaluated")

    tab_scope, tab_retrieval, tab_audit = st.tabs([
        "1. Monte Carlo Scope", "2. Retrieval Diagnostics", "3. Grounding & Router"
    ])
    with tab_scope:
        left, right = st.columns([3, 2])
        with left:
            st.dataframe(pd.DataFrame(monte_carlo["subsystems"]), use_container_width=True, hide_index=True)
            st.info(
                f"The simulated median is {monte_carlo['p50_hours']}h; its estimated overrun probability is "
                f"{monte_carlo['naive_overrun_probability_pct']}%. P80 is {monte_carlo['p80_hours']}h. "
                "Validate estimates with real delivery data before quoting a client."
            )
        with right:
            histogram = pd.DataFrame(
                list(monte_carlo["histogram"].items()), columns=["Hours bucket", "Trials"]
            )
            st.bar_chart(histogram.set_index("Hours bucket"))
    with tab_retrieval:
        st.dataframe(pd.DataFrame([
            {
                "Rank": item["rank"],
                "Project ID": item["project_id"],
                "Title": item["title"],
                "Fusion score": item["rrf_score"],
                "BM25 score/rank": f"{item['bm25_score']} (#{item['bm25_rank']})",
                "Dense score/rank": f"{item['dense_score']} (#{item['dense_rank']})",
            }
            for item in retrieved
        ]), use_container_width=True, hide_index=True)
    with tab_audit:
        st.markdown(f"**Grounded architecture summary**\n\n{audit['blueprint_summary']}")
        st.markdown("**Recommended stack:** " + " • ".join(f"`{item}`" for item in audit["recommended_stack"]))
        st.json({"router_decision": route_result, "grounding_audit": audit})
