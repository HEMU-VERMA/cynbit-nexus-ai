"""Smoke and invariant tests for the Cynbit-Nexus proof of concept."""
import json
from pathlib import Path

import pytest

from src.retrieval import HybridRRFRetriever
from src.router import CalibratedSemanticRouter
from src.simulator import BetaPertMonteCarloSimulator, HallucinationAuditor

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus():
    return json.loads((ROOT / "data" / "corpus.json").read_text(encoding="utf-8"))


def test_corpus_schema_is_present(corpus):
    assert len(corpus["projects"]) >= 2
    assert len(corpus["eval_queries"]) >= 1
    ids = {project["project_id"] for project in corpus["projects"]}
    assert all(item["expected_id"] in ids for item in corpus["eval_queries"])
    for project in corpus["projects"]:
        assert project["subsystem_hours"]
        assert all(hours > 0 for hours in project["subsystem_hours"].values())


def test_retriever_finds_expected_project(corpus):
    retriever = HybridRRFRetriever(corpus["projects"])
    for item in corpus["eval_queries"]:
        results = retriever.search(item["query"], top_k=3, mode="hybrid_rrf")
        assert results
        assert results[0]["project_id"] == item["expected_id"]


def test_simulator_quantiles_are_ordered(corpus):
    retriever = HybridRRFRetriever(corpus["projects"])
    results = retriever.search(corpus["eval_queries"][0]["query"], top_k=3)
    summary = BetaPertMonteCarloSimulator(n_trials=500, seed=42).simulate(results)
    assert summary["p50_hours"] <= summary["p80_hours"] <= summary["p95_hours"]
    assert summary["sprints_2wk"] >= 1
    assert len(summary["subsystems"]) == 4


def test_router_trains_and_routes():
    router = CalibratedSemanticRouter()
    metrics = router.train_and_evaluate_cv()
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    result = router.route("Design a zero-trust OAuth2 mTLS gateway with anomaly detection.")
    assert 0.0 <= result["p_complex"] <= 1.0
    assert result["selected_tier"].startswith("Tier-")


def test_auditor_handles_two_retrieved_projects(corpus):
    retriever = HybridRRFRetriever(corpus["projects"])
    results = retriever.search(corpus["eval_queries"][0]["query"], top_k=3)
    summary = BetaPertMonteCarloSimulator(n_trials=250, seed=42).simulate(results)
    audit = HallucinationAuditor.verify_synthesis(results, summary)
    assert audit["recommended_stack"]
    assert audit["grounding_status"] in {"PASS (100% Grounded in Cynbit Corpus)", "WARN"}
