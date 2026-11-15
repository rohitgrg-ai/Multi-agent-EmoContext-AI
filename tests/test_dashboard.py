"""Dashboard tests: architecture boundary, client logic, and rendering with Streamlit's AppTest.

The API responses used here are produced by the real FastAPI app (TestClient) running the
real graph with a fake predictor and a fake LLM, so they follow the actual API contract.
The dashboard's HTTP calls are then answered with those responses by patching httpx.
"""
import ast
from pathlib import Path

import httpx
import pytest
from conftest import GOOD_CONTEXT, GOOD_INTENT, GOOD_RESPONSE, FakeClient, FakePredictor, build_prediction
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import dashboard_client as client
from agents.graph import build_graph
from agents.llm import NullClient
from api.main import Services, create_app

SRC = Path(__file__).resolve().parents[1] / "src"
DASHBOARD = str(SRC / "dashboard.py")
SARCASTIC = build_prediction({"anger": 0.8, "sadness": 0.15, "joy": 0.05}, before_sarcasm="joy", p_sarcastic=0.99)

HEALTH_GEMMA = {"status": "ok", "model_loaded": True, "model_error": None,
                "llm": {"provider": "ollama", "model": "gemma4:latest", "available": True, "reason": ""}}
HEALTH_NULL = {"status": "ok", "model_loaded": True, "model_error": None,
               "llm": {"provider": "none", "model": None, "available": False, "reason": "Ollama not reachable: ConnectError"}}
HEALTH_NO_MODEL = {"status": "degraded", "model_loaded": False, "model_error": "trained model not found",
                   "llm": {"provider": "none", "model": None, "available": False, "reason": "LLM disabled"}}


def api_result(llm, prediction=SARCASTIC, text="I just love it when my phone dies at 2%"):
    """A real /agents/analyze JSON body, produced by the FastAPI app itself."""
    predictor = FakePredictor(prediction)
    services = Services(predictor=predictor, graph=build_graph(predictor, llm, lambda t: "joy", lambda t: "positive"),
                        llm=llm, emotion_baseline=lambda t: "joy", sentiment_baseline=lambda t: "positive")
    with TestClient(create_app(services)) as api:
        response = api.post("/agents/analyze", json={"text": text})
    assert response.status_code == 200
    return response.json()


GEMMA_RESULT = api_result(FakeClient({"ContextReading": [dict(GOOD_CONTEXT, proposed_emotion="sadness")],
                                      "IntentReading": [GOOD_INTENT], "ResponseText": [GOOD_RESPONSE]}))
FALLBACK_RESULT = api_result(NullClient())


def fake_api(monkeypatch, health=HEALTH_GEMMA, result=None, status_code=200, post_error=None, health_error=False):
    """Answer the dashboard's HTTP calls. Returns the list of POST bodies sent."""
    posted = []

    def fake_get(url, **kwargs):
        if health_error:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json=health, request=httpx.Request("GET", url))

    def fake_post(url, json=None, **kwargs):
        posted.append(json)
        if post_error:
            raise post_error
        body = result if status_code == 200 else {"detail": "Traceback (most recent call last): secret internals"}
        return httpx.Response(status_code, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)
    return posted


def html_text(app):
    return " ".join(element.proto.body for element in app.get("html"))


def run_example(app, index=1):
    app.button(key=f"example_{index}").click().run()
    return app


@pytest.fixture
def app():
    return AppTest.from_file(DASHBOARD, default_timeout=30)


# ---------- architecture boundary ----------

@pytest.mark.parametrize("filename", ["dashboard.py", "dashboard_client.py"])
def test_dashboard_never_imports_ml_agents_or_backend(filename):
    tree = ast.parse((SRC / filename).read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    forbidden = {"model", "agents", "baseline", "api", "torch", "transformers", "langgraph", "sklearn", "train"}
    assert not imported & forbidden, f"{filename} imports {imported & forbidden}"


# ---------- client logic ----------

def test_status_labels():
    assert client.describe_status(None).overall == "Offline"
    assert client.describe_status(HEALTH_NO_MODEL).overall == "Degraded"
    null = client.describe_status(HEALTH_NULL)
    assert (null.overall, null.llm_active, null.llm_label) == ("LLM unavailable", False, "Not available")
    gemma = client.describe_status(HEALTH_GEMMA)
    assert (gemma.overall, gemma.llm_active, gemma.llm_label) == ("Online", True, "ollama · gemma4:latest")


def test_status_tolerates_missing_fields():
    assert client.describe_status({}).overall == "Degraded"
    assert client.describe_status({"model_loaded": True}).overall == "LLM unavailable"


@pytest.mark.parametrize("code, expected", [(422, "between 1 and 2000"), (503, "not loaded"), (500, "internal error"),
                                            (418, "unexpected status (418)")])
def test_http_errors_become_fixed_messages(code, expected):
    response = httpx.Response(code, json={"detail": "Traceback: secret"}, request=httpx.Request("POST", "http://x"))
    result, message = client.read_response(response)
    assert result is None and expected in message and "secret" not in message


def test_unreadable_success_body():
    response = httpx.Response(200, content=b"not json", request=httpx.Request("POST", "http://x"))
    assert client.read_response(response) == (None, "The API returned a response that could not be read.")


def test_connection_failure_and_timeout(monkeypatch):
    def refuse(*args, **kwargs):
        raise httpx.ConnectError("refused")

    def slow(*args, **kwargs):
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(httpx, "post", refuse)
    assert "Cannot reach" in client.analyse_post("hi")[1]
    monkeypatch.setattr(httpx, "post", slow)
    assert "longer than" in client.analyse_post("hi")[1]
    monkeypatch.setattr(httpx, "get", refuse)
    assert client.fetch_health() is None


def test_validate_text():
    assert client.validate_text("   ") == "Type or pick a post first."
    assert "limit" in client.validate_text("x" * 2001)
    assert client.validate_text("  fine  ") is None


def test_agent_mode_labels_are_honest():
    assert client.agent_mode("emotion", False) == "Trained model"
    assert client.agent_mode("context", True, "llm") == "LLM"
    assert client.agent_mode("context", False, "fallback") == "Rule-based fallback"
    assert client.agent_mode("critic", False) == "Rules only"
    assert client.agent_mode("critic", True) == "Rules + LLM review"


def test_fallback_agents_and_source_labels():
    assert client.fallback_agents(FALLBACK_RESULT) == ["context", "intent", "response"]
    assert client.fallback_agents(GEMMA_RESULT) == []
    assert client.source_label("fallback", "ollama · gemma4:latest").startswith("Rule-based fallback")
    assert client.source_label("llm", "ollama · gemma4:latest") == "Generated by the LLM (ollama · gemma4:latest)"
    assert client.agent_output({"critic": {"approved": True}, "revision_count": 1}, "critic")["revision_count"] == 1
    assert client.top_scores(None) == []


# ---------- rendering ----------

def test_api_unavailable_shows_offline_and_disables_analysis(app, monkeypatch):
    fake_api(monkeypatch, health_error=True)
    app.run()
    assert not app.exception
    page = html_text(app)
    assert "Offline" in page
    assert app.button(key="analyze").disabled and app.button(key="example_0").disabled
    assert any("uvicorn" in block.value for block in app.code)


def test_degraded_when_model_missing(app, monkeypatch):
    fake_api(monkeypatch, health=HEALTH_NO_MODEL)
    app.run()
    assert "Degraded" in html_text(app) and app.button(key="analyze").disabled


def test_gemma_result_renders_all_six_agents_and_real_outputs(app, monkeypatch):
    posted = fake_api(monkeypatch, health=HEALTH_GEMMA, result=GEMMA_RESULT)
    run_example(app.run())
    assert not app.exception and not app.error
    page = html_text(app)
    assert posted == [{"text": "Wow, I just LOVE waiting two hours for a bus 🙄"}]
    for title in ("Emotion Agent", "Sentiment Agent", "Context Agent", "Intent Agent", "Response Agent", "Critic Agent"):
        assert title in page
    assert "ollama · gemma4:latest" in page and "Full multi-agent (LLM)" in page
    assert "Generated by the LLM (ollama · gemma4:latest)" in page
    assert GOOD_CONTEXT["implied_meaning"] in page and GOOD_RESPONSE["suggested_reply"] in page
    assert "Anger" in page                                   # RoBERTa's label
    assert "Context Agent alternative" in page and "does not replace" in page
    assert "Rule-based fallback (no LLM output)" not in page


def test_nullclient_fallback_is_never_presented_as_llm_output(app, monkeypatch):
    fake_api(monkeypatch, health=HEALTH_NULL, result=FALLBACK_RESULT)
    run_example(app.run())
    page = html_text(app)
    assert not app.exception
    assert "LLM unavailable" in page and "Fallback (rule-based)" in page and "Not available" in page
    assert "Rule-based fallback (no LLM output)" in page
    assert "Generated by the LLM" not in page and "gemma4" not in page
    assert not app.warning   # fallback was expected, so no "LLM failed" warning


def test_warns_when_llm_was_expected_but_agents_fell_back(app, monkeypatch):
    fake_api(monkeypatch, health=HEALTH_GEMMA, result=FALLBACK_RESULT)
    run_example(app.run())
    assert any("context, intent, response" in w.value for w in app.warning)


def test_empty_input_is_not_sent(app, monkeypatch):
    posted = fake_api(monkeypatch, result=GEMMA_RESULT)
    app.run()
    app.text_area(key="post_text").input("   ").run()
    app.button(key="analyze").click().run()
    assert posted == []
    assert app.error[0].value == "Type or pick a post first."


@pytest.mark.parametrize("code, expected", [(422, "rejected"), (503, "not loaded"), (500, "internal error")])
def test_api_error_codes_show_safe_messages(app, monkeypatch, code, expected):
    fake_api(monkeypatch, result=None, status_code=code)
    run_example(app.run())
    assert expected in app.error[0].value
    assert "Traceback" not in app.error[0].value and "Traceback" not in html_text(app)


def test_timeout_shows_message(app, monkeypatch):
    fake_api(monkeypatch, post_error=httpx.ReadTimeout("slow"))
    run_example(app.run())
    assert "longer than" in app.error[0].value


def test_missing_optional_fields_still_render(app, monkeypatch):
    sparse = {key: value for key, value in GEMMA_RESULT.items() if key not in ("errors", "trace")}
    sparse["emotion"] = {k: v for k, v in GEMMA_RESULT["emotion"].items() if k not in ("baseline_label", "emojis_used")}
    sparse["sentiment"] = {k: v for k, v in GEMMA_RESULT["sentiment"].items() if k != "baseline_label"}
    sparse["critic"] = {k: v for k, v in GEMMA_RESULT["critic"].items() if k != "alternative_emotion"}
    sparse["context"] = {k: v for k, v in GEMMA_RESULT["context"].items() if k != "proposed_emotion"}
    fake_api(monkeypatch, result=sparse)
    run_example(app.run())
    assert not app.exception
    page = html_text(app)
    assert "No trace was returned." in page and "not available" in page and "Context Agent alternative" not in page


REVISED_DRAFT = {"explanation": "The post reads as anger; the praise is not meant.", "suggested_reply": "That sounds frustrating."}


def revised_result(responses=(GOOD_RESPONSE, REVISED_DRAFT)):
    return api_result(FakeClient({"ContextReading": [dict(GOOD_CONTEXT, is_sarcastic=False)], "IntentReading": [GOOD_INTENT],
                                  "ResponseText": list(responses),
                                  "CriticReview": [{"approved": False, "issues": ["too casual"]}]}),
                      prediction=build_prediction({"anger": 0.6, "sadness": 0.4}, p_sarcastic=0.7))


def test_revision_is_reported_from_the_trace_without_inventing_details(app, monkeypatch):
    fake_api(monkeypatch, result=revised_result())
    run_example(app.run())
    page = html_text(app)
    assert "Response Agent (revision)" in page and "Critic Agent (revision)" in page
    # P2: the API now returns both passes, so the dashboard shows them instead of a trace summary
    assert "Revision history" in page and "Pass 1 · Draft (rejected)" in page
    assert "Pass 2 · Revised draft · final response" in page
    assert "Revision feedback sent to the Response Agent" in page and "too casual" in page
    assert GOOD_RESPONSE["explanation"] in page and REVISED_DRAFT["explanation"] in page
    assert "approved after revision" in page


def test_older_api_without_history_falls_back_to_the_trace(app, monkeypatch):
    old = {k: v for k, v in revised_result().items() if k not in ("response_history", "critic_history")}
    fake_api(monkeypatch, result=old)
    run_example(app.run())
    assert any("sent back for revision" in caption.value for caption in app.caption)
    assert "Revision history" not in html_text(app)


def test_second_rejection_shows_not_approved_outcome(app, monkeypatch):
    unnamed = {"explanation": "This is a post.", "suggested_reply": "Oh no."}
    fake_api(monkeypatch, result=revised_result(responses=(unnamed,)))
    run_example(app.run())
    page = html_text(app)
    assert "kept after revision, still not approved" in page and "Not approved (no revisions left)" in page
    assert "explanation names emotion" in page


def test_one_pass_result_shows_no_revision_history(app, monkeypatch):
    fake_api(monkeypatch, result=GEMMA_RESULT)
    run_example(app.run())
    page = html_text(app)
    assert "Revision history" not in page and "Pass 1" not in page
    assert "Rule checks" in page and "Final response." in page


def test_revision_passes_pairs_drafts_with_reviews():
    passes = client.revision_passes(revised_result())
    assert [(p["number"], p["sent_back"], p["is_final"]) for p in passes] == [(1, True, False), (2, False, True)]
    assert passes[0]["review"]["issues"] == ["too casual"]
    assert client.revision_passes(GEMMA_RESULT)[0]["is_final"] is True
    assert client.revision_passes({}) == []
