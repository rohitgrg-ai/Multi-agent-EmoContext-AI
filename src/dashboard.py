"""EmoContext AI dashboard. Start the API first, then:

    uvicorn --app-dir src api.main:app --port 8000
    streamlit run src/dashboard.py

Architecture: Streamlit dashboard -> FastAPI -> LangGraph -> six agents. This file only
sends HTTP requests (through dashboard_client) and displays the JSON that comes back.
The original single-model app, src/app.py, is unchanged and still runs on its own.
"""
from html import escape
from pathlib import Path

import streamlit as st

from dashboard_client import (AGENT_TITLES, MAX_TEXT_CHARS, START_API_COMMAND, agent_mode, agent_output,
                              analyse_post, describe_status, fallback_agents, fetch_health, revision_passes,
                              source_label, top_scores, validate_text)

# Same emotion colours as the original app (each pair meets WCAG AA contrast).
EMOTION_COLOURS = {
    "anger": ("#d62828", "#ffffff"), "fear": ("#6a2cb8", "#ffffff"), "joy": ("#ffc233", "#15213b"),
    "love": ("#ff7aa8", "#15213b"), "neutral": ("#c3c9d8", "#15213b"), "sadness": ("#2563d9", "#ffffff"),
    "surprise": ("#17c7b2", "#15213b"),
}
EXAMPLES = [
    ("Genuine happiness", "I got the job!!!"),
    ("Obvious sarcasm", "Wow, I just LOVE waiting two hours for a bus 🙄"),
    ("Subtle sarcasm, no emoji", "Thanks so much for replying to my email after three weeks, really appreciate it"),
    ("Ambiguous", "Well, that went exactly as planned."),
    ("Negative", "The landlord ignored my repair request again. I'm furious."),
    ("May need support", "I can't sleep, everything feels hopeless lately"),
]
STATUS_TONE = {"Online": "ok", "LLM unavailable": "warn", "Degraded": "warn", "Offline": "bad"}


# ---------- small HTML helpers (all API text is escaped) ----------

def pill(text, tone="neutral"):
    return f'<span class="pill {tone}">{escape(str(text))}</span>'


def emotion_chip(emotion):
    background, colour = EMOTION_COLOURS.get(emotion, ("#c3c9d8", "#15213b"))
    return f'<span class="chip" style="background:{background};color:{colour}">{escape(str(emotion).title())}</span>'


def score_bars(scores):
    rows = "".join(
        f'<div class="bar-row"><span class="bar-name">{escape(name.title())}</span>'
        f'<span class="bar-track"><span class="bar-fill" style="width:{value * 100:.1f}%;'
        f'background:{EMOTION_COLOURS.get(name, ("#8d97b0",))[0]}"></span></span>'
        f'<span class="bar-value">{value:.0%}</span></div>'
        for name, value in top_scores(scores))
    return f'<div class="bars">{rows}</div>'


def section(title, caption=None):
    st.html(f'<h2 class="section">{escape(title)}</h2>' + (f'<p class="caption">{escape(caption)}</p>' if caption else ""))


# ---------- state and actions ----------

def request_analysis():
    st.session_state.pending = True


def use_example(text):
    st.session_state.post_text = text
    st.session_state.pending = True


def run_pending_analysis():
    if not st.session_state.get("pending"):
        return
    st.session_state.pending = False
    text = st.session_state.get("post_text", "")
    problem = validate_text(text)
    if problem:
        st.session_state.result, st.session_state.error = None, problem
        return
    with st.spinner("Running the six agents… with the LLM active this usually takes 10–20 seconds."):
        result, error = analyse_post(text.strip())
    st.session_state.result, st.session_state.error = result, error
    st.session_state.analysed_text = text.strip()


# ---------- page sections ----------

def show_header():
    st.html('<header class="masthead"><h1>EmoContext AI</h1>'
            '<p class="tagline">Multi-Agent Emotional Intelligence System</p>'
            '<p class="intro">A fine-tuned RoBERTa model reads the emotion, sentiment and sarcasm in a social '
            'media post. Four further agents, backed by a local LLM, interpret the context, identify the '
            "writer's intent, draft a reply and review the result. The model's prediction is never "
            'overridden.</p></header>')


def show_status(status):
    with st.container(key="status"):
        cells = [
            ("System", pill(status.overall, STATUS_TONE[status.overall])),
            ("FastAPI", pill("Connected", "ok") if status.api_online else pill("Offline", "bad")),
            ("ML model (RoBERTa)", pill("Loaded", "ok") if status.model_loaded
             else pill("Unknown" if not status.api_online else "Not loaded", "bad")),
            ("LLM", pill(status.llm_label, "ok") if status.llm_active else pill("Not available", "warn")),
            ("Mode", pill("Full multi-agent (LLM)", "ok") if status.llm_active
             else pill("Fallback (rule-based)", "warn") if status.api_online else pill("—", "neutral")),
        ]
        st.html('<div class="status-grid">' + "".join(
            f'<div class="status-cell"><span class="label">{escape(label)}</span>{value}</div>' for label, value in cells)
            + f'</div><p class="caption">{escape(status.detail)}</p>')
        if not status.api_online:
            st.code(START_API_COMMAND, language="bash")
        st.button("Refresh status", key="refresh")


def show_input(status):
    can_analyse = status.api_online and status.model_loaded
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.text_area("Social media post", key="post_text", height=130, max_chars=MAX_TEXT_CHARS,
                     placeholder="Paste or type a post…")
        st.button("Analyze", key="analyze", type="primary", on_click=request_analysis, disabled=not can_analyse)
        if not can_analyse:
            st.caption("Analysis is unavailable until the API is online with the model loaded.")
    with right:
        st.html('<p class="field-label">Try an example</p>')
        with st.container(key="examples"):
            for index, (label, text) in enumerate(EXAMPLES):
                st.button(f"{label}: “{text}”", key=f"example_{index}", on_click=use_example, args=(text,),
                          disabled=not can_analyse, width="stretch")


def show_model_prediction(result):
    emotion, sentiment = result.get("emotion") or {}, result.get("sentiment") or {}
    section("Model prediction",
            "From the fine-tuned RoBERTa model, including its emoji vote and sarcasm rule. "
            "Sentiment follows from the emotion. The LLM agents never change these labels.")
    label = emotion.get("label", "unknown")
    col_emotion, col_sentiment, col_sarcasm = st.columns(3, gap="medium")
    with col_emotion, st.container(border=True, key="card_emotion"):
        st.html(f'<p class="label">Emotion</p>{emotion_chip(label)}'
                f'<p class="small">Score {emotion.get("scores", {}).get(label, 0):.0%} · margin over next '
                f'{emotion.get("margin", 0):.0%} · words-only confidence {emotion.get("text_confidence", 0):.0%}</p>')
    with col_sentiment, st.container(border=True, key="card_sentiment"):
        consistent = sentiment.get("consistent_with_scores", True)
        st.html(f'<p class="label">Sentiment</p><p class="big">{escape(str(sentiment.get("label", "unknown")).title())}</p>'
                + ("" if consistent else '<p class="small warn-text">The summed scores favour a different sentiment.</p>'))
    with col_sarcasm, st.container(border=True, key="card_sarcasm"):
        probability = float(emotion.get("sarcasm_probability", 0))
        verdict = "Flagged as sarcastic" if emotion.get("sarcasm_flagged") else "Not flagged as sarcastic"
        st.html(f'<p class="label">Sarcasm probability</p><p class="big">{probability:.0%}</p>'
                f'<p class="small">{verdict} by the model.</p>')
        st.progress(min(max(probability, 0.0), 1.0))
        if emotion.get("sarcasm_rule_applied"):
            st.caption(f"The words alone read as {emotion.get('emotion_before_sarcasm')}; the sarcasm rule moved "
                       f"the emotion to {label}.")
    col_scores, col_sentiment_scores = st.columns([3, 2], gap="medium")
    with col_scores:
        st.html('<p class="label">Emotion scores</p>' + score_bars(emotion.get("scores")))
    with col_sentiment_scores:
        sentiment_rows = "".join(f"<li>{escape(name.title())} <b>{value:.0%}</b></li>"
                                 for name, value in top_scores(sentiment.get("scores")))
        emojis = ", ".join(f"{escape(v.get('char', ''))} = {escape(str(v.get('emotion', '')))}"
                           for v in emotion.get("emojis_used") or []) or "none"
        st.html(f'<p class="label">Sentiment scores</p><ul class="plain">{sentiment_rows}</ul>'
                f'<p class="small">Emojis recognised: {emojis}</p>')


def show_trace(result, status):
    trace = result.get("trace") or []
    section("Six-agent execution trace", "Every step the LangGraph workflow ran, in order, as reported by the API.")
    fell_back = fallback_agents(result)
    if fell_back and status.llm_active:
        st.warning("The LLM was reported available, but these agents used their rule-based fallback "
                   f"for this post: {', '.join(fell_back)}. See the pipeline messages below.")
    sources = {name: (result.get(name) or {}).get("source") for name in ("context", "intent", "response")}
    seen = set()
    rows = []
    for number, step in enumerate(trace, 1):
        agent = step.get("agent", "?")
        revision = agent in seen
        seen.add(agent)
        mode = agent_mode(agent, bool(step.get("used_llm")), sources.get(agent) if not revision else None)
        tone = "ok" if mode in ("LLM", "Rules + LLM review") else "warn" if mode == "Rule-based fallback" else "neutral"
        rows.append(f'<div class="trace-row"><span class="step">{number}</span>'
                    f'<span class="agent">{escape(AGENT_TITLES.get(agent, agent))}{" (revision)" if revision else ""}</span>'
                    f'{pill(mode, tone)}<span class="time">{float(step.get("duration_ms", 0)) / 1000:.1f} s</span>'
                    f'<span class="summary">{escape(str(step.get("summary", "")))}</span></div>')
    st.html(f'<div class="trace">{"".join(rows) or "<p>No trace was returned.</p>"}</div>')
    with st.expander("Raw output of each agent (as returned by the API)"):
        for agent, title in AGENT_TITLES.items():
            st.markdown(f"**{title}**")
            st.json(agent_output(result, agent), expanded=False)


def show_context_and_intent(result, status):
    context, intent, emotion = result.get("context") or {}, result.get("intent") or {}, result.get("emotion") or {}
    critic = result.get("critic") or {}
    section("Context and intent", "Interpretation by the LLM agents. RoBERTa's prediction above stays the final label.")
    col_context, col_intent = st.columns(2, gap="medium")
    with col_context, st.container(border=True, key="card_context"):
        cues = "".join(pill(cue, "neutral") for cue in context.get("cues") or []) or '<span class="small">none given</span>'
        st.html(f'<p class="label">Context Agent</p><p class="source">{escape(source_label(context.get("source"), status.llm_label))}</p>'
                f'<p>{escape(str(context.get("implied_meaning", "—")))}</p>'
                f'<p class="small"><b>Reads as sarcastic:</b> {"yes" if context.get("is_sarcastic") else "no"} '
                f'(model: {"flagged" if emotion.get("sarcasm_flagged") else "not flagged"}, '
                f'{float(emotion.get("sarcasm_probability", 0)):.0%})</p>'
                f'<p class="small"><b>Cues:</b></p><div class="cues">{cues}</div>'
                f'<p class="small"><b>Reason:</b> {escape(str(context.get("reason", "—")))}</p>')
        alternative = critic.get("alternative_emotion") or context.get("proposed_emotion")
        if alternative:
            eligible = critic.get("override_eligible")
            st.html(f'<div class="note"><b>Context Agent alternative:</b> {emotion_chip(alternative)} '
                    f'<span class="small">Shown for comparison only; it does not replace the model\'s '
                    f'{escape(str(emotion.get("label", "")))}. Override checks: '
                    f'{"would pass" if eligible else "would not pass"} (overrides are disabled).</span></div>')
    with col_intent, st.container(border=True, key="card_intent"):
        support = intent.get("needs_support")
        st.html(f'<p class="label">Intent Agent</p><p class="source">{escape(source_label(intent.get("source"), status.llm_label))}</p>'
                f'<p class="big">{escape(str(intent.get("intent", "unknown")).replace("_", " ").title())}</p>'
                f'<p class="small"><b>Urgency:</b> {escape(str(intent.get("urgency", "—")))} · '
                f'<b>Needs support:</b> {"yes" if support else "no"}</p>'
                f'<p class="small"><b>Reason:</b> {escape(str(intent.get("reason", "—")))}</p>')
        if support:
            st.info("The Intent Agent thinks this writer may need support. This is an automated reading, "
                    "not a clinical judgement.")


def show_response(result, status):
    response = result.get("response") or {}
    revised = len(revision_passes(result)) > 1
    section("Generated response", "Final response, revised after the Critic's feedback (see Critic review)."
            if revised else "Final response.")
    with st.container(border=True, key="card_response"):
        st.html(f'<p class="source">{escape(source_label(response.get("source"), status.llm_label))}</p>'
                f'<p class="label">Explanation</p><p>{escape(str(response.get("explanation", "—")))}</p>'
                f'<p class="label">Suggested reply</p><blockquote class="reply">'
                f'{escape(str(response.get("suggested_reply", "—")))}</blockquote>')


def bullet_list(items, empty="None"):
    return "".join(f"<li>{escape(str(item))}</li>" for item in items) or f"<li>{escape(empty)}</li>"


def pass_html(item):
    draft, review = item["draft"], item["review"]
    if item["sent_back"]:
        decision, tone, title = "Sent back for revision", "warn", "Draft (rejected)"
    else:
        decision = "Approved" if review.get("approved") else "Not approved (no revisions left)"
        tone, title = ("ok" if review.get("approved") else "bad"), "Revised draft · final response" \
            if item["number"] > 1 else "Draft · final response"
    feedback_label = "Revision feedback sent to the Response Agent" if item["sent_back"] else "Issues"
    return (f'<div class="pass"><p class="label">Pass {item["number"]} · {escape(title)}</p>'
            f'<p>{escape(str(draft.get("explanation", "—")))}</p>'
            f'<blockquote class="reply">{escape(str(draft.get("suggested_reply", "—")))}</blockquote>'
            f'<p class="small"><b>Critic decision:</b> {pill(decision, tone)} · <b>LLM review:</b> '
            f'{"yes" if review.get("llm_reviewed") else "no"}</p>'
            f'<div class="two-col"><div><p class="label">Failed rule checks</p><ul class="plain">'
            f'{bullet_list([n.replace("_", " ") for n in item["failed_checks"]])}</ul></div>'
            f'<div><p class="label">{feedback_label}</p><ul class="plain">{bullet_list(review.get("issues") or [])}</ul>'
            f'</div></div></div>')


def show_critic(result):
    critic, revisions = result.get("critic") or {}, int(result.get("revision_count") or 0)
    passes = revision_passes(result)
    section("Critic review")
    with st.container(border=True, key="card_critic"):
        if len(passes) > 1:
            outcome = "approved after revision" if critic.get("approved") else "kept after revision, still not approved"
            st.html(f'<p class="big">{"Approved" if critic.get("approved") else "Not approved"}</p>'
                    f'<p class="small"><b>Revisions:</b> {revisions} · <b>Final outcome:</b> {escape(outcome)}</p>'
                    '<p class="label">Revision history</p>' + "".join(pass_html(item) for item in passes))
            return
        checks = "".join(f'<li>{"✓" if passed else "✗"} {escape(name.replace("_", " "))}</li>'
                         for name, passed in (critic.get("rule_checks") or {}).items())
        issues = "".join(f"<li>{escape(str(issue))}</li>" for issue in critic.get("issues") or []) or "<li>None</li>"
        st.html(f'<p class="big">{"Approved" if critic.get("approved") else "Not approved"}</p>'
                f'<p class="small"><b>Revisions:</b> {revisions} · <b>LLM review in the final pass:</b> '
                f'{"yes" if critic.get("llm_reviewed") else "no"}</p>'
                f'<div class="two-col"><div><p class="label">Rule checks</p><ul class="plain">{checks}</ul></div>'
                f'<div><p class="label">Issues in the final review</p><ul class="plain">{issues}</ul></div></div>')
        if revisions:   # an older API without history: report only what the trace records
            first_pass = next((s.get("summary") for s in result.get("trace") or [] if s.get("agent") == "critic"), None)
            st.caption("The draft was sent back once. This API version keeps only the final review's details; the "
                       f"first review is recorded in the trace as: “{first_pass}”.")


def show_comparison(result):
    emotion, sentiment = result.get("emotion") or {}, result.get("sentiment") or {}
    section("Model versus baseline", "The TF-IDF + SVM baseline is shown for comparison only. No accuracy is "
            "computed here; see results/ for the evaluated metrics.")
    rows = []
    for task, model_label, baseline_label in [("Emotion", emotion.get("label"), emotion.get("baseline_label")),
                                              ("Sentiment", sentiment.get("label"), sentiment.get("baseline_label"))]:
        agree = "—" if baseline_label is None else ("agree" if baseline_label == model_label else "differ")
        rows.append(f"<tr><td>{task}</td><td>{escape(str(model_label))}</td>"
                    f"<td>{escape(str(baseline_label)) if baseline_label else 'not available'}</td><td>{agree}</td></tr>")
    st.html('<table class="compare"><thead><tr><th></th><th>RoBERTa (used)</th><th>TF-IDF + SVM baseline</th>'
            f'<th></th></tr></thead><tbody>{"".join(rows)}</tbody></table>')


def show_messages(result):
    errors = result.get("errors") or []
    if errors:
        with st.expander(f"Pipeline messages ({len(errors)})"):
            for message in errors:
                st.write(f"- {message}")


def show_result(result, status):
    st.html(f'<blockquote class="post">{escape(st.session_state.get("analysed_text", ""))}</blockquote>')
    show_model_prediction(result)
    show_trace(result, status)
    show_context_and_intent(result, status)
    show_response(result, status)
    show_critic(result)
    show_comparison(result)
    show_messages(result)


def main():
    st.set_page_config(page_title="EmoContext AI", layout="wide")
    st.html(f"<style>{(Path(__file__).parent / 'dashboard.css').read_text(encoding='utf-8')}</style>")
    show_header()
    status = describe_status(fetch_health())
    show_status(status)
    show_input(status)
    run_pending_analysis()
    if st.session_state.get("error"):
        st.error(st.session_state.error)
    if st.session_state.get("result"):
        show_result(st.session_state.result, status)


main()
