"""Run one post through all six agents from the terminal.

    python src/cli.py "I just love it when my phone dies at 2%"
    python src/cli.py --no-llm "..."      # deterministic fallback, no Ollama needed
    python src/cli.py --json "..."        # print the whole shared state as JSON
"""
import argparse
import json
import sys
import warnings

from agents.graph import analyse, build_graph, state_to_dict
from agents.llm import NullClient, get_llm_client
from baseline import load_keyword_model, load_sentiment_baseline
from model import Predictor


def print_report(state, llm) -> None:
    final, emotion = state["final"], state["emotion"]
    if llm.provider == "none":
        print(f"LLM: off ({llm.reason}); LLM agents used their fallbacks")
    else:
        print(f"LLM: {llm.provider} {llm.model}")
    print(f"\nPost: {state['text']}\n\nAgent trace:")
    for number, step in enumerate(state["trace"], 1):
        mode = "LLM " if step.used_llm else "rule"
        print(f"  {number}. {step.agent:<9} [{mode}] {step.duration_ms:8.1f} ms  {step.summary}")

    print(f"\nEmotion:   {final.emotion} ({emotion.scores[final.emotion]:.0%}, from the trained model)")
    print(f"Sentiment: {final.sentiment}")
    print(f"Sarcasm:   {emotion.sarcasm_probability:.0%} likely")
    print(f"Intent:    {final.intent}, urgency {final.urgency}" + (", may need support" if final.needs_support else ""))
    if final.alternative_emotion:
        eligible = "passes" if state["critic"].override_eligible else "does not pass"
        print(f"Context Agent alternative: {final.alternative_emotion} (shown only; {eligible} the override checks)")
    print(f"Baseline (TF-IDF + SVM, comparison only): emotion {emotion.baseline_label}, "
          f"sentiment {state['sentiment'].baseline_label}")
    print(f"\nExplanation: {final.explanation}")
    print(f"Suggested reply: {final.suggested_reply}")
    critic = state["critic"]
    print(f"\nCritic: {'approved' if critic.approved else 'not approved'}"
          + ("".join(f"\n  - {issue}" for issue in critic.issues)))
    for error in state["errors"]:
        print(f"Warning: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", help="the post to analyse")
    parser.add_argument("--no-llm", action="store_true", help="do not use Ollama")
    parser.add_argument("--json", action="store_true", help="print the full shared state as JSON")
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("the post is empty")

    with warnings.catch_warnings():
        # The baseline .joblib files were saved with a newer scikit-learn (see docs/limitations).
        warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
        emotion_baseline, sentiment_baseline = load_keyword_model(), load_sentiment_baseline()
    try:
        predictor = Predictor()
    except OSError as error:
        print(f"Could not load the trained model from models/emotion_sarcasm: {error}", file=sys.stderr)
        return 1
    llm = NullClient("--no-llm was given") if args.no_llm else get_llm_client()

    state = analyse(build_graph(predictor, llm, emotion_baseline, sentiment_baseline), args.text)
    if args.json:
        print(json.dumps(state_to_dict(state), ensure_ascii=False, indent=2))
    else:
        print_report(state, llm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
