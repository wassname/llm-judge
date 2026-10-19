"""Jev or evidence-first numerical judgments, with durable Inspect traces. -- PI/OpenAI"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import statistics
from pathlib import Path
from unittest.mock import patch

from inspect_ai import Epochs, Task, eval as inspect_eval
from inspect_ai.hooks import Hooks, hooks
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, mean, scorer
from inspect_ai.solver import solver

from . import generative as gen
from . import feedback as talk
from .jev import FAILURES, JEV, make_request, require, validate
from .trace import FIRST_PAIR, Decisions, Trace, digest

ACTIVE = None


@hooks(name="durable_judge", description="Append complete samples and events before the eval finishes")
class DurableJudge(Hooks):
    def enabled(self):
        return ACTIVE is not None

    async def on_sample_event(self, data):
        ACTIVE.event("inspect_event", sample_id=data.sample_id, event=data.event.model_dump(mode="json"))

    async def on_sample_end(self, data):
        ACTIVE.sample(data.sample.model_dump(mode="json"))


def jev_request(pair, order, reverse, vocabulary, criterion):
    compact = {"request": pair["context"], "answer_A": pair["response_a"], "answer_B": pair["response_b"]}
    request = make_request(compact, order == "AB", reverse, vocabulary["diagnostic"])
    questions = request["questions"]
    questions["preference"]["instructions"] = criterion
    questions["change"]["instructions"] = criterion + " How does B compare with A?"
    # PI/OpenAI: Build the vent from the final questions actually sent, including the caller's criterion.
    questions["vent"]["instructions"]["questions_to_review"] = copy.deepcopy({k: v for k, v in questions.items() if k != "vent"})
    return request


@solver
def measure(mode, decisions, trace, vocabulary, criterion, dev, dev_budget):
    numerical = gen.generate_judgment()
    async def solve(state, generate):
        meta = state.metadata
        if mode in {"gen", "both"}:
            # PI/OpenAI: The generative judge sees the original pair, never Jev's answers.
            state = await numerical(state, generate)
            trace.model("gen", state.output.model)
        if mode in {"jev", "both"}:
            request = jev_request(meta["pair"], meta["order"], meta["reverse_options"], vocabulary, criterion)
            if meta["pair"]["id"] == trace.first_id:
                trace.first_keys.add(digest(request))
            body = await decisions(request)
            meta["jev"] = {"request": request, "body": body}
            if dev and meta["pair"]["id"] == trace.first_id and meta["order"] == "AB" and not meta["reverse_options"] and state.epoch == 1:
                charges = []
                async def diagnostic(request):
                    trace.first_keys.add(digest(request))
                    body, charge = await decisions.charged(request)
                    charges.append(charge)
                    return body
                try:
                    meta["dev"] = await talk.feedback({"judged_setup": request, "returned_answers": body["answers"]},
                                                  vocabulary, diagnostic, model=JEV, served=body["model"], budget=dev_budget, charge_of=lambda _: charges[-1])
                except talk.FeedbackLimit as stop:
                    meta["dev"] = stop.result
                trace.markdown("Unscored experimental feedback (all attempts, including budget stops)", meta["dev"])
        return state
    async def traced(state, generate):
        token = FIRST_PAIR.set(state.metadata["pair"]["id"] == trace.first_id)
        try:
            return await solve(state, generate)
        finally:
            FIRST_PAIR.reset(token)
    return traced


@scorer(metrics={"gen_preference": [mean()], "gen_quality_change": [mean()], "jev_change": [mean()]})
def measurement_scores(mode):
    async def score(state, target):
        meta, values = state.metadata, {}
        sign = 1 if meta["order"] == "AB" else -1
        if mode in {"gen", "both"}:
            rating = gen.PairJudgment.model_validate_json(state.output.completion)
            gen.validate_quotes(rating, meta["pair"], meta["order"])
            values.update(gen_preference=sign * {"A": -1, "tie": 0, "B": 1}[rating.preference],
                          gen_quality_change=sign * (rating.response_B.quality - rating.response_A.quality))
        if mode in {"jev", "both"}:
            values["jev_change"] = sign * (meta["jev"]["body"]["answers"]["change"]["score"] - 2)
        return Score(value=values, answer=json.dumps(values))
    return score


def task(pairs, mode, passes, parallel, trace, vocabulary, criterion, *, dev=False, dev_budget=.05, transport=None, model=None):
    require(mode in {"jev", "gen", "both"}, "mode must be jev, gen or both")
    require(passes > 0 and parallel > 0, "passes and parallel must be positive")
    source = digest({str(p.relative_to(Path(__file__).parent)): p.read_text() for p in sorted(Path(__file__).parent.rglob("*.py"))})
    data = gen.dataset(pairs, criterion)
    for sample in data:
        sample.metadata["source"] = source
    decisions = Decisions(trace, validate, parallel=parallel, transport=transport)
    return Task(dataset=data, solver=measure(mode, decisions, trace, vocabulary, criterion, dev, dev_budget),
                scorer=measurement_scores(mode), model=model or ("mockllm/model" if mode == "jev" else gen.MODEL),
                epochs=Epochs(passes if mode != "jev" else 1, "mean"), fail_on_error=True,
                config=GenerateConfig(temperature=.7, max_tokens=32000, max_connections=parallel, max_retries=6,
                                      attempt_timeout=180, timeout=600, extra_body=gen.OPTIONS, response_schema=gen.schema(False)),
                metadata={"mode": mode, "criterion": criterion, "source": source, "provider": gen.PROVIDER})


def discrepancy(samples, value, pair):
    by_order = [statistics.mean(value(s) for s in samples if s.metadata["order"] == order) for order in ("AB", "BA")]
    options = statistics.mean(abs(statistics.mean(value(s) for s in samples if s.metadata["order"] == order and not s.metadata["reverse_options"])
                                  - statistics.mean(value(s) for s in samples if s.metadata["order"] == order and s.metadata["reverse_options"])) for order in ("AB", "BA"))
    return {"response_order_discrepancy": abs(by_order[0] - by_order[1]) if pair["response_a"] != pair["response_b"] else None,
            "response_order_flip": by_order[0] * by_order[1] < 0 if pair["response_a"] != pair["response_b"] else None,
            "option_order_discrepancy": options}


def summarize(log, pairs, mode, passes, trace):
    expected = len(pairs) * 4 * (passes if mode != "jev" else 1)
    require(log.status == "success" and len(log.samples) == expected and all(not s.error for s in log.samples),
            "Incomplete run; inspect traces, do not score surviving rows")
    results = []
    for pair in pairs:
        samples = [s for s in log.samples if s.metadata["pair"]["id"] == pair["id"]]
        row = {"id": pair["id"], "identical_responses": pair["response_a"] == pair["response_b"]}
        if mode in {"gen", "both"}:
            value = lambda s: s.scores["measurement_scores"].value["gen_preference"]
            row["gen"] = {"B_minus_A_preference": statistics.mean(value(s) for s in samples),
                          "B_minus_A_quality": statistics.mean(s.scores["measurement_scores"].value["gen_quality_change"] for s in samples),
                          **discrepancy(samples, value, pair)}
            ratings = [(gen.PairJudgment.model_validate_json(s.output.completion), s.metadata["order"]) for s in samples]
            row["gen"]["failures"] = {f"{failure}_{original}": statistics.mean(getattr(getattr(r, "response_" + (letter if order == "AB" else other)), failure) for r, order in ratings)
                                      for failure in FAILURES for original, letter, other in (("a", "A", "B"), ("b", "B", "A"))}
        if mode in {"jev", "both"}:
            views = [s for s in samples if s.epoch == 1]
            value = lambda s: (s.metadata["jev"]["body"]["answers"]["preference"]["probabilities"]["B"] - s.metadata["jev"]["body"]["answers"]["preference"]["probabilities"]["A"]) * (1 if s.metadata["order"] == "AB" else -1)
            row["jev"] = {"B_minus_A_preference": statistics.mean(value(s) for s in views),
                          "B_minus_A_change": statistics.mean(s.scores["measurement_scores"].value["jev_change"] for s in views),
                          "views": [s.metadata["jev"] for s in views], **discrepancy(views, value, pair)}
            row["jev"]["failures"] = {f"{failure}_{original}": statistics.mean(s.metadata["jev"]["body"]["answers"][f"{failure}_{letter if s.metadata['order'] == 'AB' else other}"]["noul"] for s in views)
                                      for failure in FAILURES for original, letter, other in (("a", "A", "B"), ("b", "B", "A"))}
        results.append(row)
    summary = {backend: {"mean_preference": statistics.mean(row[backend]["B_minus_A_preference"] for row in results),
                        "stderr_across_pairs": statistics.stdev(row[backend]["B_minus_A_preference"] for row in results) / math.sqrt(len(results)) if len(results) > 1 else None}
               for backend in ("jev", "gen") if backend in results[0]}
    # PI/OpenAI: Publish aggregate files only after every intended pair succeeds; samples.jsonl is incremental.
    pending = trace.output / "results.jsonl.tmp"
    pending.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results))
    pending.replace(trace.output / "results.jsonl")
    trace.markdown("Results (B minus A; views/repeats are not additional items)", {"pairs": len(pairs), "summary": summary, "eval": log.location})
    with (trace.output / "run.md").open("a") as stream:
        stream.write("\nRaw evidence: events.jsonl; full samples: samples.jsonl; aggregates: results.jsonl.\n\n" + str((trace.output / "run.md").resolve()) + "\n")
    print(json.dumps({"pairs": len(pairs), "summary": summary, "run": str(trace.output / "run.md"), "eval": log.location}, indent=2))
    print((trace.output / "run.md").resolve())
    return results


def run(args, *, pairs=None, transport=None, model=None):
    global ACTIVE
    if pairs is None:
        pairs = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()] if args.pairs else [gen.DEMO]
    require(pairs and all(set(pair) == {"id", "context", "response_a", "response_b"} for pair in pairs), "Pair fields: id, context, response_a, response_b")
    require(all(all(isinstance(v, str) for v in pair.values()) and pair["id"] for pair in pairs), "Pair fields must be strings; IDs must be nonempty")
    require(len({p["id"] for p in pairs}) == len(pairs), "duplicate pair IDs")
    require(not args.dev or args.mode in {"jev", "both"}, "--dev is Jev's experimental follow-up")
    trace = Trace(args.output)
    trace.first_id = pairs[0]["id"]
    previous = args.output / "results.jsonl"
    if previous.exists():
        previous.rename(args.output / f"results.{trace.run_id}.previous.jsonl")
    trace.status("running", mode=args.mode, criterion=args.criterion)
    ACTIVE = gen.TRACE = trace
    vocabulary = json.loads(Path(__file__).with_name("vocabulary.json").read_text())
    try:
        evaluation = task(pairs, args.mode, args.passes, args.parallel, trace, vocabulary, args.criterion,
                          dev=args.dev, dev_budget=args.dev_budget, transport=transport, model=model)
        trace.markdown("Configuration and expected distinctions", {"criterion": args.criterion, "mode": args.mode, "passes": args.passes,
                       "parallel": args.parallel, "gen_model": gen.MODEL_ID, "jev_model": JEV, "provider": gen.PROVIDER,
                       "expectation": "Identical responses should tie; switching response/option order should not change mapped judgments. Inspect deviations."})
        # PI/OpenAI: Pinned Inspect searches parent .env files; require explicit credential selection instead.
        with patch("inspect_ai._eval.context.init_dotenv", new=lambda: None):
            logs = inspect_eval(evaluation, log_dir=str(args.output / "logs"), log_model_api=True, max_samples=args.parallel, display="none")
        results = summarize(logs[0], pairs, args.mode, args.passes, trace)
        trace.status("success", results="results.jsonl", eval=logs[0].location)
        return results
    except BaseException as error:
        trace.event("run_failed", error=repr(error))
        trace.status("failed", error=repr(error))
        trace.markdown("Failed run: no valid aggregate", {"error": repr(error), "read": "events.jsonl, samples.jsonl, and the Inspect log"})
        raise
    finally:
        ACTIVE = gen.TRACE = None


def smoke(args):
    import httpx
    def reply(request):
        payload = json.loads(request.content)
        answers = {}
        for name, q in payload["questions"].items():
            if q["type"] == "noul":
                answers[name] = {"type": "noul", "noul": .1}
            else:
                labels = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
                chosen = "tie" if name == "preference" else "2" if name == "change" else "none"
                answers[name] = {"type": q["type"], "probabilities": {k: float(k == chosen) for k in labels},
                                 **({"choice": chosen} if q["type"] == "choice" else {"score": 2.0})}
        return httpx.Response(200, json={"answers": answers, "model": "smoke/jev", "usage": {"cost": 0}})
    rating = {"explanation": "Both are readable, but B refuses the requested explanation.",
              "response_A": {"quotes": [gen.DEMO["response_a"]], "quality": 3, "refusal": False, "incoherence": False, "repetition": False, "off_topic": False},
              "response_B": {"quotes": [gen.DEMO["response_b"]], "quality": 0, "refusal": True, "incoherence": False, "repetition": False, "off_topic": False},
              "preference": "A", "judge_note": "Smoke fixture, not evidence about real judge accuracy."}
    from inspect_ai.model import ModelOutput
    outputs = []
    for order, reverse in itertools.product(("AB", "BA"), (False, True)):
        mapped = rating if order == "AB" else rating | {"response_A": rating["response_B"], "response_B": rating["response_A"], "preference": "B"}
        outputs.append(ModelOutput.from_content("smoke/gen", json.dumps(mapped)))
    environment = {key: os.environ[key] for key in ("OPENROUTER_API_KEY", "INSPECT_CACHE_DIR") if key in os.environ}
    # PI/OpenAI: Isolate fixture caches without changing the caller's credentials or later cache location.
    args.output = args.output / "offline-smoke"
    try:
        os.environ["OPENROUTER_API_KEY"] = "offline-smoke"
        os.environ["INSPECT_CACHE_DIR"] = str(args.output / "inspect-cache")
        args.mode, args.passes, args.parallel, args.dev = "both", 1, 1, False
        return run(args, transport=httpx.MockTransport(reply), model=get_model("mockllm/model", custom_outputs=outputs))
    finally:
        for key in ("OPENROUTER_API_KEY", "INSPECT_CACHE_DIR"):
            os.environ.pop(key, None)
        os.environ.update(environment)




def compare(pairs: list[dict], *, mode="jev", criterion=gen.CRITERION, passes=3, parallel=4,
            output="outputs/judge", dev=False, dev_budget=.05) -> list[dict]:
    """Compare supplied pairs; output owns full traces, caches and atomic aggregate status. -- PI/OpenAI"""
    args = argparse.Namespace(pairs=None, mode=mode, criterion=criterion, passes=passes, parallel=parallel,
                              output=Path(output), dev=dev, dev_budget=dev_budget)
    return run(args, pairs=pairs)
