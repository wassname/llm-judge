"""Batch questions about one state; retain both unordered-option presentations. -- PI/OpenAI"""
from __future__ import annotations

import asyncio
import copy
import json
from importlib.resources import files

from .jev import JEV, require, validate, with_vent
from .trace import Decisions, FIRST_PAIR, Trace, digest


def vocabulary() -> dict:
    return json.loads(files("llm_judge").joinpath("vocabulary.json").read_text())


def reverse_choices(request: dict) -> dict:
    request = copy.deepcopy(request)
    for question in request["questions"].values():
        if question["type"] == "choice":
            question["criteria"] = dict(reversed(list(question["criteria"].items())))
    # PI/OpenAI: The vent reviews the exact presentation sent; Score levels keep their semantic order.
    request["questions"]["vent"]["instructions"]["questions_to_review"] = copy.deepcopy({k:v for k,v in request["questions"].items() if k != "vent"})
    return request


async def label(state, questions: dict, *, trace: Trace, model: str = JEV, decide=None) -> dict:
    """Two option-order views; means are derived summaries, not new independent judgments. -- PI/OpenAI"""
    require(bool(questions), "empty question set")
    if decide is None:
        if trace.decide is None:
            trace.decide = Decisions(trace, validate)
        decide = trace.decide
    request = with_vent({"model": model, "state": state, "questions": questions}, vocabulary()["diagnostic"])
    requests = [request, reverse_choices(request)]
    first = trace.label_count == 0
    trace.label_count += 1
    if first:
        trace.first_keys.update(digest(r) for r in requests)
    token = FIRST_PAIR.set(first)
    try:
        bodies = await asyncio.gather(*(decide(r) for r in requests))
        for request, body in zip(requests, bodies, strict=True):
            validate(body, request)
            trace.model("jev", body["model"])
        means = {}
        for name, question in questions.items():
            answers = [b["answers"][name] for b in bodies]
            if question["type"] == "noul":
                means[name] = {"noul": sum(a["noul"] for a in answers) / 2}
            else:
                means[name] = {"probabilities": {k:sum(a["probabilities"][k] for a in answers)/2 for k in answers[0]["probabilities"]}}
                if question["type"] == "score":
                    means[name]["score"] = sum(a["score"] for a in answers) / 2
        result = {"means": means, "views": [{"request":r, "body":b} for r,b in zip(requests,bodies,strict=True)],
                  "unscored_vent": [b["answers"]["vent"] for b in bodies]}
        trace.event("label_complete", result=result)
        return result
    finally:
        FIRST_PAIR.reset(token)
