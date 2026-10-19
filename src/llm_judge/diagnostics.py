"""Optional Jev controls and post-label feedback; never add these outputs to scores. -- PI/OpenAI"""
from __future__ import annotations

import copy
import math

from . import feedback
from .jev import JEV, make_request, require, validate
from .questions import vocabulary
from .trace import Decisions, FIRST_PAIR, Trace, digest


def control_requests():
    """Keep scored inputs fixed while changing only the vent's access to their definitions. -- PI/OpenAI"""
    pair = {"request": "Name the capital of France.", "answer_A": "Paris.", "answer_B": "Lyon."}
    for case in ("clear", "missing_reference", "truncated_answer"):
        request = make_request(pair, True, False, vocabulary()["diagnostic"])
        if case == "missing_reference":
            request["questions"]["change"] = {"type": "score", "instructions":
                "Apply the five-level rubric in rubric_reference_v7. This required reference defines each level and must be supplied to assign a score.",
                "criteria": [f"Reference level {i}" for i in range(5)]}
        if case == "truncated_answer":
            request["state"]["answer_B"] = "I think the capital of France is"
        vent = request["questions"]["vent"]
        vent["instructions"]["questions_to_review"] = copy.deepcopy({k:v for k,v in request["questions"].items() if k != "vent"})
        hidden = copy.deepcopy(request)
        del hidden["questions"]["vent"]["instructions"]["questions_to_review"]
        yield case, "no_rubric_copy", hidden
        yield case, "rubric_copy", request


async def controls(*, trace: Trace, budget: float, repeats: int = 2, decide=None) -> dict:
    """Constructed controls, not a validated detector; fresh repeats measure serving variation. -- PI/OpenAI"""
    require(math.isfinite(budget) and budget > 0 and repeats > 0, "budget must be finite and positive; repeats must be positive")
    decide = decide or Decisions(trace, validate)
    rows, cost = [], 0.0
    token = FIRST_PAIR.set(True)
    try:
        for case, view, request in control_requests():
            for repeat in range(repeats):
                if cost >= budget:
                    result = {"status": "budget_exhausted", "rows": rows, "calls": len(rows), "cost": cost, "unscored": True}
                    trace.markdown("Partial diagnostic controls", result)
                    return result
                # PI/OpenAI: Cache replay cannot measure repeat noise; keep the prompt byte-identical.
                body = await decide(request, refresh=True)
                validate(body, request)
                trace.model("jev", body["model"])
                row = {"case": case, "view": view, "repeat": repeat, "request": request, "body": body}
                rows.append(row)
                cost += body["usage"]["cost"]
                trace.event("diagnostic_control", row=row)
        result = {"status": "complete", "rows": rows, "calls": len(rows), "cost": cost, "unscored": True,
                  "limits": "These cases do not estimate detection rates; lexical echo can explain a missing-reference flag. none is not clearance."}
        trace.markdown("Diagnostic controls", result)
        return result
    finally:
        FIRST_PAIR.reset(token)


async def interview(request: dict, body: dict, *, trace: Trace, budget: float, max_calls: int = 75, decide=None) -> dict:
    """Supply actual returned answers explicitly; a same-call vent cannot inspect them. -- PI/OpenAI"""
    validate(body, request)
    engine = decide or Decisions(trace, validate)
    charges = []
    async def tracked(request):
        body, charge = await engine.charged(request)
        charges.append(charge)
        return body
    token = FIRST_PAIR.set(True)
    try:
        try:
            result = await feedback.feedback({"judged_setup": request, "returned_answers": body["answers"]},
                vocabulary(), tracked, model=request["model"], served=body["model"], budget=budget, max_calls=max_calls, charge_of=lambda _: charges[-1])
        except feedback.FeedbackLimit as stop:
            result = stop.result
        trace.markdown("Experimental post-label interview (all attempts)", result)
        return result
    finally:
        FIRST_PAIR.reset(token)
