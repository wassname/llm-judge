"""Experimental unscored feedback composition; jev-talks algorithm, not internal reasoning. -- PI/OpenAI

Algorithm reference: https://github.com/zie1ony/jev-talks/tree/50bc6df463202f0b7f329077ea688ef97b9a473d
Grouped Choice shortlists, separate Noul continuation/finish/undo decisions, and rejected-prefix memory.
"""
from __future__ import annotations

import itertools
import random
import json
import re
import math


RULES = (
    "Compose one short, natural English sentence answering question about judged_setup and returned_answers. "
    "These are the only evidence: do not invent missing facts or claim access to internal reasoning. "
    "Words are placed literally, not used as labels. Keep extending the sentence already started. "
    "Avoid repeating fragments; recent_moves shows what you have already tried. "
    "You may remove a wrong final word. A single word or an unfinished fragment is not a completed sentence. "
    "If nothing apparent needs checking, say so; otherwise describe one concrete concern or unmet need. "
    "Write a declarative statement, not an instruction telling the reader to check or review something. "
    "Simply summarizing the judged answer, its target disposition or its score is not feedback. "
    "Do not invent a concern; say no apparent issue when none is apparent."
)


async def compose_feedback(payload: dict, vocabulary: list[str], decide, *, model: str,
                           seed: int = 0, max_rounds: int = 12, max_words: int = 24) -> dict:
    """Every continuation is judged separately; never greedily append a Choice winner. -- PI/OpenAI"""
    assert vocabulary and len(vocabulary) == len(set(vocabulary))
    assert max_rounds > 0 and max_words > 0
    question = ("Describe what needs attention in the judged setup or its returned answers, if anything. "
                "Use one declarative sentence rather than telling the reader to check or review something.")
    context_words = re.findall(r"[A-Za-z][A-Za-z0-9_'\-]*", json.dumps(payload) + RULES + question)
    words = list(dict.fromkeys(vocabulary + context_words))
    random.Random(seed).shuffle(words)
    placed, history, rounds, rejected = [], [], [], set()
    for round_index in range(max_rounds):
        draft = " ".join(placed)
        state = {"rules": RULES, "question": question, **payload, "draft": draft, "recent_moves": history[-10:]}
        if len(placed) >= 15:
            state["length_note"] = "The draft is already long. Finish when it is a complete sentence; add words only if needed to complete it."
        groups = [words[i:i + 250] for i in range(0, len(words), 250)]
        scout_questions = {f"words_{i}": {"type": "choice",
            "instructions": "Which literal vocabulary word could appear among the next three words of the sentence answering question?",
            "criteria": dict.fromkeys(group)} for i, group in enumerate(groups)}
        scout_request = {"model": model, "state": state, "questions": scout_questions}
        scout = await decide(scout_request)
        probabilities = {w: p for answer in scout["answers"].values() for w, p in answer["probabilities"].items()}
        shortlist = sorted(probabilities, key=probabilities.__getitem__, reverse=True)[:30]
        candidates = shortlist + [" ".join(pair) for pair in itertools.permutations(shortlist[:15], 2)]
        candidates = [p for p in candidates if len(placed) + len(p.split()) <= max_words
                      and " ".join(placed + p.split()[:1]) not in rejected]
        checks = {}
        for i, phrase in enumerate(candidates):
            candidate = " ".join(placed + phrase.split())
            checks[f"append_{i}"] = {"type": "noul", "instructions":
                f"Appending {phrase!r} makes the draft {candidate!r}. Would this read as natural, correct English "
                f"and be a good {'continuation' if placed else 'beginning'} of a complete, informative answer to question?"}
        if placed:
            checks["undo"] = {"type": "noul", "instructions":
                f"Should the final word of {draft!r} be removed, leaving {' '.join(placed[:-1])!r}?",
                "criteria": {"true": "The final word is wrong; removing it improves the sentence.",
                             "false": "The draft is a good start; the final word should stay."}}
            checks["finish"] = {"type": "noul", "instructions":
                "Is draft already a complete, natural and informative sentence answering question, so writing should stop?",
                "criteria": {"true": "Complete sentence; adding words would make it worse.",
                             "false": "Single word, fragment, repetition, unfinished thought or incorrect claim."}}
        assert checks, "no valid next moves"
        checked = await decide({"model": model, "state": state, "questions": checks})
        values = {name: answer["noul"] for name, answer in checked["answers"].items()}
        move = max(values, key=values.__getitem__)
        rounds.append({"round": round_index, "draft": draft, "move": move,
                       "move_probability": values[move], "phrase": candidates[int(move[7:])] if move.startswith("append_") else None})
        if move == "finish":
            return {"text": draft, "status": "model_stopped", "rounds": rounds, "unscored": True,
                    "note": "Composed feedback, not recovered internal reasoning or independent verification."}
        if move == "undo":
            rejected.add(draft)
            removed = placed.pop()
            history.append(f"Removed {removed!r}; draft is now {' '.join(placed)!r}.")
        else:
            phrase = candidates[int(move[7:])]
            placed.extend(phrase.split())
            history.append(f"Appended {phrase!r}; draft is now {' '.join(placed)!r}.")
    return {"text": " ".join(placed), "status": "round_limit", "rounds": rounds, "unscored": True,
            "note": "Unfinished at the explicit limit; do not treat as a completed explanation."}


class FeedbackLimit(AssertionError):
    """An expected diagnostic stop with its partial work attached. -- PI/OpenAI"""
    def __init__(self, result):
        super().__init__("Development call/observed-cost limit reached")
        self.result = result


async def feedback(payload, vocabulary, decide, *, model, served, budget=.05, max_calls=75, charge_of=None):
    """Keep every attempt and charge; self-grading is a heuristic, not verification. -- PI/OpenAI"""
    assert math.isfinite(budget) and budget > 0 and max_calls > 0
    charges, attempts, grades = [], [], {}
    async def limited(request):
        if len(charges) >= max_calls or sum(charges) >= budget:
            raise FeedbackLimit({"status": "budget_exhausted", "attempts": attempts, "grades": grades,
                                 "partial_state": request["state"], "calls": len(charges), "cost": sum(charges),
                                 "charges": charges, "unscored": True})
        body = await decide(request)
        charge = body["usage"]["cost"] if charge_of is None else charge_of(body)
        assert math.isfinite(charge) and charge >= 0, "invalid new charge"
        charges.append(charge)
        assert body["model"] == served, "Follow-up changed served model"
        return body
    words = list(dict.fromkeys(vocabulary["english"] + vocabulary["diagnostic"]))
    for seed in range(3):
        attempts.append(await compose_feedback(payload, words, limited, model=model, seed=seed))
    for attempt in attempts:
        text = attempt["text"]
        if text in grades:
            continue
        result = await limited({"model": model, "state": payload | {"candidate": text, "status": attempt["status"]},
            "questions": {"grade": {"type": "score", "instructions":
                "Grade diagnostic usefulness, not accuracy at summarizing the judged content. Merely describing the target or score "
                "is not feedback: level 0. A short statement that no issue is apparent is valid feedback. Do not reward length. "
                "Cap at level 1 for ungrammatical, inaccurate, contradictory or unfinished output. Check claims against the supplied material.",
                "criteria": ["Empty, repetitive, irrelevant or merely a content summary.",
                             "Inaccurate, ungrammatical, contradictory or unfinished.",
                             "A grammatical concern, but vague or weakly grounded.",
                             "Supported and useful; names a concern or says no apparent concern.",
                             "Supported and useful, with a precise reference to supplied evidence."]}}})
        grades[text] = result["answers"]["grade"]
    best = max(range(3), key=lambda i: grades[attempts[i]["text"]]["score"])
    return {"selected": best, "text": attempts[best]["text"], "status": attempts[best]["status"],
            "attempts": attempts, "grades": grades, "calls": len(charges), "cost": sum(charges), "unscored": True}
