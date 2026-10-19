"""Small command-line entry points over the same Python functions. -- PI/OpenAI"""
import argparse
import asyncio
import json
from pathlib import Path

from . import diagnostics, generative, pairs
from .jev import validate
from .trace import Decisions, Trace


def _offline_decisions(request):
    import httpx
    payload = json.loads(request.content)
    state, questions = payload["state"], payload["questions"]
    visible = questions["vent"]["instructions"].get("questions_to_review", {})
    missing = "rubric_reference_v7" in json.dumps(visible)
    truncated = state["answer_B"].endswith("France is")
    flag = "missing" if missing else "truncated" if truncated else "none"
    answers = {}
    for name, q in questions.items():
        if q["type"] == "noul":
            answers[name] = {"type": "noul", "noul": .1}
        else:
            options = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
            winner = flag if name == "vent" else "tie" if name == "preference" else "2"
            answers[name] = {"type": q["type"], "probabilities": {k:float(k == winner) for k in options},
                             **({"choice": winner} if q["type"] == "choice" else {"score": 2.0})}
    return httpx.Response(200, json={"answers": answers, "model": "offline/fixture", "usage": {"cost": 0}})


async def diagnose(args):
    output = args.output
    if args.offline:
        output = output / "offline-controls"
    with Trace(output) as trace:
        if args.kind == "controls":
            if args.offline:
                import httpx
                decide = Decisions(trace, validate, transport=httpx.MockTransport(_offline_decisions), api_key="offline-fixture")
            else:
                decide = None
            result = await diagnostics.controls(trace=trace, budget=args.budget, repeats=args.repeats, decide=decide)
        else:
            rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
            if args.id:
                by_id = {row["id"]: row for row in rows}
                if args.id not in by_id:
                    raise ValueError(f"Pair ID {args.id!r} not found in {args.input}")
                row = by_id[args.id]
            else:
                row = rows[0]
            view = row["jev"]["views"][0]
            result = await diagnostics.interview(view["request"], view["body"], trace=trace, budget=args.budget)
        print(json.dumps({"status": result["status"], "cost": result["cost"], "unscored": True}))
        print((output / "run.md").resolve())


def main():
    parser = argparse.ArgumentParser(description="wassname's LLM-as-judge defaults: compare, label via Python, inspect and debug")
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare", help="Compare JSONL pairs in four mapped presentations")
    compare.add_argument("--pairs", type=Path, help="JSONL id/context/response_a/response_b; default is a built-in example")
    compare.add_argument("--mode", choices=("jev", "gen", "both"), default="jev")
    compare.add_argument("--criterion", default=generative.CRITERION)
    compare.add_argument("--passes", type=int, default=3, help="Generative draws per view, not extra Jev repeats")
    compare.add_argument("--parallel", type=int, default=4)
    compare.add_argument("--output", type=Path, default=Path("outputs/judge"))
    compare.add_argument("--dev", action="store_true", help="Paid unscored post-label feedback on the first pair only; not a dataset audit")
    compare.add_argument("--dev-budget", type=float, default=.05, help="Observed-cost stop, not a provider billing cap")
    smoke = commands.add_parser("smoke", help="Run the evaluation pipeline with simulated replies; not an accuracy test")
    smoke.add_argument("--output", type=Path, default=Path("outputs/smoke"))
    smoke.set_defaults(pairs=None, criterion=generative.CRITERION, mode="both", passes=1, parallel=1, dev=False, dev_budget=.05)
    debug = commands.add_parser("diagnose", help="Additional unscored calls: controls or a post-label interview")
    kinds = debug.add_subparsers(dest="kind", required=True)
    controls = kinds.add_parser("controls", help="Clear/missing-reference/truncated inputs; fresh serving repeats")
    controls.add_argument("--repeats", type=int, default=2)
    controls.add_argument("--offline", action="store_true", help="Simulated replies test execution, not accuracy")
    interview = kinds.add_parser("feedback", help="Experimental interview from saved Jev pair results")
    interview.add_argument("--input", type=Path, required=True, help="results.jsonl from compare --mode jev/both")
    interview.add_argument("--id", help="Pair ID; default: first pair")
    interview.set_defaults(offline=False)
    for command in (controls, interview):
        command.add_argument("--budget", type=float, required=True, help="Explicit observed-cost allowance; a final reply can exceed it")
        command.add_argument("--output", type=Path, default=Path("outputs/diagnostics"))
    args = parser.parse_args()
    if args.command == "smoke":
        pairs.smoke(args)
    elif args.command == "compare":
        pairs.run(args)
    else:
        asyncio.run(diagnose(args))


if __name__ == "__main__":
    main()
