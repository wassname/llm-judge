# LLM-as-judge: wassname's defaults

Practical defaults and guidance for research agents: compare model outputs, label data against multiple rubrics, and debug the measurement itself. Small Python functions with readable implementations, rather than a configurable evaluation framework. The defaults are starting points, not universal accuracy claims.

## Install and try

```sh
uv add --exclude-newer "8 days" git+https://github.com/wassname/llm-judge@v0.1.0
# Or clone, then: uv sync --locked
uv run llm-judge smoke                         # offline execution check, not an accuracy test
uv run llm-judge diagnose controls --offline --budget .01
```

To work from a clone, run `git clone https://github.com/wassname/llm-judge`, `cd llm-judge`, then `uv sync --locked`. Keep a release-age policy in the consuming project too: a dependency's `[tool.uv]` settings do not carry over. The install command applies the eight-day hold to that resolution; keep any stricter existing policy.

Paid calls use `OPENROUTER_API_KEY` from your own environment. Neither smoke command reads a project's key file. Run logs and caches stay local.

## Compare outputs

```python
from llm_judge import compare

results = compare([
    {"id": "capital", "context": "Name the capital of France.",
     "response_a": "Paris.", "response_b": "Lyon."},
], mode="both", output="outputs/capital")
```

`compare` is synchronous, owns its output directory/caches, and defaults to **Jev only**. Explanations and quoted numerical judgments require `mode="gen"` or `mode="both"`; they cost additional model calls. Jev supplies narrow metrics and probabilities. The generative judge explains evidence **before** numerical ratings and quotes each answer: an attention procedure as well as an audit trail. They see the original inputs, blind to each other's labels; their scores stay separate.

The default criterion is answer correctness/quality. Set `criterion=` for your research question; good-looking scores on the wrong criterion are not useful. [Defaults and their reasons](docs/defaults.md) explain scales, provider restrictions, order controls and uncertainty.

For a dataset, put one pair per JSONL line:

```sh
uv run llm-judge compare --pairs examples/pairs.jsonl --mode both --passes 3
uv run inspect view --log-dir outputs/judge/logs
```

Each original pair has four presentations: two response orders × two unordered-option orders, mapped back before averaging. Ordered Score levels stay ordered. Generative draws average within the original pair; uncertainty is across pairs, not presentations. Repeating cached Jev results adds no independent evidence.

## Label one state against many rubrics

```python
import asyncio
from llm_judge import Trace, label

async def main():
    questions = {
        "correct": {"type": "noul", "instructions": "Does the supplied answer correctly name France's capital?"},
        "quality": {"type": "score", "instructions": "How well does the answer satisfy the request?",
                    "criteria": ["Wrong or no answer.", "Partly answers.", "Correct and direct."]},
    }
    with Trace("outputs/labels") as trace:
        result = await label({"request": "Name the capital of France.", "answer": "Paris."}, questions, trace=trace)
    print(result["means"])

asyncio.run(main())
```

`label` is asynchronous and Jev-specific; the caller owns the `Trace`. All questions share one state per call. It returns two option-order views, their mean vectors/scores and the unscored vent. Retain `result["views"]` and `result["unscored_vent"]`, not only the printed means. Several rubrics can apply simultaneously: do not force one winning tradition or virtue. Question IDs are lookup keys, not instructions; put the criterion in `instructions`.

## Debug the judge

An unscored broad vent is included by default. Jev isolates questions, so its vent gets an exact copy of the scoring questions inside **its own instructions**. It sees supplied setup, not returned sibling answers. Keep all 216 diagnostic words/emojis and the full vector: `none` is not clearance, and `consent` can describe valid content rather than defective data.

Read the complete first pair/state in `run.md`, including prompts, schemas, exposed reasoning, raw replies, failures and order variants. `events.jsonl` and `samples.jsonl` are incremental; Inspect `.eval` gives the viewer; `status.json` distinguishes complete runs from interrupted ones. Rerunning the same output directory reuses validated caches.

Additional diagnostics are separate, explicitly unscored:

```sh
# Live clear/missing-reference/truncated controls, 12 fresh calls by default:
uv run llm-judge diagnose controls --budget .01 --output outputs/controls
# Experimental post-label talk-to-Jev interview from saved results:
uv run llm-judge diagnose feedback --input outputs/judge/results.jsonl --budget .05
```

The optional `compare --dev` shortcut does paid post-label feedback for **only the first pair, first presentation**; it is not a dataset-wide audit. The separate saved-result command above makes this extra work explicit. Only controls/smoke have an offline CLI mode; feedback is tested with fixture transports through the Python API.

A budget is an observed-cost stop, not a provider billing cap. Controls are constructed examples, not a validated detector. Interviews can falsely say no issue; every attempt remains visible. A code reviewer or oracle can instead read the same full trace and original rubric. For an independent opinion, withhold prior labels; for post-label critique, explicitly include them. Give a reasoning/chat reviewer the saved state and rubric as data in its prompt—not an unchanged Jev decisions-API request.

[Hierarchy and Jev visibility](docs/jev.md) distinguish nested data, parallel questions and adaptive routing. [Source](src/llm_judge) is organized into the small API (`compare`, `label`, `Trace`), backends, and optional `diagnostics`/`feedback` modules. No plugin registry or automatic rubric selection.

For custom offline transports, construct `Decisions(trace, validate, transport=..., api_key="offline")`; Noul questions may omit `criteria`. Production defaults require `OPENROUTER_API_KEY` rather than inventing a fallback credential.

Development: `uv run python -m unittest discover -s tests/regression -v`. Package resolution keeps an eight-day release-age hold; use the committed lockfile. Dependency updates are deliberate, not a reason to disable that hold.

<!-- Guidance/code prepared by PI/OpenAI for wassname; adapted from the earlier judge gist. -->
