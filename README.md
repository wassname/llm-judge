# LLM-as-judge: wassname's defaults

Compare model answers or label data against several rubrics, with these defaults:

- Use an [information-theory view](https://en.wikipedia.org/wiki/Information_theory): seek information about the behavior of interest; check saturation, redundancy, correlated errors and distribution overlap. [Measurement guide](docs/defaults.md#start-with-a-difference-you-can-check).
- When a judge hits its token limit, retain the returned, truncated reasoning and follow up for an immediate final answer where possible. [Recovery and API limits](docs/defaults.md#recover-explicitly-preserve-the-failed-attempt).
- Use Jev for cheap probabilities and a broad, unscored flag for problems with the supplied data, rubric or setup. [What Jev can see](docs/jev.md#what-the-vent-can-see).
- [Ask Jev for an exit interview on the first sample](docs/jev.md#experimental-post-label-interview), enabled by default: give it the actual returned labels and keep the feedback experimental and unscored.
- Switch to an open reasoning model to debug framing, setup or supplied harness issues through its returned reasoning and exit note. [Example](docs/defaults.md#debug-with-an-open-reasoning-judge).
- Ask for explanations and quoted evidence before numerical ratings, to direct the judge's attention and make its interpretation inspectable. [Procedure](docs/defaults.md#ask-for-evidence-before-ratings).
- Use a judge from a different model family than the student to reduce [self-preference bias](https://arxiv.org/abs/2506.02592).
- Check [position bias](https://arxiv.org/abs/2406.07791): swap answers and unordered options, then map judgments back to the original answers. [Implementation](docs/defaults.md#check-whether-position-changes-the-verdict).
- Pin OpenRouter to the model author (unknown native precision is acceptable), or prefer DeepInfra, Fireworks or Groq with documented FP8-or-higher floating-point precision. Disable silent fallback. [Endpoint-accuracy evidence](https://artificialanalysis.ai/methodology/endpoint-accuracy-index); [routing rules](docs/defaults.md#choose-credentials-and-a-model-explicitly).
- Reuse Inspect AI for generative-call retries, concurrency, repeated evaluations and the trace viewer. [Execution](docs/defaults.md#read-one-complete-pair-before-the-aggregate).
- Keep full probability vectors and complete traces, including partial replies, errors and follow-ups. [Saved evidence](docs/defaults.md#read-one-complete-pair-before-the-aggregate).
- Resume with validated caches and share duplicate in-flight requests instead of repeating paid work. [Caching](docs/defaults.md#reuse-work-without-counting-it-as-new-evidence).

## Install and try without a key

In a Python 3.12+ project:

```sh
uv add --exclude-newer "8 days" git+https://github.com/wassname/llm-judge@main
uv run llm-judge smoke
```

The smoke test runs the pipeline with simulated replies and prints the path to `run.md`. Open it to see the inputs, judgments and saved evidence. This checks execution, not a real judge's accuracy.

## Compare your answers

Save one pair per line in `answers.jsonl`:

```json
{"id":"capital","context":"Name the capital of France.","response_a":"Paris.","response_b":"Lyon."}
```

Select your project's `OPENROUTER_API_KEY` explicitly. For example, if it is in that project's `.env` file:

```sh
uv run --env-file /path/to/your-project/.env llm-judge compare --pairs answers.jsonl
```

This makes paid calls to **Jev**, TypeSafe's model for probabilities over supplied choices and rating levels. Each pair is shown as A/B and B/A; the unordered answer options are also reversed to check whether their position changes the verdict. Results are mapped back to the original A/B identities; positive differences favor B. [Why these controls matter](docs/defaults.md#check-whether-position-changes-the-verdict).

Read `outputs/judge/run.md` before using the aggregate scores. It contains the complete first pair, not just a winning label, and ends with an instruction to check the full trace against expectations and report the checks and problems to the user. To browse the Inspect logs:

```sh
uv run inspect view --log-dir outputs/judge/logs
```

To add a generative judge that explains the evidence before giving numerical ratings, append `--mode both`. Its ratings stay separate from Jev's; `--mode gen` runs it alone. By default, `both` adds twelve generative judgments per pair: four presentations × three draws. The initial output allowance is 64k tokens, with no forced low reasoning effort; use `--max-tokens` to change it. For models that cannot disable reasoning, use `--followup-reasoning`. Opaque/signed reasoning replay is still [unverified](docs/defaults.md#recover-explicitly-preserve-the-failed-attempt). [Model controls, scales and repeated judgments](docs/defaults.md).

## Use the Python functions

`compare(pairs, ...)` runs the same paired evaluation synchronously and owns its output directory. For example:

```python
from llm_judge import compare

results = compare([
    {"id": "capital", "context": "Name the capital of France.",
     "response_a": "Paris.", "response_b": "Lyon."},
], criterion="Which answer correctly names France's capital?", output="outputs/capital")
```

For several questions about one shared input, use asynchronous `label(state, questions, trace=trace)`. Questions can apply simultaneously; no single winning category is required. [A complete example and Jev's visibility rules](docs/jev.md#label-one-input-against-several-rubrics).

## Check the measurement

Scores can move because behavior changed, because the judge preferred the presentation, or because the input was lost or truncated. [Choose controls that distinguish these explanations](docs/defaults.md#start-with-a-difference-you-can-check).

Jev labeling and comparison calls include an unscored **vent**: an extra question that can flag an issue with the supplied data or rubric. It receives its own exact rubric copy because Jev questions cannot read each other. Keep the full diagnostic probabilities; `none` is not clearance. [What the vent can see](docs/jev.md#what-the-vent-can-see).

The first Jev sample also gets a bounded post-label interview by default; generative judging writes its unscored exit note. The Jev interview uses an observed-cost allowance of $.05, which the final reply can exceed. Set `--feedback-budget`, or explicitly skip it with `--no-feedback` (`dev=False` in `compare`, `feedback=False` in `label`). Offline smokes skip model interviews. Additional control runs and interviews of other saved samples remain explicit work. [Commands and limitations](docs/jev.md#additional-diagnostics). The interview is not yet a reliable critic.

## Updating from v0.1.1

The root imports `Trace`, `compare` and `label` remain. Internal submodule imports now come from `llm_judge.judge`; diagnostics come from `llm_judge.talk_to_jev`. The vocabulary resource is `jev-vocab.json`.

Jev comparison and labeling now attempt an extra, paid interview on the first sample by default, within the stated observed-cost allowance. Use the explicit opt-outs above when needed. Custom `label` decision clients need `charged()` for that interview; a plain callback must set `feedback=False`.

The initial output allowance changes from 32k to 64k, with no forced low reasoning effort or `min_p`. Provider selection now applies the creator/third-party precision rule. CLI comparisons require `--pairs`; non-1 `--passes` is rejected in Jev-only mode. These are deliberate behavior changes, not just file moves. Do not pool old and new generative scores as if they used the same measurement procedure; the source-keyed generative cache invalidates old entries.

## Reference and development

- [Defaults and their reasons](docs/defaults.md): score meanings, uncertainty, caching, recovery, credentials and provider restrictions.
- [Jev questions and diagnostics](docs/jev.md): runnable labeling example, typed answers, visibility and hierarchical decisions.
- [Read the implementation](src/llm_judge/judge.py): the complete normal procedure is in one file, including questions, judging, order controls, recovery, caching and traces. [talk_to_jev.py](src/llm_judge/talk_to_jev.py) contains the optional controls and experimental interview.

To work from source, clone this repository, run `uv sync --locked`, then `uv run python -m unittest discover -s tests/regression -v`. Keep the dependency-age hold in the consuming project too; a dependency's uv settings do not carry over. The install command above applies an eight-day hold to that resolution.

Logs and caches stay local. The paired runner blocks Inspect's automatic search for parent `.env` files, so credentials must be selected explicitly. [Execution and privacy details](docs/defaults.md#choose-credentials-and-a-model-explicitly).

<!-- PI/OpenAI, prepared for wassname. -->
