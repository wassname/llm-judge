# Jev questions and diagnostics

Jev does not generate a written answer. You supply an input (`state`) and questions with predefined choices or rating levels. It returns probabilities. This makes it useful for several narrow checks about the same material.

- [Label one input against several rubrics](#label-one-input-against-several-rubrics)
- [Choose an answer type](#choose-an-answer-type)
- [Understand what the vent sees](#what-the-vent-can-see)
- [Run additional diagnostics](#additional-diagnostics)
- [Distinguish nested data from hierarchical decisions](#nested-data-and-hierarchical-decisions)

For paired comparisons, start with the [README](../README.md). For the reasons behind order controls, scoring and logging, see [defaults](defaults.md).

## Label one input against several rubrics

After [installing and selecting your project's key](../README.md#compare-your-answers), save this as `label_answer.py` and run it with `uv run python label_answer.py`:

```python
import asyncio
from llm_judge import Trace, label

async def main():
    state = {"request": "Name the capital of France.", "answer": "Paris."}
    questions = {
        "correct": {
            "type": "noul",
            "instructions": "Does the supplied answer correctly name France's capital?",
        },
        "quality": {
            "type": "score",
            "instructions": "How well does the answer satisfy the request?",
            "criteria": ["Wrong or no answer.", "Partly answers.", "Correct and direct."],
        },
    }
    with Trace("outputs/labels") as trace:
        result = await label(state, questions, trace=trace)
    print(result["means"])  # Mean probability vectors and expected scores. -- PI/OpenAI

asyncio.run(main())
```

`label` is asynchronous and Jev-specific. You own the `Trace`; it saves the requests and full results. `result["views"]` contains both raw presentations, and `result["unscored_vent"]` contains their diagnostic vectors. The first `label` call in a Trace also gets an unscored post-label interview in `result["feedback"]`, with a default $.05 observed-cost allowance. Later states do not repeat it. Set `feedback_budget=` to change the allowance, or `feedback=False` for an explicit caller budget/fixture. An injected decision client must provide `charged()` for interview cost accounting. Read `outputs/labels/run.md` before interpreting the means.

All rubric questions share one state per call. The function also sends a second presentation with unordered Choice options reversed, to check option-order sensitivity. It does not reverse ordered Score levels. These are two views of the same input, not two independent examples.

Several questions may be true together. For twenty virtues, supply twenty rubric questions; do not force a single winning virtue. Question IDs such as `quality` are lookup keys, not instructions: put the actual definition in `instructions`.

Implementation: [label and option reversal](../src/llm_judge/judge.py). Contract: [TypeSafe API](https://docs.typesafe.ai/api).

## Choose an answer type

- **Noul:** a yes/no proposition; returns the probability of yes. `criteria` can be omitted, or give optional `true` and `false` descriptions.
- **Choice:** unordered alternatives; returns a probability distribution and selected option. At most 255 options.
- **Score:** 2–10 ordered levels; returns their probabilities and expected level index. A mean of 1.5 is between levels, not a new textual verdict.

These are Jev's native answer types, not fields in an arbitrary generative JSON Schema. The [primitives guide](https://docs.typesafe.ai/primitives) explains when to use each. [Validation](../src/llm_judge/judge.py) retains the vectors and checks their values without silently normalizing them.

## What the vent can see

The vent is an extra, unscored Choice question: “What needs attention in this judging setup?” It is intended to surface problems beyond the scored rubric, including data, context, harness or supplied code issues.

Jev evaluates questions in isolation. Sharing `state` does **not** let one question read its siblings. Therefore [with_vent](../src/llm_judge/judge.py) puts an exact copy of the scored question definitions inside the vent's own `instructions`, excluding the vent itself. After option reversal or a criterion change, that copy is rebuilt from the final questions actually sent.

This lets the vent inspect the supplied rubric. It still cannot inspect the *returned answers* of other questions in that call. A post-label interview must receive those answers explicitly in a later request.

Keep a short task signpost in shared state. Do not copy the whole question set there as a universal default. Per-question definitions belong in that question's instructions. The [API](https://docs.typesafe.ai/api) permits structured instructions; for example:

```json
{"instructions": {"task": "Inspect these supplied questions for missing context.",
                  "questions_to_review": {"correct": {"type": "noul", "instructions": "..."}}}}
```

The example illustrates visibility, not a replacement vent implementation. Use `with_vent` to generate the exact copy rather than maintaining one by hand.

The [Jev vocabulary](../src/llm_judge/jev-vocab.json) contains 216 diagnostic words/emojis. With `none`, `idk` and `review_this`, the vent has 219 options. Keep the vocabulary broad for unanticipated needs and preserve the full vector. Warnings can split probability while `none` wins; `none` is not clearance. A word such as `consent` can describe the content without identifying a data defect.

Do not assume other models have this isolation rule. [Decider v1.2](https://huggingface.co/perplexity-ai/pplx-decider-v1.2#when-to-group-questions) describes noncausal cross-question attention. And question isolation never establishes statistical independence of errors.

## Additional diagnostics

These commands make extra, unscored checks. They do not change the numerical judging criterion or certify its answers.

### Clear and defective controls

```sh
uv run llm-judge diagnose controls --offline --budget .01
```

The offline version uses simulated replies to test execution. To query the real judge, select your key and omit `--offline`:

```sh
uv run llm-judge diagnose controls --budget .01 --output outputs/controls
```

The default is twelve fresh calls: clear, missing-reference and truncated-answer cases, each with/without the vent's rubric copy, each repeated twice. Within each case the scored input stays fixed; only the vent's access to the rubric changes. Cache refresh is explicit because replay cannot measure serving variation.

Inspect every request/reply. These constructed cases do not establish a general detection rate. A missing-reference flag might reflect literal word matching rather than understanding the defect.

Implementation: [control_requests and controls](../src/llm_judge/talk_to_jev.py).

### Experimental post-label interview

To investigate saved Jev pair results:

```sh
uv run llm-judge diagnose feedback --input outputs/judge/results.jsonl --budget .05
```

The input must come from `compare --mode jev` or `--mode both`. The command defaults to the first pair; use `--id capital` to select another. Normal `compare` runs already interview the first Jev pair's first presentation by default, not every dataset row; the legacy `--dev` flag merely makes this explicit. Set `--feedback-budget` to change its $.05 observed-cost allowance, or `--no-feedback` to skip it. Generative-only mode instead retains the judge's own unscored exit note, without silently adding Jev calls. Offline smokes do not call a live interview.

The interview receives the original setup and actual returned answers. Following [jev-talks](https://github.com/zie1ony/jev-talks/tree/50bc6df463202f0b7f329077ea688ef97b9a473d), it shortlists words, asks separate Noul questions about appending, undoing or finishing, composes three candidate sentences, then self-grades distinct texts. The vocabulary's 3,000 English words are used here, not as one giant Choice question in normal judging.

This is **not yet a reliable critic**. The current decoder chooses the highest-probability move even if none exceeds .5, so `model_stopped` does not prove the sentence is complete. Selection uses the highest expected self-grade without a completion-eligibility filter. It can select a poor fragment or falsely report no issue. Read all attempts and grades; do not interpret the selected sentence as clearance or recovered internal reasoning.

`--budget` is a check on observed spending before another call, not a provider billing cap; the last reply can exceed it. Expected budget exhaustion retains partial work without invalidating otherwise valid scores. Unexpected errors, including a changed served model, still fail rather than being hidden.

Implementation: [interview wrapper](../src/llm_judge/talk_to_jev.py), [composition and selection](../src/llm_judge/talk_to_jev.py).

## Nested data and hierarchical decisions

A nested input and a multi-step decision are different things. The [API reference](https://docs.typesafe.ai/api) says:

> The `instructions` property can be a string, an object, or an array. You can break up a long question that has extra context, or data it needs to reference, into a structured object. Put the question in one field and the data in the others, and refer to the data fields by name in backticks, the same way you point a question at a nested `state` value:

Inputs can contain nested JSON, up to 64 levels; criteria retain the structure required by their answer type. Answers still use Noul, Choice or Score. To produce `{tradition: {virtue: score}}`, ask the relevant questions and assemble that shape in Python. Nesting itself saves no money; sharing the state across questions can.

If the *next* question depends on a returned answer, the application needs another request. For example, choosing a topic and then asking its subquestions is adaptive routing. Known questions can instead be batched in advance and assembled afterwards.

The [hierarchical-classification cookbook](https://docs.typesafe.ai/cookbooks/hierarchical_classification) describes one search method:

> Beam search keeps the best `K` paths by a geometric-mean edge probability: `product(edge_probabilities) ** (1 / decisions)`, and prunes the rest. The probability is length-normalized so that shallow and deep leaves are compared fairly.

Its displayed code calls `client.system_one` separately, with threads over frontier nodes; that is not one shared-state HTTP batch per round. Its path scores are search heuristics, not calibrated probabilities that a leaf is correct.

Routing can save work by skipping branches, but an early mistake can hide relevant answers. Do not use it to select only one tradition or one problem when several may apply. This package does not perform taxonomy search or silently prune rubrics.

<!-- PI/OpenAI, prepared for wassname. -->
