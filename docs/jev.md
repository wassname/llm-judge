# Jev structure, visibility and hierarchical decisions

PI/OpenAI. Contract checked against the official API/cookbook; no new live model accuracy test is implied.

## Nested data is supported; output types are fixed

The [API reference](https://docs.typesafe.ai/api) says:

> The `instructions` property can be a string, an object, or an array. You can break up a long question that has extra context, or data it needs to reference, into a structured object. Put the question in one field and the data in the others, and refer to the data fields by name in backticks, the same way you point a question at a nested `state` value:

The request's `state`, question `instructions` and `criteria` can contain nested JSON, up to 64 levels. This is supplied context, not an arbitrary generative response schema. Native answers remain Noul (yes probability), Choice (category distribution, at most 255 options), or Score (expected value over 2–10 ordered levels). Question IDs are lookup keys; the docs say they are not sent to the underlying model.

For multiple traditions/virtues, ask independent rubric questions in one shared-state batch, then assemble `{tradition: {virtue: score}}` in Python. Several can apply together. JSON nesting itself does not save money; sharing state across questions can.

## Adaptive traversal is application code

The [hierarchical-classification cookbook](https://docs.typesafe.ai/cookbooks/hierarchical_classification) says:

> Beam search keeps the best `K` paths by a geometric-mean edge probability: `product(edge_probabilities) ** (1 / decisions)`, and prunes the rest. The probability is length-normalized so that shallow and deep leaves are compared fairly.

Adaptive traversal needs rounds **when the next question depends on a returned decision**. Known node questions can instead be batched in advance and reconstructed afterwards. Nested subtree descriptions do not turn the immediate Choice answer into a nested generated answer.

Source caveat: its displayed `choose` calls `client.system_one` separately, with threads over frontiers. That code does not share state in one HTTP batch per beam round. Our interfaces let callers batch known questions explicitly.

Pruning can skip relevant branches. Path scores and separation ratios are search heuristics, not calibrated leaf-correctness probabilities. This package does not silently route away rubrics or implement taxonomy search. Keep the broad vent or multiple issue questions when diagnostic needs can coexist.

## A same-call vent cannot inspect sibling answers

Jev questions share state but are evaluated in isolation. `with_vent` deep-copies the actual scored questions into the vent's own instructions, excluding the vent itself. This makes supplied rubrics visible. It does not make the returned sibling labels or probability vectors visible. A post-label interview must receive those explicitly.

Keep a short task signpost in shared input. Do not duplicate the full schema there as a universal default. Jev's typed question definitions are not JSON Schema reply-field descriptions. [Decider v1.2](https://huggingface.co/perplexity-ai/pplx-decider-v1.2#when-to-group-questions) describes noncausal cross-question attention, so swapping models can change this contract. Question isolation never establishes statistical independence of errors.

Inspect the exact requests produced by `diagnostics.control_requests()` to see the with/without-rubric-copy comparison. It changes only the vent's visibility; scored questions/state stay fixed within a case.
