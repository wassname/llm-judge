# What each default is for

These are wassname's practical starting points, implemented by PI/OpenAI. Adapt the criterion and check known examples before trusting an evaluation.

## Distinguish behaviors, not merely scores

Ask which distributions you need to separate: a genuine decision change, vocabulary mimicry, lost task ability, or a judge responding to length/style. A virtue score alone may fit all four. Add measurements and controls that distinguish those explanations, rather than optimizing an average with unclear meaning.

Ten constant binary labels contain zero bits of variation; ten copies of one balanced bit contain one bit jointly; ten independent balanced bits contain ten bits. Partial correlation is not equivalence. Jev's question isolation is an attention contract, not statistical independence. Variation is useful only insofar as it distinguishes the behavior or hypotheses you care about. A constant judge has low noise and no between-item discrimination.

Keep full distributions. `[.5,0,0,0,.5]` and `[0,0,1,0,0]` have the same expected level 2 but describe different uncertainty. Probabilities near 0/1 can discriminate well when they vary between items; saturation matters when the needed distinction disappears. Priors and controls concern the measurement procedure as well as the evaluated model.

## Evidence before ratings

The generative judge explains the comparison, quotes exact answer spans, then rates each answer 0–3 and chooses a preference. This directs attention before the decision, like a checklist. It also makes mistakes inspectable. Substring checks catch invented quotes, not false interpretations. The final setup note is unscored; a prose-only critique does not replace numerical judging.

`both` runs independent procedures on the original inputs, not a generative judge anchored on Jev labels. A later post-label critic is useful, but it answers a different question. Prefer a different model family from the evaluated model and inspect disagreements yourself.

## Position controls and repeated draws

Four presentations change nuisances while holding the answers fixed: A/B and B/A crossed with normal/reversed unordered Choice options. Map back to original identity before averaging. This exposes position/option sensitivity; it does not guarantee its removal. Never reverse Score levels because their order defines the scale.

Generative `passes=3` samples each presentation three times, averages within the original pair, then reports standard error across pair means. Related scenarios need grouped uncertainty rather than an independent-pair interpretation. More draws reduce sampling noise, not shared errors. Jev supplies probability vectors and is cached once per presentation; repeat fresh Jev calls only when measuring serving variation. Views and recovery calls do not increase the item count.

All pair scores are original B minus A. Preference is P(B)−P(A), −1..1 (empirical frequencies for the generative judge); Jev comparison is an expected rubric index centered on zero, −2..2; generative quality change is the difference of 0–3 ratings, −3..3. These are different ordinal scales, not interchangeable utilities or P(correct). Refusal/incoherence/repetition/off-topic checks remain separate. Disobedience inside a story is not task refusal.

## A small study recipe

Name the distinction before reading the scores: for example, changed decisions versus virtue-word mimicry versus lost task ability. Keep stable pair IDs and include an identical-answer control, a vocabulary-only case, an ordinary quality contrast, and relevant failure cases. Run `compare` with the target criterion, then use `label` on the same material for explicit narrow checks. Keep the full vectors and disagreement cases alongside the means.

Inspect the first full trace and a seeded random sample, plus targeted disagreements. Record how samples were selected. When several rows come from one scenario or source, analyze those groups rather than treating each presentation as independent. Revise the rubric as you learn, record the change, and rerun the same controls. This is an example-level recipe, not another measurement abstraction.

## Explicit credentials

Select credentials in the calling environment (or pass an explicit key to `Decisions`). The paired runner disables Inspect's implicit `.env` lookup: otherwise the SDK searches parent directories and can select another project's configuration. This is a scoped guard on the pinned Inspect initialization hook, not a silent fallback. Offline checks use fixture credentials/transports and preserve the caller's environment. Inspect's evaluation context and this scoped hook are process-global; use separate processes for concurrent independent evaluations.

## Durable execution

`Trace` appends requests before transport and raw replies/errors before validation, flushes writes, and preserves a torn final JSONL fragment separately on resume. Complete/failed Inspect samples are appended incrementally with the current run ID, including cache replays. The full first pair/state also goes to `run.md`; do not skim just its selected label. Inputs may be sensitive: outputs and credentials are ignored, not published.

Caches include exact model/request/schema/order/config, with separate generative epoch entries. Raw cached bodies retain their original usage, while current diagnostic charge accounting treats cache hits and duplicate waiters as zero new spending. A failed in-flight request is removed so the caller can explicitly retry it later. Concurrent duplicate decisions share an in-flight request. A changed rubric misses cache. Model identity is checked across cached/live replies and persisted between reruns. The paired runner archives old aggregates at startup; a failed comparison cannot present a prior successful `results.jsonl` as current. `status.json` marks the current run. Unresolved measurement failures fail the aggregate, not silently drop hard rows.

Concurrency defaults to four. Transient HTTP/network failures get bounded backoff/retries; permanent errors fail with the raw body retained. Models are pinned; generative routing requires `deepinfra/fp8`, FP8, requested-parameter support and no fallback. These are concrete route requirements, not evidence that a model is calibrated for your task.

If reasoning reaches the output limit, preserve the provider-returned reasoning and request one structured final answer with reasoning disabled. This cannot finish incomplete reasoning. An unfinished final reply still fails. One malformed JSON/field/quote gets a bounded validation-error retry; keep the invalid reply, never patch scores or invent defaults. Jev vectors tolerate two-decimal total-mass error up to .02 and check score/vector consistency without silently normalizing.

## Additional diagnostics

The normal broad vent is cheap, unscored and enabled by default. It names potential needs beyond the anticipated failures, including data, rubric, harness and supplied code issues. Preserve its full probabilities: warnings can split mass while `none` wins. A content trait is not automatically a bad input.

`diagnostics.controls` deliberately compares clear/missing-reference/truncated cases with and without the vent's own rubric copy. Identical fresh repeats test serving variation, so cache refresh is explicit. These few cases cannot establish detection rates; literal word echo competes with real understanding.

`diagnostics.interview` supplies actual returned answers to experimental talk-to-Jev composition. Grouped vocabulary shortlists and separate Noul continuation/finish/undo decisions compose three candidates, then self-grade distinct texts. It can miss explicit defects; the self-grader is not independent verification, and composed text is not recovered internal reasoning. Expected budget exhaustion retains partial work without invalidating otherwise valid scores. Unexpected invalid measurements still fail, including a served-model change during optional feedback; optional does not mean invalid metadata is silently accepted.

Sources: [Jev API](https://docs.typesafe.ai/api), [primitives](https://docs.typesafe.ai/primitives), [Jev 1.13 limits](https://docs.typesafe.ai/model-jaggedness/jev-1.13), [Miller, Adding Error Bars to Evals](https://www.anthropic.com/research/statistical-approach-to-model-evals), [Wang et al., order calibration](https://arxiv.org/abs/2305.17926), [Inspect](https://inspect.aisi.org.uk/), [jev-talks algorithm](https://github.com/zie1ony/jev-talks/tree/50bc6df463202f0b7f329077ea688ef97b9a473d).
