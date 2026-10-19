# Defaults and their reasons

The [README](../README.md) shows how to run a comparison. This page explains what the results mean and which failures the defaults try to expose. They are starting points for research, not a claim that the judge is accurate on your task.

- [Choose a difference to measure](#start-with-a-difference-you-can-check)
- [Produce and interpret judgments](#ask-for-evidence-before-ratings)
- [Check position and repeat noise](#check-whether-position-changes-the-verdict)
- [Read the trace](#read-one-complete-pair-before-the-aggregate)
- [Understand caching and recovery](#reuse-work-without-counting-it-as-new-evidence)
- [Choose credentials and models](#choose-credentials-and-a-model-explicitly)

## Start with a difference you can check

Suppose a steered model gets a higher virtue score. Did its decisions change, did it learn virtue-related vocabulary, or did the judge merely prefer longer answers? A single average score can fit all three explanations.

Choose cases that distinguish them: a new decision scenario, an answer containing virtue words without virtuous conduct, and the same decision expressed at different lengths. Also check ordinary task ability and refusal. Write down what each explanation predicts before looking at the scores. Set `criterion=` to the behavior you actually want to compare.

Keep stable pair IDs and an identical-answer control. Read a seeded random sample as well as cases selected for disagreement, and record how you selected them. When several rows come from one scenario, analyze those rows as a group rather than independent evidence.

### More questions need not mean more information

Ten copies of one balanced yes/no label carry one bit jointly, not ten. A label that never changes carries no variation across items. Conversely, a judge returning .5 for every item expresses uncertainty without distinguishing the items.

The useful question is what an extra measurement distinguishes *after* you know the existing measurements. Jev's isolation between questions does not guarantee independent errors. Probabilities near 0 or 1 can be useful when they change with the behavior of interest; a skewed histogram alone is not a defect.

Keep the vectors as well as their means. Over levels 0–4, `[.5,0,0,0,.5]` and `[0,0,1,0,0]` both have expected score 2. The first splits between extremes; the second puts all probability on the middle. Averaging discards that distinction.

For the fuller information-theory framing, see the [ML-debug measurement guide](https://github.com/wassname/ml-debug/blob/main/references/llm_judges.md#useful-information-redundancy-and-saturation).

## Ask for evidence before ratings

The generative judge follows this order:

1. Explain the comparison and quote exact spans from each answer.
2. Rate each answer from 0 to 3, check specific failures, and choose a preference.
3. Add an unscored note about the judging setup.

This is an attention procedure, like a checklist, as well as an audit trail. The code verifies that quotes occur in the answers. That catches invented quotations; it cannot establish that an interpretation is sound.

With `mode="both"`, Jev and the generative judge see the original inputs, not each other's labels. Their scores remain separate. A critic shown Jev's returned labels answers a different, post-label question and may be influenced by them. Use a judge from a different model family than the student model whose answers it grades. This reduces a known self-preference risk; it does not guarantee independent errors or correctness. See Chen et al., [*Beyond the Surface: Measuring Self-Preference in LLM Judgments*](https://arxiv.org/abs/2506.02592), EMNLP 2025. Inspect disagreements rather than treating a second judge as ground truth.

Implementation: [generative schema, quote checks and recovery](../src/llm_judge/judge.py). For known judge biases and source evidence: [ML-debug judge guide](https://github.com/wassname/ml-debug/blob/main/references/llm_judges.md).

### Debug with an open reasoning judge

When Jev's probability looks wrong, use the generative judge to make the comparison easier to inspect. The default generative model is open-weight DeepSeek. For example, after selecting your project's key:

```sh
uv run llm-judge compare --pairs answers.jsonl --mode gen --passes 1 --output outputs/debug-judge
```

Read the complete first pair in `run.md`: original input, rubric, reasoning returned by the provider, quoted evidence, ratings and final `judge_note`. That last, unscored setup note serves as a short exit interview. Look for a misread action, missing context or an unclear criterion before accepting the score.

To investigate a harness or rendering problem, supply the relevant code/configuration as part of the judged context; the judge cannot inspect files or settings it was never given. For a more focused follow-up, give a reviewer the saved trace and explicitly ask what in the framing, setup or supplied harness prevented a reliable judgment. Keep that feedback separate from the numerical scores. A blind second judgment gets the original inputs without prior labels; a post-label interview gets the actual labels explicitly.

Provider-returned reasoning can help locate a misunderstanding, but it is not an independent correctness check. Some APIs expose no readable reasoning, and opaque/signed replay remains [unverified](#recover-explicitly-preserve-the-failed-attempt). The optional [talk-to-Jev interview](jev.md#experimental-post-label-interview) composes feedback through decisions; it is a different, experimental procedure, not Jev's hidden reasoning.

## Check whether position changes the verdict

A judge that always chooses the first answer appears to favor A until you swap the answers. To expose this, each pair has four presentations: A/B and B/A, each with normal and reversed unordered answer options.

Map the verdicts back to the original answer identities **before averaging**. If Paris was originally A but appeared second, a preference for Paris still counts for original A. Keep the order discrepancy alongside the mean; averaging need not remove all bias.

Reverse unordered Choice options, not ordered Score levels. Reversing “much worse … much better” would change the meaning of the scale rather than just its presentation.

Implementation in [judge.py](../src/llm_judge/judge.py): `make_request`, `schema`, `measurement_scores` and `discrepancy`. Background: Shi et al., [*Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge*](https://arxiv.org/abs/2406.07791), IJCNLP-AACL 2025; [Wang et al., *Large Language Models are not Fair Evaluators*](https://arxiv.org/abs/2305.17926); and [Jev 1.13 limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13).

### Repeat judgments, not the item count

For the generative judge, the default `passes=3` samples each presentation three times. Jev-only mode uses one call per presentation and rejects a different `--passes` value rather than replaying cached answers as if they were fresh draws. Ten original pairs therefore produce 120 generative judgments, but still only ten pair-level results. Average within each original pair, then estimate standard error across pair means. Standard error describes uncertainty in that average under independent-pair assumptions; related scenarios need grouped analysis.

More draws can reduce sampling noise, not a shared misreading. Jev already returns probability vectors and reuses cached answers for a presentation. Fresh repeated Jev calls are useful when testing serving variation; replaying a cached answer is not fresh evidence. Recovery calls and order variants never increase the number of original items.

Background: [Miller, *Adding Error Bars to Evals*](https://www.anthropic.com/research/statistical-approach-to-model-evals).

### Read the score on its own scale

All pair differences refer to **original B minus original A**. Positive favors B:

- `B_minus_A_preference`: −1 to 1. For Jev, P(B)−P(A); for the generative judge, the mean of A=−1, tie=0, B=1.
- Jev `B_minus_A_change`: −2 to 2, the expected comparison-level index centered on zero.
- Generative `B_minus_A_quality`: −3 to 3, the difference between the two 0–3 ratings.

These scales are not interchangeable and none is P(correct). Expected indices use the numerical spacing assigned by the implementation. Separate refusal, incoherence, repetition and off-topic checks help explain a change that one broad score would hide. Disobedience *inside a story* is not refusal of the user's task.

Implementation in [judge.py](../src/llm_judge/judge.py): `measurement_scores`, `summarize` and `FAILURES`.

## Read one complete pair before the aggregate

Start with `run.md`: it includes the full first pair/state, prompts, schemas, exposed provider reasoning, raw replies, errors and order variants, without clipping the sample. Check that the judge saw the intended material and applied the intended criterion. On normal completion or handled failure, the last line instructs the reading LLM to read the entire log, check the full first-sample trace against the stated expectations, and tell the user what it checked and what failed or remains uncertain. A hard process kill can interrupt that final write; a `running` status is not completion.

The other files serve specific purposes:

- `events.jsonl`: incremental requests, replies, errors and cache events, with run IDs.
- `samples.jsonl`: complete Inspect samples from paired comparisons, with run IDs.
- `logs/*.eval`: Inspect's viewer format. Run `uv run inspect view --log-dir outputs/judge/logs`.
- `results.jsonl`: per-pair summaries after every intended judgment succeeds.
- `status.json`: whether the current operation is running, successful or failed.
- `decisions_cache.jsonl`: validated Jev responses and their exact requests.
- `served_models.json` and `endpoint.json`: persisted served identities and the selected generative endpoint.
- `results.<run_id>.previous.jsonl`: an archived aggregate from before the current comparison attempt.

`label` uses the same trace transport without running an Inspect evaluation; its full result is saved in `events.jsonl`. Requests are saved before transport, and raw responses before validation. Writes are flushed to disk; a torn final JSONL fragment is preserved separately when resuming.

Outputs may contain private inputs. Keep them local. Implementation: [Trace and durable transport](../src/llm_judge/judge.py), [paired Inspect integration](../src/llm_judge/judge.py), [Inspect log documentation](https://inspect.aisi.org.uk/).

## Reuse work without counting it as new evidence

Rerun the same output directory to reuse validated caches. Decisions are keyed by the exact request, including model, schema, rubric and option order. Generative cache entries also distinguish source configuration and sampling epoch. Changing the criterion requires new judgments.

Inspect caches model replies before our JSON, range and quote checks. When a reply fails those checks, the runner removes only its matching cache entry and records the key and reason. The raw reply stays in the trace. A later rerun can request a fresh judgment while valid cached judgments remain reusable. This uses the pinned Inspect cache/event API.

Concurrent duplicate decisions share one in-flight request. Cache hits and duplicate waiters incur zero new charges; raw saved responses retain their original usage. Failed requests leave the in-flight table so a later explicit retry can try again. Served-model identity is checked across cached/live replies and persisted between runs.

The paired runner archives a previous aggregate at startup. A failed run cannot leave that old result looking like a current success. Unresolved judging failures invalidate the aggregate instead of silently dropping difficult rows. Concurrency defaults to four. Generative sampling uses temperature .7 to permit repeated draws; this is an example setting, not a task-independent optimum. Inspect gets up to six retries, a 180-second attempt timeout and a 600-second total timeout. Jev transport makes at most six HTTP attempts, each with a 180-second timeout. These are bounded operational defaults; record failures rather than treating timeout rows as valid scores.

Implementation: [cache and charge accounting](../src/llm_judge/judge.py), [run lifecycle](../src/llm_judge/judge.py).

### Recover explicitly; preserve the failed attempt

Transient HTTP/network failures get bounded retries with backoff. Permanent errors retain their raw response and fail.

The initial output allowance defaults to 64,000 tokens, with no forced reasoning effort or `min_p` setting. This is a configurable starting allowance, not a minimum the model must consume. For models that count reasoning against output, a typical 30k-token thought can require substantial extra allowance for longer cases and the final answer. Raise `--max-tokens` for the task; inspect reported usage and truncations rather than assuming one budget suits every model.

If a generative model reaches that limit, save its exposed reasoning and partial answer, then ask once for the structured final answer. For a model documented to support disabling reasoning, the default follow-up requests reasoning off with a 2,048-token final-answer allowance. This cannot finish missing reasoning. An unfinished final answer still fails.

Some models cannot disable reasoning or do not accept a requested effort level. Set `--followup-reasoning` for those models: the follow-up keeps the supported settings and the full initial output allowance rather than sending an unsupported disable switch. An endpoint advertising no reasoning parameter gets this no-switch behavior automatically. Only set `--reasoning-effort` to a value documented for the selected model. The normal finalization phase deliberately replaces the initial effort request with reasoning off; `--followup-reasoning` preserves it instead. There is no automatic retry that drops a rejected control or switches models, but a successful API response cannot prove the provider honored a control. Record every phase and its usage; a mandatory-reasoning follow-up still costs reasoning tokens.

TODO: verify opaque reasoning replay for OpenAI encrypted reasoning and Anthropic signed/redacted thinking through the pinned SDK and adapter. Preserve the complete provider-issued blocks, opaque payloads, signatures and required order unchanged; do not reconstruct them from visible text. Current offline tests establish text-reasoning preservation only, not native-format compatibility. Do not assume follow-up works for those formats until that path is checked. -- PI/OpenAI, from wassname

A malformed JSON reply, invalid field or false quotation gets one validation-error retry. Save the invalid reply; never invent a score, patch it with a regex or turn an unresolved judgment into a clean outcome.

Jev's displayed vectors can have rounding error. Validation allows total mass within .02 of one and checks the Score against its vector with a rounding allowance. It does not silently normalize probabilities.

Implementation in [judge.py](../src/llm_judge/judge.py): `Decisions._fetch`, `generate_judgment` and `validate`.

## Choose credentials and a model explicitly

Paid calls use the selected environment's `OPENROUTER_API_KEY`; custom transports may pass an explicit key to `Decisions`. The paired runner blocks Inspect's automatic `.env` search because it can load another project's parent-directory configuration. Select the intended file yourself with `uv run --env-file /path/to/your-project/.env ...`.

The defaults pin Jev to `typesafe/jev-1.13` and the generative judge to `deepseek/deepseek-v4.1-flash`. Prefer the model creator's endpoint when it supports the required features. Choose for task accuracy and parameter support, not latency alone. A creator-operated endpoint is acceptable even when its quantization is reported as `unknown`. This includes the native endpoints for OpenAI, Anthropic, DeepSeek, Moonshot/Kimi, TypeSafe/Jev, Perplexity and Z.AI. For a third-party host, prefer DeepInfra, Fireworks or Groq, but require documented floating-point precision of at least eight bits: FP8/MXFP8, BF16, FP16 or FP32. Reject unknown precision and lower-bit/integer formats on third-party hosts. A trusted company hosting someone else's model is still a third-party host.

Why care about the endpoint? [Artificial Analysis's endpoint-accuracy methodology](https://artificialanalysis.ai/methodology/endpoint-accuracy-index) describes differences due to quantization, sampling, context handling, token limits and prompt parsers—not precision alone. Its [gpt-oss-120b provider results](https://artificialanalysis.ai/models/gpt-oss-120b/providers?endpoint-accuracy=by-eval#endpoint-accuracy) and [Kimi K3 results](https://artificialanalysis.ai/models/kimi-k3/providers?endpoint-accuracy=by-eval#endpoint-accuracy) illustrate why one provider ranking need not transfer between models.

For precision evidence, [*Give Me BF16 or Give Me Death?*](https://arxiv.org/abs/2411.02355v4), ACL 2025, reports effectively lossless FP8 in its Llama-3.1 experiments. That is evidence for a practical prior, not proof that every FP8 endpoint is accurate or every lower-bit format is bad. The creator/host rule above is wassname's chosen default; native formats and endpoint behavior still need to be understood.

OpenRouter has one flat quantization filter, which cannot express “unknown is acceptable only on a creator endpoint.” Our `select_endpoint` and `provider_options` functions enforce that distinction first. They select a compatible creator endpoint when available, otherwise prefer the named hosts among qualifying endpoints. Required parameters and any advertised output cap must also fit. An unspecified cap is not proof that the full allowance is supported; the API may still reject the request. `--creator` must name the actual model creator as shown in the catalog, not merely a provider you trust.

On the first generative run, the code reads the [model's endpoint catalog](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4.1-flash/endpoints), logs the response and pins a selected endpoint in `endpoint.json`. The public catalog check found DeepInfra FP8 with strict structured outputs and a 131,072-token output allowance; Fireworks listed unknown precision and therefore did not meet the third-party rule. This is advertised capability evidence, not an accuracy comparison. Reruns reuse the saved selection so cache replay requires no new catalog call; use a new output directory to deliberately reselect.

In the request, `only` pins the selected endpoint; `allow_fallbacks=False` prevents another provider silently taking over; and `require_parameters=True` excludes endpoints that cannot honor the requested parameters. Third-party requests also include their verified quantization filter. Creator requests omit that filter so `unknown` is not accidentally excluded. The code checks the returned provider name and records the served identity. These restrictions keep the measuring procedure stable; they do not establish calibration on your task. See [OpenRouter's routing fields](https://openrouter.ai/docs/features/provider-routing#quantization).

Offline commands use fixture credentials/transports and preserve the caller's environment. Inspect's evaluation state and the scoped dotenv guard are process-global, so run concurrent independent evaluations in separate processes. Keep the eight-day dependency-age hold when installing or updating; dependency uv settings do not propagate to consuming projects.

Implementation in [judge.py](../src/llm_judge/judge.py): `select_endpoint`, `provider_options`, `Decisions._fetch` and `run`.

## Additional diagnostics are leads, not scores

The normal vent flags possible issues with the supplied setup. The first Jev comparison/labeling sample also gets a bounded experimental interview by default, with the actual returned answers supplied explicitly; it is separate from the scored questions and their means. The allowance defaults to $.05 observed cost and 75 calls, not a hard billing cap. An expected budget stop retains partial work and valid scores. Callers can explicitly opt out (`--no-feedback`, `compare(dev=False)`, or `label(feedback=False)`); offline smokes do not call live interviews. Generative-only judgments already include their own unscored exit note and do not silently invoke another backend. Additional controls test specific defects. None of these feedback outputs is added to the scores or treated as proof that a judgment is correct.

See [Jev visibility and diagnostic commands](jev.md#what-the-vent-can-see) for examples, explicit spending limits and the interview's known failure modes.

<!-- PI/OpenAI, prepared for wassname. -->
