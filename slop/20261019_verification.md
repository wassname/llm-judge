# Initial release verification

PI/OpenAI. Functional checks, not a model-accuracy or calibration benchmark.

- Locked install with the eight-day package-age hold; built wheel contains the broad216 diagnostic vocabulary and all package modules. A separate noneditable installation also ran the offline smoke successfully.
- 21 maintained regression tests pass. They cover mapped response/option views, independent generative epochs, zero-call cache replay, criterion invalidation, quote/JSON recovery, preserved provider reasoning, provider restrictions, persistent model identity, charge accounting, explicit retry after exhaustion, run attribution, diagnostic budgets and fixture credential isolation.
- Both documented offline CLI commands pass: `uv run llm-judge smoke` and `uv run llm-judge diagnose controls --offline --budget .01`. They use simulated replies, not paid inference.
- A fresh consumer independently executed the CLI checks and twenty simultaneous rubrics over nested input, assembled a 4x5 result, and inspected interview-budget and truncated/invalid trace cases. This checks usability and execution, not rubric validity.
- An independent Kimi reviewer reran the tests and identified two P2 issues: failed requests stayed in the pending table, and samples lacked run IDs. Both were reproduced as failing regressions, repaired, and independently rechecked. Follow-up verdict on 82d90cf: **OK**, with no new scoring, mapping or cache-key defect.
- Astra's oracle opinion led to clearer backend/lifecycle documentation, explicit paid first-pair-only feedback coverage, and a short study recipe rather than a universal measurement abstraction. The earlier requested `--dev` shortcut remains opt-in; separate saved-result feedback is available.
- The final minor CLI change uses the explicit fixture key instead of mutating the caller's environment; its added regression passes. Raw requests/provider replies remain untouched; cache replay and duplicate waiters have zero new charges.

Public material is source, tests, examples, concise guidance, lockfile and provenance. Generated/private traces and credentials are excluded. Existing gist revisions remain historical records; the repository is the maintained source of truth. No dataset credentials or new paid judge calls were used in this migration.
