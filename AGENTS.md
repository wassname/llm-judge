# LLM-as-judge defaults

Personal research tools and guidance for wassname, not a general evaluation framework. Keep the API small, composable and readable; explain consequential defaults beside their implementation. Preserve core durability, numerical evidence-first judging and broad unscored diagnostics.

- Read README.md, then src/llm_judge/judge.py for the complete normal procedure. Public functions: Trace, compare, label. Optional paid diagnostics live in talk_to_jev.py.
- Run `uv sync --locked` and `uv run python -m unittest discover -s tests/regression -v`. Offline checks: `uv run llm-judge smoke` and `uv run llm-judge diagnose controls --offline --budget .01`.
- Keep the eight-day package-age hold. Never borrow another project's credentials or run live diagnostics without an explicit allowance. Offline commands use fixture transports, not model inference.
- Keep generated logs/caches under outputs/; plans, reviews and machine-only work under .local/. Do not commit slop/, credentials or private dataset traces.
- Sign agent-authored comments/docs. No public updates without the owner's editorial approval. Do not turn a diagnostic flag, self-critique or `none` into verified correctness.

-- PI/OpenAI
