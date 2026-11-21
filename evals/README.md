# A/B eval: plain Hermes vs Hermes + sol-pi

This directory runs the same tasks through plain Hermes and through Hermes with sol-pi, on the same model. It reports pass rate and provider-reported input tokens for each. Results live in `results/`.

## What is compared

| arm | toolsets | plugin |
|---|---|---|
| `plain` | `file`, `terminal` | none |
| `sol-pi` | `file`, `terminal`, `sol-pi` | this repo, `sol-pi.json` = ObservationPack on, Action Fusion on |

Both arms get the same prompt, model, iteration cap (24), and generated working directory. Each row gets a fresh `HERMES_HOME` holding only a generated `config.yaml` (plus the plugin and `sol-pi.json` in the sol-pi arm). No credentials, memories, sessions or skills are copied. The child process calls Hermes's own `run_agent.AIAgent.run_conversation`, so plugin discovery, middleware and tool dispatch are Hermes code, not harness code. Arm order rotates per task and repeat.

## Tasks (`tasks.py`)

Each task is generated deterministically and includes at least one file over ObservationPack's 10 KiB threshold:

- `fix_parser`: fix a separator bug in a 23 KB module.
- `multi_fix`: fix one bug in each of two modules (21 KB and 14 KB).
- `config_audit`: pick 53 services out of a 39 KB YAML registry.
- `log_trace`: follow one request id through a 205 KB log, then look up the user in a CSV.
- `chained_edits`: four sequential edits to a 25 KB module, running the tests after each. This is the long-session case ObservationPack is built for.

Grading never trusts files the agent could edit. Test tasks rerun pristine copies of the original tests against the agent's modules. `tests/test_ab_tasks.py` checks that every task fails unsolved, passes with a reference fix, and generates byte-identical files.

## Run it

```sh
export AB_PROVIDER_KEY=...   # an OpenRouter key; :free models cost $0
python3 evals/abharness.py run --tasks evals.tasks --out evals/results/<name> --work /tmp/ab-work \
  --hermes-source ~/.hermes/hermes-agent --hermes-python ~/.hermes/hermes-agent/venv/bin/python --repeats 3
python3 evals/abharness.py report evals/results/<name>
```

The runner refuses a model without a `:free` suffix unless you pass `--allow-paid`. A row that ends on HTTP 429/5xx or a timeout is marked `infra_error` and rerun on resume. The report keeps only the last row per key. `protocol.json` records the harness, task and plugin hashes, the repo and Hermes commits, and the model.

## What the numbers do and don't mean

- **Three token measures**, all sums of provider-reported usage: **prompt** (everything sent), **cache read** (the part served from the provider's prompt cache), and **uncached** (prompt − cache read, which is Hermes's own `input_tokens` counter). ObservationPack lowers prompt but rewrites old history, which costs cache hits, so the report shows all three plus the cache price at which the plugin breaks even.
- **Wall time** depends on the free route's queueing and isn't a result.
- **n is small.** With 5 tasks and 3 repeats, a one-run pass difference is noise. The token ratios are paired (same task, same repeat), and those carry the signal.
- ObservationPack only saves tokens when a large result is still in history at the third and later provider requests after it. If the model greps instead of reading a large file, or the task ends soon after the read, the plugin's three extra tool schemas cost more than it saves. Per-task ratios show which regime each task was in.
