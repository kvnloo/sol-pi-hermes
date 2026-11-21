# sol-pi vs plain Hermes, 2026-11-21

Nemotron 3 Ultra (`nvidia/nemotron-3-ultra-550b-a55b:free` on OpenRouter, $0), Hermes `791df314d2`, 5 tasks × 2 arms × 3 repeats = 30 runs. Full tables are in [RESULTS.md](RESULTS.md) and per-run data is in `rows.jsonl`.

## Headline

**sol-pi sends fewer prompt tokens but loses prompt-cache hits, so it costs more on any provider whose cache discount is deep.** Accuracy didn't change: plain passed 15/15 and sol-pi 14/15.

| measure, summed over 30 runs | plain | sol-pi | change |
|---|---|---|---|
| prompt tokens sent | 1,931,064 | 1,649,200 | −14.6% |
| served from the provider's prompt cache | 1,359,040 (70%) | 815,424 (49%) | |
| uncached prompt tokens | 572,024 | 833,776 | +45.8% |

Paired by task and repeat, the median sol-pi run sent 0.71× plain's prompt tokens and 1.19× plain's uncached tokens. Priced as uncached + p × cached, sol-pi is cheaper only when a cached token costs more than 48% of an uncached one.

## Why the numbers come out this way

1. **The mechanism works on the wire.** `wire-probe.txt` is a scripted run against a local fake OpenAI server (`evals/wire_probe.py`, no network): one 46 KB `read_file`, then five small calls. Plain Hermes resends the 46 KB result on every request. sol-pi sends it twice, then a 516-char placeholder. Request body bytes over 7 requests drop 43% (measured).
2. **It costs about 490 tokens per request.** sol-pi adds three tool schemas (`obs_recall`, `sol_pi_write`, `sol_pi_patch`): 1,948 bytes on every request, measured in the probe.
3. **The placeholder breaks the prompt cache.** The provider caches request prefixes. ObservationPack rewrites an old tool result in place when it turns into a placeholder, so everything after it misses the cache. Hermes's own `input_tokens` counter is the uncached part (prompt − cache read, checked on every row). That is why the uncached total rises while the prompt total falls.
4. **One task flipped on model behavior, not on packing.** In `log_trace`, the sol-pi runs read the whole 205 KB log in 2 of 3 repeats, while plain runs grepped it every time (84K vs 1.8K mean tool-result chars). That one task adds 324K prompt tokens to sol-pi's total. Whether sol-pi's tool descriptions caused the choice is a guess; n = 3 can't tell it apart from sampling noise. Without `log_trace`, sol-pi sends 32.7% fewer prompt tokens, sends 14.3% more uncached tokens, and breaks even at a cache price of 11.4%.

So on a route that discounts cached input to 10% (for example Anthropic's cache-read price), sol-pi is roughly break-even at best on these tasks. On a route without prompt caching, it saves 15–33% of input. Both claims are inferred from the token counts above, not measured on those routes.

## Accuracy

The one failure (`config_audit`, sol-pi, repeat 0) wrote 65 names instead of 53. It wrote `answer.txt` on its second request, while the file contents were still in full view (the ledger shows the placeholder first appearing on request 3). So packing didn't hide the data it needed (inferred from the ledger order).

## What limits these numbers

- n = 3 per task and arm. One run can move a task's mean by a third.
- Synthetic tasks, one model, one free route. Free-route caching behavior may differ from paid routes.
- Wall time reflects free-route queueing and is not a result.
- `prompt_tokens` and `cache_read_tokens` were added to `rows.jsonl` after the run, from each run's saved Hermes result (`backfilled_from`). It's the same dict the harness now records directly. `input_tokens` matched on all 30 rows before backfilling. The report code changed after the run, and the run itself used harness commit `824e7ae`.
