# SoL-Pi matched live A/B protocol (spec only; dry run available)

**No live runner or provider access is implemented.** No benchmark instances
were fetched, run or scored. The checked-in example emits fixed fake telemetry
and illustrative pass rates; it makes no efficacy or non-inferiority claim.

```sh
python -B scripts/hermes_ab_dry_run.py --dry-run > /tmp/sol-pi-ab-fake.json
python -B -m unittest discover -s tests -p test_hermes_ab_dry_run.py
```

The report format is [schema v1](../schema/hermes-ab-dry-run.v1.schema.json).
It contains three invented benign task labels and four matched replicates per
task (48 jobs total). Neither the tasks nor their fabricated outcomes are
SWE-bench results. In each block, arms share the task and scheduling seed and
appear in rotated order; the seed is not a Hermes CLI or provider seed.

## Freeze before any live execution

Choose a reviewed **SWE-bench Verified** subset restricted to ordinary
application/library fixes (for example, parsing, formatting and sorting).
Exclude security exploitation, intrusion, credential discovery, attack payloads
and tasks requiring network access from the tool sandbox. Freeze dataset
revision, instance IDs, grading harness/container digests, prompts and their
hashes, Hermes and SoL-Pi SHAs, provider/model/revision/route, inference settings,
approved budget, replicate count, seeds and execution order before any outcome.
Decide sample size with a task-level power simulation from historical or
holdout data before opening the evaluation set; if too few reviewed tasks are
available, label the resulting experiment exploratory. Four replicates per task
are planned; changes require a new pre-registration before collecting results.

Use isolated pristine workspaces and separate disposable Hermes profiles per
job. No host mounts, secrets exposed to tools, write/rollback guard bypasses,
or tool network access. Enforce filesystem/process boundary outside the agent,
including nested Action Fusion dispatch. Check this independently before any live
job; the plugin's nested dispatch is not a substitute for host isolation.
Keep the grade harness and gold patches inaccessible during the job and score
resulting workspace diffs with the pinned external benchmark tests. Quit if
provider authorization, cost ceiling, sandbox or collector validation is absent.

| Arm | observationPack | actionFusion |
| --- | --- | --- |
| base | false | false |
| observation_pack | true | false |
| action_fusion | false | true |
| both | true | true |

All arms install the same plugin pin and use version-1 policy;
`evidencePreservingReducer` and `onlineContextCompact` stay false. At the
SoL-Pi baseline `a892e7b9f363612f0739b5d3bc2561e15cce7fa2`, `obs_recall` is
registered even when packing is off. Retain that schema in all four arms to
compare flag configurations on the same harness, not stock Hermes versus a
plugin install. Fusion tools are absent in arms with Fusion off, and that
tool-schema difference is part of the treatment. Check the effective schema
and configuration for each arm once before recording outcomes.

At Hermes pin `32dbc1dcda2255fd35969ec99cfc0170988c3eca`, the
[chat parser](https://github.com/NousResearch/hermes-agent/blob/32dbc1dcda2255fd35969ec99cfc0170988c3eca/hermes_cli/_parser.py)
supports the following command shape. This is illustrative, not executed;
the dry run only serializes the variable-free argv preview:

```sh
# Future authorized harness only: set up isolated profile and sandbox first.
HERMES_HOME="$JOB_PROFILE" hermes chat -q "$REVIEWED_PROMPT" --oneshot \
  --format stream-json --toolsets terminal,file,sol-pi --max-turns 30 --run-budget 600
```

Freeze 30 tool iterations and 600-second Hermes budget for all arms; enforce
the same hard outer wall-clock limit independently. Pin non-treatment knobs,
identical prompts per task, and randomize blocks reproducibly, balancing order
within each task. The schedule seed does not imply provider determinism; pass
a provider seed only if the pinned route actually supports it, and record if
not. Preserve within-job prompt caching. Require documented between-job cache
isolation for a causal cold-start cache comparison; otherwise report shared
cache interference and do not claim independent cold-start effects.

## Capture contract for the future harness

Join `pre_api_request`, `post_api_request` and `api_request_error` by the
same session and request ID. Store only the allowlisted usage fields, status,
turn/request identifiers and timing. Never copy the hook's prompt, content,
tool arguments, raw provider payload, secrets or credentials.
[Hook normalization](https://github.com/NousResearch/hermes-agent/blob/32dbc1dcda2255fd35969ec99cfc0170988c3eca/agent/api_request_hooks.py)
derives four token buckets from response usage. Retain normalized
`input_tokens`, `cache_read_tokens`, `cache_write_tokens`, and
`output_tokens` per observed request; absent provider fields are unknown,
not inferred zeros. At this pin, hook normalization can turn absent raw
buckets into scalar zero and does not expose raw field presence. Confirm
bucket support for the frozen provider route independently; if presence
cannot be established, mark that bucket unknown or omit the comparison.
Freeze/document bucket semantics and validate retry visibility before
admitting data; do not sum input and cache buckets without verifying the
pinned route's semantics. Disable or
separately account for auxiliary requests via the
[observer contract](https://github.com/NousResearch/hermes-agent/blob/32dbc1dcda2255fd35969ec99cfc0170988c3eca/website/docs/developer-guide/observer-hooks.md).
Report collector gaps rather than pretending SDK-internal retries are visible.

[Stream JSON](https://github.com/NousResearch/hermes-agent/blob/32dbc1dcda2255fd35969ec99cfc0170988c3eca/hermes_cli/stream_json.py)
reports aggregate token stats in the terminal result, not per-request usage.
Use it for completion/exit reconciliation, not to reconstruct individual
requests. Keep stream content private; the report publishes only metrics.
Record one user turn per one-shot job, count completed model-response turns
separately from request attempts, and use monotonic elapsed time from process
start to exit for job wall time. Grade pass from the external SWE-bench
harness, not the assistant's text or CLI exit. Timeouts and ordinary agent
failures count as failures. Predefine infrastructure-only faults; if one
invalidates a matched block, retry the entire block once under the same
settings, then exclude an incomplete block consistently across arms and report
all exclusions and sensitivity treating missing outcomes as failures.

## Primary decision rule (pre-registered proposal)

For each task, average the paired run-level differences in binary pass
(`both - base`) over its matched replicates, then take the equally weighted
mean of the task differences. Use a paired cluster percentile bootstrap:
resample **tasks** with replacement, preserving all paired arms and replicates
inside each sampled task; 10,000 draws with fixed bootstrap seed 20261120.
Use the 5th percentile of draws as the one-sided 95% lower confidence bound.
Declare non-inferiority only if that bound is strictly greater than
**−3 percentage points**. Precheck method performance and sample size without
inspecting evaluation outcomes; insufficient precision is inconclusive.
Report the point estimate, bound, sample size, failures and missingness.
Report both single-feature arms and all other contrasts descriptively, without
secondary non-inferiority claims. The dry run computes fake pass rates only;
its inferential fields stay null regardless of those rates.
