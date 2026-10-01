# sol-pi-hermes

Standalone **Hermes Python plugin** port of [NVlabs/SoL-Pi](https://github.com/NVlabs/SoL-Pi). It is not a PR against `NousResearch/hermes-agent` and it does not import TypeScript.

Install into `~/.hermes/plugins/sol-pi/` (copy or clone this repo there) and enable it:

```bash
hermes plugins enable sol-pi
```

All four mechanisms default **off**, same as SoL-Pi. Write `~/.hermes/sol-pi.json` or `.hermes/sol-pi.json` in a trusted project:

```json
{
  "version": 1,
  "actionFusion": true,
  "observationPack": true,
  "evidencePreservingReducer": false,
  "onlineContextCompact": false
}
```

## Mapping

| Mechanism | Hermes seam | Must not |
| --- | --- | --- |
| ObservationPack | Profile/session-scoped archive + `obs_recall` + `llm_request` middleware (request-time projection) | Private engine lookup; mutating stored history; flattening provider content blocks |
| Action Fusion | `sol_pi_write` / `sol_pi_patch` with optional `then_run` → `write_file`/`patch` + `terminal` | Assume Pi `edit`/`write` names; override built-ins unless the operator sets `allow_tool_override` |
| Evidence-Preserving Reducer | `transform_tool_result` + quote-verify; fail open; skip likely secrets | Replace ObservationPack; send secret-ish logs |
| Online Context Compact | Plan-boundary intent; **compact only on `agent_settled`** around the built-in compressor | Register a second `ContextEngine`; compact on `post_llm_call` |

This plugin **never** calls `ctx.register_context_engine`. ObservationPack uses the existing session-aware request middleware, so Hermes compression and engine selection stay in place. It does not depend on an interactive CLI agent or wrap a shared engine prototype.

Projection currently supports OpenAI-style `messages` with string tool results. Native Responses `input` envelopes, requests containing non-string tool content (including screenshots and cache-decorated text blocks), and requests without a session/task ID pass through unchanged. Other request fields and message metadata are preserved. Config and archives resolve through Hermes's active profile home; a session's archive is never captured at plugin-registration time.

Packing excludes existing Pi-style error flags and Hermes JSON content with a nonempty top-level `error`, `success: false`, or an integer nonzero `exit_code` alongside `output`. JSON-object-looking content that cannot be parsed, including truncated bodies or host-appended subdirectory hints, conservatively stays inline too. Successful JSON logs mentioning errors, and error fields nested inside returned data, remain eligible for packing. This guard does not classify arbitrary plain-text failures or unwrap other provider formats.

`obs_recall` caps each successful UTF-8 JSON response at 16 KiB, including escaping and metadata, while retaining the 400-line content cap. Its offsets always count original archived bytes; smaller pages still reconstruct the full original text exactly.

OCC without `agent_settled` is a no-op. Stock Hermes does not emit that Pi event; enabling `onlineContextCompact` records the gate but will not compact until the host fires the hook.

## Offline validation

Run the standalone suite with `python3 -m unittest discover -s tests`.
The real-host integration tests additionally require `HERMES_SOURCE` pointing to a Hermes checkout and its import dependencies. CI pins Hermes to [`bda33b601d194fb8a43e5660c57542aad14aa372`](https://github.com/NousResearch/hermes-agent/commit/bda33b601d194fb8a43e5660c57542aad14aa372) and builds a disposable Python environment with Hermes PM. Run with that interpreter:

```bash
HERMES_SOURCE=/path/to/hermes-agent /path/to/test-python -m unittest discover -s tests -v
```

The integration test loads this directory plugin through real Hermes discovery, applies real request middleware, and dispatches the registered `obs_recall` tool. It reuses the five existing synthetic CaptureResult fixtures across profiles A→B→A and two session IDs, checking exact first-two sends, smaller third sends, byte-exact paged recall, unchanged request metadata/history, isolation, and unload. On the original PR3 head `b81af04`, the new projection invariant failed: the third provider request was still 70,754 bytes because the non-CLI host had no engine wrapper. A separate legacy-wrapper reproduction packed into `default/` and could not recall from the actual session.

`python3 -m experiments.capture_result_shadow` reports placeholder retention separately from exact recall. These are deterministic integration/storage checks, not live desktop runs, model decisions, tokenizer measurements, cache-cost measurements, or long-session compaction benchmarks. The first two sends count provider attempts; rewriting old tool content later changes the cache prefix. No real-model quality or end-to-end cost improvement is claimed.

Remaining rollout blockers: send counters currently live for the plugin lifetime without eviction. A fresh repeated tool call with the same provider call ID and identical content can inherit an older call's send count and be packed early. Occurrence-safe accounting and session/restart cleanup still need validation before a rollout or model-quality/cost comparison; the current fixes do not resolve those cases.

## Provenance

Algorithms follow NVlabs/SoL-Pi @ `22277b7e` (MIT). Linear: [PER-1507](https://linear.app/0ism/issue/PER-1507). Research: [kvnloo/frontier-kb#10](https://github.com/kvnloo/frontier-kb/pull/10).
