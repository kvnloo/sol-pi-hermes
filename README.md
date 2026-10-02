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
| ObservationPack | Archive on disk + `obs_recall` + wrap live engine `select_context` (request-time projection) | `transform_tool_result` as the only rewrite; mutating stored history |
| Action Fusion | `sol_pi_write` / `sol_pi_patch` with optional `then_run` → `write_file`/`patch` + `terminal` | Assume Pi `edit`/`write` names; override built-ins unless the operator sets `allow_tool_override` |
| Evidence-Preserving Reducer | `transform_tool_result` + quote-verify; fail open; skip likely secrets | Replace ObservationPack; send secret-ish logs |
| Online Context Compact | Plan-boundary intent; **compact only on `agent_settled`** around the built-in compressor | Register a second `ContextEngine`; compact on `post_llm_call` |

This plugin **never** calls `ctx.register_context_engine`. ObservationPack composes by binding `select_context` on the already-active engine so Hermes compression stays in place.

OCC without `agent_settled` is a no-op. Stock Hermes does not emit that Pi event; enabling `onlineContextCompact` records the gate but will not compact until the host fires the hook.

Action Fusion treats returned Hermes JSON errors, `success: false`, and integer nonzero exit codes as failures. A failed write or patch skips the chained command even when the old target file still exists. A failed command retains its output and reports `[then_run:failed]`; successful and exception-based paths keep their existing behavior. Tests use temporary files and harmless callbacks, with no provider or live shell calls.

Commands that yield to the background or return a nonblank string process `session_id` report `[then_run:pending]`. A background launch's `exit_code: 0` only confirms that it started; it does not prove completion. Reported failures still take precedence. The complete terminal receipt, including the process ID and follow-up instructions, is retained in `output`.

Hermes's existing process notifications or `process` tool own background completion. Action Fusion does not wait, poll, rerun the command, or manage a new process lifecycle. Successful completed foreground results and legacy unstructured callbacks keep their existing behavior.

## Provenance

Algorithms follow NVlabs/SoL-Pi @ `22277b7e` (MIT). Linear: [PER-1507](https://linear.app/0ism/issue/PER-1507). Research: [kvnloo/frontier-kb#10](https://github.com/kvnloo/frontier-kb/pull/10).
