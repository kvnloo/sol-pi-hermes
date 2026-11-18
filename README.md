# sol-pi-hermes

A standalone Hermes plugin that ports two mechanisms from [NVlabs/SoL-Pi](https://github.com/NVlabs/SoL-Pi) to Hermes.

- **ObservationPack.** A tool result over 10 KiB is sent to the model in full twice. On later provider requests it is replaced by a short placeholder: an id plus a 1 KiB excerpt. The model reads it back with `obs_recall` (up to 16 KiB or 400 lines per call, continuing from `next_offset`). Only the outgoing request is changed. Stored history keeps the full result.
- **Action Fusion.** `sol_pi_write` and `sol_pi_patch` run Hermes' own `write_file` / `patch` and, when the call carries `then_run`, run one follow-up command through Hermes' `terminal` tool. The edit and the check then cost one model turn instead of two.

## Install

```bash
hermes plugins install kvnloo/sol-pi-hermes
hermes plugins enable sol-pi
```

Both mechanisms are **off** until you turn them on, as in SoL-Pi. Write `sol-pi.json` in your Hermes home (the active profile's home):

```json
{ "version": 1, "observationPack": true, "actionFusion": true }
```

## How it plugs in

| Mechanism | Hermes surface |
| --- | --- |
| ObservationPack | `llm_request` middleware rewrites `messages` in the provider request copy; `obs_recall` tool |
| Action Fusion | `sol_pi_write` / `sol_pi_patch` tools, dispatched through Hermes' tool registry |

The plugin never registers a context engine, so Hermes' compressor stays in charge. It never patches Hermes code.

SoL-Pi's Evidence-Preserving Reducer and Online Context Compact are ported in `epr.py` / `occ.py` with tests, but they are **not wired** in this release. EPR needs a reducer model, and OCC needs an `agent_settled` hook that Hermes doesn't have.

## What it touches

- **Writes**: archived tool results under `<hermes home>/sol-pi/<session>/observation-pack/` (directory mode 0700, new files 0600), plus a newly created 0600 `ledger.jsonl` next to them. Existing archives/ledgers are not permission-migrated; installations upgrading from older versions should inspect their permissions before enabling ObservationPack. Action Fusion additionally writes the requested target through Hermes write_file/patch in the originating task/session backend.
- **Reads**: `sol-pi.json` in the Hermes home, no host-side reads of Action Fusion target files. PluginContext does not attest backend locality, so the plugin does not hash a host path or promise an atomic edit/check transaction.
- **Commands**: `then_run` runs the model's command through Hermes' `terminal` tool, with the originating task/session identity. Dangerous-command approval and the write denylist still apply, but nested dispatch currently bypasses ACP/Zed editor edit approval, file checkpoints (`/rollback`), and other plugins' `pre_tool_call` guards. Do not enable Action Fusion where those protections are required. Core PR NousResearch/hermes-agent#138933 addresses the hook portion, not all these protections.
- **Network**: none.
- **Credentials**: none.

Action Fusion tools are registered only when `actionFusion` is enabled. Returned mutation failures skip the command; returned command failures are labeled failed. Command results without a completed integer exit code are labeled `then_run:unknown`, not success; no background polling is added. Restart/reload the plugin after changing configuration.

## Provenance

Failure-result handling and task/session forwarding reuse the focused logic from retained downstream commits `015d10fc` (#4) and `12e03247`, without restoring their older host hooks.

The algorithms follow NVlabs/SoL-Pi @ `22277b7e` (MIT).

## Tests

```bash
PYTHONPATH=. python -m unittest discover -s tests
```

### Recall response budget

Successful `obs_recall` responses fit the 16 KiB limit as serialized UTF-8 JSON,
including its envelope and escaped content. Unicode, control characters, quotes
and backslashes can require smaller pages than plain text. The returned `bytes`
and `next_offset` still count original UTF-8 content bytes, not JSON escaping;
continue with `next_offset` until `eof` to recover the complete observation.
The page selection conservatively halves the content budget when needed rather
than maximizing each page. This does not claim a model-token or billing limit.
