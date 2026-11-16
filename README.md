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

- **Writes**: archived tool results under `<hermes home>/sol-pi/<session>/observation-pack/` (directory mode 0700, files 0600), plus a `ledger.jsonl` next to them. Nothing outside the Hermes home.
- **Reads**: `sol-pi.json` in the Hermes home, and the file it just wrote (to hash it before `then_run`).
- **Commands**: `then_run` runs the model's command through Hermes' `terminal` tool, so Hermes' approvals apply to it as they do to any terminal call.
- **Network**: none.
- **Credentials**: none.

## Provenance

The algorithms follow NVlabs/SoL-Pi @ `22277b7e` (MIT).

## Tests

```bash
PYTHONPATH=. python -m unittest discover -s tests
```
