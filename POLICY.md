# SoL-Pi shared host-port policy (v1)

This tree is a **host port**, not a third plugin ABI.

Pi and Oh My Pi load TypeScript `ExtensionAPI` factories. Hermes loads Python
`register(ctx)` with `plugin.yaml`. Those surfaces drift independently. Share
**policy** — the `sol-pi.json` object and ObservationPack constants — across
[`kvnloo/sol-pi-omp`](https://github.com/kvnloo/sol-pi-omp) and this repo. Do
not invent a Hermes↔Pi adapter, a shared binary plugin format, or extra
operator keys that NVlabs/SoL-Pi would reject.

Vault: `perm-20260911-hermes-pi-plugin-adapter-is-host-port-not-abi`.

Machine-readable contract: [`schema/sol-pi.policy.v1.schema.json`](schema/sol-pi.policy.v1.schema.json).
Conservative instance (local mechanisms only): [`schema/sol-pi.policy.v1.json`](schema/sol-pi.policy.v1.json).

## Operator file: `sol-pi.json`

Version **1** keys match [NVlabs/SoL-Pi](https://github.com/NVlabs/SoL-Pi)
`src/sol-pi/config.ts` / `sol-pi.example.json`:

| Key | Type | Role |
| --- | --- | --- |
| `version` | `1` | Only supported version. Required when the file exists. |
| `actionFusion` | boolean | Optional follow-up `then_run` on file mutation. |
| `observationPack` | boolean | Archive large tool results; project placeholders after full sends. |
| `evidencePreservingReducer` | boolean | Quote-verified reduction of long diagnostic logs. |
| `evidencePreservingReducerProvider` | non-empty string | Optional EPR model route (Pi registry). |
| `evidencePreservingReducerModel` | non-empty string | Optional EPR model id. |
| `onlineContextCompact` | boolean | Compact at a plan boundary only after settle. |
| `cacheWriteReadRatio` | finite `>= 0` number | OCC economics; default `12.5`. |

Omitted feature keys default to `false` in SoL-Pi and in this Hermes port.
Unknown keys, a version other than `1`, or wrong types fail the config loader.
Do **not** add `FULL_SENDS`, `thresholdBytes`, or host-specific fields to the
operator file.

Host search paths differ; the object does not:

- Hermes: `.hermes/sol-pi.json` (project) or `~/.hermes/sol-pi.json`
- OMP: `.omp/sol-pi.json` (trusted project) or `~/.omp/agent/sol-pi.json`
- Stock SoL-Pi: `.pi/sol-pi.json` or `~/.pi/agent/sol-pi.json`

Missing-file defaults may differ by host (SoL-Pi and Hermes: all off; OMP
phase-0: Action Fusion + ObservationPack on so a loadable subset is useful
without `findCutPoint`). The **file schema** is the same.

## ObservationPack constants (not operator keys)

NVlabs/SoL-Pi hardcodes these. Both host ports must match. They are **not**
`sol-pi.json` keys.

| Constant | Value | Meaning |
| --- | --- | --- |
| `FULL_SENDS` | `2` | Provider requests that still carry the full payload before the placeholder. |
| `thresholdBytes` | `10240` | Only tool results larger than `10 * 1024` bytes participate. |

See `$defs.observationPackConstants` in the JSON Schema. Do not remint the
ObservationPack algorithm to expose these as tunables.

## What this is not

- Not a PR against `NVlabs/SoL-Pi`, `can1357/oh-my-pi`, or `NousResearch/hermes-agent`.
- Not a reminted ObservationPack. Projection stays request-time; stored history
  is not truncated; `transform_tool_result` is not the pack seam.
- Not quote-verify or fusion-queue as a new wire protocol. Those remain
  mechanism internals (EPR quotes, Action Fusion `then_run` queue).
