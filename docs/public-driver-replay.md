# Public Driver recording replay

This adapter targets Sol-Pi #11 at `252ac9c0d6b360b7b1be4f111ce958d8ea5a9d5d`
and Hermes `09ded63c4fedcc8fe975d37f9765797d904fc75d`. It imports the real
structured-element parser and AX serializer, then calls the offline evaluator.
It does not call a backend, provider, desktop action or mutation tool.

Donors are the sanitized GTK3 and AppKit density-24 recordings at
[trycua/cua c7db4c8](https://github.com/trycua/cua/tree/c7db4c85f65b1575cb4b3fffb9b8effa04bc796f/libs/cua-driver/examples/jev-use/fixtures/native).
The pinned repository's [MIT license](https://github.com/trycua/cua/blob/c7db4c85f65b1575cb4b3fffb9b8effa04bc796f/LICENSE.md)
is Copyright (c) 2025 Cua AI, Inc.; preserve that license if redistributing donor data.
No donor raw data is vendored here. Hashes and exact URLs are in the adapter.

Download the two named JSON files from those exact pinned origins, preserving
bytes. Run with an already installed compatible Hermes interpreter and source:

```sh
"$HERMES_PYTHON" -B -X pycache_prefix="$(mktemp -d)" scripts/replay_public_driver.py gtk3 /tmp/gtk3-window-state-density-24-v1.json --host-root "$HERMES_SRC" --output /tmp/gtk3-replay-new
"$HERMES_PYTHON" -B -X pycache_prefix="$(mktemp -d)" scripts/replay_public_driver.py appkit /tmp/appkit-window-state-density-24-v1.json --host-root "$HERMES_SRC" --output /tmp/appkit-replay-new
```

Output directories must be new. The script verifies the donor SHA-256, host
commit, clean tracked host files and imported module paths. Pin checks also run
under Python `-O`. It uses a disposable Hermes home and copies the public-data
spill to the output only if the spill resolves inside that home.
The control's temporary spill path is deliberately not rewritten. Consequently
control hashes and byte counts may vary with that path. The output is a replay,
**not a frozen original Hermes/provider pair**, native-platform validation, model
or action-equivalence result. `max_elements` explicitly equals donor count, not
Hermes's default cap. No screenshots are invented; absent-image dimensions are 0x0.

Prior disposable replay: GTK3 61 elements, 8,508 control bytes below the 10KiB
threshold; all three sends unchanged, zero savings. AppKit 127 elements, 14,897
control bytes: first two unchanged, third 423 bytes; 14,474 bytes saved across
three sends (32.39%), exact recall true. Estimated tokens were 3,725 each before,
3,725/3,725/106 after. These are character-count estimates, not provider billing.

Read each report's per-send fields, not only totals. AppKit's packed send exposed
zero element indices (127 before), and all top-level structured control fields
were missing until recall. Byte-exact recall does not establish decision equivalence.
The host serializer already omits opaque element tokens (61/127 normalized tokens,
zero in either control). AppKit's long label triggers normal 120-character clipping
and a full-label spill; this is host serialization, not Sol-Pi loss.

No remote publication, catalog admission, policy settlement, or issue closure is
implied. Real original paired-capture and semantic-equivalence gates remain open.

The adapter supplies `control.raw_text_payload`: the exact UTF-8 JSON-object
string returned by the pinned serializer, without parsing and reserializing the
control before evaluation. Its whitespace, Unicode escapes and number spelling
therefore contribute unchanged to byte/token estimates and recall comparison.
Fixtures must select exactly one of `raw_text_payload` (a JSON-object string),
legacy `text_payload` (an object serialized with `json.dumps` defaults), or
`content_blocks` (a native block list). Existing fixture forms retain their
behavior. This preserves an offline string representation, not proof that a
provider received it, nor an original same-call capture or decision-equivalence
claim. The parser still reads the control separately to locate its public spill.

## Handoff for a future original pair (issue #2)

For [issue #2](https://github.com/kvnloo/sol-pi-hermes/issues/2), start with one
owner-authorized AX-only capture of a synthetic window. Use the pinned Hermes
commit above, ordinary capture settings, and no ObservationPack projection on the
executed request. This section describes evidence to obtain, not evidence already
collected. Use an already authorized recorder; if it cannot observe both boundaries,
leave the case unpaired rather than instrumenting production or reconstructing it.

1. At [the normalized capture boundary][capture-boundary], record the exact
   `CaptureResult` returned by `backend.capture`, before `_capture_response`.
   Associate it with the enclosing session and tool call. For this first slice,
   use a direct `capture` call, not an action-triggered follow-up capture.
2. From the same call's actual submitted provider request, extract just its
   tool-result content and correlate the outbound tool-call ID with the same
   capture/session. The [tool-message assembly][tool-message] may transform the
   tool result before it reaches the [non-streaming chat transport][transport];
   neither a persisted transcript nor a fresh serializer invocation proves what
   was sent. For text, preserve the exact UTF-8 bytes of the decoded outbound
   content string, including whitespace, Unicode escapes within the string and
   number spelling. This is the tool-content byte sequence, not the wire JSON
   envelope or proof that the provider received or acted on it.
3. In a private fixture, record the host SHA, capture mode/settings, pairing
   identifiers, and the outbound content's SHA-256 and UTF-8 byte length.
   Use `capture` for the normalized object, `tool_call_id` for the correlated
   call, and `control.raw_text_payload` for the original JSON-object string;
   include the case `name`. At top level, set `schema_version` to 1, record
   the pinned commit as `host_main_sha`, and include `provenance` and a nonempty
   `cases` array. Recheck the
   byte length and digest after reading the fixture back. A separate native
   block case would use only `control.content_blocks`, preserving list order
   and block fields; it is not part of this AX-only first acquisition. Do not
   rewrite control strings or strip fields to fit the schema: label an
   incompatible control unsupported. Do not record headers, credentials,
   unrelated messages, or actual user/desktop data. Keep fixtures private; if a
   pair needs redaction, collect again against a safe synthetic window instead
   of treating the edited output as byte-exact original evidence.

Manually reject mismatched or uncorrelatable event IDs, byte/hash mismatches,
and serializer-only/reconstructed controls from the original-pair evidence set.
The evaluator enforces fixture shape and replay invariants, not acquisition
provenance. Once a pair qualifies, run the offline shadow evaluator with that
fixture: the original request stays the control, the first two simulated sends
must stay unchanged, and any third-send projection and recall are measured
separately. Compare bytes, element/index/token retention, latency and exact
recall; do not claim native-platform, model, or action equivalence from these
checks. Keep issue #2 open until its remaining evidence gates are met.

[capture-boundary]: https://github.com/NousResearch/hermes-agent/blob/09ded63c4fedcc8fe975d37f9765797d904fc75d/tools/computer_use/tool.py#L398-L404
[tool-message]: https://github.com/NousResearch/hermes-agent/blob/09ded63c4fedcc8fe975d37f9765797d904fc75d/agent/tool_executor.py#L1154-L1164
[transport]: https://github.com/NousResearch/hermes-agent/blob/09ded63c4fedcc8fe975d37f9765797d904fc75d/agent/chat_completion_helpers.py#L756-L789
