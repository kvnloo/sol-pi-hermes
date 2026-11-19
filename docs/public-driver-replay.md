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
