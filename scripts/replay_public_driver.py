"""Offline replay, not a recorded Hermes/provider pair or action-equivalence test."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HOST = '09ded63c4fedcc8fe975d37f9765797d904fc75d'
DONOR = 'c7db4c85f65b1575cb4b3fffb9b8effa04bc796f'
FIXTURES = {
    'gtk3': ('gtk3-window-state-density-24-v1.json', 'dc5efba2f0eb2df166d3361c20838b427485fedd97c2fa60c59e08d619776730'),
    'appkit': ('appkit-window-state-density-24-v1.json', 'b7e44f38a653b41515bf05c1b6a0465ad6ecb30c5d691734badd8a9a6af07f64'),
}

def verify_digest(raw, expected):
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Donor bytes differ from pin')


def verify_host(host, expected=HOST):
    host = host.resolve(strict=True)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(host), *args], text=True).strip()
    if git('rev-parse', '--show-toplevel') != str(host) or git('rev-parse', 'HEAD') != expected:
        raise ValueError('Host differs from pin')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Host tracked worktree/index is not clean')
    return host


def verify_import(module, root, relative):
    if Path(module.__file__).resolve() != root / relative:
        raise ValueError(f'Unexpected import origin: {module.__name__}')


def spill_bytes(path, home):
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_relative_to(Path(home).resolve()) or not resolved.is_file():
        raise ValueError('Spill is outside temporary Hermes home or is not a file')
    return resolved.read_bytes()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('kind', choices=FIXTURES)
    ap.add_argument('fixture', type=Path)
    ap.add_argument('--host-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    name, expected = FIXTURES[args.kind]
    raw = args.fixture.read_bytes()
    verify_digest(raw, expected)
    host = verify_host(args.host_root)
    plugin_root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(host), str(plugin_root)]
    donor = json.loads(raw)
    with tempfile.TemporaryDirectory(prefix='sol-pi-public-replay-') as home:
        os.environ['HERMES_HOME'] = home
        from tools.computer_use import backend, cua_backend_parse, tool
        from sol_pi_hermes import observation_eval as evaluator
        for module, relative in [(tool, 'tools/computer_use/tool.py'),
                                 (backend, 'tools/computer_use/backend.py'),
                                 (cua_backend_parse, 'tools/computer_use/cua_backend_parse.py')]:
            verify_import(module, host, relative)
        verify_import(evaluator, plugin_root, 'sol_pi_hermes/observation_eval.py')
        elements = cua_backend_parse._parse_elements_from_structured(donor['elements'])
        cap = backend.CaptureResult(mode='ax', width=0, height=0, elements=elements,
                            app=donor['app_name'], window_title=donor['window_title'])
        control_text = tool._capture_response(cap, max_elements=len(elements))
        control = json.loads(control_text)
        provenance = {
            'kind': 'public Driver recording replay, NOT original Hermes/provider capture',
            'source': f'https://github.com/trycua/cua/blob/{DONOR}/libs/cua-driver/examples/jev-use/fixtures/native/{name}',
            'source_sha256': expected, 'host_sha': HOST,
            'control_sha256': hashlib.sha256(control_text.encode()).hexdigest(),
            'mode': 'AX, no image, 0x0 as screenshot-absent backend, uncapped element count',
            'limits': 'No provider, desktop action, native platform validation, or decision equivalence. Spill path is temporary and control hash may vary.',
        }
        recordings = {'schema_version': 1, 'host_main_sha': HOST, 'provenance': provenance,
            'cases': [{'name': f'public-{args.kind}-density24-replay',
                       'capture': dataclasses.asdict(cap), 'control': {'raw_text_payload': control_text},
                       'tool_call_id': 'offline-public-replay'}]}
        report = evaluator.evaluate_recordings(recordings)
        spill = control.get('elements_file')
        spill_content = spill_bytes(spill, home) if spill else None
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'fixture.json').write_text(json.dumps(recordings, indent=2)+'\n', encoding='utf-8')
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        (args.output/'control.json').write_bytes(control_text.encode('utf-8'))
        if spill_content is not None:
            # Preserve the public donor's host-generated full-label spill, not a private file.
            (args.output/'host-spill.json').write_bytes(spill_content)
        print(json.dumps({'case': args.kind, 'elements': len(elements), 'totals': report['totals']}))

if __name__ == '__main__':
    main()
