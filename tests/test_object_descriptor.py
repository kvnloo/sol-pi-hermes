import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from sol_pi_hermes.observation_pack import create_observation, ensure_stored


class ArchiveDescriptorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.obs = create_observation({'role': 'tool', 'content': 'a' * 20000}, self.root)
        ensure_stored(self.obs)

    def test_validation_does_not_reopen_path(self):
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('path reopened')):
            ensure_stored(self.obs)

    def test_oversized_archive_rejected_before_read(self):
        self.obs.file_path.write_bytes(b'a' * 20001)
        with patch('os.fdopen', side_effect=AssertionError('oversized archive read')):
            with self.assertRaises(OSError):
                ensure_stored(self.obs)

    def test_replaced_regular_file_rejected(self):
        original_open = os.open
        def swap(path, flags, *args, **kwargs):
            if Path(path) == self.obs.file_path:
                replacement = self.root / 'replacement'
                replacement.write_text(self.obs.text)
                replacement.replace(path)
            return original_open(path, flags, *args, **kwargs)
        with patch('os.open', side_effect=swap):
            with self.assertRaises(OSError):
                ensure_stored(self.obs)

    @unittest.skipUnless(hasattr(os, 'O_NONBLOCK'), 'requires nonblocking file open')
    def test_fifo_replacement_rejected_without_blocking(self):
        original_open = os.open
        def swap(path, flags, *args, **kwargs):
            if Path(path) == self.obs.file_path:
                self.assertTrue(flags & os.O_NONBLOCK)
                Path(path).unlink()
                os.mkfifo(path)
            return original_open(path, flags, *args, **kwargs)
        with patch('os.open', side_effect=swap):
            with self.assertRaises(OSError):
                ensure_stored(self.obs)

    def test_missing_nonblocking_support_fails_safe(self):
        flag = os.O_NONBLOCK
        del os.O_NONBLOCK
        try:
            with patch('os.open', side_effect=AssertionError('unsafe open')):
                with self.assertRaises(OSError):
                    ensure_stored(self.obs)
        finally:
            os.O_NONBLOCK = flag

    def test_fstat_failure_closes_descriptor(self):
        original_open = os.open
        opened = []
        def track(*args, **kwargs):
            fd = original_open(*args, **kwargs)
            opened.append(fd)
            return fd
        with patch('os.open', side_effect=track), patch('os.fstat', side_effect=OSError('failed')):
            with self.assertRaises(OSError):
                ensure_stored(self.obs)
        self.assertEqual(len(opened), 1)
        with self.assertRaises(OSError):
            os.fstat(opened[0])

    @unittest.skipUnless(hasattr(os, 'O_NONBLOCK'), 'requires nonblocking file open')
    def test_read_is_bounded_when_inode_grows_after_fstat(self):
        original_fstat, original_fdopen, sizes = os.fstat, os.fdopen, []
        def grow(fd):
            result = original_fstat(fd)
            with open(self.obs.file_path, 'ab') as handle:
                handle.write(b'a' * 100000)
            return result
        class Spy:
            def __init__(self, stream): self.stream = stream
            def __enter__(self): return self
            def __exit__(self, *exc): return self.stream.__exit__(*exc)
            def read(self, size=-1):
                sizes.append(size)
                return self.stream.read(size)
        with patch('os.fstat', side_effect=grow), patch('os.fdopen', side_effect=lambda *a, **k: Spy(original_fdopen(*a, **k))):
            with self.assertRaises(OSError):
                ensure_stored(self.obs)
        self.assertEqual(sizes, [self.obs.bytes + 1])
