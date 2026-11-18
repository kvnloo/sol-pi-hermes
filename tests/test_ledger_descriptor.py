import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from sol_pi_hermes.observation_pack import append_ledger, ledger_path


@unittest.skipUnless(hasattr(os, 'mkfifo') and hasattr(os, 'O_NONBLOCK'), 'POSIX FIFO checks')
class LedgerDescriptorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = ledger_path(self.root)
        self.path.parent.mkdir(parents=True)

    def test_fifo_rejected_before_lock_or_write(self):
        os.mkfifo(self.path)
        reader = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            with patch('sol_pi_hermes.observation_pack._try_lock', side_effect=AssertionError('nonregular lock')):
                with self.assertRaises(OSError):
                    append_ledger(self.root, {'event': 'synthetic'})
            self.assertEqual(os.read(reader, 4096), b'')
        finally:
            os.close(reader)

    def test_fifo_substituted_at_open_is_rejected_and_fd_closed(self):
        self.path.write_bytes(b'')
        real_open = os.open
        fds = []
        def swap(path, flags, *args, **kwargs):
            self.assertTrue(flags & os.O_NONBLOCK)
            self.path.unlink()
            os.mkfifo(self.path)
            fd = real_open(path, flags, *args, **kwargs)
            fds.append(fd)
            return fd
        with patch('os.open', side_effect=swap):
            with self.assertRaises(OSError):
                append_ledger(self.root, {'event': 'synthetic'})
        self.assertEqual(len(fds), 1)
        with self.assertRaises(OSError):
            os.fstat(fds[0])

    def test_missing_nonblocking_support_fails_before_open(self):
        flag = os.O_NONBLOCK
        del os.O_NONBLOCK
        try:
            with patch('os.open', side_effect=AssertionError('unsafe open')):
                self.assertIsNone(append_ledger(self.root, {'event': 'synthetic'}))
        finally:
            os.O_NONBLOCK = flag
