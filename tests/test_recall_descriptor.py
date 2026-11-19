import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
from sol_pi_hermes.observation_pack import ledger_path, observation_path, read_recall_chunk, recall

class RecallDescriptorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.path=self.root/'archive';self.other=self.root/'outside'
        self.path.write_text('archive');self.other.write_text('synthetic outside sentinel')

    def read(self):return read_recall_chunk(self.path,0,max_bytes=100,max_lines=100)

    def test_symlink_swapped_at_open_is_never_read(self):
        real_path_open=Path.open;real_os_open=os.open;swapped=False
        def swap():
            nonlocal swapped
            if not swapped:self.path.unlink();self.path.symlink_to(self.other);swapped=True
        def path_open(path,*args,**kw):
            if path==self.path:swap()
            return real_path_open(path,*args,**kw)
        def os_open(path,*args,**kw):
            if Path(path)==self.path:swap()
            return real_os_open(path,*args,**kw)
        with patch.object(Path,'open',path_open),patch('os.open',os_open):
            with self.assertRaises(OSError):self.read()
        self.assertTrue(swapped)
        self.assertEqual(self.other.read_text(),'synthetic outside sentinel')

    def test_regular_replacement_at_open_is_rejected(self):
        real_path_open=Path.open;real_os_open=os.open;swapped=False
        def swap():
            nonlocal swapped
            if not swapped:self.other.replace(self.path);swapped=True
        def path_open(path,*args,**kw):
            if path==self.path:swap()
            return real_path_open(path,*args,**kw)
        def os_open(path,*args,**kw):
            if Path(path)==self.path:swap()
            return real_os_open(path,*args,**kw)
        with patch.object(Path,'open',path_open),patch('os.open',os_open):
            with self.assertRaises(OSError):self.read()
        self.assertTrue(swapped)

    def test_static_nonregular_objects_rejected(self):
        self.path.unlink();self.path.symlink_to(self.other)
        with self.assertRaises(OSError):self.read()
        self.path.unlink();self.path.mkdir()
        with self.assertRaises(OSError):self.read()

    def test_normal_file_reads_and_eof(self):
        result=self.read()
        self.assertEqual(result['text'],'archive');self.assertTrue(result['eof'])

    def test_symlink_swap_without_nofollow_still_rejected(self):
        real_open=os.open;opened=[]
        def swapped_open(path,flags,*args,**kw):
            self.path.unlink();self.path.symlink_to(self.other)
            fd=real_open(path,flags,*args,**kw);opened.append(fd);return fd
        with patch.object(os,'O_NOFOLLOW',0,create=True),patch('os.open',swapped_open):
            with self.assertRaises(OSError):self.read()
        self.assertEqual(len(opened),1)
        with self.assertRaises(OSError):os.fstat(opened[0])

    def test_opened_nonregular_mode_with_matching_identity_rejected(self):
        import stat
        real_fstat = os.fstat
        def chr_fstat(fd):
            r = real_fstat(fd)
            return os.stat_result((stat.S_IFCHR | 0o600,) + tuple(r)[1:])
        with patch('os.fstat', chr_fstat):
            with self.assertRaisesRegex(OSError, 'changed while opening'):
                self.read()

    @unittest.skipUnless(hasattr(os,'mkfifo') and hasattr(os,'O_NONBLOCK'),'POSIX FIFO required')
    def test_fifo_swap_is_nonblocking_and_descriptor_closed(self):
        real_open=os.open;opened=[]
        def swapped_open(path,flags,*args,**kw):
            self.assertTrue(flags & os.O_NONBLOCK)  # Fail safely rather than hang under a mutant.
            self.path.unlink();os.mkfifo(self.path)
            fd=real_open(path,flags,*args,**kw);opened.append(fd);return fd
        with patch('os.open',swapped_open):
            with self.assertRaises(OSError):self.read()
        self.assertEqual(len(opened),1)
        with self.assertRaises(OSError):os.fstat(opened[0])


class RecallEndToEndTests(unittest.TestCase):
    def test_recall_swapped_symlink_returns_error_without_outside_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rt"
            obs_id = "obs_" + "a" * 24
            path = observation_path(root, obs_id)
            path.parent.mkdir(parents=True)
            path.write_text("archive")
            outside = Path(tmp) / "outside"
            outside.write_text("synthetic outside sentinel")
            real_open, real_path_open, swaps = os.open, Path.open, []

            def swap(p):
                if Path(p) == path and not swaps:
                    path.unlink(); path.symlink_to(outside); swaps.append(1)

            def swapped_open(p, *args, **kw):
                swap(p)
                return real_open(p, *args, **kw)

            def swapped_path_open(p, *args, **kw):
                swap(p)
                return real_path_open(p, *args, **kw)

            with patch("os.open", swapped_open), patch.object(Path, "open", swapped_path_open):
                result = recall(root, obs_id)
            self.assertNotIn("synthetic outside sentinel", result)
            self.assertTrue(swaps)
            self.assertIn("error", json.loads(result))
            self.assertFalse(ledger_path(root).exists())
