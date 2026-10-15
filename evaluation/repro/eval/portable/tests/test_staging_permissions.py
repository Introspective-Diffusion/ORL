import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from repro.eval.portable.reproduce import stage_scoring_artifacts


class StagingPermissionsTest(unittest.TestCase):
    def test_nonroot_scorer_can_read_staged_artifacts_under_umask077(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source';source.mkdir(mode=0o700)
            item=source/'source.py';item.write_text('# trusted inert fixture\n');item.chmod(0o600)
            recipe={'artifacts':[{'root':'runtime','path':'nested/source.py','sha256':hashlib.sha256(item.read_bytes()).hexdigest()}]}
            nested=source/'nested';nested.mkdir();item.rename(nested/'source.py')
            old=os.umask(0o077)
            try:
                staged=stage_scoring_artifacts(recipe,{'runtime':source},root/'pinned')
            finally:
                os.umask(old)
            for name in ('runtime','inputs'):
                self.assertEqual(staged[name].stat().st_mode & 0o777,0o755)
            self.assertEqual((staged['runtime']/'nested').stat().st_mode & 0o777,0o755)
            self.assertEqual((staged['runtime']/'nested/source.py').stat().st_mode & 0o777,0o444)
            self.assertEqual(source.stat().st_mode & 0o777,0o700)
            self.assertEqual((nested/'source.py').stat().st_mode & 0o777,0o600)


if __name__=='__main__':unittest.main()
