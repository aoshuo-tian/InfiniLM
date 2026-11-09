"""Restore the same SHA-pinned source bundle and enter the separate TP8 profile."""
import hashlib
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
bundle = ROOT / 'profile-sources.zip'
if hashlib.sha256(bundle.read_bytes()).hexdigest() != 'b815bc413c9fcc95f6f40872358b2a55bb61d51632f3ecc2c76f521f8b092d8b':
    raise ValueError('Profile bundle identity mismatch')
dest = ROOT / 'profile-sources'
dest.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as archive:
    for info in archive.infolist():
        target = (dest / info.filename).resolve()
        if not target.is_relative_to(dest.resolve()) or info.is_dir():
            raise ValueError('Unexpected profile member')
        data = archive.read(info)
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError('Existing source differs')
        else:
            target.write_bytes(data)
sys.path.insert(0, str(dest))
from h3_tp8_profile import main
main(__file__)
