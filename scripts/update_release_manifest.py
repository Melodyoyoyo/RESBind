"""Refresh current tracked-file checksums; retain original extraction provenance."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
path = root / 'release_manifest.json'
manifest = json.loads(path.read_text(encoding='utf-8'))
# Model artifacts must remain identical to the original checked bundle.
for name, expected in manifest['copied_sources'].items():
    if name.startswith('models/'):
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
names = subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
manifest['files'] = {name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                     for name in sorted(names) if name and name != path.name}
path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(f"Recorded {len(manifest['files'])} tracked files; bundle bytes unchanged.")
