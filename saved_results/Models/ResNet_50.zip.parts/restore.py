from pathlib import Path
import hashlib, json, shutil
folder = Path(__file__).resolve().parent
manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8-sig'))
target = folder.parent / manifest['original']
if target.exists():
    raise SystemExit(f'Already exists: {target}')
with target.open('wb') as out:
    for name in manifest['parts']:
        with (folder / name).open('rb') as src:
            shutil.copyfileobj(src, out, 8 * 1024 * 1024)
h = hashlib.sha256()
with target.open('rb') as src:
    for chunk in iter(lambda: src.read(8 * 1024 * 1024), b''):
        h.update(chunk)
if h.hexdigest().lower() != manifest['sha256'].lower():
    raise SystemExit('Checksum mismatch')
print(f'Restored and verified: {target}')
