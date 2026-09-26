"""Download, verify and safely restore PRISM release assets (Python standard library)."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def safe_target(root, name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe archive path: {name}')
    target = (root / Path(*p.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f'Archive path escapes destination: {name}')
    return target

def restore(archive, package, root):
    root = Path(root).resolve()
    if archive.stat().st_size != package['size'] or sha256(archive) != package['sha256']:
        raise ValueError(f'Archive checksum mismatch: {archive.name}')
    expected = {f['path']: f for f in package['files']}
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError('Archive file list does not match manifest')
        # Check all conflicts before extracting any file from this package.
        targets = {}
        for name in names:
            target = safe_target(root, name)
            item = expected[name]
            if z.getinfo(name).file_size != item['size']:
                raise ValueError(f'Unexpected size: {name}')
            if target.exists() and (not target.is_file() or sha256(target) != item['sha256']):
                raise FileExistsError(f'Refusing to replace different existing content: {target}')
            targets[name] = target
        for name, target in targets.items():
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix='.prism-', dir=target.parent)
            temporary = Path(temporary)
            try:
                with os.fdopen(fd, 'wb') as out, z.open(name) as source:
                    shutil.copyfileobj(source, out, 8 * 1024 * 1024)
                if sha256(temporary) != expected[name]['sha256']:
                    raise ValueError(f'File checksum mismatch: {name}')
                # Exclusive creation: an existing destination can never be overwritten.
                with target.open('xb') as out, temporary.open('rb') as source:
                    shutil.copyfileobj(source, out, 8 * 1024 * 1024)
            finally:
                temporary.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category', choices=['all','models','data','caches'], default='all')
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1]/'release/manifest.json')
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive-dir', type=Path, default=Path(__file__).resolve().parents[1]/'.downloads')
    parser.add_argument('--verify-only', action='store_true', help='Verify archive and restored files without writing')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    packages = [p for p in manifest['packages'] if args.category in ('all',p['category'])]
    for package in packages:
        archive = args.archive_dir/package['name']
        if not archive.exists():
            if args.verify_only: raise FileNotFoundError(archive)
            if not package.get('url'): raise ValueError(f"Download URL not yet published: {package['name']}")
            args.archive_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix='.download-', dir=args.archive_dir)
            try:
                request = urllib.request.Request(package['url'],headers={'User-Agent':'PRISM-reproduction/1.0'})
                with os.fdopen(fd,'wb') as out, urllib.request.urlopen(request,timeout=120) as response:
                    shutil.copyfileobj(response,out,8*1024*1024)
                if Path(tmp).stat().st_size != package['size'] or sha256(tmp) != package['sha256']:
                    raise ValueError('Downloaded archive failed integrity check')
                if archive.exists(): raise FileExistsError(archive)
                Path(tmp).rename(archive)
            finally:
                Path(tmp).unlink(missing_ok=True)
        if args.verify_only:
            if archive.stat().st_size != package['size'] or sha256(archive) != package['sha256']:
                raise ValueError(f'Archive checksum mismatch: {archive}')
            for item in package['files']:
                target = safe_target(args.destination,item['path'])
                if not target.is_file() or sha256(target) != item['sha256']:
                    raise ValueError(f'Missing or differing restored file: {target}')
        else:
            restore(archive,package,args.destination)
        print(f"Verified {package['name']}",flush=True)

if __name__=='__main__':
    main()
