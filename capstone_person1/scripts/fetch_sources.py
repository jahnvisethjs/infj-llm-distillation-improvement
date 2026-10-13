"""Download immutable upstream train Parquet assets; never substitute a newer revision."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'capstone_person1'


def get(url):
    return urlopen(Request(url, headers={'User-Agent': 'person1-data-release/3'}), timeout=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args()
    config = json.loads((PROJECT / 'configs/person1_data_config.json').read_text())
    root = PROJECT / 'sources'
    root.mkdir(exist_ok=True)
    inventory = {}
    for source, spec in config['sources'].items():
        base = 'https://huggingface.co'
        with get(f"{base}/api/datasets/{spec['dataset_id']}/tree/{spec['revision']}?recursive=true&limit=1000") as response:
            tree = json.load(response)
        assets = sorted([entry for entry in tree if entry['type'] == 'file' and entry['path'].endswith('.parquet')
                  and Path(entry['path']).name.startswith('train-')],key=lambda entry:entry['path'])
        if not assets: raise ValueError('No pinned train Parquet files found for ' + source)
        folder = root / source / spec['revision']
        folder.mkdir(parents=True, exist_ok=True)
        entries = []
        for entry in assets:
            # Names are source-relative, validated against the selected local root.
            path = (folder / entry['path']).resolve()
            if not path.is_relative_to(folder.resolve()): raise ValueError('Unsafe upstream asset path')
            record = {'path': entry['path'], 'bytes': entry['size'], 'lfs_sha256': entry.get('lfs', {}).get('oid'),
                      'local_path': str(path.relative_to(PROJECT)).replace('\\', '/')}
            entries.append(record)
        inventory[source] = dict(spec, files=entries, total_bytes=sum(item['bytes'] for item in entries))
        print(source, len(entries), 'files', inventory[source]['total_bytes'], 'bytes', flush=True)
        (root / 'source_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
        if args.inventory_only: continue
        for record in entries:
            path = PROJECT / record['local_path']
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size == record['bytes']:
                digest_obj = hashlib.sha256()
                with path.open('rb') as cached:
                    for chunk in iter(lambda: cached.read(1024 * 1024), b''): digest_obj.update(chunk)
                digest = digest_obj.hexdigest()
                if record['lfs_sha256'] and digest != record['lfs_sha256']: raise ValueError('Cached asset hash mismatch')
                record['sha256'] = digest
                print('Reused:', record['path'], flush=True)
                continue
            pending = path.with_suffix(path.suffix + '.partial')
            url = f"{base}/datasets/{spec['dataset_id']}/resolve/{spec['revision']}/{quote(record['path'])}"
            print('Downloading:', source, record['path'], flush=True)
            digest, count, stamp = hashlib.sha256(), 0, time.monotonic()
            with get(url) as response, pending.open('wb') as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
                    if time.monotonic() - stamp > 20:
                        print(source, record['path'], count, '/', record['bytes'], flush=True)
                        stamp = time.monotonic()
            if count != record['bytes'] or (record['lfs_sha256'] and digest.hexdigest() != record['lfs_sha256']):
                raise ValueError('Upstream asset size/hash mismatch: ' + record['path'])
            pending.replace(path)
            record['sha256'] = digest.hexdigest()
            (root / 'source_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
        for name in ['README.md', 'LICENSE']:
            try:
                with get(f"{base}/datasets/{spec['dataset_id']}/resolve/{spec['revision']}/{name}") as response:
                    (folder / name).write_bytes(response.read())
            except Exception as error:
                print('Optional source card/license unavailable:', source, name, type(error).__name__, flush=True)
    (root / 'source_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print('Saved pinned source inventory.', flush=True)


if __name__ == '__main__': main()
