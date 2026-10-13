"""Package the verified release without changing its immutable files."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    release = project / 'releases/person1_v3_20261012_ready'
    metadata = json.loads((release/'metadata/release_manifest.json').read_text(encoding='utf-8'))
    assert metadata['immutable_after_finalization'] and metadata['readiness']['data_ready_for_trainer']
    assert release not in output.parents and output != release
    output.mkdir(parents=True, exist_ok=True)
    groups = {name: [] for name in ['01-training-and-handoff', '02-canonical-orca',
              '03-canonical-numina', '04-numina-canonical-shards', '05-processing-provenance',
              '06-pinned-upstream-sources', '07-pinned-tokenizer']}
    for path in sorted(release.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(release).as_posix()
        if relative.startswith('processed/canonical_orca'):
            group = '02-canonical-orca'
        elif relative == 'processed/canonical_numina.parquet':
            group = '03-canonical-numina'
        elif relative.startswith('processed/canonical_numina/'):
            group = '04-numina-canonical-shards'
        elif relative.startswith('processed/') and path.name not in {'row_status_audit.parquet', 'problem_group_mapping.parquet'}:
            group = '05-processing-provenance'
        else:
            group = '01-training-and-handoff'
        expected = metadata['files'].get(relative)
        if expected and (path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']):
            raise ValueError('Original release artifact differs: ' + relative)
        groups[group].append((path, 'capstone_person1/releases/'+release.name+'/'+relative))
    for path in sorted((project/'sources').rglob('*')):
        if path.is_file():
            groups['06-pinned-upstream-sources'].append((path, 'capstone_person1/'+path.relative_to(project).as_posix()))
    snapshot = project/'cache/huggingface/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987'
    for name in ['tokenizer.json', 'tokenizer_config.json']:
        path = snapshot/name
        groups['07-pinned-tokenizer'].append((path, 'capstone_person1/'+path.relative_to(project).as_posix()))

    def package(item):
        group, paths = item
        target = output/('person1-v3-'+group+'.zip')
        if target.exists():
            raise ValueError('Archive already exists; do not overwrite distribution evidence')
        if sum(path.stat().st_size for path, _ in paths) >= 2*1024**3:
            raise ValueError('Archive grouping exceeds GitHub per-asset limit')
        with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for path, name in paths:
                archive.write(path, name)
        entries = []
        with zipfile.ZipFile(target) as archive:
            assert archive.namelist() == [name for _, name in paths]
            for path, name in paths:
                with archive.open(name) as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                expected = digest(path)
                if actual != expected:
                    raise ValueError('Archive round-trip changed ' + name)
                entries.append(dict(path=name, bytes=path.stat().st_size, sha256=actual))
        result = dict(name=target.name, bytes=target.stat().st_size, sha256=digest(target),
                      uncompressed_bytes=sum(x['bytes'] for x in entries), files=entries)
        assert result['bytes'] < 2*1024**3
        print(f'Packaged and verified {target.name}: {result["bytes"]/1024**2:.1f} MiB, {len(entries)} files', flush=True)
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        assets = sorted(pool.map(package, groups.items()), key=lambda item: item['name'])
    release_files = [x for a in assets for x in a['files'] if '/releases/'+release.name+'/' in x['path']]
    assert len(release_files) == len([p for p in release.rglob('*') if p.is_file()])
    result = dict(release_id=release.name, status='all_archives_round_trip_verified',
                  original_release_manifest_sha256=digest(release/'metadata/release_manifest.json'),
                  assets=assets, extraction='Extract all ZIP files into the repository root; paths recreate the complete original release, upstream sources and pinned tokenizer.')
    (output/'DATA_DOWNLOADS.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    (output/'SHA256SUMS.txt').write_text(''.join(a['sha256']+'  '+a['name']+'\n' for a in assets), encoding='utf-8')
    print('Complete data bundle packaged, including pinned upstream sources and tokenizer.', flush=True)


if __name__ == '__main__':
    main()
