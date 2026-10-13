"""Upload verified data assets using existing GitHub Git credentials in memory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.parse import quote, urlparse
import requests

REPOSITORY = 'jahnvisethjs/infj-llm-distillation-improvement'
TAG = 'person1-data-v3-20261012'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def headers(repo):
    environment = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')
    result = subprocess.run(['git', 'credential', 'fill'], cwd=repo,
        input='protocol=https\nhost=github.com\n\n', capture_output=True, text=True,
        env=environment, timeout=60)
    fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if result.returncode or not fields.get('password'):
        raise RuntimeError('No usable saved GitHub credential; authenticate GitHub before retrying.')
    return {'Authorization': 'Bearer '+fields['password'], 'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2026-03-10'}


def check(response, expected):
    if response.status_code not in expected:
        raise RuntimeError(f'GitHub request failed: HTTP {response.status_code}; credential/response bodies are not printed.')
    return response.json() if response.status_code != 204 else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--check-auth', action='store_true')
    args = parser.parse_args()
    repo = args.repo.resolve()
    auth = headers(repo)
    base = 'https://api.github.com/repos/'+REPOSITORY
    info = check(requests.get(base, headers=auth, timeout=60), {200})
    if not info.get('permissions', {}).get('push'):
        raise RuntimeError('Saved GitHub account lacks push permission to the requested repository.')
    print('GitHub repository write access verified; credentials remain in memory.', flush=True)
    if args.check_auth:
        return
    assets_root = args.assets.resolve()
    distribution = json.loads((assets_root/'DATA_DOWNLOADS.json').read_text(encoding='utf-8'))
    assert distribution['status'] == 'all_archives_round_trip_verified'
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    remote = check(requests.get(base+'/branches/person1%2Fdata-alignment', headers=auth, timeout=60), {200})
    if remote['commit']['sha'] != head:
        raise RuntimeError('Push the current commit to person1/data-alignment before creating its release.')
    readme = (repo/'capstone_person1/docs/DATA_DOWNLOADS.md').read_text(encoding='utf-8')
    releases = check(requests.get(base+'/releases?per_page=100', headers=auth, timeout=60), {200})
    matches = [r for r in releases if r['tag_name'] == TAG]
    if matches:
        release = matches[0]
        if not release['draft']:
            raise RuntimeError('Release is already published; do not overwrite it.')
    else:
        release = check(requests.post(base+'/releases', headers=auth, json=dict(tag_name=TAG,
            target_commitish=head, name='Person 1 complete data release v3', body=readme,
            draft=True, prerelease=False), timeout=60), {201})
    print('Draft release ready; uploading verified datasets.', flush=True)
    existing = {item['name']: item for item in release.get('assets', [])}
    candidates = [(assets_root/a['name'], a['sha256']) for a in distribution['assets']]
    candidates += [(assets_root/name, digest(assets_root/name)) for name in ['DATA_DOWNLOADS.json', 'SHA256SUMS.txt']]
    upload = release['upload_url'].split('{', 1)[0]
    assert urlparse(upload).hostname == 'uploads.github.com'
    assert urlparse(upload).path == '/repos/'+REPOSITORY+'/releases/'+str(release['id'])+'/assets'

    def upload_asset(item):
        path, sha = item
        if digest(path) != sha:
            raise ValueError('Local archive changed: '+path.name)
        old = existing.get(path.name)
        if old:
            if old.get('state') == 'uploaded' and old.get('digest') == 'sha256:'+sha and old['size'] == path.stat().st_size:
                print('Reusing verified uploaded asset '+path.name, flush=True)
                return old
            if old.get('state') == 'starter' or old['size'] == 0:
                check(requests.delete(base+'/releases/assets/'+str(old['id']), headers=auth, timeout=60), {204})
            else:
                raise RuntimeError('Existing asset does not match '+path.name)
        size = path.stat().st_size
        print(f'Uploading {path.name}: {size/1024**2:.1f} MiB', flush=True)
        for attempt in range(3):
            try:
                with path.open('rb') as stream:
                    response = requests.post(upload+'?name='+quote(path.name), headers=dict(auth,
                        **{'Content-Type': 'application/zip' if path.suffix == '.zip' else 'application/octet-stream',
                           'Content-Length': str(size)}), data=stream, timeout=(60, 1800))
                result = check(response, {201})
                if result['size'] != size or result.get('digest') != 'sha256:'+sha:
                    raise RuntimeError('Uploaded asset digest or size differs: '+path.name)
                print('Upload verified '+path.name, flush=True)
                return result
            except requests.RequestException:
                if attempt == 2:
                    raise RuntimeError('Upload network failure for '+path.name) from None
                # Inspect the draft before retrying to avoid duplicate assets.
                items = check(requests.get(release['assets_url'], headers=auth, timeout=60), {200})
                found = next((x for x in items if x['name'] == path.name), None)
                if found and found.get('digest') == 'sha256:'+sha and found['size'] == size:
                    return found
                if found:
                    raise RuntimeError('Partial upload requires inspection before retry: '+path.name) from None
                time.sleep(2)
        raise RuntimeError('Upload incomplete')

    with ThreadPoolExecutor(max_workers=3) as pool:
        uploaded = list(pool.map(upload_asset, candidates))
    published = check(requests.patch(base+'/releases/'+str(release['id']), headers=auth,
        json={'draft': False, 'make_latest': 'false'}, timeout=60), {200})
    result = dict(status='published', html_url=published['html_url'], tag=TAG, commit=head,
        assets=[{key: item[key] for key in ['name', 'size', 'digest', 'browser_download_url']} for item in uploaded])
    (assets_root/'PUBLICATION_RESULT.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit(str(error)) from None
