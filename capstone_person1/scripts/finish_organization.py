"""Repair relocated support paths and verify preserved runtime/run artifacts."""
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'capstone_person1'
EVIDENCE = PROJECT / 'evidence'
plan = json.loads((EVIDENCE / 'ORGANIZATION_MANIFEST.json').read_text(encoding='utf-8-sig'))
if plan.get('paths_repaired'):
    raise SystemExit('This organization was already applied. Use the README for current commands.')
mapping = {str((ROOT / item['old']).resolve()): (ROOT / item['new']).resolve() for item in plan['moves']}

for item in plan['moves']:
    path = ROOT / item['new']
    if path.suffix != '.py': continue
    text = path.read_text(encoding='utf-8')
    depth = len(Path(item['new']).parts) - 1
    text = re.sub(r'Path\(__file__\)\.resolve\(\)\.parent\.parent\b', f'Path(__file__).resolve().parents[{depth}]', text)
    text = re.sub(r'Path\(__file__\)\.resolve\(\)\.parent\b', f'Path(__file__).resolve().parents[{depth - 1}]', text)
    text = text.replace('from capstone_person1.verify_smoke_run import verify',
                        'from capstone_person1.scripts.verify_smoke_run import verify')
    for filename in ['IMPLEMENTATION_STATUS.json','FIRST_BATCH_VERIFICATION.json','NEXT_STEP_VERIFICATION.json',
                     'REFERENCE_CONTRACT_VERIFICATION.json','SNAPSHOT_CONTRACT_SAMPLE.json']:
        text = text.replace("/ '" + filename + "'", "/ 'evidence/" + filename + "'")
    path.write_text(text, encoding='utf-8')

for folder in [PROJECT, PROJECT / 'tests', PROJECT / 'scripts', PROJECT / 'scripts/history']:
    init = folder / '__init__.py'
    if not init.exists():
        init.write_text('"""' + ('Historical one-time maintenance tools; see README.md.' if folder.name == 'history' else 'Person 1 data preparation package.') + '"""\n', encoding='utf-8')

# Resolve links from their old locations and rewrite only local Markdown targets.
reverse = {str(new): Path(old) for old, new in mapping.items()}
md_paths = [ROOT / item['new'] for item in plan['moves'] if item['new'].endswith('.md')]
md_paths += [PROJECT / 'CODEBASE_ALIGNMENT_PLAN.md', PROJECT / 'PROGRESS_AND_COMPLETION_PLAN.md']
for path in md_paths:
    old_path = reverse.get(str(path.resolve()), path)
    text = path.read_text(encoding='utf-8')
    def link(match):
        label, target = match.group(1), match.group(2)
        raw = target.strip('<>')
        if '://' in raw or raw.startswith('#') or re.match(r'^[A-Za-z]:', raw): return match.group(0)
        base, marker, fragment = raw.partition('#')
        resolved = (old_path.parent / base).resolve()
        relocated = mapping.get(str(resolved), resolved)
        relative = Path(os.path.relpath(relocated, path.parent)).as_posix()
        return '[' + label + '](' + relative + (marker + fragment if marker else '') + ')'
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, text)
    if '/docs/milestones/' in path.as_posix():
        text += '\n> Historical milestone: execution commands and paths describe that milestone. Use [the current package README](../../README.md) for today\'s layout and commands.\n'
    path.write_text(text, encoding='utf-8')

for relative, digest in plan['stable_code_sha256'].items():
    if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
        raise ValueError('Stable scientific code/notebook changed: ' + relative)
for item in plan['artifact_inventory']:
    stat = (ROOT / item['path']).stat()
    # NTFS ticks have 100ns precision; convert the native stat representation.
    ticks = stat.st_mtime_ns // 100 + 621355968000000000
    if stat.st_size != item['bytes'] or abs(ticks - item['modified_ticks']) > 2:
        raise ValueError('Saved artifact changed: ' + item['path'])

status_path = EVIDENCE / 'IMPLEMENTATION_STATUS.json'
status = json.loads(status_path.read_text(encoding='utf-8'))
status['recorded_test_command_before_organization'] = status['test_command']
status['test_command'] = 'python -m unittest discover -s capstone_person1/tests -t . -p "test_*.py"'
status['first_batch_evidence'] = 'evidence/FIRST_BATCH_VERIFICATION.json'
status['evidence']['handoff_readiness'] = 'runs/' + status['successful_diagnostic']['run'] + '/outputs/reports/handoff_readiness.json'
status['workspace_organization'] = {
    'manifest': 'evidence/ORGANIZATION_MANIFEST.json', 'moved_files': len(plan['moves']),
    'stable_runtime_code_and_notebooks_unchanged': True, 'saved_artifact_inventory_unchanged': True,
    'protected_artifact_files': len(plan['artifact_inventory']),
    'implementation_scope': 'file organization; no additional release phase implemented',
}
status_path.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
plan['paths_repaired'] = True
(EVIDENCE / 'ORGANIZATION_MANIFEST.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
print(f'Repaired support paths; checked {len(plan["stable_code_sha256"])} stable code/notebook hashes and {len(plan["artifact_inventory"])} saved artifacts.')
