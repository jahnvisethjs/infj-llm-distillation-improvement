"""Read-only layout/code preservation checks; save their evidence separately."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'capstone_person1'
EVIDENCE = PROJECT / 'evidence'
plan = json.loads((EVIDENCE / 'ORGANIZATION_MANIFEST.json').read_text(encoding='utf-8-sig'))

for relative, expected in plan['stable_code_sha256'].items():
    if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
        raise ValueError('Stable runtime/notebook hash changed: ' + relative)
for item in plan['artifact_inventory']:
    stat = (ROOT / item['path']).stat()
    ticks = stat.st_mtime_ns // 100 + 621355968000000000
    if stat.st_size != item['bytes'] or abs(ticks - item['modified_ticks']) > 2:
        raise ValueError('Saved artifact differs from its pre-organization inventory: ' + item['path'])
for item in plan['moves']:
    if (ROOT / item['old']).exists() or not (ROOT / item['new']).is_file():
        raise ValueError('Relocation incomplete: ' + item['old'])

owned_python = list(PROJECT.glob('*.py')) + list((PROJECT / 'tests').glob('*.py'))
owned_python += list((PROJECT / 'scripts').glob('*.py')) + list((PROJECT / 'scripts/history').glob('*.py'))
for path in owned_python:
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

md_paths = list(ROOT.glob('*.md')) + list(PROJECT.glob('*.md'))
md_paths += list((PROJECT / 'docs').rglob('*.md')) + list((ROOT / 'project_documents').glob('*.md'))
md_paths += list((PROJECT / 'scripts/history').glob('*.md'))
link_count = 0
for path in md_paths:
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
        target = target.strip('<>')
        if '://' in target or target.startswith('#') or re.match(r'^[A-Za-z]:', target): continue
        file_part = target.split('#', 1)[0]
        if not (path.parent / file_part).exists():
            raise ValueError('Broken local Markdown link: ' + str(path.relative_to(ROOT)) + ' -> ' + target)
        link_count += 1

reference = PROJECT / 'docs/reference/Person1_Codex_Implementation_Plan.md'
download = Path('C:/Users/jahnv/Downloads/Person1_Codex_Implementation_Plan.md')
reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
if reference_hash != hashlib.sha256(download.read_bytes()).hexdigest():
    raise ValueError('Copied reference plan differs from the downloaded original')
report = {
    'status': 'passed', 'moved_files': len(plan['moves']),
    'stable_code_and_notebook_hashes_checked': len(plan['stable_code_sha256']),
    'saved_artifact_paths_sizes_and_timestamps_checked': len(plan['artifact_inventory']),
    'python_files_syntax_checked': len(owned_python), 'local_markdown_links_checked': link_count,
    'reference_plan_sha256': reference_hash,
    'implementation_readiness': 'DIAGNOSTIC_ONLY',
    'note': 'Organization does not add completed data-release phases. The scientific pipeline/notebooks and saved run artifacts remain unchanged.',
}
(EVIDENCE / 'ORGANIZATION_VALIDATION.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
plan['paths_repaired'] = True
(EVIDENCE / 'ORGANIZATION_MANIFEST.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
