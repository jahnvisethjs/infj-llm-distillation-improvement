"""Preserve the actual pre-alignment inputs and create the distinct v2 entrypoint once."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from capstone_person1.pipeline_controls import file_sha256

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / 'capstone_person1'
source = ROOT / 'Person1_Full_Run.ipynb'
target = ROOT / 'Person1_Full_Run_v2.ipynb'
digest = file_sha256(source)
archive = PROJECT / 'preserved_inputs' / ('pre_alignment_' + digest[:16])
archive.mkdir(parents=True, exist_ok=True)
copies = [source, PROJECT / 'pipeline_controls.py', PROJECT / 'streaming_rebuild.py', PROJECT / 'snapshot_stream.py']
for path in copies:
    destination = archive / path.name
    if destination.exists():
        if file_sha256(destination) != file_sha256(path):
            raise ValueError(f'Preserved input would be overwritten: {destination}')
    else:
        shutil.copy2(path, destination)
metadata = archive / 'preservation.json'
if not metadata.exists():
    metadata.write_text(json.dumps({
        'preserved_utc': datetime.now(timezone.utc).isoformat(),
        'notebook_sha256': digest,
        'meaning': 'Actual locally available, previously modified pre-alignment input; not the untouched original Colab notebook',
        'untouched_original_notebook_recovery': 'pending; unavailable in the local workspace',
        'files': {path.name: file_sha256(path) for path in copies},
    }, indent=2), encoding='utf-8')
if not target.exists():
    notebook = json.loads(source.read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'], cell['execution_count'] = [], None
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('Preserved input:', archive)
print('Corrected notebook entrypoint:', target)
