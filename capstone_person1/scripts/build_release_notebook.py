"""Build a thin local/Colab entrypoint; preserve the former embedded diagnostic notebook."""
import json
import shutil
from pathlib import Path

from capstone_person1.pipeline_controls import file_sha256

ROOT=Path(__file__).resolve().parents[2]
archive=ROOT/'capstone_person1/preserved_inputs/pre_named_stages_v3'
archive.mkdir(parents=True,exist_ok=True)
path=ROOT/'Person1_Full_Run_v2.ipynb'
if not (archive/path.name).exists():
    shutil.copy2(path,archive/path.name)
    (archive/'preservation.json').write_text(json.dumps({'notebook_sha256':file_sha256(path),'description':'Previous embedded v2 diagnostic entrypoint, preserved before named-stage migration'},indent=2),encoding='utf-8')
cells=[]
def add(kind,source):
    cell={'cell_type':kind,'metadata':{},'source':source.splitlines(keepends=True)}
    if kind=='code': cell.update(execution_count=None,outputs=[])
    cells.append(cell)
add('markdown', '''# Person 1 — pinned upstream data release

This notebook uses the same importable, tested stage functions as the local CLI. It prepares data without loading model weights. The original notebook and former diagnostic v2 are preserved. Read `capstone_person1/README.md` and the resulting `PERSON2_HANDOFF.md`.

Local: open from the workspace root. Colab: upload/unpack the `code/` folder of the package and set `PROJECT_ROOT` to that folder. Enable the dependency cell in a clean kernel. Download pinned sources and tokenizer assets with `PREPARE_INPUTS=True`. Mount Drive only when explicitly enabled below. Start with `PILOT_ROWS=10000`; use `None` for a new full release. Complete the exported review queues, then rerun the same unchanged release to finalize. Finalized releases are immutable.
''')
add('code', '''# Run in a clean Colab environment when dependencies are not already installed.
%pip install numpy==2.3.5 pandas==2.3.3 pyarrow==21.0.0 transformers==4.57.6 huggingface-hub==0.36.0 tokenizers==0.22.2 datasketch==2.0.0 jsonschema==4.25.0
''')
add('code', '''import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get('CAPSTONE_PROJECT_ROOT', str(Path.cwd()))).resolve()
MOUNT_DRIVE = os.environ.get('CAPSTONE_MOUNT_DRIVE', '0') == '1'
if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
sys.path.insert(0, str(PROJECT_ROOT))
DATA_PROJECT = PROJECT_ROOT / 'capstone_person1'
assert (DATA_PROJECT / 'configs/person1_data_config.json').is_file(), 'Set PROJECT_ROOT to the workspace or extracted package code directory'
''')
add('code', '''PREPARE_INPUTS = False
if PREPARE_INPUTS:
    import subprocess
    subprocess.run([sys.executable, '-m', 'capstone_person1.scripts.fetch_sources'], cwd=PROJECT_ROOT, check=True)
    from huggingface_hub import hf_hub_download
    from capstone_person1.person1_data_contract import TOKENIZER_ID, PINNED_TOKENIZER_REVISION
    for filename in ['tokenizer.json', 'tokenizer_config.json']:
        hf_hub_download(TOKENIZER_ID, filename, revision=PINNED_TOKENIZER_REVISION, cache_dir=DATA_PROJECT / 'cache/huggingface')
''')
add('code', '''from datetime import datetime, timezone
from capstone_person1.release_pipeline import run_release

PILOT_ROWS = 10000  # None for full upstream processing
RELEASE_NAME = 'person1_v3_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
RELEASE_ROOT = DATA_PROJECT / 'releases' / RELEASE_NAME
# To resume, set RELEASE_ROOT to the previous unfinalized directory.
audit = run_release(RELEASE_ROOT, pilot=PILOT_ROWS)
audit['readiness']
''')
add('markdown', '''Review `reports/formatted_target_review.csv`, `reports/answer_extraction_review.csv` and `reports/duplicate_and_overlap_reports/pair_policy_review.csv`. Record actual identity/time and case-specific decisions; do not auto-populate approvals. Rerun the same release with unchanged code/config after review. A pilot remains `DIAGNOSTIC_ONLY`. A full release can reach `DATA_READY_FOR_TRAINER` without Person 2 actual labels or a parent. Frozen Person 3 evaluation inputs and trainer verification remain separate prerequisites for main runs.
''')
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'person1_entrypoint':'named_upstream_stages_v3'},'nbformat':4,'nbformat_minor':4}
path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
print(path)
