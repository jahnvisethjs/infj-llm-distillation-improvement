import ast
import importlib.util
import json
from pathlib import Path
import jsonschema
from capstone_person1.pipeline_controls import file_sha256

root=Path(__file__).resolve().parents[2]
notebook=json.loads((root/'Person1_Full_Run_v2.ipynb').read_text(encoding='utf-8'))
schema_root=Path(importlib.util.find_spec('nbformat').origin).parent/'v4'
schema=json.loads((schema_root/f'nbformat.v4.{notebook["nbformat_minor"]}.schema.json').read_text(encoding='utf-8'))
jsonschema.Draft4Validator(schema).validate(notebook)
assert notebook['metadata']['person1_entrypoint']=='named_upstream_stages_v3'
code=''
for cell in notebook['cells']:
    if cell['cell_type']!='code':continue
    assert cell['execution_count'] is None and not cell['outputs']
    source=''.join(cell['source'])
    ast.parse('\n'.join(line for line in source.splitlines() if not line.startswith('%')))
    code+=source
assert 'from capstone_person1.release_pipeline import run_release' in code
assert 'exec(' not in code and 'apply_chat_template' not in code
preservation=json.loads((root/'capstone_person1/preserved_inputs/pre_alignment_dce3ef7348754718/preservation.json').read_text(encoding='utf-8'))
assert file_sha256(root/'Person1_Full_Run.ipynb')==preservation['notebook_sha256']
archive=root/'capstone_person1/preserved_inputs/pre_named_stages_v3'
assert file_sha256(archive/'Person1_Full_Run_v2.ipynb')==json.loads((archive/'preservation.json').read_text(encoding='utf-8'))['notebook_sha256']
print('Notebook schema/syntax, named-stage imports, cleared outputs and preserved notebook hashes passed.')
