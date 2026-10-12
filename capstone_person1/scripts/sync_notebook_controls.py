"""Keep the standalone Colab notebook's embedded controls equal to the tested module."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / 'Person1_Full_Run_v2.ipynb'
n = json.loads(path.read_text(encoding='utf-8'))
module = (root / 'capstone_person1/pipeline_controls.py').read_text(encoding='utf-8')
for cell in n['cells']:
    source = ''.join(cell['source'])
    if '# BEGIN EMBEDDED PIPELINE CONTROLS\n' not in source:
        continue
    start = source.index('CONTROL_SOURCE = ')
    end = source.index('# END EMBEDDED PIPELINE CONTROLS') + len('# END EMBEDDED PIPELINE CONTROLS')
    replacement = 'CONTROL_SOURCE = ' + repr(module) + '\n# BEGIN EMBEDDED PIPELINE CONTROLS\n' + module + '\n# END EMBEDDED PIPELINE CONTROLS'
    source = source[:start] + replacement + source[end:]
    cell['source'] = source.splitlines(keepends=True)
    cell['outputs'], cell['execution_count'] = [], None
contract = (root / 'capstone_person1/person1_data_contract.py').read_text(encoding='utf-8')
source = ''.join(n['cells'][8]['source'])
if '\n# BEGIN SHARED DATA CONTRACT\n' in source:
    before, block = source.split('\n# BEGIN SHARED DATA CONTRACT\n', 1)
    source = before + block.split('\n# END SHARED DATA CONTRACT\n', 1)[1]
block = '\n# BEGIN SHARED DATA CONTRACT\nCONTRACT_SOURCE = ' + repr(contract) + '\n' + contract + '\n# END SHARED DATA CONTRACT\n'
source = source.replace('# Freeze the entire preprocessing configuration', block + '\n# Freeze the entire preprocessing configuration', 1)
n['cells'][8]['source'] = source.splitlines(keepends=True)
benchmark = (root / 'capstone_person1/benchmark_manifests.py').read_text(encoding='utf-8')
if '\n# BEGIN BENCHMARK MANIFEST INTERFACE\n' in source:
    before, block = source.split('\n# BEGIN BENCHMARK MANIFEST INTERFACE\n', 1)
    source = before + block.split('\n# END BENCHMARK MANIFEST INTERFACE\n', 1)[1]
block = '\n# BEGIN BENCHMARK MANIFEST INTERFACE\nBENCHMARK_SOURCE = ' + repr(benchmark) + '\n' + benchmark + '\n# END BENCHMARK MANIFEST INTERFACE\n'
source = source.replace('# Freeze the entire preprocessing configuration', block + '\n# Freeze the entire preprocessing configuration', 1)
n['cells'][8]['source'] = source.splitlines(keepends=True)
path.write_text(json.dumps(n, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('Notebook controls synchronized.')
