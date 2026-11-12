"""Shared filesystem/provenance helpers. Standard library only."""
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode('utf-8')


def write_new(path, content):
    path = Path(path)
    if not isinstance(content, bytes):
        content = json_bytes(content)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f'Different output exists: {path}; preserve it and use a new version')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.incomplete')
    temporary.write_bytes(content)
    temporary.replace(path)


def load_rows(path):
    with Path(path).open(encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def config(path=None):
    return read(path or HERE / 'config.json')


def registry():
    return read(HERE / 'parent_registry.json')


def data_paths(root):
    root = Path(root)
    choices = {
        'parent': [root/'data/train_n1000.jsonl', HERE/'inputs_snapshot/train_n1000.jsonl'],
        'universe': [root/'outputs/approach_b_seed42_v1/common_candidate_universe.jsonl', HERE/'inputs_snapshot/common_candidate_universe.jsonl'],
        'contract': [root/'inputs/formatting_contract.json', HERE/'inputs_snapshot/formatting_contract.json'],
        'protected': [root/'inputs/protected_questions.csv', HERE/'inputs_snapshot/protected_questions.csv'],
    }
    found = {}
    for key, paths in choices.items():
        found[key] = next((p for p in paths if p.is_file()), None)
        if found[key] is None:
            raise FileNotFoundError(f'Missing {key}; restore the resume bundle or this package inputs_snapshot')
    return found


def work_root(root):
    return Path(root) / 'outputs' / 'B_workbench_v2_500'


def budget_name(multiplier):
    if type(multiplier) not in (int, float) or multiplier not in [0.5, 1, 2, 4]:
        raise ValueError('Choose a declared continuation budget: 0.5, 1, 2, or 4')
    return f'{multiplier:g}x'


def parse_budget(text):
    choices={'0.5':0.5,'1':1,'2':2,'4':4}
    if text not in choices:
        raise ValueError('Choose a declared continuation budget: 0.5, 1, 2, or 4')
    return choices[text]


def budget_target(parent_tokens, multiplier):
    budget_name(multiplier)
    exact=Fraction(parent_tokens)*Fraction(str(multiplier))
    # All supported targets are integers or exact binary halves; no rounding.
    return exact.numerator if exact.denominator==1 else float(exact)


def profile(size, seed):
    key = f'{size}_seed{seed}'
    if key not in registry():
        raise ValueError(f'Unknown parent profile: {key}')
    return key, registry()[key]


def sources_valid():
    provenance = read(HERE/'source_provenance.json')
    for group in ['vendor_sha256', 'parent_reference_records_sha256']:
        for name, expected in provenance[group].items():
            if sha(HERE/name) != expected:
                raise ValueError(f'Pinned source/record changed: {name}')
    return provenance


def load_person2():
    sources_valid()
    for path in [HERE/'vendor', HERE/'vendor/person2']:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import person2
    return person2


# V3 explicit data routing; original helpers remain intact above.
def data_paths(root):
    cfg=config()
    return {k:Path(v) for k,v in cfg["resolved_data_paths"].items()}
def work_root(root):
    return Path(config()["resolved_work_root"])
def budget_target(parent_tokens,multiplier):
    value=Fraction(config()["budget_reference_tokens"])*Fraction(str(multiplier))
    return value.numerator if value.denominator==1 else float(value)
