"""Method B: parent-derived, right-closed strata and constrained random utility.

No student scores, embeddings, or evaluation accuracy enter selection.
The only third-party dependency is the separately installed OR-Tools solver.
"""
import bisect
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

VERSION = 'method-b-document-alignment-v3'
ORTOOLS_VERSION = '9.14.6206'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def write(path, value):
    path = Path(path)
    data = value if isinstance(value, bytes) else json_bytes(value)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f'Preserved existing different file: {path}. Use a new study version.')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.incomplete')
    tmp.write_bytes(data)
    tmp.replace(path)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def jsonl(data):
    return b''.join((json.dumps(r, ensure_ascii=False) + '\n').encode() for r in data)


def csv_bytes(data):
    if not data:
        return b''
    out = io.StringIO(newline='')
    w = csv.DictWriter(out, fieldnames=list(data[0]))
    w.writeheader()
    w.writerows(data)
    return out.getvalue().encode()


def fraction(value):
    return value if isinstance(value, Fraction) else Fraction(str(value))


def number(value):
    return value.numerator if isinstance(value, Fraction) and value.denominator == 1 else float(value)


def bounds(target, tolerance):
    target, tolerance = fraction(target), fraction(tolerance)
    return math.ceil(target * (1 - tolerance)), math.floor(target * (1 + tolerance))


def hash32(text):
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:4], 'big')


def utility(row_id, seed=42):
    # floor(1_000_000 * first32 / 2**32 + 0.5), using integer arithmetic.
    return (hash32(f'{seed}|random|{row_id}') * 1_000_000 + 2**31) // 2**32


def tie_utility(row_id, seed=42):
    return hash32(f'{seed}|tie|{row_id}')


def parent_edges(parent):
    result = {}
    for source in sorted({r['dataset_id'] for r in parent}):
        values = sorted(r['supervised_tokens'] for r in parent if r['dataset_id'] == source)
        if not values:
            raise ValueError(f'Empty parent source: {source}')
        cuts = []
        for j in range(1, 10):
            position = Fraction((len(values) - 1) * j, 10)
            low, high = math.floor(position), math.ceil(position)
            cut = values[low] + (values[high] - values[low]) * (position - low)
            if cut not in cuts:
                cuts.append(cut)
        result[source] = cuts
    return result


def rebin(data, edges):
    out = []
    for r in data:
        if r['dataset_id'] not in edges:
            raise ValueError('Candidate source absent from parent: ' + r['dataset_id'])
        # Equality belongs to the lower interval: (-inf, e1], (e1, e2], ...
        out.append(dict(r, length_bin=bisect.bisect_left(edges[r['dataset_id']], r['supervised_tokens'])))
    return out


def edge_report(edges):
    return {'reference_population': 'actual R1000 parent, separately by source',
            'quantiles': [j / 10 for j in range(1, 10)], 'interpolation': 'linear',
            'endpoint_rule': 'right-closed; bisect_left; outer edges -inf and +inf',
            'source_inner_edges': {s: [str(x) for x in cuts] for s, cuts in edges.items()}}


def cell(row):
    return row['dataset_id'], row['length_bin']


def validate(data, cap=2048):
    if not data:
        raise ValueError('Empty dataset')
    for key in ('row_id', 'problem_group_id'):
        ids = [r[key] for r in data]
        if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
            raise ValueError(f'Empty/repeated {key}')
    for r in data:
        if r['eos_tokens'] != 1 or r['supervised_tokens'] != r['completion_tokens'] + 1:
            raise ValueError('Completion/EOS count mismatch: ' + r['row_id'])
        if r['total_tokens'] != r['prompt_tokens'] + r['supervised_tokens']:
            raise ValueError('Total token count mismatch: ' + r['row_id'])
        if r['total_tokens'] > cap or r.get('boundary_crossing_token') is True:
            raise ValueError('Overlength or boundary-crossing row: ' + r['row_id'])


def quotas(parent, multiplier):
    """Exact parent counts at 1x/2x/4x; explicit largest-remainder 0.5x extension."""
    m = fraction(multiplier)
    counts = Counter(cell(r) for r in parent)
    ideals = {h: n * m for h, n in counts.items()}
    target_count = len(parent) * m
    if target_count.denominator != 1:
        raise ValueError('Requested total example count is fractional')
    result = {h: math.floor(x) for h, x in ideals.items()}
    order = sorted(ideals, key=lambda h: (-(ideals[h] - result[h]), h))
    for h in order[:int(target_count) - sum(result.values())]:
        result[h] += 1
    return result


def solve(parent, bank, multiplier=1, *, seed=42, deterministic_limit=120,
          tie_limit=10, global_tolerance='0.005', stratum_tolerance='0.05'):
    import ortools
    from ortools.sat.python import cp_model
    if ortools.__version__ != ORTOOLS_VERSION:
        raise ValueError(f'Expected OR-Tools {ORTOOLS_VERSION}, found {ortools.__version__}')
    validate(parent); validate(bank)
    for key in ('row_id', 'problem_group_id'):
        if {r[key] for r in parent} & {r[key] for r in bank}:
            raise ValueError('Parent/bank overlap: ' + key)
    ordered = sorted(bank, key=lambda r: r['row_id'])
    counts = quotas(parent, multiplier)
    reference = defaultdict(int)
    for r in parent:
        reference[cell(r)] += r['supervised_tokens']
    groups = defaultdict(list)
    for i, r in enumerate(ordered):
        groups[cell(r)].append(i)
    # Some quantile intervals can be empty in a tied/discrete parent. Candidates
    # in such intervals remain in the bank but have a strict zero-row quota.
    for h in set(groups) - set(counts):
        counts[h] = 0
        reference[h] = 0
    target = sum(reference.values()) * fraction(multiplier)
    low, high = bounds(target, global_tolerance)
    report = {'version': VERSION, 'ortools_version': ortools.__version__,
              'multiplier': number(fraction(multiplier)), 'selection_seed': seed,
              'target_rows': sum(counts.values()), 'target_supervised_tokens': number(target),
              'global_integer_bounds': [low, high], 'global_tolerance': global_tolerance,
              'stratum_tolerance': stratum_tolerance,
              'random_utility': 'floor(1000000 * first32(SHA256(seed|random|row_id))/2^32 + 0.5)',
              'strata': [], 'student_scores_used': False, 'embeddings_used': False,
              'uniform_over_feasible_subsets_claimed': False}
    model = cp_model.CpModel()
    variables = [model.new_bool_var(f'x_{i:06d}') for i in range(len(ordered))]
    for h in sorted(counts):
        ix = groups.get(h, [])
        lo, hi = bounds(reference[h] * fraction(multiplier), stratum_tolerance)
        record = {'source': h[0], 'length_bin': h[1], 'required_rows': counts[h],
                  'candidate_rows': len(ix), 'parent_tokens': reference[h], 'token_bounds': [lo, hi]}
        report['strata'].append(record)
        if len(ix) < counts[h]:
            report.update(status='INFEASIBLE_CAPACITY', infeasibility_proven=True)
            return None, report
        lengths = sorted(ordered[i]['supervised_tokens'] for i in ix)
        minimum = sum(lengths[:counts[h]])
        maximum = sum(lengths[-counts[h]:]) if counts[h] else 0
        if minimum > hi or maximum < lo:
            report.update(status='INFEASIBLE_STRATUM_TOKEN_BOUNDS', infeasibility_proven=True)
            record.update(minimum_possible_tokens=minimum, maximum_possible_tokens=maximum)
            return None, report
        model.add(sum(variables[i] for i in ix) == counts[h])
        tokens = sum(variables[i] * ordered[i]['supervised_tokens'] for i in ix)
        model.add(tokens >= lo); model.add(tokens <= hi)
    all_tokens = sum(x * r['supervised_tokens'] for x, r in zip(variables, ordered))
    model.add(all_tokens >= low); model.add(all_tokens <= high)
    utilities = [utility(r['row_id'], seed) for r in ordered]
    primary = sum(x * u for x, u in zip(variables, utilities))
    model.maximize(primary)

    def make_solver(limit):
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = seed
        solver.parameters.max_deterministic_time = limit
        return solver

    solver = make_solver(deterministic_limit)
    print(f'Solving {len(ordered)} candidates; {sum(counts.values())} required rows; '
          f'{deterministic_limit} deterministic time units (not wall-clock seconds).', flush=True)
    status = solver.solve(model)
    report.update(status=solver.status_name(status), wall_seconds=solver.wall_time,
                  deterministic_time=solver.response_proto.deterministic_time,
                  infeasibility_proven=status == cp_model.INFEASIBLE)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        report['note'] = 'UNKNOWN/time limit is not proof of infeasibility. No tolerance or count was relaxed.'
        return None, report
    selected = [i for i, x in enumerate(variables) if solver.value(x)]
    achieved = sum(utilities[i] for i in selected)
    bound = solver.best_objective_bound
    report.update(objective=achieved, best_bound=bound,
                  relative_gap=max(0.0, (bound - achieved) / max(1, abs(achieved))),
                  approximate_optimization=status != cp_model.OPTIMAL)
    model.add(primary == achieved)
    model.minimize(sum(x * tie_utility(r['row_id'], seed) for x, r in zip(variables, ordered)))
    selected_set = set(selected)
    for i, x in enumerate(variables):
        model.add_hint(x, int(i in selected_set))
    tie_solver = make_solver(tie_limit)
    tie_status = tie_solver.solve(model)
    report['tie_break'] = {'status': tie_solver.status_name(tie_status),
                           'wall_seconds': tie_solver.wall_time,
                           'deterministic_time': tie_solver.response_proto.deterministic_time,
                           'primary_incumbent_retained': tie_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE)}
    if tie_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected = [i for i, x in enumerate(variables) if tie_solver.value(x)]
    chosen = [ordered[i] for i in selected]
    actual = sum(r['supervised_tokens'] for r in chosen)
    if len(chosen) != sum(counts.values()) or not low <= actual <= high:
        raise AssertionError('Solver output failed global controls')
    for record in report['strata']:
        subset = [r for r in chosen if cell(r) == (record['source'], record['length_bin'])]
        record.update(actual_rows=len(subset), actual_tokens=sum(r['supervised_tokens'] for r in subset))
        if record['actual_rows'] != record['required_rows'] or not record['token_bounds'][0] <= record['actual_tokens'] <= record['token_bounds'][1]:
            raise AssertionError('Solver output failed stratum controls')
    report.update(passed=True, actual_rows=len(chosen), actual_tokens=actual,
                  difference_percent=100 * float((actual - target) / target),
                  whole_rows_preserved=True, tolerance_relaxed=False)
    return chosen, report
