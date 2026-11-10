#!/usr/bin/env python3
"""Unattended Method B queue for an existing one-GPU Sol allocation.

Upload this single file to the project and run it once. It launches a detached
worker, resumes matching completed work, logs exceptions, and writes a morning
report. It never promises that the whole study fits one allocation.
"""
import argparse
import ast
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback
import uuid

# Replaced by the artifact builder with the reviewed, conversation-scoped code.
EMBEDDED_FILES = {'select_b.py': '"""Method B: parent-derived, right-closed strata and constrained random utility.\n\nNo student scores, embeddings, or evaluation accuracy enter selection.\nThe only third-party dependency is the separately installed OR-Tools solver.\n"""\nimport bisect\nimport csv\nimport hashlib\nimport io\nimport json\nimport math\nfrom collections import Counter, defaultdict\nfrom fractions import Fraction\nfrom pathlib import Path\n\nVERSION = \'method-b-document-alignment-v3\'\nORTOOLS_VERSION = \'9.14.6206\'\n\n\ndef digest(path):\n    h = hashlib.sha256()\n    with Path(path).open(\'rb\') as f:\n        for chunk in iter(lambda: f.read(1024 * 1024), b\'\'):\n            h.update(chunk)\n    return h.hexdigest()\n\n\ndef read(path):\n    return json.loads(Path(path).read_text(encoding=\'utf-8\'))\n\n\ndef json_bytes(value):\n    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + \'\\n\').encode()\n\n\ndef write(path, value):\n    path = Path(path)\n    data = value if isinstance(value, bytes) else json_bytes(value)\n    if path.exists():\n        if path.read_bytes() != data:\n            raise FileExistsError(f\'Preserved existing different file: {path}. Use a new study version.\')\n        return\n    path.parent.mkdir(parents=True, exist_ok=True)\n    tmp = path.with_suffix(path.suffix + \'.incomplete\')\n    tmp.write_bytes(data)\n    tmp.replace(path)\n\n\ndef rows(path):\n    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]\n\n\ndef jsonl(data):\n    return b\'\'.join((json.dumps(r, ensure_ascii=False) + \'\\n\').encode() for r in data)\n\n\ndef csv_bytes(data):\n    if not data:\n        return b\'\'\n    out = io.StringIO(newline=\'\')\n    w = csv.DictWriter(out, fieldnames=list(data[0]))\n    w.writeheader()\n    w.writerows(data)\n    return out.getvalue().encode()\n\n\ndef fraction(value):\n    return value if isinstance(value, Fraction) else Fraction(str(value))\n\n\ndef number(value):\n    return value.numerator if isinstance(value, Fraction) and value.denominator == 1 else float(value)\n\n\ndef bounds(target, tolerance):\n    target, tolerance = fraction(target), fraction(tolerance)\n    return math.ceil(target * (1 - tolerance)), math.floor(target * (1 + tolerance))\n\n\ndef hash32(text):\n    return int.from_bytes(hashlib.sha256(text.encode(\'utf-8\')).digest()[:4], \'big\')\n\n\ndef utility(row_id, seed=42):\n    # floor(1_000_000 * first32 / 2**32 + 0.5), using integer arithmetic.\n    return (hash32(f\'{seed}|random|{row_id}\') * 1_000_000 + 2**31) // 2**32\n\n\ndef tie_utility(row_id, seed=42):\n    return hash32(f\'{seed}|tie|{row_id}\')\n\n\ndef parent_edges(parent):\n    result = {}\n    for source in sorted({r[\'dataset_id\'] for r in parent}):\n        values = sorted(r[\'supervised_tokens\'] for r in parent if r[\'dataset_id\'] == source)\n        if not values:\n            raise ValueError(f\'Empty parent source: {source}\')\n        cuts = []\n        for j in range(1, 10):\n            position = Fraction((len(values) - 1) * j, 10)\n            low, high = math.floor(position), math.ceil(position)\n            cut = values[low] + (values[high] - values[low]) * (position - low)\n            if cut not in cuts:\n                cuts.append(cut)\n        result[source] = cuts\n    return result\n\n\ndef rebin(data, edges):\n    out = []\n    for r in data:\n        if r[\'dataset_id\'] not in edges:\n            raise ValueError(\'Candidate source absent from parent: \' + r[\'dataset_id\'])\n        # Equality belongs to the lower interval: (-inf, e1], (e1, e2], ...\n        out.append(dict(r, length_bin=bisect.bisect_left(edges[r[\'dataset_id\']], r[\'supervised_tokens\'])))\n    return out\n\n\ndef edge_report(edges):\n    return {\'reference_population\': \'actual R1000 parent, separately by source\',\n            \'quantiles\': [j / 10 for j in range(1, 10)], \'interpolation\': \'linear\',\n            \'endpoint_rule\': \'right-closed; bisect_left; outer edges -inf and +inf\',\n            \'source_inner_edges\': {s: [str(x) for x in cuts] for s, cuts in edges.items()}}\n\n\ndef cell(row):\n    return row[\'dataset_id\'], row[\'length_bin\']\n\n\ndef validate(data, cap=2048):\n    if not data:\n        raise ValueError(\'Empty dataset\')\n    for key in (\'row_id\', \'problem_group_id\'):\n        ids = [r[key] for r in data]\n        if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):\n            raise ValueError(f\'Empty/repeated {key}\')\n    for r in data:\n        if r[\'eos_tokens\'] != 1 or r[\'supervised_tokens\'] != r[\'completion_tokens\'] + 1:\n            raise ValueError(\'Completion/EOS count mismatch: \' + r[\'row_id\'])\n        if r[\'total_tokens\'] != r[\'prompt_tokens\'] + r[\'supervised_tokens\']:\n            raise ValueError(\'Total token count mismatch: \' + r[\'row_id\'])\n        if r[\'total_tokens\'] > cap or r.get(\'boundary_crossing_token\') is True:\n            raise ValueError(\'Overlength or boundary-crossing row: \' + r[\'row_id\'])\n\n\ndef quotas(parent, multiplier):\n    """Exact parent counts at 1x/2x/4x; explicit largest-remainder 0.5x extension."""\n    m = fraction(multiplier)\n    counts = Counter(cell(r) for r in parent)\n    ideals = {h: n * m for h, n in counts.items()}\n    target_count = len(parent) * m\n    if target_count.denominator != 1:\n        raise ValueError(\'Requested total example count is fractional\')\n    result = {h: math.floor(x) for h, x in ideals.items()}\n    order = sorted(ideals, key=lambda h: (-(ideals[h] - result[h]), h))\n    for h in order[:int(target_count) - sum(result.values())]:\n        result[h] += 1\n    return result\n\n\ndef solve(parent, bank, multiplier=1, *, seed=42, deterministic_limit=120,\n          tie_limit=10, global_tolerance=\'0.005\', stratum_tolerance=\'0.05\'):\n    import ortools\n    from ortools.sat.python import cp_model\n    if ortools.__version__ != ORTOOLS_VERSION:\n        raise ValueError(f\'Expected OR-Tools {ORTOOLS_VERSION}, found {ortools.__version__}\')\n    validate(parent); validate(bank)\n    for key in (\'row_id\', \'problem_group_id\'):\n        if {r[key] for r in parent} & {r[key] for r in bank}:\n            raise ValueError(\'Parent/bank overlap: \' + key)\n    ordered = sorted(bank, key=lambda r: r[\'row_id\'])\n    counts = quotas(parent, multiplier)\n    reference = defaultdict(int)\n    for r in parent:\n        reference[cell(r)] += r[\'supervised_tokens\']\n    groups = defaultdict(list)\n    for i, r in enumerate(ordered):\n        groups[cell(r)].append(i)\n    # Some quantile intervals can be empty in a tied/discrete parent. Candidates\n    # in such intervals remain in the bank but have a strict zero-row quota.\n    for h in set(groups) - set(counts):\n        counts[h] = 0\n        reference[h] = 0\n    target = sum(reference.values()) * fraction(multiplier)\n    low, high = bounds(target, global_tolerance)\n    report = {\'version\': VERSION, \'ortools_version\': ortools.__version__,\n              \'multiplier\': number(fraction(multiplier)), \'selection_seed\': seed,\n              \'target_rows\': sum(counts.values()), \'target_supervised_tokens\': number(target),\n              \'global_integer_bounds\': [low, high], \'global_tolerance\': global_tolerance,\n              \'stratum_tolerance\': stratum_tolerance,\n              \'random_utility\': \'floor(1000000 * first32(SHA256(seed|random|row_id))/2^32 + 0.5)\',\n              \'strata\': [], \'student_scores_used\': False, \'embeddings_used\': False,\n              \'uniform_over_feasible_subsets_claimed\': False}\n    model = cp_model.CpModel()\n    variables = [model.new_bool_var(f\'x_{i:06d}\') for i in range(len(ordered))]\n    for h in sorted(counts):\n        ix = groups.get(h, [])\n        lo, hi = bounds(reference[h] * fraction(multiplier), stratum_tolerance)\n        record = {\'source\': h[0], \'length_bin\': h[1], \'required_rows\': counts[h],\n                  \'candidate_rows\': len(ix), \'parent_tokens\': reference[h], \'token_bounds\': [lo, hi]}\n        report[\'strata\'].append(record)\n        if len(ix) < counts[h]:\n            report.update(status=\'INFEASIBLE_CAPACITY\', infeasibility_proven=True)\n            return None, report\n        lengths = sorted(ordered[i][\'supervised_tokens\'] for i in ix)\n        minimum = sum(lengths[:counts[h]])\n        maximum = sum(lengths[-counts[h]:]) if counts[h] else 0\n        if minimum > hi or maximum < lo:\n            report.update(status=\'INFEASIBLE_STRATUM_TOKEN_BOUNDS\', infeasibility_proven=True)\n            record.update(minimum_possible_tokens=minimum, maximum_possible_tokens=maximum)\n            return None, report\n        model.add(sum(variables[i] for i in ix) == counts[h])\n        tokens = sum(variables[i] * ordered[i][\'supervised_tokens\'] for i in ix)\n        model.add(tokens >= lo); model.add(tokens <= hi)\n    all_tokens = sum(x * r[\'supervised_tokens\'] for x, r in zip(variables, ordered))\n    model.add(all_tokens >= low); model.add(all_tokens <= high)\n    utilities = [utility(r[\'row_id\'], seed) for r in ordered]\n    primary = sum(x * u for x, u in zip(variables, utilities))\n    model.maximize(primary)\n\n    def make_solver(limit):\n        solver = cp_model.CpSolver()\n        solver.parameters.num_search_workers = 1\n        solver.parameters.random_seed = seed\n        solver.parameters.max_deterministic_time = limit\n        return solver\n\n    solver = make_solver(deterministic_limit)\n    print(f\'Solving {len(ordered)} candidates; {sum(counts.values())} required rows; \'\n          f\'{deterministic_limit} deterministic time units (not wall-clock seconds).\', flush=True)\n    status = solver.solve(model)\n    report.update(status=solver.status_name(status), wall_seconds=solver.wall_time,\n                  deterministic_time=solver.response_proto.deterministic_time,\n                  infeasibility_proven=status == cp_model.INFEASIBLE)\n    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):\n        report[\'note\'] = \'UNKNOWN/time limit is not proof of infeasibility. No tolerance or count was relaxed.\'\n        return None, report\n    selected = [i for i, x in enumerate(variables) if solver.value(x)]\n    achieved = sum(utilities[i] for i in selected)\n    bound = solver.best_objective_bound\n    report.update(objective=achieved, best_bound=bound,\n                  relative_gap=max(0.0, (bound - achieved) / max(1, abs(achieved))),\n                  approximate_optimization=status != cp_model.OPTIMAL)\n    model.add(primary == achieved)\n    model.minimize(sum(x * tie_utility(r[\'row_id\'], seed) for x, r in zip(variables, ordered)))\n    selected_set = set(selected)\n    for i, x in enumerate(variables):\n        model.add_hint(x, int(i in selected_set))\n    tie_solver = make_solver(tie_limit)\n    tie_status = tie_solver.solve(model)\n    report[\'tie_break\'] = {\'status\': tie_solver.status_name(tie_status),\n                           \'wall_seconds\': tie_solver.wall_time,\n                           \'deterministic_time\': tie_solver.response_proto.deterministic_time,\n                           \'primary_incumbent_retained\': tie_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE)}\n    if tie_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):\n        selected = [i for i, x in enumerate(variables) if tie_solver.value(x)]\n    chosen = [ordered[i] for i in selected]\n    actual = sum(r[\'supervised_tokens\'] for r in chosen)\n    if len(chosen) != sum(counts.values()) or not low <= actual <= high:\n        raise AssertionError(\'Solver output failed global controls\')\n    for record in report[\'strata\']:\n        subset = [r for r in chosen if cell(r) == (record[\'source\'], record[\'length_bin\'])]\n        record.update(actual_rows=len(subset), actual_tokens=sum(r[\'supervised_tokens\'] for r in subset))\n        if record[\'actual_rows\'] != record[\'required_rows\'] or not record[\'token_bounds\'][0] <= record[\'actual_tokens\'] <= record[\'token_bounds\'][1]:\n            raise AssertionError(\'Solver output failed stratum controls\')\n    report.update(passed=True, actual_rows=len(chosen), actual_tokens=actual,\n                  difference_percent=100 * float((actual - target) / target),\n                  whole_rows_preserved=True, tolerance_relaxed=False)\n    return chosen, report\n', 'run_method_b.py': '#!/usr/bin/env python3\n"""Run document-aligned Method B in the existing Sol project.\n\nOriginal workbenches, parents, datasets, environments, and results are preserved.\nCP-SAT lives in its own CPU environment; training uses the tested Python 3.12 venv.\n"""\nimport argparse\nimport csv\nimport hashlib\nimport importlib\nimport json\nimport os\nfrom pathlib import Path\nimport shutil\nimport subprocess\nimport sys\nimport tarfile\nimport time\nimport unicodedata\nfrom datetime import datetime, timezone\nfrom fractions import Fraction\nfrom collections import Counter, defaultdict\n\nfrom select_b import (ORTOOLS_VERSION, VERSION, bounds, cell, csv_bytes, digest,\n                      edge_report, fraction, hash32, json_bytes, jsonl, number,\n                      parent_edges, read, rebin, rows, solve, utility, validate, write)\n\nHERE = Path(__file__).resolve().parent\nSTUDY = \'B_method_B_document_v3\'\n\n\ndef output(root):\n    return Path(root) / \'outputs\' / STUDY\n\n\ndef stamp():\n    return datetime.now(timezone.utc).strftime(\'%Y%m%dT%H%M%S%fZ\')\n\n\ndef update(path, value):\n    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)\n    tmp = path.with_suffix(path.suffix + \'.incomplete\')\n    tmp.write_bytes(json_bytes(value)); tmp.replace(path)\n\n\ndef call(command, root, label, *, capture=False):\n    log = output(root) / \'logs\' / f\'{label}_{stamp()}.log\'\n    log.parent.mkdir(parents=True, exist_ok=True)\n    started = time.monotonic(); lines = []\n    print(\'Running:\', \' \'.join(map(str, command)), flush=True)\n    print(\'Log:\', log, flush=True)\n    with log.open(\'w\', encoding=\'utf-8\') as f:\n        env = dict(os.environ, PYTHONUNBUFFERED=\'1\', TOKENIZERS_PARALLELISM=\'false\')\n        with subprocess.Popen(list(map(str, command)), cwd=str(HERE), env=env,\n                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n                              text=True, bufsize=1) as p:\n            for line in p.stdout:\n                print(line, end=\'\', flush=True); f.write(line); f.flush()\n                if capture:\n                    lines.append(line)\n            code = p.wait()\n    update(log.with_suffix(\'.status.json\'), {\'command\': list(map(str, command)),\n           \'exit_code\': code, \'wall_seconds\': time.monotonic() - started, \'log\': str(log)})\n    if code:\n        raise RuntimeError(f\'Stage failed (exit {code}). See {log}. Earlier outputs were preserved.\')\n    return \'\'.join(lines)\n\n\ndef training_python(root):\n    p = Path(root) / \'venv_b_training_py312/bin/python\'\n    if not p.is_file():\n        raise FileNotFoundError(\'Existing training environment is missing: \' + str(p))\n    return p\n\n\ndef source(root):\n    p = Path(root) / \'b_workbench_500\'\n    manifest = read(HERE / \'source_hashes.json\')\n    for name, expected in manifest.items():\n        if digest(p / name) != expected:\n            raise ValueError(f\'Your working source differs from the reviewed copy: {p/name}. \'\n                             \'No files were replaced; share the changed source before adapting it.\')\n    for name, expected in read(p / \'source_provenance.json\')[\'vendor_sha256\'].items():\n        if digest(p / name) != expected:\n            raise ValueError(\'Pinned vendor source/data differs: \' + str(p / name))\n    for name, expected in read(p / \'source_provenance.json\')[\'parent_reference_records_sha256\'].items():\n        if digest(p / name) != expected:\n            raise ValueError(\'Pinned parent record differs: \' + str(p / name))\n    return p\n\n\ndef solver_python(root):\n    return Path(root) / \'venv_b_solver_v3/bin/python\'\n\n\ndef install_solver(root):\n    target = solver_python(root)\n    if not target.is_file():\n        call([training_python(root), \'-m\', \'venv\', target.parent.parent], root, \'create_solver_venv\')\n    probe = subprocess.run([str(target), \'-c\', \'import ortools; print(ortools.__version__)\'],\n                           capture_output=True, text=True)\n    if probe.returncode or probe.stdout.strip() != ORTOOLS_VERSION:\n        call([target, \'-m\', \'pip\', \'install\', \'--only-binary=:all:\',\n              f\'ortools=={ORTOOLS_VERSION}\'], root, \'install_ortools\')\n    call([target, HERE / \'test_method_b.py\', \'--require-solver\'], root, \'solver_acceptance_tests\')\n    freeze = subprocess.check_output([str(target), \'-m\', \'pip\', \'freeze\'])\n    write(output(root) / \'solver_requirements.lock.txt\', freeze)\n    print(\'SOLVER_READY. Your training environment was not changed.\', flush=True)\n\n\ndef normalize(text):\n    return \' \'.join(unicodedata.normalize(\'NFKC\', text).split())\n\n\ndef paired_item_interval(parent_path, b_path, replicates=2000):\n    """A fixed-checkpoint item bootstrap; it does not estimate training variance."""\n    import random\n    parent={r[\'problem_id\']:r[\'correct\'] for r in rows(parent_path)}\n    b={r[\'problem_id\']:r[\'correct\'] for r in rows(b_path)}\n    if set(parent)!=set(b) or not parent:raise ValueError(\'Cannot bootstrap unpaired/empty predictions\')\n    differences=[int(b[k])-int(parent[k]) for k in sorted(parent)]\n    rng=random.Random(1729)\n    sampled=sorted(100*sum(rng.choices(differences,k=len(differences)))/len(differences)\n                   for _ in range(replicates))\n    return {\'paired_item_ci95_low_pp\':sampled[int(0.025*(replicates-1))],\n            \'paired_item_ci95_high_pp\':sampled[math_ceil(0.975*(replicates-1))],\n            \'bootstrap_replicates\':replicates,\n            \'uncertainty_scope\':\'paired items for these fixed checkpoints; training variability is separate\'}\n\n\ndef math_ceil(value):\n    import math\n    return math.ceil(value)\n\n\ndef build_banks(root):\n    old = source(root); cfg = read(old / \'config.json\')\n    root = Path(root); shared = output(root) / \'shared\'\n    parent_path = root / \'data/train_n1000.jsonl\'\n    pool_path = root / \'outputs/common_pool_review_v1/candidate_pool.jsonl\'\n    summary_path = pool_path.parent / \'pool_summary.json\'\n    if digest(parent_path) != cfg[\'parent_manifest_sha256\']:\n        raise ValueError(\'Actual parent manifest differs from the verified R1000 manifest\')\n    if digest(pool_path) != cfg[\'screened_pool_sha256\']:\n        raise ValueError(\'Screened pool differs from the reviewed 9,909-example bank\')\n    summary = read(summary_path)\n    if summary.get(\'review_queue_complete\') is not True or summary.get(\'pool_sha256\') != cfg[\'screened_pool_sha256\']:\n        raise ValueError(\'Missing/incompatible completed screening evidence\')\n    parent, pool = rows(parent_path), rows(pool_path)\n    if len(parent) != 1000 or len(pool) != 9909:\n        raise ValueError(\'Unexpected verified parent/pool row counts\')\n    validate(parent); validate(pool)\n    contract = root / \'inputs/formatting_contract.json\'\n    if digest(contract) != cfg[\'formatting_contract_sha256\']:\n        raise ValueError(\'Renderer/tokenization contract changed\')\n    # Check the UNION of actual groups seen by every supplied parent profile.\n    union = set()\n    for entry in read(old / \'parent_registry.json\').values():\n        with (old / entry[\'records_relative_path\'] / \'seen_groups.csv\').open(newline=\'\') as f:\n            union.update(r[\'problem_group_id\'] for r in csv.DictReader(f))\n    if union != {r[\'problem_group_id\'] for r in parent}:\n        raise ValueError(\'The supplied parents have different seen manifests; rebuild the shared exclusions first\')\n    if union & {r[\'problem_group_id\'] for r in pool}:\n        raise ValueError(\'Screened pool overlaps actual parent-seen groups\')\n    if {r[\'row_id\'] for r in parent} & {r[\'row_id\'] for r in pool}:\n        raise ValueError(\'Screened pool overlaps actual parent row IDs\')\n    protected = root / \'inputs/protected_questions.csv\'\n    if not protected.is_file():\n        protected = old / \'inputs_snapshot/protected_questions.csv\'\n    with protected.open(newline=\'\', encoding=\'utf-8\') as f:\n        protected_text = {normalize(r[\'problem\']) for r in csv.DictReader(f)}\n    if protected_text & {normalize(r[\'problem\']) for r in pool}:\n        raise ValueError(\'Screened pool has an exact protected-question overlap\')\n    edges = parent_edges(parent)\n    parent_rebinned, pool = rebin(parent, edges), rebin(pool, edges)\n    numina_parent = [r for r in parent_rebinned if r[\'dataset_id\'] == \'numina\']\n    if not numina_parent:\n        raise ValueError(\'Cannot reserve the documented transfer condition: parent has no Numina rows\')\n    # The old parents have unverified development protection. Reserve the\n    # documented clean fallback: 1,000 screened, group-disjoint examples,\n    # 500/source. Export references for the common evaluator to ingest.\n    development=[]\n    for source_name in sorted({r[\'dataset_id\'] for r in parent_rebinned}):\n        source_parent=[r for r in parent_rebinned if r[\'dataset_id\']==source_name]\n        source_counts=Counter(cell(r) for r in source_parent)\n        if len(source_parent)!=500:\n            raise ValueError(\'This fallback development reserve expects the verified 500/source parent\')\n        for h,n in sorted(source_counts.items()):\n            candidates=sorted((r for r in pool if cell(r)==h),\n                              key=lambda r:(hash32(\'1729|clean-development|\'+r[\'row_id\']),r[\'row_id\']))\n            if len(candidates)<n: raise ValueError(\'Insufficient clean development capacity: \'+str(h))\n            development.extend(candidates[:n])\n    dev_groups={r[\'problem_group_id\'] for r in development}\n    remaining=[r for r in pool if r[\'problem_group_id\'] not in dev_groups]\n    # Reserve 5x each source-specific parent stratum before any mixed-source draw.\n    reserve = []\n    by = defaultdict(list)\n    for r in remaining:\n        by[cell(r)].append(r)\n    for h, n in sorted(Counter(cell(r) for r in numina_parent).items()):\n        candidates = sorted(by[h], key=lambda r: (hash32(\'1729|transfer-reserve|\' + r[\'row_id\']), r[\'row_id\']))\n        if len(candidates) < 5 * n:\n            raise ValueError(f\'Insufficient screened transfer capacity in {h}: need {5*n}, have {len(candidates)}. \'\n                             \'No main-bank draw was made and no raw candidates were substituted.\')\n        reserve.extend(candidates[:5*n])\n    reserved = {r[\'problem_group_id\'] for r in reserve}\n    mixed = sorted((r for r in remaining if r[\'problem_group_id\'] not in reserved), key=lambda r: r[\'row_id\'])\n    reserve.sort(key=lambda r: r[\'row_id\'])\n    write(shared / \'parent_length_bins.json\', edge_report(edges))\n    write(shared / \'parent_original_manifest.jsonl\', parent_path.read_bytes())\n    write(shared / \'parent_seen_groups.csv\', (\'problem_group_id\\n\'+\'\\n\'.join(sorted(union))+\'\\n\').encode())\n    write(shared / \'parent_rebinned_reference.jsonl\', jsonl(parent_rebinned))\n    write(shared / \'clean_development_reference.jsonl\', jsonl(sorted(development,key=lambda r:r[\'row_id\'])))\n    write(shared / \'mixed_candidate_bank.jsonl\', jsonl(mixed))\n    write(shared / \'transfer_numina_bank.jsonl\', jsonl(reserve))\n    report = {\'version\': VERSION, \'source_document_sha256\': read(HERE/\'source_hashes_meta.json\')[\'method_document_sha256\'],\n              \'parent_manifest_sha256\': digest(parent_path), \'parent_rows\': len(parent),\n              \'parent_supervised_tokens\': sum(r[\'supervised_tokens\'] for r in parent),\n              \'screened_pool_rows\': len(pool), \'screened_pool_sha256\': digest(pool_path),\n              \'clean_development_rows\':len(development),\n              \'clean_development_sha256\':digest(shared/\'clean_development_reference.jsonl\'),\n              \'clean_development_evaluator_status\':\'Reference rows reserved and exported; ingestion/parser validation by the shared evaluator is still required.\',\n              \'mixed_bank_rows\': len(mixed), \'mixed_source_rows\': dict(Counter(r[\'dataset_id\'] for r in mixed)),\n              \'mixed_bank_sha256\': digest(shared/\'mixed_candidate_bank.jsonl\'),\n              \'transfer_rows\': len(reserve), \'transfer_bank_sha256\': digest(shared/\'transfer_numina_bank.jsonl\'),\n              \'transfer_reference_rows\': len(numina_parent),\n              \'transfer_reference_tokens\': sum(r[\'supervised_tokens\'] for r in numina_parent),\n              \'length_bins_sha256\': digest(shared/\'parent_length_bins.json\'),\n              \'protected_questions_sha256\': digest(protected), \'parent_union_checked\': True,\n              \'parent_seen_group_overlap\':0,\n              \'exact_protected_overlap\': 0, \'screening_evidence_sha256\': digest(summary_path),\n              \'near_overlap_scope\': \'The pinned screened pool and its completed review; no exhaustive pretraining claim\',\n              \'bank_capacity_note\': \'10,000 is a target subject to screened capacity. \'\n                  \'This study uses all eligible remaining rows after reserving disjoint development and transfer banks.\',\n              \'shared_with_other_methods\': \'These artifacts define common inputs for A/B/C; adoption by other runs is not asserted.\',\n              \'parent_related_development_groups_to_disclose\': cfg[\'scientific_scope\'].get(\'parent_related_development_groups_to_disclose\'),\n              \'evaluation_status\': \'Preliminary until the shared evaluator and parent/development protection are verified.\'}\n    write(shared / \'bank_audit.json\', report)\n    print(json.dumps(report, indent=2), flush=True)\n    print(\'BANKS_READY. Preserve/share this bank, bins, and audit for all fresh selectors.\', flush=True)\n\n\ndef setting_root(root, setting, budget):\n    return output(root) / setting / f\'budget_{float(budget):g}x\'\n\n\ndef bridge_root(root, setting, budget):\n    return setting_root(root, setting, budget) / \'runtime\'\n\n\ndef make_bridge(root, setting, budget, selection, report, parent_seeds):\n    old = source(root); shared = output(root) / \'shared\'; audit = read(shared/\'bank_audit.json\')\n    dest = bridge_root(root, setting, budget); dest.mkdir(parents=True, exist_ok=True)\n    for name in [\'common.py\', \'workbench.py\', \'continuation.py\', \'compare.py\', \'allocator.py\',\n                 \'parent_registry.json\', \'source_provenance.json\']:\n        content = (old/name).read_text()\n        if name == \'common.py\':\n            content += \'\\n\\n# V3 explicit data routing; original helpers remain intact above.\\n\'\n            content += \'def data_paths(root):\\n    cfg=config()\\n    return {k:Path(v) for k,v in cfg["resolved_data_paths"].items()}\\n\'\n            content += \'def work_root(root):\\n    return Path(config()["resolved_work_root"])\\n\'\n            content += \'def budget_target(parent_tokens,multiplier):\\n    value=Fraction(config()["budget_reference_tokens"])*Fraction(str(multiplier))\\n    return value.numerator if value.denominator==1 else float(value)\\n\'\n        if name == \'continuation.py\':\n            needle = \'stop_after=1 if smoke else None\'\n            if content.count(needle) != 1:\n                raise ValueError(\'Inspected smoke-stop expression changed\')\n            content = content.replace(needle, \'stop_after=4 if smoke else None\')\n            label="\'controlled_continuation\'"\n            if content.count(label)!=1:raise ValueError(\'Inspected experiment-status expression changed\')\n            content=content.replace(label,"\'preliminary_document_aligned_B\'")\n        write(dest/name, content.encode())\n    for directory in (\'vendor\', \'reference_records\'):\n        link = dest/directory\n        if link.is_symlink():\n            if link.resolve() != (old/directory).resolve():\n                raise ValueError(\'Runtime link points to a different pinned source: \'+str(link))\n        elif link.exists():\n            raise FileExistsError(\'Unexpected runtime source directory: \'+str(link))\n        else:\n            link.symlink_to(old/directory, target_is_directory=True)\n    cfg = read(old/\'config.json\')\n    cfg.update(version=VERSION, protocol_status=\'frozen\', protocol_evidence=str(dest/\'execution_protocol.json\'),\n               parent_training_seeds=list(parent_seeds), continuation_training_seeds=[42,137,2026],\n               budget_multipliers=[float(budget)],\n               common_universe_sha256=audit[\'mixed_bank_sha256\'] if setting==\'mixed\' else audit[\'transfer_bank_sha256\'],\n               resolved_work_root=str(setting_root(root,setting,budget)),\n               budget_reference_tokens=audit[\'parent_supervised_tokens\'] if setting==\'mixed\' else audit[\'transfer_reference_tokens\'],\n               reference_scope=\'full R1000 parent\' if setting==\'mixed\' else \'Numina subset of actual R1000 parent\',\n               models=[\'0.5B\',\'1.5B\'] if setting==\'mixed\' else [\'0.5B\'],\n               selection_implementation=\'document Appendix A CP-SAT\',\n               per_stratum_token_tolerance=\'0.05\',\n               source_document_sha256=read(HERE/\'source_hashes_meta.json\')[\'method_document_sha256\'])\n    protected = Path(root)/\'inputs/protected_questions.csv\'\n    if not protected.is_file(): protected=old/\'inputs_snapshot/protected_questions.csv\'\n    cfg[\'resolved_data_paths\'] = {\'parent\':str(Path(root)/\'data/train_n1000.jsonl\'),\n          \'universe\':str(shared/(\'mixed_candidate_bank.jsonl\' if setting==\'mixed\' else \'transfer_numina_bank.jsonl\')),\n          \'contract\':str(Path(root)/\'inputs/formatting_contract.json\'), \'protected\':str(protected)}\n    cfg[\'notes\'] = [\'Execution settings locked for this B version from the supplied design and user instruction.\',\n                    \'This is a reproducibility lock for B; it does not certify adoption by A/C or evaluator clearance.\',\n                    \'0.5x and 2x are explicitly additional conditions. 0.5x uses largest-remainder stratum counts.\',\n                    \'Every run starts independently from the original matching R1000 parent.\',\n                    \'Training seeds 42/137/2026 are different from parent seeds 17/123.\',\n                    \'All evaluation remains preliminary until the shared development/test protection is verified.\']\n    write(dest/\'config.json\', cfg)\n    fields = [\'common_universe_sha256\',\'parent_manifest_sha256\',\'models\',\'parent_training_seeds\',\n              \'continuation_training_seeds\',\'budget_multipliers\',\'selection_seed\',\'universe_seed\',\n              \'quota_seed\',\'training_order_seed\',\'relative_token_tolerance\',\'training\',\'evaluation\']\n    protocol = {k:cfg[k] for k in fields}\n    protocol.update(status=\'frozen\', recorded_by=\'Suraj Shinde — requested execution of the supplied Method B design\',\n                    scope=\'B execution settings only; cross-method compatibility must be checked against these artifacts\',\n                    selection_audit_sha256=digest(setting_root(root,setting,budget)/\'selection_audit.json\'),\n                    length_bins_sha256=audit[\'length_bins_sha256\'],\n                    source_document_sha256=cfg[\'source_document_sha256\'],\n                    evaluation_clearance=\'not established by a protocol lock\')\n    write(dest/\'execution_protocol.json\',protocol)\n    return dest\n\n\ndef prepare(root, setting, budget, parent_seeds):\n    shared = output(root)/\'shared\'; audit=read(shared/\'bank_audit.json\')\n    parent = rows(shared/\'parent_rebinned_reference.jsonl\')\n    if setting == \'transfer\':\n        if float(budget) != 1: raise ValueError(\'The documented transfer condition uses its own 1x source-specific budget\')\n        parent = [r for r in parent if r[\'dataset_id\']==\'numina\']\n    bank_path = shared/(\'mixed_candidate_bank.jsonl\' if setting==\'mixed\' else \'transfer_numina_bank.jsonl\')\n    if digest(bank_path) != audit[\'mixed_bank_sha256\' if setting==\'mixed\' else \'transfer_bank_sha256\']:\n        raise ValueError(\'Common bank changed after audit\')\n    bank = rows(bank_path); dest=setting_root(root,setting,budget)\n    report_path=dest/\'selection_audit.json\'\n    selected_path=dest/f\'{float(budget):g}x/train_B.jsonl\'\n    if report_path.is_file() and selected_path.is_file():\n        report=read(report_path); selection=rows(selected_path)\n        if report.get(\'passed\') is not True or report.get(\'selected_manifest_sha256\')!=digest(selected_path) or report.get(\'bank_sha256\')!=digest(bank_path):\n            raise ValueError(\'Existing selection failed identity checks\')\n        print(\'Reusing canonical selected dataset:\',selected_path,flush=True)\n    else:\n        selection, report=solve(parent,bank,budget)\n        report.update(setting=setting,bank_sha256=digest(bank_path),parent_manifest_sha256=audit[\'parent_manifest_sha256\'],\n                      parent_length_bins_sha256=audit[\'length_bins_sha256\'])\n        if selection is None:\n            update(dest/f\'selection_failure_{stamp()}.json\',report)\n            raise RuntimeError(\'No conforming selection was produced: \'+report[\'status\']+\n                               \'. Counts and tolerances were preserved; inspect the saved feasibility report.\')\n        selection.sort(key=lambda r:(hash32(\'42|training-order|\'+r[\'row_id\']),r[\'row_id\']))\n        write(selected_path,jsonl(selection))\n        report[\'selected_manifest_sha256\']=digest(selected_path)\n        write(report_path,report)\n        write(dest/f\'{float(budget):g}x/selected_ids.txt\', (\'\\n\'.join(sorted(r[\'row_id\'] for r in selection))+\'\\n\').encode())\n        selection_ids={s[\'row_id\'] for s in selection}\n        write(dest/f\'{float(budget):g}x/candidate_priorities.csv\',csv_bytes([\n              {\'row_id\':r[\'row_id\'],\'utility_integer\':utility(r[\'row_id\']),\n               \'raw_first32\':hash32(\'42|random|\'+r[\'row_id\']),\n               \'selected\':r[\'row_id\'] in selection_ids}\n              for r in sorted(bank,key=lambda r:r[\'row_id\'])]))\n    # The old continuation engine expects these accounting fields; no data are changed.\n    write(selected_path.parent/\'budget_report.json\',{\n          \'budget_multiplier\':float(budget),\'target_supervised_tokens\':report[\'target_supervised_tokens\'],\n          \'selected_rows\':len(selection),\'selected_supervised_tokens\':report[\'actual_tokens\'],\n          \'selected_manifest_sha256\':digest(selected_path),\'quota_deviations\':0,\n          \'difference_percent\':report[\'difference_percent\'],\'tolerance_relaxed\':False,\n          \'selection_audit_sha256\':digest(report_path),\'reference_scope\':setting})\n    bridge=make_bridge(root,setting,budget,selection,report,parent_seeds)\n    print(f\'SELECTION_PASSED: {setting} {float(budget):g}x; {len(selection)} examples; \'\n          f\'{report["actual_tokens"]} tokens; {report["difference_percent"]:+.5f}%.\',flush=True)\n    call([training_python(root),HERE/\'run_method_b.py\',\'engine\',\'--root\',root,\'--setting\',setting,\n          \'--budget\',str(budget),\'--action\',\'verify\'],root,f\'{setting}_{budget}_verify\')\n    print(\'DOCUMENT_ALIGNED_PREPARATION_PASSED\',flush=True)\n\n\ndef checked_runtime(root,setting,budget):\n    dest=bridge_root(root,setting,budget)\n    if not (dest/\'config.json\').is_file():\n        raise FileNotFoundError(\'Prepare this condition first: \'+str(dest))\n    sys.path.insert(0,str(dest))\n    import common,workbench,continuation\n    cfg=common.config()\n    common.sources_valid()\n    continuation.validate_protocol(root,cfg)\n    selection=setting_root(root,setting,budget)/f\'{float(budget):g}x/train_B.jsonl\'\n    audit=read(setting_root(root,setting,budget)/\'selection_audit.json\')\n    if digest(selection)!=audit[\'selected_manifest_sha256\'] or digest(common.data_paths(root)[\'universe\'])!=audit[\'bank_sha256\']:\n        raise ValueError(\'Data changed since constrained selection\')\n    return common,workbench,continuation,cfg\n\n\ndef engine(args):\n    root=args.root\n    os.environ.update(HF_HUB_OFFLINE=\'1\',HF_HUB_CACHE=str(root/\'hf_cache\'),TOKENIZERS_PARALLELISM=\'false\')\n    common,wb,cont,cfg=checked_runtime(root,args.setting,args.budget)\n    if args.action==\'verify\':\n        wb.verify(root,cfg,offline=True)\n        for size in cfg[\'models\']:\n            verification=read(common.work_root(root)/\'label_checks\'/size/\'verification.json\')\n            if verification[\'actual_parent_supervised_tokens\']!=read(output(root)/\'shared/bank_audit.json\')[\'parent_supervised_tokens\']:\n                raise ValueError(\'Actual parent label counts differ from the verified reference\')\n            with (common.work_root(root)/\'label_checks\'/size/\'rows.csv\').open(newline=\'\') as f:\n                actual={r[\'row_id\']:int(r[\'actual_supervised_tokens\']) for r in csv.DictReader(f)}\n            for row in rows(common.data_paths(root)[\'parent\'])+rows(common.data_paths(root)[\'universe\']):\n                if actual[row[\'row_id\']]!=row[\'supervised_tokens\']:\n                    raise ValueError(\'Actual labels differ from stored selection tokens: \'+row[\'row_id\'])\n        write(common.work_root(root)/\'label_counts_and_strata_verified.json\',\n              {\'passed\':True,\'all_stored_counts_match_actual_labels\':True,\n               \'selection_audit_sha256\':digest(common.work_root(root)/\'selection_audit.json\'),\n               \'models\':cfg[\'models\']})\n        return\n    import torch\n    verified=read(common.work_root(root)/\'label_counts_and_strata_verified.json\')\n    if verified.get(\'passed\') is not True or verified.get(\'selection_audit_sha256\')!=digest(common.work_root(root)/\'selection_audit.json\'):\n        raise ValueError(\'Actual label/stratum reconciliation is missing or stale\')\n    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:\n        raise RuntimeError(\'This stage requires a session exposing exactly one NVIDIA GPU\')\n    key,entry=common.profile(args.model,args.parent_seed)\n    if args.parent_seed not in cfg[\'parent_training_seeds\'] or args.model not in cfg[\'models\']:\n        raise ValueError(\'Profile absent from this condition\')\n    checked=wb.parent_check(root,key,entry)\n    if not checked[\'ready\']: raise ValueError(json.dumps(checked))\n    work=common.work_root(root)\n    if args.action==\'smoke\':\n        path=work/\'smokes\'/f\'{key}_t{args.training_seed}/smoke_report.json\'\n        if path.is_file():\n            old=read(path)\n            if old.get(\'passed\') is not True or old.get(\'actual_gpu_execution\') is not True or old.get(\'reload_probe_matched\') is not True or old.get(\'fingerprint\')!=cont.run_fingerprint(entry,cfg,args.training_seed) or old.get(\'optimizer_updates\')!=4:\n                raise ValueError(\'Existing smoke is incompatible; preserve it and use a new study version\')\n            print(\'Reusing successful four-update smoke:\',key,flush=True);return\n        cont.train(root,cfg,args.model,args.parent_seed,args.budget,args.training_seed,True,\n                   run_id=f\'B_{key}_t{args.training_seed}_{args.budget:g}x_smoke32\')\n        result=read(path)\n        summary=read(Path(result[\'checkpoint\']).parent/\'training_summary.json\')\n        if result[\'optimizer_updates\']!=4 or summary.get(\'unique_examples\')!=32:\n            raise ValueError(\'Smoke completed but did not establish the documented 32-example exposure; inspect summary\')\n        print(\'32_EXAMPLE_GPU_SMOKE_PASSED:\',key,flush=True);return\n    run_name=f\'B_{args.model}_p{args.parent_seed}_t{args.training_seed}_{args.budget:g}x\'\n    index_path=work/\'run_index.json\'; index=read(index_path) if index_path.is_file() else {}\n    current=work/\'runs\'/index.get(run_name,run_name)\n    if args.action==\'train\':\n        if current.exists():\n            completion=current/\'B_completion_audit.json\'\n            if completion.is_file() and read(completion).get(\'passed\') is True:\n                run_cfg=read(current/\'run_config.json\')\n                if run_cfg.get(\'parent_profile\')!=key or run_cfg.get(\'continuation_training_seed\')!=args.training_seed:\n                    raise ValueError(\'Completed run identity differs\')\n                if run_cfg.get(\'manifest_sha256\')!=digest(work/f\'{args.budget:g}x/train_B.jsonl\'):\n                    raise ValueError(\'Completed run used a different dataset\')\n                print(\'Already completed:\',current,flush=True);return\n            checkpoints=[p.parent for p in current.glob(\'*/continuation_checkpoint.json\')]\n            if not checkpoints:\n                # A killed process before its first checkpoint is a fresh restart.\n                resume=None\n                print(\'No completed checkpoint: restart from original parent; previous folder preserved.\',flush=True)\n            else:\n                resume=max(checkpoints,key=lambda p:(p/\'continuation_checkpoint.json\').stat().st_mtime)\n            new_name=run_name+\'_resume_\'+stamp()\n        else:\n            resume=None;new_name=run_name\n        index[run_name]=new_name;update(index_path,index)\n        cont.train(root,cfg,args.model,args.parent_seed,args.budget,args.training_seed,False,\n                   resume=resume,run_id=new_name)\n        print(\'B_TRAINING_COMPLETED:\',new_name,flush=True);return\n    if args.action==\'evaluate\':\n        if not (current/\'B_completion_audit.json\').is_file() or read(current/\'B_completion_audit.json\').get(\'passed\') is not True:\n            raise ValueError(\'A complete audited continuation is required before evaluation\')\n        run_cfg=read(current/\'run_config.json\')\n        if run_cfg.get(\'parent_profile\')!=key or run_cfg.get(\'continuation_training_seed\')!=args.training_seed or run_cfg.get(\'manifest_sha256\')!=digest(work/f\'{args.budget:g}x/train_B.jsonl\'):\n            raise ValueError(\'Completed run identity/manifest differs from the requested evaluation\')\n        cont.validate_resume(root,current/\'final\',entry,digest(work/f\'{args.budget:g}x/train_B.jsonl\'),\n                             cont.run_fingerprint(entry,cfg,args.training_seed)[\'effective_train_source_sha256\'])\n        from compare import validate_pair\n        plan=wb.evaluate(root,cfg,args.model,args.parent_seed,args.budget,trained_run=current,execute=False,quiet=True)\n        results={}\n        eval_dir=work/\'evaluation\'/current.name\n        signature_keys=[\'precision\',\'batch_size\',\'max_new_tokens\',\'question_text\',\'decoding\',\'adapters\']\n        for command in plan[\'commands\']:\n            command=list(command)\n            source_eval=Path(command[command.index(\'--eval_file\')+1]);dataset=source_eval.stem\n            parent_run=Path(command[command.index(\'--adapter\')+1]).resolve()==Path(checked[\'adapter\']).resolve()\n            if parent_run:\n                sig={\'adapter_sha256\':entry[\'expected_adapter_sha256\'],\'eval_file_sha256\':digest(source_eval),\n                     \'evaluator_sha256\':digest(common.HERE/\'vendor/evaluation/evaluate.py\'),\n                     \'settings\':{k:cfg[\'evaluation\'][k] for k in signature_keys},\n                     \'gpu\':torch.cuda.get_device_name(0),\'torch\':torch.__version__}\n                cachekey=hashlib.sha256(json_bytes(sig)).hexdigest()[:20]\n                pred=output(root)/\'parent_evaluation_cache\'/f\'{key}_{dataset}_{cachekey}.jsonl\'\n                command[command.index(\'--out\')+1]=str(pred)\n                command[command.index(\'--run_id\')+1]=f\'{key}_parent_{dataset}_{cachekey}\'\n            else:\n                pred=Path(command[command.index(\'--out\')+1])\n                sig={\'run\':str(current),\'run_adapter_sha256\':digest(current/\'final/adapter/adapter_model.safetensors\'),\n                     \'eval_file_sha256\':digest(source_eval),\'evaluator_sha256\':digest(common.HERE/\'vendor/evaluation/evaluate.py\')}\n            marker=pred.with_suffix(\'.validated.json\')\n            valid=False\n            if marker.is_file() and pred.is_file():\n                prior=read(marker)\n                valid=prior.get(\'signature\')==sig and prior.get(\'predictions_sha256\')==digest(pred) and prior.get(\'config_sha256\')==digest(pred.with_suffix(\'.config.json\'))\n            if valid:\n                validate_pair(pred,pred)\n                print(\'Reusing complete predictions:\',pred,flush=True)\n            else:\n                if pred.exists() or marker.exists():\n                    raise FileExistsError(\'Partial/different predictions were preserved: \'+str(pred)+\n                                          \'. Inspect them before starting a separately named retry.\')\n                pred.parent.mkdir(parents=True,exist_ok=True)\n                call(command,root,f\'eval_{current.name}_{"parent" if parent_run else "B"}_{dataset}\')\n                validate_pair(pred,pred)\n                update(marker,{\'signature\':sig,\'predictions_sha256\':digest(pred),\n                               \'config_sha256\':digest(pred.with_suffix(\'.config.json\'))})\n            results.setdefault(dataset,{})[\'parent\' if parent_run else \'B\']=str(pred)\n        comparisons=[]\n        for dataset,pair in results.items():\n            result=validate_pair(pair[\'parent\'],pair[\'B\'])\n            result.update(paired_item_interval(pair[\'parent\'],pair[\'B\']))\n            result.update(dataset=dataset,parent_profile=key,training_seed=args.training_seed,\n                          budget_multiplier=args.budget,evidence_status=\'preliminary development comparison\',\n                          development_protection=\'Parent/evaluator clearance still requires shared verification\')\n            comparisons.append(result)\n        aggregate=sum(x[\'B_accuracy\'] for x in comparisons)/len(comparisons)\n        update(eval_dir/\'B_vs_parent.json\',{\'comparisons\':comparisons,\n               \'equal_weight_mean_B_accuracy\':aggregate,\n               \'equal_weight_mean_parent_accuracy\':sum(x[\'parent_accuracy\'] for x in comparisons)/len(comparisons),\n               \'final_test_used\':False,\'parent_development_protection_verified\':False})\n        print(json.dumps(comparisons,indent=2),flush=True)\n\n\ndef gpu_stage(root,setting,budget,action,parent_seeds,training_seeds):\n    cfg=read(bridge_root(root,setting,budget)/\'config.json\')\n    for size in cfg[\'models\']:\n        for parent_seed in parent_seeds:\n            for training_seed in training_seeds:\n                command=[training_python(root),HERE/\'run_method_b.py\',\'engine\',\'--root\',root,\n                      \'--setting\',setting,\'--budget\',str(budget),\'--action\',action,\'--model\',size,\n                      \'--parent-seed\',str(parent_seed),\'--training-seed\',str(training_seed)]\n                call(command,root,f\'{setting}_{budget}_{action}_{size}_p{parent_seed}_t{training_seed}\')\n\n\ndef summarize(root):\n    all_rows=[];checks=[]\n    for path in sorted(output(root).glob(\'*/budget_*/evaluation/*/B_vs_parent.json\')):\n        record=read(path)\n        for r in record[\'comparisons\']:\n            all_rows.append(dict(setting=path.relative_to(output(root)).parts[0],**r))\n    for path in sorted(output(root).glob(\'*/budget_*/runs/*/B_completion_audit.json\')):\n        audit=read(path);cfg=read(path.parent/\'run_config.json\')\n        checks.append({\'run\':str(path.parent),\'passed\':audit.get(\'passed\') is True,\n                       \'parent_profile\':cfg.get(\'parent_profile\'),\n                       \'training_seed\':cfg.get(\'continuation_training_seed\'),\n                       \'additional_budget_multiplier\':cfg.get(\'additional_budget_multiplier\'),\n                       \'actual_consumed_tokens\':audit.get(\'actual_consumed_tokens\')})\n    import statistics\n    grouped=defaultdict(list)\n    for r in all_rows:\n        grouped[(r[\'setting\'],r[\'parent_profile\'],r[\'budget_multiplier\'],r[\'dataset\'])].append(r)\n    seed_summary=[]\n    for group,data in sorted(grouped.items()):\n        deltas=[r[\'change_percentage_points\'] for r in data]\n        seed_summary.append({\'setting\':group[0],\'parent_profile\':group[1],\'budget_multiplier\':group[2],\n              \'dataset\':group[3],\'training_seeds\':[r[\'training_seed\'] for r in data],\n              \'mean_change_percentage_points\':statistics.mean(deltas),\n              \'sample_sd_percentage_points\':statistics.stdev(deltas) if len(deltas)>1 else None,\n              \'combined_item_and_training_confidence_interval\':\'not estimated; limited seed count\'})\n    update(output(root)/\'progress_summary.json\',{\'completed_training_runs\':checks,\'development_comparisons\':all_rows,\n            \'training_seed_summaries\':seed_summary,\n            \'final_test_used\':False,\'remaining_shared_work\':[\'A/C compatibility with the exported bank, bins, and controls\',\n                  \'Clean development/evaluator protection evidence; disclose prior overlap/exposure\',\n                  \'Frozen final model list and common final evaluation\',\n                  \'B-minus-A / C-minus-B paired comparisons after other methods finish\']})\n    update(output(root)/\'development_results.json\',all_rows)\n    (output(root)/\'development_results.csv\').write_bytes(csv_bytes(all_rows))\n    print(\'Completed training runs:\',len(checks),flush=True)\n    for item in checks: print(item,flush=True)\n    print(\'Development comparisons:\',len(all_rows),flush=True)\n    print(\'Summary:\',output(root)/\'progress_summary.json\',flush=True)\n\n\ndef estimate(root,setting,budget):\n    work=setting_root(root,setting,budget)\n    selected=read(work/\'selection_audit.json\')[\'actual_tokens\']\n    estimates=[]\n    for path in sorted((work/\'runs\').glob(\'*smoke32/training_log.jsonl\')):\n        ticks=[r for r in rows(path) if \'optimizer_update\' in r and r.get(\'elapsed_seconds\',0)>0]\n        if not ticks:continue\n        last=ticks[-1]\n        processed=last.get(\'cumulative_processed_tokens\',0)\n        if processed<=0:continue\n        cfg=read(path.parent/\'run_config.json\')\n        seconds=selected*last[\'elapsed_seconds\']/processed\n        estimates.append({\'profile\':cfg.get(\'parent_profile\'),\'selected_supervised_tokens\':selected,\n               \'measured_smoke_training_seconds\':last[\'elapsed_seconds\'],\n               \'measured_smoke_tokens\':processed,\n               \'projected_training_minutes\':round(seconds/60,1),\n               \'rough_range_minutes\':[round(seconds*0.5/60,1),round(seconds*2/60,1)],\n               \'excludes\':\'Model loading, checkpoint IO, evaluation, queueing, and long-run variability\'})\n    update(work/\'training_time_projection.json\',estimates)\n    print(json.dumps(estimates,indent=2),flush=True)\n    if not estimates:print(\'No readable training timing records yet. Timing remains unmeasured for this selection.\',flush=True)\n\n\ndef export(root):\n    summarize(root)\n    target=Path(root)/f\'method_b_handoff_{stamp()}.tar.gz\'\n    base=output(root)\n    # Package selection/evidence/logs/source and final adapters; omit duplicate\n    # intermediate checkpoints, base models, caches, environments, and vendor copies.\n    with tarfile.open(target,\'w:gz\') as archive:\n        for path in sorted(base.rglob(\'*\')):\n            if not path.is_file() or path.is_symlink(): continue\n            relative=path.relative_to(base)\n            if any(x in relative.parts for x in (\'vendor\',\'reference_records\',\'__pycache__\')): continue\n            if \'runs\' in relative.parts:\n                parts=list(relative.parts);i=parts.index(\'runs\')\n                if len(parts)>i+2 and parts[i+2] not in (\'final\',\'training_summary.json\',\'training_log.jsonl\',\n                                                      \'run_config.json\',\'B_completion_audit.json\'):\n                    continue\n                if path.name==\'training_state.pt\':continue\n            archive.add(path,arcname=str(Path(STUDY)/relative),recursive=False)\n        for path in sorted(HERE.glob(\'*\')):\n            if path.is_file(): archive.add(path,arcname=str(Path(\'method_b_code\')/path.name),recursive=False)\n    print(\'HANDOFF_ARCHIVE:\',target,flush=True)\n\n\ndef main():\n    p=argparse.ArgumentParser(description=__doc__)\n    p.add_argument(\'stage\',choices=[\'install-solver\',\'banks\',\'prepare\',\'smoke\',\'train\',\'evaluate\',\'summary\',\'estimate\',\'export\',\'engine\'])\n    p.add_argument(\'--root\',type=Path,required=True)\n    p.add_argument(\'--setting\',choices=[\'mixed\',\'transfer\'],default=\'mixed\')\n    p.add_argument(\'--budget\',type=float,choices=[0.5,1,2,4],default=1)\n    p.add_argument(\'--parent-seeds\',type=int,nargs=\'+\',default=[17,123])\n    p.add_argument(\'--training-seeds\',type=int,nargs=\'+\',default=[42])\n    p.add_argument(\'--action\',choices=[\'verify\',\'smoke\',\'train\',\'evaluate\'])\n    p.add_argument(\'--model\',choices=[\'0.5B\',\'1.5B\'])\n    p.add_argument(\'--parent-seed\',type=int)\n    p.add_argument(\'--training-seed\',type=int,default=42)\n    a=p.parse_args();a.root=a.root.resolve()\n    if not set(a.parent_seeds)<={17,123} or not set(a.training_seeds)<={42,137,2026}:\n        raise ValueError(\'Seeds must come from the declared parent/continuation lists\')\n    if a.stage==\'install-solver\':install_solver(a.root)\n    elif a.stage==\'banks\':build_banks(a.root)\n    elif a.stage==\'prepare\':prepare(a.root,a.setting,a.budget,a.parent_seeds)\n    elif a.stage==\'engine\':engine(a)\n    elif a.stage in (\'smoke\',\'train\',\'evaluate\'):gpu_stage(a.root,a.setting,a.budget,a.stage,a.parent_seeds,a.training_seeds)\n    elif a.stage==\'summary\':summarize(a.root)\n    elif a.stage==\'estimate\':estimate(a.root,a.setting,a.budget)\n    elif a.stage==\'export\':export(a.root)\n\n\nif __name__==\'__main__\':main()\n', 'test_method_b.py': '"""Acceptance checks; run with --require-solver on Sol before selecting real data."""\nimport hashlib\nimport importlib.util\nimport itertools\nimport json\nimport math\nfrom pathlib import Path\nimport random\nimport sys\nimport tempfile\nimport unittest\nfrom fractions import Fraction\n\nfrom select_b import (bounds, cell, edge_report, parent_edges, quotas, rebin,\n                      solve, tie_utility, utility, validate, write)\n\n\ndef example(identity, tokens, source=\'orca\', length_bin=0):\n    return {\'row_id\':identity,\'problem_group_id\':\'group_\'+identity,\'dataset_id\':source,\n            \'length_bin\':length_bin,\'supervised_tokens\':tokens,\'completion_tokens\':tokens-1,\n            \'prompt_tokens\':5,\'total_tokens\':5+tokens,\'eos_tokens\':1,\n            \'problem\':identity,\'completion\':\'solution\',\'final_answer\':\'1\'}\n\n\ndef fixture():\n    parent=[example(f\'p_{source}_{i}\',t,source) for source in [\'orca\',\'numina\']\n            for i,t in enumerate([10,12])]\n    bank=[example(f\'b_{source}_{i}\',t,source) for source in [\'orca\',\'numina\']\n          for i,t in enumerate([10,12,10,12,10,12])]\n    return parent,bank\n\n\nclass Accounting(unittest.TestCase):\n    def test_linear_quantiles(self):\n        p=[example(str(i),10*(i+1)) for i in range(10)]\n        self.assertEqual(parent_edges(p)[\'orca\'],[Fraction(19+9*i) for i in range(9)])\n    def test_right_closed_equal_edge(self):\n        data=[example(\'x\',10),example(\'y\',11)]\n        self.assertEqual([r[\'length_bin\'] for r in rebin(data,{\'orca\':[Fraction(10)]})],[0,1])\n    def test_duplicate_edges_removed(self):\n        self.assertEqual(parent_edges([example(str(i),7) for i in range(10)])[\'orca\'],[Fraction(7)])\n    def test_bins_are_source_specific(self):\n        p=[example(\'a\',10,\'orca\'),example(\'b\',100,\'numina\')]\n        self.assertEqual(parent_edges(p),{\'numina\':[Fraction(100)],\'orca\':[Fraction(10)]})\n    def test_report_has_outer_infinities_as_text(self):\n        report=edge_report({\'orca\':[Fraction(3,2)]})\n        self.assertIn(\'right-closed\',report[\'endpoint_rule\'])\n        json.dumps(report,allow_nan=False)\n    def test_fractional_bounds(self):\n        self.assertEqual(bounds(Fraction(318845,2),\'0.005\'),(158626,160219))\n    def test_integer_bounds(self):\n        self.assertEqual(bounds(318845,\'0.005\'),(317251,320439))\n    def test_stratum_bounds(self):\n        self.assertEqual(bounds(22,\'0.05\'),(21,23))\n    def test_exact_primary_counts(self):\n        p,_=fixture();self.assertEqual(sum(quotas(p,1).values()),4)\n    def test_exact_fourfold_counts(self):\n        p,_=fixture();self.assertEqual(sum(quotas(p,4).values()),16)\n    def test_half_extension_total_count(self):\n        p,_=fixture();self.assertEqual(sum(quotas(p,0.5).values()),2)\n    def test_hash_utility_formula(self):\n        raw=int(hashlib.sha256(b\'42|random|abc\').hexdigest()[:8],16)\n        self.assertEqual(utility(\'abc\'),math.floor(Fraction(raw*1000000,2**32)+Fraction(1,2)))\n    def test_tie_hash_formula(self):\n        self.assertEqual(tie_utility(\'abc\'),int(hashlib.sha256(b\'42|tie|abc\').hexdigest()[:8],16))\n    def test_duplicate_group_rejected(self):\n        a,b=example(\'a\',10),example(\'b\',10);b[\'problem_group_id\']=a[\'problem_group_id\']\n        with self.assertRaises(ValueError):validate([a,b])\n    def test_overlength_rejected(self):\n        with self.assertRaises(ValueError):validate([example(\'a\',2050)])\n    def test_wrong_eos_rejected(self):\n        a=example(\'a\',10);a[\'eos_tokens\']=2\n        with self.assertRaises(ValueError):validate([a])\n    def test_write_preserves_other_version(self):\n        with tempfile.TemporaryDirectory() as d:\n            p=Path(d)/\'x\';write(p,{\'a\':1});write(p,{\'a\':1})\n            with self.assertRaises(FileExistsError):write(p,{\'a\':2})\n            self.assertEqual(json.loads(p.read_text()),{\'a\':1})\n\n\nHAS_SOLVER=bool(importlib.util.find_spec(\'ortools\'))\n\n\n@unittest.skipUnless(HAS_SOLVER,\'OR-Tools not installed in this execution environment\')\nclass Solver(unittest.TestCase):\n    def test_global_and_stratum_controls(self):\n        p,b=fixture();chosen,r=solve(p,b,1,deterministic_limit=2,tie_limit=1)\n        self.assertTrue(r[\'passed\']);self.assertEqual(len(chosen),4)\n        self.assertEqual(sum(x[\'supervised_tokens\'] for x in chosen),44)\n        self.assertTrue(all(s[\'actual_rows\']==s[\'required_rows\'] for s in r[\'strata\']))\n    def test_objective_matches_brute_force(self):\n        p,b=fixture();chosen,r=solve(p,b,1,deterministic_limit=2,tie_limit=1)\n        feasible=[]\n        for subset in itertools.combinations(b,4):\n            if sum(x[\'supervised_tokens\'] for x in subset)!=44:continue\n            if any(sum(x[\'dataset_id\']==s for x in subset)!=2 or\n                   sum(x[\'supervised_tokens\'] for x in subset if x[\'dataset_id\']==s)!=22\n                   for s in (\'orca\',\'numina\')):continue\n            feasible.append(sum(utility(x[\'row_id\']) for x in subset))\n        self.assertEqual(sum(utility(x[\'row_id\']) for x in chosen),max(feasible))\n    def test_order_independent(self):\n        p,b=fixture();a,_=solve(p,b,1,deterministic_limit=2,tie_limit=1)\n        random.Random(91).shuffle(b);c,_=solve(p,b,1,deterministic_limit=2,tie_limit=1)\n        self.assertEqual([x[\'row_id\'] for x in a],[x[\'row_id\'] for x in c])\n    def test_capacity_failure(self):\n        p,b=fixture();chosen,r=solve(p,b,4,deterministic_limit=2,tie_limit=1)\n        self.assertIsNone(chosen);self.assertTrue(r[\'infeasibility_proven\'])\n    def test_stratum_failure_does_not_relax(self):\n        p,b=fixture();chosen,r=solve(p,b,0.5,deterministic_limit=2,tie_limit=1)\n        self.assertIsNone(chosen)\n        self.assertIn(r[\'status\'],[\'INFEASIBLE\',\'INFEASIBLE_STRATUM_TOKEN_BOUNDS\'])\n        self.assertEqual(r[\'global_tolerance\'],\'0.005\');self.assertEqual(r[\'stratum_tolerance\'],\'0.05\')\n    def test_timeout_not_called_infeasible(self):\n        p,b=fixture();chosen,r=solve(p,b,1,deterministic_limit=0,tie_limit=0)\n        self.assertIsNone(chosen);self.assertFalse(r[\'infeasibility_proven\'])\n        self.assertEqual(r[\'status\'],\'UNKNOWN\')\n    def test_parent_group_overlap_rejected(self):\n        p,b=fixture();b[0][\'problem_group_id\']=p[0][\'problem_group_id\']\n        with self.assertRaises(ValueError):solve(p,b,1,deterministic_limit=1)\n\n\nif __name__==\'__main__\':\n    if \'--require-solver\' in sys.argv:\n        sys.argv.remove(\'--require-solver\')\n        if not HAS_SOLVER:raise SystemExit(\'The real solver is required for Sol acceptance checks\')\n    unittest.main(verbosity=2)\n', 'source_hashes.json': '{\n  "common.py": "c7b3be80ad94eec2a489850986b915ca4379e338e0b607bec3eb0bcd877eef82",\n  "workbench.py": "3502356f6e46e44c8b9ab8cc4c52ddd434abe3870112a517e983a7f774e2bf0f",\n  "continuation.py": "ab447b748dcb826f9dbfa1da5fcc515c295cf98bb3530c0f0e1a11b215a00437",\n  "compare.py": "31fb9db20ddd6661145a4bc1f4c5fff857c25b23e2c1fe61443b57b0cbc6609c",\n  "allocator.py": "683ad70be2a66134673d5f5746ff9c42de5bc202f9edf4a215a3a99d2e0cc9b2",\n  "config.json": "d2e85308e21b5dc3ca4b7f47ad4fae5fab838345028a917986d4343ec5e54806",\n  "parent_registry.json": "7327837a0596ed1cdfeab8fe5590cde310ea50fd3197421ce755b11cc0a3c259",\n  "source_provenance.json": "14105c5e744279d6358e603adc31a1bb7a27e3e39edf636f7a3f12096689c3c8"\n}\n', 'source_hashes_meta.json': '{\n  "method_document_sha256": "8c23b94193f02d21f8e512329ab63a0b7ac98864237912e093d92a1dda365f69",\n  "original_one_update_smokes": "All four passed in user-provided output; these are separate from v3 checks.",\n  "document_source": "User attachment Pasted markdown(7).md"\n}\n', 'evaluation_recovery.py': '"""Preserve invalid evaluation files and prevent overlapping recovery launches.\n\nNo rows are deduplicated or selected by correctness. Invalid/unvalidated outputs\nare archived whole, and the unchanged evaluator regenerates that output.\n"""\nfrom collections import Counter, defaultdict\nfrom contextlib import contextmanager\nfrom datetime import datetime, timezone\nimport fcntl\nimport hashlib\nimport json\nimport os\nfrom pathlib import Path\nimport uuid\n\nSTUDY = \'B_method_B_document_v3\'\n\n\ndef sha(path):\n    h=hashlib.sha256()\n    with Path(path).open(\'rb\') as f:\n        for chunk in iter(lambda:f.read(1024*1024),b\'\'):h.update(chunk)\n    return h.hexdigest()\n\n\ndef inspect_predictions(path):\n    path=Path(path)\n    if not path.is_file():return {\'exists\':False,\'path\':str(path)}\n    counts=Counter();variants=defaultdict(set);invalid=[];ids=[]\n    differing_fields=Counter();by_id={}\n    for line_no,line in enumerate(path.read_text().splitlines(),1):\n        if not line.strip():continue\n        try:\n            record=json.loads(line)\n            identity=record[\'problem_id\']\n            hash(identity)\n        except (ValueError,KeyError,TypeError) as error:\n            invalid.append({\'line\':line_no,\'error\':type(error).__name__});continue\n        counts[identity]+=1;ids.append(identity)\n        variants[identity].add(json.dumps(record,sort_keys=True,ensure_ascii=False))\n        if identity in by_id:\n            old=by_id[identity]\n            for key in set(old)|set(record):\n                if old.get(key)!=record.get(key):differing_fields[key]+=1\n        else:by_id[identity]=record\n    halfway=len(ids)//2\n    return {\'exists\':True,\'path\':str(path),\'sha256\':sha(path),\'valid_json_records\':len(ids),\n            \'unique_question_ids\':len(counts),\'duplicated_question_ids\':sum(n>1 for n in counts.values()),\n            \'conflicting_question_ids\':sum(len(v)>1 for v in variants.values()),\n            \'different_record_fields\':dict(differing_fields),\n            \'invalid_lines\':invalid,\'matching_two_halves\':bool(ids) and len(ids)%2==0 and ids[:halfway]==ids[halfway:]}\n\n\ndef evaluation_processes(root,proc_root=Path(\'/proc\')):\n    """Read only our own project workers; no process is killed."""\n    root=str(Path(root).resolve());found=[]\n    for path in Path(proc_root).glob(\'[0-9]*\'):\n        try:\n            pid=int(path.name)\n            if pid==os.getpid() or path.stat().st_uid!=os.getuid():continue\n            argv=[x.decode(\'utf-8\',errors=\'replace\') for x in (path/\'cmdline\').read_bytes().split(b\'\\0\') if x]\n            if not any(root in arg for arg in argv):continue\n            names={Path(arg).name for arg in argv}\n            is_evaluator=\'evaluate.py\' in names\n            is_runner=bool(names & {\'run_method_b.py\',\'run_method_b_eval_recovery.py\'})\n            is_evaluation_stage=\'evaluate\' in argv or (\'--action\' in argv and\n                    argv.index(\'--action\')+1<len(argv) and argv[argv.index(\'--action\')+1]==\'evaluate\')\n            if is_evaluator or (is_runner and is_evaluation_stage):\n                found.append({\'pid\':pid,\'program\':\'evaluate.py\' if is_evaluator else \'Method B evaluation runner\'})\n        except (OSError,ValueError):continue\n    return found\n\n\n@contextmanager\ndef single_evaluation(root):\n    folder=Path(root)/\'outputs\'/STUDY\n    folder.mkdir(parents=True,exist_ok=True)\n    path=folder/\'evaluation_recovery.lock\'\n    with path.open(\'a+\') as handle:\n        try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)\n        except BlockingIOError:\n            raise RuntimeError(\'Another recovery evaluation is running. Leave it to finish; no files were changed.\')\n        active=evaluation_processes(root)\n        if active:\n            raise RuntimeError(\'Existing evaluation workers are still active: \'+json.dumps(active)+\n                \'. No files were changed. Wait for those workers to exit before starting recovery.\')\n        try:yield\n        finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)\n\n\ndef archive_prediction_bundle(root,prediction,reason=\'Existing output lacks a matching verified completion record\'):\n    root=Path(root).resolve();base=root/\'outputs\'/STUDY\n    prediction=Path(prediction)\n    if prediction.is_symlink():raise ValueError(\'Refusing to move a symlink prediction: \'+str(prediction))\n    relative=prediction.resolve().relative_to(base.resolve())\n    if not (\'parent_evaluation_cache\' in relative.parts or \'evaluation\' in relative.parts):\n        raise ValueError(\'Recovery is restricted to evaluation outputs: \'+str(prediction))\n    if prediction.suffix!=\'.jsonl\':raise ValueError(\'Expected a prediction JSONL file\')\n    paths=[prediction,prediction.with_suffix(\'.config.json\'),prediction.with_suffix(\'.validated.json\')]\n    existing=[p for p in paths if p.exists()]\n    if not existing:return None\n    audit=inspect_predictions(prediction)\n    folder=base/\'evaluation_recovery_archives\'/(\n           datetime.now(timezone.utc).strftime(\'%Y%m%dT%H%M%S%fZ\')+\'_\'+uuid.uuid4().hex[:8])/relative.parent\n    folder.mkdir(parents=True,exist_ok=False)\n    moved=[]\n    for path in existing:\n        if path.is_symlink():raise ValueError(\'Refusing symlink metadata: \'+str(path))\n        identity=sha(path)\n        target=folder/path.name\n        path.rename(target)\n        if sha(target)!=identity:raise RuntimeError(\'Archived file checksum differs: \'+str(target))\n        moved.append({\'original\':str(path),\'archived\':str(target),\'sha256\':identity})\n    report={\'reason\':reason,\'inspection\':audit,\'moved\':moved,\n            \'records_cherry_picked\':False,\'training_changed\':False,\n            \'next_action\':\'Regenerate this output using the unchanged pinned evaluator\'}\n    (folder/\'recovery_audit.json\').write_text(json.dumps(report,indent=2)+\'\\n\')\n    print(\'Archived previous evaluation output:\',folder,flush=True)\n    print(\'Regenerating this prediction file. No training is repeated.\',flush=True)\n    return report\n\n\ndef enter(main):\n    import sys\n    args=sys.argv[1:]\n    if not args:raise ValueError(\'Specify evaluate, summary, or export\')\n    if \'--root\' not in args:raise ValueError(\'Expected --root PROJECT\')\n    root=Path(args[args.index(\'--root\')+1])\n    if args[0]==\'evaluate\':\n        with single_evaluation(root):main()\n    elif args[0]==\'engine\' and \'--action\' in args and args[args.index(\'--action\')+1]==\'evaluate\':\n        main()\n    elif args[0] in (\'summary\',\'export\'):main()\n    else:raise ValueError(\'This recovery runner only permits evaluation, summary, and export\')\n', 'patch_runner.py': '"""Create an evaluation-only recovery copy; the original helper is not edited."""\nimport ast\n\n\ndef replace_once(source,old,new):\n    if source.count(old)!=1:raise ValueError(\'Source differs from reviewed evaluation runner: \'+old[:65])\n    return source.replace(old,new,1)\n\n\ndef patch_runner(source):\n    source=replace_once(source,"from select_b import (ORTOOLS_VERSION, VERSION, bounds, cell, csv_bytes, digest,",\n                        "from evaluation_recovery import archive_prediction_bundle, enter\\n\\nfrom select_b import (ORTOOLS_VERSION, VERSION, bounds, cell, csv_bytes, digest,")\n    old="""                if pred.exists() or marker.exists():\n                    raise FileExistsError(\'Partial/different predictions were preserved: \'+str(pred)+\n                                          \'. Inspect them before starting a separately named retry.\')\n"""\n    new="""                if pred.exists() or marker.exists() or pred.with_suffix(\'.config.json\').exists():\n                    archive_prediction_bundle(root,pred)\n"""\n    source=replace_once(source,old,new)\n    old="""            if valid:\n                validate_pair(pred,pred)\n"""\n    new="""            if valid:\n                try:\n                    saved_runtime=read(pred.with_suffix(\'.config.json\'))\n                    valid=(saved_runtime.get(\'gpu\')==torch.cuda.get_device_name(0) and\n                           saved_runtime.get(\'torch\')==torch.__version__ and\n                           saved_runtime.get(\'cuda\')==torch.version.cuda)\n                except (OSError,ValueError):\n                    valid=False\n            if valid:\n                validate_pair(pred,pred)\n"""\n    source=replace_once(source,old,new)\n    old="""            if marker.is_file() and pred.is_file():\n                prior=read(marker)\n                valid=prior.get(\'signature\')==sig and prior.get(\'predictions_sha256\')==digest(pred) and prior.get(\'config_sha256\')==digest(pred.with_suffix(\'.config.json\'))\n"""\n    new="""            if marker.is_file() and pred.is_file():\n                try:\n                    prior=read(marker)\n                    valid=prior.get(\'signature\')==sig and prior.get(\'predictions_sha256\')==digest(pred) and prior.get(\'config_sha256\')==digest(pred.with_suffix(\'.config.json\'))\n                except (OSError,ValueError,KeyError):\n                    valid=False\n"""\n    source=replace_once(source,old,new)\n    if source.count("HERE/\'run_method_b.py\'")!=2:\n        raise ValueError(\'Unexpected number of subprocess script references\')\n    source=source.replace("HERE/\'run_method_b.py\'","Path(__file__).resolve()")\n    source=replace_once(source,"if __name__==\'__main__\':main()","if __name__==\'__main__\':enter(main)")\n    ast.parse(source)\n    return source\n'}
EXPECTED_REFERENCE_SHA256 = 'dc743c10ca08f1df00cfb89d5b80a6ce8d1571a6693e2ad0a4762bb587869b06'

STUDY='B_method_B_document_v3'
DEFAULT_ROOT=Path('/scratch/sshind38/capstone_approach_b')
PARENT_SEEDS=[17,123]
TRAINING_SEEDS=[42,137,2026]
BUDGETS=[1,0.5,2,4]
STOP_REQUESTED=False


def now_stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str)+'\n');tmp.replace(path)


def atomic_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(text);tmp.replace(path)


def condition(root,setting,budget):
    return Path(root)/'outputs'/STUDY/setting/f'budget_{budget:g}x'


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def bootstrap(root):
    code=root/'method_b_document_v3_code';code.mkdir(parents=True,exist_ok=True)
    for name,text in EMBEDDED_FILES.items():
        path=code/name;data=text.encode()
        if path.exists() and path.read_bytes()!=data:
            raise RuntimeError('Existing helper differs and was preserved: '+str(path))
        if not path.exists():path.write_bytes(data)
    original=code/'run_method_b.py'
    if sha(original)!=EXPECTED_REFERENCE_SHA256:
        raise RuntimeError('Original workflow source differs from the reviewed version')
    sys.path.insert(0,str(code))
    patcher=load_module('overnight_eval_patcher',code/'patch_runner.py')
    eval_source=patcher.patch_runner(original.read_text())
    train_source=original.read_text()
    needle="run_id=f'B_{key}_t{args.training_seed}_{args.budget:g}x_smoke32'"
    if train_source.count(needle)!=1:raise RuntimeError('Smoke run-id expression changed')
    train_source=train_source.replace(needle,"run_id=f'B_{key}_t{args.training_seed}_{args.budget:g}x_{stamp()}_smoke32'")
    needle="""        else:
            resume=None;new_name=run_name
        index[run_name]=new_name;update(index_path,index)
"""
    replacement="""        else:
            resume=None;new_name=run_name
            if (work/'runs'/new_name).exists():
                new_name=run_name+'_restart_'+stamp()
        index[run_name]=new_name;update(index_path,index)
"""
    if train_source.count(needle)!=1:raise RuntimeError('Inspected training-resume expression changed')
    train_source=train_source.replace(needle,replacement)
    train_source=train_source.replace("HERE/'run_method_b.py'","Path(__file__).resolve()")
    for name,text in [('run_method_b_eval_unattended.py',eval_source),
                      ('run_method_b_train_unattended.py',train_source)]:
        ast.parse(text)
        path=code/name
        if path.exists() and path.read_text()!=text:raise RuntimeError('Different unattended helper preserved: '+str(path))
        if not path.exists():path.write_text(text)
    this_file=Path(__file__).resolve()
    archived_source=code/('overnight_source_'+sha(this_file)[:12]+'.py')
    if not archived_source.exists():archived_source.write_bytes(this_file.read_bytes())
    return code


def parse_slurm_time(text):
    text=text.strip()
    if text in ('UNLIMITED','NOT_SET','INVALID','N/A',''):return None
    days=0
    if '-' in text:
        d,text=text.split('-',1);days=int(d)
    parts=[int(x) for x in text.split(':')]
    if len(parts)==3:h,m,s=parts
    elif len(parts)==2:h,m,s=0,*parts
    elif len(parts)==1:h,m,s=0,0,parts[0]
    else:raise ValueError('Unrecognized Slurm time: '+text)
    return days*86400+h*3600+m*60+s


def allocation_seconds():
    job=os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOBID')
    if not job or not re.fullmatch(r'[0-9_+.]+',job):return None
    try:
        p=subprocess.run(['squeue','-h','-j',job,'-o','%L'],capture_output=True,text=True,timeout=15)
        lines=[s for s in p.stdout.splitlines() if s.strip()]
        values=[parse_slurm_time(s) for s in lines]
        return min(x for x in values if x is not None) if p.returncode==0 and any(x is not None for x in values) else None
    except (OSError,ValueError,subprocess.SubprocessError):return None


def gpu_probe(python):
    source="""import json,torch
n=torch.cuda.device_count()
print(json.dumps({'available':torch.cuda.is_available(),'count':n,
 'gpu':torch.cuda.get_device_name(0) if n else None,'torch':str(torch.__version__),
 'cuda':torch.version.cuda}))
"""
    p=subprocess.run([str(python),'-c',source],capture_output=True,text=True,timeout=90)
    if p.returncode:raise RuntimeError('GPU/runtime check failed: '+p.stderr[-2500:])
    info=json.loads(p.stdout.strip().splitlines()[-1])
    if not info['available'] or info['count']!=1:
        raise RuntimeError('Run this file inside an active session exposing exactly one NVIDIA GPU')
    return info


def legacy_workers(root):
    found=[];root=str(root.resolve())
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if int(proc.name)==os.getpid() or proc.stat().st_uid!=os.getuid():continue
            args=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            if not any(root in s for s in args):continue
            names={Path(s).name for s in args}
            if any(n.startswith('run_method_b') and n.endswith('.py') for n in names) or 'evaluate.py' in names:
                found.append({'pid':int(proc.name),'program':next((n for n in names if n.startswith('run_method_b') or n=='evaluate.py'),'worker')})
        except (OSError,ValueError):pass
    return found


def capacity(root,setting,budget):
    import select_b as s
    shared=root/'outputs'/STUDY/'shared'
    reference=jsonl(shared/'parent_rebinned_reference.jsonl')
    if setting=='transfer':reference=[r for r in reference if r['dataset_id']=='numina']
    bank=jsonl(shared/('mixed_candidate_bank.jsonl' if setting=='mixed' else 'transfer_numina_bank.jsonl'))
    required=s.quotas(reference,budget);available=Counter(s.cell(r) for r in bank)
    ref_tokens=Counter()
    lengths={}
    for r in reference:ref_tokens[s.cell(r)]+=r['supervised_tokens']
    for r in bank:lengths.setdefault(s.cell(r),[]).append(r['supervised_tokens'])
    details=[];failed=[]
    for h,n in sorted(required.items()):
        lo,hi=s.bounds(Fraction(ref_tokens[h])*Fraction(str(budget)),'0.05')
        values=sorted(lengths.get(h,[]));minimum=sum(values[:n]);maximum=sum(values[-n:]) if n else 0
        reason=None
        if available[h]<n:reason='not enough distinct screened rows'
        elif minimum>hi or maximum<lo:reason='whole-row token bounds cannot satisfy the required stratum'
        record={'source':h[0],'bin':h[1],'required_rows':n,'available_rows':available[h],
                'token_lower':lo,'token_upper':hi,'minimum_possible':minimum,'maximum_possible':maximum}
        if reason:record['blocker']=reason;failed.append(record)
        details.append(record)
    return {'setting':setting,'budget':budget,'feasible_by_necessary_bounds':not failed,
            'infeasibility_proven':bool(failed),'blockers':failed,'strata':details,
            'note':'Passing necessary bounds does not prove a feasible CP-SAT solution exists.'}


def make_tasks():
    tasks=[];conditions={};sequence=0
    def add(stage,setting,budget,priority,model=None,parent=None,seed=None,deps=()):
        nonlocal sequence
        identity=f'{setting}_{budget:g}x_{stage}'
        if model:identity+=f'_{model}_p{parent}_t{seed}'
        sequence+=1
        task={'id':identity,'stage':stage,'setting':setting,'budget':budget,'priority':priority,
              'sequence':sequence,'model':model,'parent_seed':parent,'training_seed':seed,
              'dependencies':list(deps),'status':'pending','attempts':0,'events':[]}
        tasks.append(task);return identity
    def case(setting,budget,priority,seed,models):
        key=(setting,budget)
        if key not in conditions:conditions[key]=add('prepare',setting,budget,priority)
        for model in models:
            for parent in PARENT_SEEDS:
                a=add('smoke',setting,budget,priority,model,parent,seed,[conditions[key]])
                b=add('train',setting,budget,priority,model,parent,seed,[a])
                add('evaluate',setting,budget,priority,model,parent,seed,[b])
    case('mixed',1,0,42,['0.5B','1.5B'])
    for seed in [137,2026]:case('mixed',1,1,seed,['0.5B','1.5B'])
    case('transfer',1,2,42,['0.5B'])
    case('mixed',0.5,3,42,['0.5B','1.5B'])
    case('mixed',2,4,42,['0.5B','1.5B'])
    case('mixed',4,5,42,['0.5B','1.5B'])
    for seed in [137,2026]:case('transfer',1,6,seed,['0.5B'])
    for priority,budget in enumerate([0.5,2,4],7):
        for seed in [137,2026]:case('mixed',budget,priority,seed,['0.5B','1.5B'])
    return tasks


def run_directory(root,task):
    work=condition(root,task['setting'],task['budget'])
    base=f'B_{task["model"]}_p{task["parent_seed"]}_t{task["training_seed"]}_{task["budget"]:g}x'
    index=read(work/'run_index.json') if (work/'run_index.json').is_file() else {}
    return work/'runs'/index.get(base,base)


def prediction_valid(path,gpu,expected_ids):
    try:
        marker=read(path.with_suffix('.validated.json'));cfg=read(path.with_suffix('.config.json'))
        if marker.get('predictions_sha256')!=sha(path) or marker.get('config_sha256')!=sha(path.with_suffix('.config.json')):return False
        if any(cfg.get(k)!=gpu[k] for k in ['gpu','torch','cuda']):return False
        data=jsonl(path);ids=[r['problem_id'] for r in data]
        return len(ids)==len(set(ids)) and set(ids)==expected_ids and all(type(r.get('correct')) is bool for r in data)
    except (OSError,ValueError,KeyError,TypeError):return False


def completed(root,task,gpu):
    try:
        work=condition(root,task['setting'],task['budget'])
        if task['stage']=='prepare':
            audit=read(work/'selection_audit.json');v=read(work/'label_counts_and_strata_verified.json')
            selected=work/f'{task["budget"]:g}x/train_B.jsonl'
            return audit.get('passed') is True and audit.get('selected_manifest_sha256')==sha(selected) and v.get('passed') is True and v.get('selection_audit_sha256')==sha(work/'selection_audit.json')
        if task['stage']=='smoke':
            profile=f'{task["model"]}_seed{task["parent_seed"]}'
            s=read(work/'smokes'/f'{profile}_t{task["training_seed"]}'/'smoke_report.json')
            return all(s.get(k) is True for k in ['passed','actual_gpu_execution','reload_probe_matched']) and s.get('optimizer_updates')==4 and s.get('fingerprint',{}).get('training_seed')==task['training_seed']
        run=run_directory(root,task)
        audit=read(run/'B_completion_audit.json');cfg=read(run/'run_config.json')
        if audit.get('passed') is not True or cfg.get('parent_profile')!=f'{task["model"]}_seed{task["parent_seed"]}' or cfg.get('continuation_training_seed')!=task['training_seed'] or cfg.get('manifest_sha256')!=sha(work/f'{task["budget"]:g}x/train_B.jsonl'):return False
        if task['stage']=='train':return True
        result=read(work/'evaluation'/run.name/'B_vs_parent.json')
        comparisons=result.get('comparisons',[])
        if {r['dataset'] for r in comparisons}!={'dev_gsm','dev_math'}:return False
        for dataset in ['dev_gsm','dev_math']:
            ids={r['problem_id'] for r in jsonl(work/'runtime/vendor/evaluation/data'/f'{dataset}.jsonl')}
            if not prediction_valid(work/'evaluation'/run.name/f'B_{dataset}.jsonl',gpu,ids):return False
            profile=f'{task["model"]}_seed{task["parent_seed"]}'
            parents=list((root/'outputs'/STUDY/'parent_evaluation_cache').glob(f'{profile}_{dataset}_*.jsonl'))
            if not any(prediction_valid(p,gpu,ids) for p in parents):return False
        return all(r.get('parent_profile')==f'{task["model"]}_seed{task["parent_seed"]}' and r.get('training_seed')==task['training_seed'] and r.get('budget_multiplier')==task['budget'] for r in comparisons)
    except (OSError,ValueError,KeyError,TypeError):return False


def tail(path,limit=4000):
    try:
        with Path(path).open('rb') as f:
            f.seek(0,2);length=f.tell();f.seek(max(0,length-limit));return f.read().decode(errors='replace')
    except OSError:return ''


def terminate_group(process):
    for sig,wait in [(signal.SIGINT,12),(signal.SIGTERM,8),(signal.SIGKILL,3)]:
        if process.poll() is not None:return
        try:os.killpg(process.pid,sig)
        except ProcessLookupError:return
        try:process.wait(timeout=wait);return
        except subprocess.TimeoutExpired:pass


def supervise(command,log,deadline,*,stall_seconds=1200,heartbeat=None):
    log.parent.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    with log.open('w',encoding='utf-8') as f:
        env=dict(os.environ,PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        p=subprocess.Popen(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,start_new_session=True,env=env)
        last_size=0;last_progress=time.monotonic();last_beat=0;reason=None
        try:
            while p.poll() is None:
                now=time.monotonic();size=log.stat().st_size
                if size!=last_size:last_progress=now;last_size=size
                if STOP_REQUESTED:reason='stop signal received'
                elif time.time()>=deadline:reason='allocation time budget reached'
                elif now-last_progress>stall_seconds:reason='no log progress within stall limit'
                if reason:terminate_group(p);break
                if now-last_beat>=30:
                    if heartbeat:heartbeat(p.pid,round(now-start,1),tail(log,900))
                    last_beat=now
                time.sleep(2)
            code=p.wait()
        finally:
            if p.poll() is None:terminate_group(p)
            # An exited supervisor must not leave an orphan evaluator writing
            # into the next attempt's file. Every task owns a distinct group.
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    return {'return_code':code,'wall_seconds':time.monotonic()-start,'stop_reason':reason,'log':str(log)}


def command_for(root,code,task):
    python=root/'venv_b_training_py312/bin/python'
    if task['stage']=='prepare':
        return [root/'venv_b_solver_v3/bin/python','-u',code/'run_method_b_train_unattended.py','prepare',
                '--root',root,'--setting',task['setting'],'--budget',str(task['budget']),
                '--parent-seeds',*map(str,PARENT_SEEDS)]
    script=code/('run_method_b_eval_unattended.py' if task['stage']=='evaluate' else 'run_method_b_train_unattended.py')
    return [python,'-u',script,'engine','--root',root,'--setting',task['setting'],'--budget',str(task['budget']),
            '--action',task['stage'],'--model',task['model'],'--parent-seed',str(task['parent_seed']),
            '--training-seed',str(task['training_seed'])]


def estimated_seconds(root,task):
    if task['stage']=='prepare':return 900
    if task['stage']=='smoke':return 150
    if task['stage']=='train':
        base=condition(root,'mixed',1)/'runs'/f'B_{task["model"]}_p{task["parent_seed"]}_t42_1x/training_log.jsonl'
        try:
            ticks=[r for r in jsonl(base) if r.get('elapsed_seconds',0)>0]
            reference=ticks[-1]['elapsed_seconds']
        except (OSError,ValueError,KeyError,IndexError):reference=600 if task['model']=='0.5B' else 900
        multiplier=task['budget']*(0.6 if task['setting']=='transfer' else 1)
        return reference*multiplier*1.4+90
    # Parent predictions are reused across conditions. The first primary pass
    # may need parent+B; later runs normally need only the two B files.
    base=900 if task['model']=='0.5B' else 1500
    profile=f'{task["model"]}_seed{task["parent_seed"]}'
    cached=list((root/'outputs'/STUDY/'parent_evaluation_cache').glob(f'{profile}_*.validated.json'))
    return base*(2 if len(cached)>=2 else 4)


class Controller:
    def __init__(self,root,run_folder,hours):
        self.root=root;self.folder=run_folder;self.hours=hours;self.tasks=make_tasks();self.gpu={}
        self.code=None;self.capacities={};self.running=None;self.start=time.time()
        remaining=allocation_seconds()
        seconds=min(hours*3600,max(0,remaining-90)) if remaining is not None else hours*3600
        self.hard_deadline=self.start+seconds;self.work_deadline=self.hard_deadline-300
        self.meta={'run_id':run_folder.name,'started_utc':datetime.now(timezone.utc).isoformat(),
          'worker_pid':os.getpid(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
          'allocation_seconds_at_start':remaining,'requested_worker_hours':hours,
          'work_deadline_utc':datetime.fromtimestamp(self.work_deadline,timezone.utc).isoformat(),
          'scope':'All configured B experiments; final tests and A/C jobs are not launched',
          'external_requirements':['Common clean/final evaluator clearance and protected final model list',
             'A/C predictions for cross-method comparisons','A larger commonly screened bank for any infeasible budget']}
    def save(self,status='running'):
        value={'status':status,'metadata':self.meta,'gpu':self.gpu,'running_task':self.running,
               'tasks':self.tasks,'capacity_checks':list(self.capacities.values())}
        atomic_json(self.folder/'status.json',value)
        atomic_json(self.root/'outputs'/STUDY/'overnight/latest_status.json',value)
        self.report(status)
    def report(self,status):
        counts=Counter(t['status'] for t in self.tasks)
        training=[t for t in self.tasks if t['stage']=='train' and t['status']=='completed']
        evaluated=[t for t in self.tasks if t['stage']=='evaluate' and t['status']=='completed']
        lines=['METHOD B MORNING REPORT',f'Run: {self.folder.name}',f'Status: {status}',
          f'Completed configured training cases: {len(training)}/54',
          f'Completed configured evaluation cases: {len(evaluated)}/54',
          'Task counts: '+json.dumps(dict(counts)),
          'Results remain preliminary until the shared evaluator/protection checks are verified.',
          '', 'BLOCKED / FAILED / DEFERRED TASKS:']
        for t in self.tasks:
            if t['status'] in ('blocked','failed','deferred'):
                lines.append(t['id']+' | '+t['status']+' | '+t.get('reason',''))
        lines+=['','EXTERNAL STUDY REQUIREMENTS:']+self.meta['external_requirements']
        lines+=['','Progress and per-task logs: '+str(self.folder),
                'Canonical result table: '+str(self.root/'outputs'/STUDY/'development_results.csv')]
        data='\n'.join(lines)+'\n'
        atomic_text(self.folder/'MORNING_REPORT.txt',data)
        atomic_text(self.root/'outputs'/STUDY/'overnight/MORNING_REPORT.txt',data)
    def heartbeat(self,pid,elapsed,last):
        self.running.update(pid=pid,elapsed_seconds=elapsed)
        self.save()
        print(f'HEARTBEAT {self.running["id"]}: {elapsed:.0f}s elapsed; '
              f'{max(0,self.work_deadline-time.time())/60:.1f} work minutes left',flush=True)
        if last.strip():print(last[-650:],flush=True)
    def preflight(self):
        self.code=bootstrap(self.root)
        reference=load_module('overnight_reference',self.code/'run_method_b.py')
        reference.source(self.root)
        audit=read(self.root/'outputs'/STUDY/'shared/bank_audit.json')
        for name,key in [('mixed_candidate_bank.jsonl','mixed_bank_sha256'),
                         ('transfer_numina_bank.jsonl','transfer_bank_sha256'),
                         ('clean_development_reference.jsonl','clean_development_sha256'),
                         ('parent_length_bins.json','length_bins_sha256')]:
            if sha(self.root/'outputs'/STUDY/'shared'/name)!=audit[key]:raise RuntimeError('Bank hash changed: '+name)
        parent=self.root/'data/train_n1000.jsonl'
        cfg=read(self.root/'b_workbench_500/config.json')
        if sha(parent)!=cfg['parent_manifest_sha256']:raise RuntimeError('Actual parent manifest changed')
        import select_b
        expected=select_b.rebin(jsonl(parent),select_b.parent_edges(jsonl(parent)))
        if expected!=jsonl(self.root/'outputs'/STUDY/'shared/parent_rebinned_reference.jsonl'):
            raise RuntimeError('Parent length-stratum reference changed')
        for entry in read(self.code/'parent_registry.json').values() if (self.code/'parent_registry.json').is_file() else read(self.root/'b_workbench_500/parent_registry.json').values():
            if sha(self.root/entry['adapter_relative_path']/'adapter_model.safetensors')!=entry['expected_adapter_sha256']:
                raise RuntimeError('A parent adapter hash differs from the verified handoff')
        self.gpu=gpu_probe(self.root/'venv_b_training_py312/bin/python')
        for setting,budget in [('mixed',1),('mixed',0.5),('mixed',2),('mixed',4),('transfer',1)]:
            self.capacities[(setting,budget)]=capacity(self.root,setting,budget)
        if self.work_deadline-time.time()<180:raise RuntimeError('Too little allocation time remains to start unattended work')
        active=legacy_workers(self.root)
        if active:raise RuntimeError('Other project workers are active; no duplicate run was started: '+json.dumps(active))
        self.save('ready')
        print('STARTUP_PASSED. GPU:',self.gpu['gpu'],flush=True)
        print('The complete queue contains 54 training/evaluation cases. Six hours may leave deferred tasks.',flush=True)
        for key,report in self.capacities.items():
            if report['infeasibility_proven']:
                print('BLOCKED DATA CAPACITY:',key,json.dumps(report['blockers']),flush=True)
    def run_task(self,task):
        for attempt in range(1,3 if task['stage']!='prepare' else 2):
            task['attempts']=attempt;task['status']='running'
            log=self.folder/'logs'/f'{task["sequence"]:03d}_{task["id"]}_attempt{attempt}.log'
            self.running={'id':task['id'],'log':str(log)};self.save()
            print('START',task['id'],'attempt',attempt,'log:',log,flush=True)
            try:
                result=supervise(command_for(self.root,self.code,task),log,self.work_deadline,
                                 stall_seconds=2400 if task['stage']=='prepare' else 1200,
                                 heartbeat=self.heartbeat)
            except Exception:
                error=traceback.format_exc()
                log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('a') as out:out.write('\nSUPERVISOR ERROR\n'+error)
                result={'return_code':1,'wall_seconds':0,'stop_reason':None,
                        'log':str(log),'supervisor_error':error}
            task['events'].append(result)
            if result['return_code']==0 and completed(self.root,task,self.gpu):
                task['status']='completed';task['reason']='Completed and canonical evidence verified';break
            message=result['stop_reason'] or tail(log,1800)
            if result['return_code']==0:message='Stage returned successfully but canonical completion evidence did not verify'
            task['reason']=message[-1600:]
            print('TASK NEEDS RETRY/REVIEW:',task['id'],task['reason'],flush=True)
            if result['stop_reason']=='allocation time budget reached' or STOP_REQUESTED:
                task['status']='deferred';break
            if attempt>=2 or task['stage']=='prepare':task['status']='failed';break
            if self.work_deadline-time.time()<estimated_seconds(self.root,task)+60:
                task['status']='deferred';task['reason']+='; retry deferred to the next allocation';break
        self.running=None;self.save()
    def queue(self):
        by_id={t['id']:t for t in self.tasks}
        # Canonical data decide what is complete; an old queue ledger alone
        # never causes a task to be treated as finished.
        for task in self.tasks:
            if completed(self.root,task,self.gpu):
                task.update(status='completed',reason='Reused verified existing artifacts')
        while not STOP_REQUESTED and time.time()<self.work_deadline:
            changed=False
            for t in self.tasks:
                if t['status']!='pending':continue
                cap=self.capacities[(t['setting'],t['budget'])]
                if cap['infeasibility_proven']:
                    t.update(status='blocked',reason='Screened-bank capacity/token bounds fail; constraints were not relaxed');changed=True;continue
                bad=[d for d in t['dependencies'] if by_id[d]['status'] in ('failed','blocked')]
                if bad:
                    t.update(status='blocked',reason='Dependency requires repair: '+', '.join(bad));changed=True
            eligible=[t for t in self.tasks if t['status']=='pending' and all(by_id[d]['status']=='completed' for d in t['dependencies'])]
            eligible.sort(key=lambda t:(t['priority'],t['sequence']))
            if not eligible:
                if changed:self.save()
                break
            chosen=None
            for t in eligible:
                remaining=self.work_deadline-time.time();estimate=estimated_seconds(self.root,t)
                if t['stage']=='train':
                    # Prefer to leave enough time to evaluate a newly trained
                    # model, rather than starting new training at the deadline.
                    eval_task=dict(t,stage='evaluate')
                    estimate+=estimated_seconds(self.root,eval_task)*0.7
                if estimate>remaining:
                    t.update(status='deferred',reason='Estimated work does not fit the remaining allocation');continue
                chosen=t;break
            if chosen is None:self.save();break
            if chosen['stage']=='prepare' and not (self.root/'venv_b_solver_v3/bin/python').is_file():
                setup_log=self.folder/'logs/solver_setup.log'
                result=supervise([self.root/'venv_b_training_py312/bin/python','-u',
                      self.code/'run_method_b_train_unattended.py','install-solver','--root',self.root],
                      setup_log,min(self.work_deadline,time.time()+900),stall_seconds=600)
                if result['return_code']:
                    chosen.update(status='failed',reason='Solver setup failed; see '+str(setup_log));self.save();continue
            self.run_task(chosen)
        for task in self.tasks:
            if task['status']=='pending':task.update(status='deferred',reason='Time limit or dependency was deferred; rerun the same file to resume')
        self.save('queue_finished')
    def finish(self):
        self.running=None;self.save('saving_results')
        python=self.root/'venv_b_training_py312/bin/python'
        for stage in ['summary','export']:
            if time.time()+15>=self.hard_deadline:break
            log=self.folder/'logs'/f'final_{stage}.log'
            result=supervise([python,'-u',self.code/'run_method_b_eval_unattended.py',stage,'--root',self.root],
                    log,self.hard_deadline-5,stall_seconds=240)
            self.meta[f'{stage}_result']=result
            if result['return_code']==0:print(tail(log,1800),flush=True)
        counts=Counter(t['status'] for t in self.tasks)
        final='all_configured_execution_completed' if all(t['status']=='completed' for t in self.tasks) else 'finished_with_remaining_work'
        self.save(final)
        print('OVERNIGHT_RUN_FINISHED:',json.dumps(dict(counts)),flush=True)
        print('MORNING_REPORT:',self.folder/'MORNING_REPORT.txt',flush=True)


def signal_stop(signum,frame):
    global STOP_REQUESTED
    STOP_REQUESTED=True


def worker(args):
    root=args.root.resolve();folder=args.run_folder
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal_stop)
    controller=Controller(root,folder,args.hours)
    try:
        code=bootstrap(root)
        recovery=load_module('night_recovery',code/'evaluation_recovery.py')
        overnight=root/'outputs'/STUDY/'overnight'
        with ExitStack() as stack:
            handle=stack.enter_context((overnight/'worker.lock').open('a+'))
            try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('Another overnight worker is already running')
            stack.enter_context(recovery.single_evaluation(root))
            controller.preflight()
            atomic_json(folder/'startup.json',{'ready':True,'pid':os.getpid(),'gpu':controller.gpu,
                 'work_deadline_utc':controller.meta['work_deadline_utc']})
            controller.queue()
            controller.finish()
    except BaseException as error:
        controller.meta['fatal_error']=str(error)
        controller.meta['traceback']=traceback.format_exc()
        controller.save('stopped_for_required_repair')
        atomic_json(folder/'startup.json',{'ready':False,'error':str(error),'pid':os.getpid()})
        traceback.print_exc();return 1
    return 0


def launch(args):
    root=args.root.resolve();python=root/'venv_b_training_py312/bin/python'
    if not root.is_dir() or not python.is_file():raise RuntimeError('Project/training environment is missing. Run this file on Sol.')
    base=root/'outputs'/STUDY/'overnight';base.mkdir(parents=True,exist_ok=True)
    with (base/'worker.lock').open('a+') as probe:
        try:fcntl.flock(probe.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('An overnight worker is already running. No second worker was started.',flush=True)
            print('Progress:',base/'latest_status.json',flush=True);return 0
        finally:
            try:fcntl.flock(probe.fileno(),fcntl.LOCK_UN)
            except OSError:pass
    folder=base/'sessions'/now_stamp()
    folder.mkdir(parents=True,exist_ok=False)
    log=folder/'console.log'
    command=[str(python),'-u',str(Path(__file__).resolve()),'--worker','--root',str(root),
             '--hours',str(args.hours),'--run-folder',str(folder)]
    with log.open('w') as f:
        process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                 start_new_session=True,cwd=str(root))
    print('STARTING UNATTENDED WORKER:',process.pid,flush=True)
    print('Console log:',log,flush=True)
    print('Waiting for startup checks; leave this command open for up to one minute.',flush=True)
    end=time.monotonic()+60
    while time.monotonic()<end:
        if (folder/'startup.json').is_file():
            status=read(folder/'startup.json')
            print(json.dumps(status,indent=2),flush=True)
            if not status.get('ready'):
                print(tail(log,4000),flush=True);return 1
            print('RUNNING IN BACKGROUND. Browser disconnection will not stop this worker inside its live allocation.',flush=True)
            print('Progress:',base/'latest_status.json',flush=True)
            print('Morning report:',base/'MORNING_REPORT.txt',flush=True)
            print('The same file can be rerun in another allocation to finish deferred work.',flush=True)
            return 0
        if process.poll() is not None:
            print(tail(log,4000),flush=True);return process.returncode or 1
        time.sleep(1)
    print('Startup is still in progress. Inspect this log before leaving it unattended:',log,flush=True)
    print(tail(log,2000),flush=True)
    return 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--hours',type=float,default=5.75)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--run-folder',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args()
    if not 0<args.hours<=24:raise ValueError('Choose a positive worker time budget of at most 24 hours')
    if args.plan_only:
        tasks=make_tasks();print(json.dumps({'tasks':len(tasks),
          'training_cases':sum(t['stage']=='train' for t in tasks),
          'evaluation_cases':sum(t['stage']=='evaluate' for t in tasks),
          'default_hours':args.hours,'six_hour_completion_guaranteed':False},indent=2));return 0
    return worker(args) if args.worker else launch(args)


if __name__=='__main__':raise SystemExit(main())
