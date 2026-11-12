"""CPU-only preview of known repairs and their B/C0 selection consequences.

This is not a replacement training runner or an eligibility clearance. The old
bridge, manifests, checkpoints and score cache remain unchanged.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

BRIDGE_SHA = '47dd44c53526ef10ca8056b65ef215988e028d2047a04b549966c849db32bc6d'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def apply_decisions(bank, decisions):
    lookup = {r['row_id']: r for r in bank}
    if len(lookup) != len(bank):
        raise ValueError('Duplicate input row IDs')
    ids = [d['row_id'] for d in decisions]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate exclusion decision')
    for decision in decisions:
        row = lookup.get(decision['row_id'])
        if not row or hashlib.sha256(row['problem'].encode()).hexdigest() != decision['question_sha256']:
            raise ValueError('Decision refers to absent or changed question')
        if decision.get('completion_sha256') and hashlib.sha256(row['completion'].encode()).hexdigest() != decision['completion_sha256']:
            raise ValueError('Decision refers to changed reference solution')
        if decision.get('retained_representative'):
            other = decision['retained_representative']
            if other not in lookup or other in ids:
                raise ValueError('Duplicate representative absent or also excluded')
    bad_groups = {lookup[rid]['problem_group_id'] for rid in ids}
    retained = [r for r in bank if r['problem_group_id'] not in bad_groups]
    return retained, [r for r in bank if r['problem_group_id'] in bad_groups]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('/scratch/sshind38/capstone_c0_c1_work'))
    parser.add_argument('--b-root', type=Path, default=Path('/scratch/sshind38/capstone_approach_b'))
    args = parser.parse_args()
    args.work = args.work.resolve()
    package = Path(__file__).resolve().parent
    decisions_path = package / 'known_exclusions.json'
    decisions = json.loads(decisions_path.read_text())['decisions']
    bridge_path = args.work / 'c0_c1_sol_bridge_v1/sol_bridge.py'
    if sha(bridge_path) != BRIDGE_SHA:
        raise ValueError('Original bridge source changed; inspect it before importing old evidence')
    spec = importlib.util.spec_from_file_location('_original_bridge_for_cpu_preview', bridge_path)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    context = bridge.load_context(SimpleNamespace(
        b_root=str(args.b_root), repo=str(args.work / 'repo'),
        output=str(args.work / 'outputs/C0_C1_B_bridge_v1'), model='0.5B', parent_seed=17,
    ))
    screen = json.loads((package / 'screening_proposal.json').read_text())
    projection = [{k: r.get(k) for k in screen['projection_fields']} for r in context.bank]
    observed = hashlib.sha256(json.dumps(projection, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if observed != screen['original_projection_sha256']:
        raise ValueError('Live bank question projection differs from screened input')
    bridge.verify_proof(context)
    old_scores, score_receipt = bridge.load_scores(context)
    old_c0, old_c0_receipt = bridge.load_selection(context, 'C0')
    if score_receipt['parent_sha256'] != context.entry['expected_adapter_sha256']:
        raise ValueError('Cached scores came from a different parent')
    full_lookup = {r['row_id']: r for r in context.bank}
    for score in old_scores:
        if (score['row_sha256'] != bridge.digest(full_lookup[score['row_id']])
                or score.get('status') != 'ok'
                or score['measurement_key'] != score_receipt['measurement_key']
                or score['parent_sha256'] != context.entry['expected_adapter_sha256']):
            raise ValueError('Cached row measurement/content identity differs')
    from capstone_person4.b_select import solve, hash32
    from capstone_person4.selection import fit_priorities
    retained, removed = apply_decisions(context.bank, decisions)
    bridge.unique(retained)
    if (len(removed) != screen['held_out_rows'] or len(retained) != screen['retained_rows']
            or {r['row_id'] for r in removed} != set(screen['held_out_ids'])):
        raise ValueError('Applied quarantine differs from the screening proposal')
    capacities = Counter((r['dataset_id'], r['length_bin']) for r in retained)
    required = Counter((r['dataset_id'], r['length_bin']) for r in context.rebinned)
    if set(capacities) != set(required) or any(capacities[k] < n for k, n in required.items()):
        raise ValueError('Provisional pool lacks a required stratum capacity; no bounds relaxed')
    retained_ids = {r['row_id'] for r in retained}
    retained_scores = [s for s in old_scores if s['row_id'] in retained_ids]
    # Within-stratum fit ranks must be recalculated after the universe changes.
    priorities = fit_priorities(retained, retained_scores, 42)
    utility = {r['row_id']: int(round(r['fit'] * 1_000_000)) for r in priorities}
    identity = bridge.digest({'version': 'independent-conservative-screen-preview-v1', 'original_identity': context.identity,
        'decisions_sha256': sha(decisions_path), 'preview_script_sha256': sha(Path(__file__).resolve()),
        'screening_proposal_sha256': sha(package/'screening_proposal.json'),
        'old_score_sha256': score_receipt['scores_sha256'], 'new_rows': bridge.digest(retained)})
    out = args.work / 'outputs/C0_C1_independent_screen_preview_v1/0.5B_seed17' / identity[:16]
    bridge.save_rows(out / 'candidate_bank.jsonl', retained)
    bridge.save_rows(out / 'imported_reference_measurements.jsonl', retained_scores)
    selections = {}
    for method, utilities, original in [('B', None, bridge.rows(context.b_path)), ('C0', utility, old_c0)]:
        chosen, solver = solve(context.rebinned, retained, 1, utilities=utilities)
        if chosen is None or solver.get('passed') is not True:
            raise ValueError(f'{method} preview not feasible: {solver.get("status")}; no bounds relaxed')
        chosen.sort(key=lambda r: (hash32('42|training-order|' + r['row_id']), r['row_id']))
        target = out / method / 'train.jsonl'
        bridge.save_rows(target, chosen)
        selected_ids = {r['row_id'] for r in chosen}
        old_ids = {r['row_id'] for r in original}
        if len(chosen) != 1000 or any(r['row_id'] not in retained_ids for r in chosen):
            raise ValueError('Preview selection membership/count differs')
        selected_counts = Counter((r['dataset_id'],r['length_bin']) for r in chosen)
        if selected_counts != required:
            raise ValueError('Preview source/length counts differ')
        details = {'method': method, 'rows': len(chosen), 'actual_tokens': sum(r['supervised_tokens'] for r in chosen),
            'target_tokens': bridge.TARGET, 'manifest_sha256': sha(target), 'solver': solver,
            'same_ids_as_original': selected_ids == old_ids,
            'same_content_and_order_as_original': chosen == original,
            'removed_from_original': sorted(old_ids-selected_ids), 'added_to_original': sorted(selected_ids-old_ids),
            'screen_quarantined_original_rows': sorted(old_ids-{r['row_id'] for r in original if r['row_id'] in retained_ids}),
            'training_started': False, 'main_training_cleared': False}
        bridge.save(out / method / 'selection_preview.json', details)
        selections[method] = {k:v for k,v in details.items() if k not in ('solver','removed_from_original','added_to_original','screen_quarantined_original_rows')}
        selections[method].update({'removed_count': len(old_ids-selected_ids), 'added_count': len(selected_ids-old_ids),
                                  'screen_quarantined_original_count': len(details['screen_quarantined_original_rows'])})
    report = {'version': 'independent-conservative-screen-preview-v1', 'identity': identity,
        'status': 'new_screening_protocol_proposal_not_adopted_for_training',
        'screening_proposal_sha256': sha(package/'screening_proposal.json'),
        'original_matcher_flags_resolved': False,
        'original_bridge_identity': context.identity, 'old_bank_sha256': bridge.BANK_SHA,
        'new_bank_sha256': sha(out/'candidate_bank.jsonl'), 'original_rows': len(context.bank),
        'new_rows': len(retained), 'removed_rows': len(removed),
        'source_rows': dict(Counter(r['dataset_id'] for r in retained)),
        'minimum_stratum_capacity': min(capacities.values()), 'selections': selections,
        'score_import': {'source_sha256': score_receipt['scores_sha256'], 'source_measurement_key': score_receipt['measurement_key'],
                         'retained_rows': len(retained_scores), 'new_GPU_scoring_performed': False,
                         'row_content_hashes_checked': True, 'ranks_recomputed': True},
        'remaining_review': 'This is a replacement screening proposal, not an adoption or certification of the unavailable 25 exclusion/572 quarantine audit. Review the declared scope and choose the screening protocol before training; inherited parent exposure remains disclosed.',
        'training_started': False, 'main_training_cleared': False, 'output_directory': str(out)}
    bridge.save(out / 'preview_report.json', report)
    # Check the original input manifests again after generating the preview.
    bridge.check_hash(context.bank_path, bridge.BANK_SHA)
    bridge.check_hash(context.b_path, bridge.B_SHA)
    bridge.check_hash(bridge.selection_root(context,'C0')/'train.jsonl', old_c0_receipt['manifest_sha256'])
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
