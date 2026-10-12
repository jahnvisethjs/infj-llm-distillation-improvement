"""Instance-level grouping: similarity retrieves evidence; only identity/decisions join groups."""
import json
import re
from collections import defaultdict

import pandas as pd

from capstone_person1.pipeline_controls import clean_text, sha256_text, scalar_fraction, answers_equivalent

MATCHING_VERSION = 'instance-decisions-v1'


def normalize_instance(problem):
    # Preserve numbers, signs, operators, variable case and mathematical expressions.
    return re.sub(r'\s+', ' ', clean_text(problem)).strip()


def instance_hash(problem): return sha256_text(normalize_instance(problem))


def context_equivalent(problem, left, right):
    assignments = re.findall(r'^\s*([a-zA-Z])\s*=', str(left)) + re.findall(r'^\s*([a-zA-Z])\s*=', str(right))
    if assignments and (len(set(assignments)) > 1 or any(not re.search(r'\b' + re.escape(var) + r'\b', problem) for var in assignments)):
        return None
    return answers_equivalent(left, right)


def context_answer_key(problem, answer):
    if re.match(r'^\s*[a-zA-Z]\s*=', str(answer)) and context_equivalent(problem, answer, '0') is None:
        return 'unresolved_assignment:' + sha256_text(str(answer))
    value = scalar_fraction(answer)
    return 'rational:' + str(value) if value is not None else 'text:' + re.sub(r'\s+', '', str(answer))


def word_shingles(text, size=5):
    tokens = normalize_instance(text).casefold().split()
    return {tuple(tokens[i:i+size]) for i in range(max(1, len(tokens)-size+1))} if tokens else set()


def pair_kind(left, right):
    if normalize_instance(left) == normalize_instance(right): return 'same_instance'
    a, b = normalize_instance(left), normalize_instance(right)
    number = r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)'
    a_shape, b_shape = re.sub(number, '<N>', a), re.sub(number, '<N>', b)
    if a_shape == b_shape and re.findall(number, a) != re.findall(number, b): return 'template_relative'
    word_shape = lambda text: re.sub(r'(?i)(letters of|word)\s+[A-Za-z]+', r'\1 <WORD>', text)
    if word_shape(a) == word_shape(b) and a != b: return 'template_relative'
    return 'unresolved_candidate'


class Groups:
    def __init__(self, ids): self.parents = {item: item for item in ids}
    def find(self, item):
        while self.parents[item] != item:
            self.parents[item] = self.parents[self.parents[item]]
            item = self.parents[item]
        return item
    def join(self, left, right):
        a, b = sorted([self.find(left), self.find(right)])
        self.parents[b] = a


def apply_instance_decisions(frame, pairs, protected_pairs, decisions=None, protected_lookup=None):
    """Apply reviewed joins/exclusions, including decisions below a retrieval threshold."""
    frame = frame.copy()
    groups = Groups(frame['row_id'])
    lookup = frame.set_index('row_id')['problem'].to_dict()
    decision_lookup = {}
    for decision in decisions or []:
        if not decision.get('reviewer') or not decision.get('reviewed_utc'):
            raise ValueError('Instance decisions need actual reviewer identity/time')
        if decision.get('policy_version') != MATCHING_VERSION:
            raise ValueError('Instance decision policy version changed')
        key = tuple(sorted([decision['row_id_a'], decision['row_id_b']]))
        if key in decision_lookup: raise ValueError('Duplicate pair decision')
        decision_lookup[key] = decision
    records, unresolved, protected_ids, used = [], [], set(), set()
    for protected, pair in [(False, p) for p in pairs] + [(True, p) for p in protected_pairs]:
        left, right = pair['row_id_a'], pair['row_id_b']
        key = tuple(sorted([left, right]))
        decision = decision_lookup.get(key)
        kind = pair_kind(pair['problem_a'], pair['problem_b'])
        action = 'retain_template_relative' if kind == 'template_relative' else 'unresolved'
        basis = 'exact_instance_identity' if kind == 'same_instance' else 'automatic_shape_diagnostic'
        if kind == 'same_instance': action = 'remove_protected_match' if protected else 'join_instances'
        if decision:
            used.add(key)
            if decision.get('problem_a_sha256') != sha256_text(pair['problem_a']) or decision.get('problem_b_sha256') != sha256_text(pair['problem_b']):
                raise ValueError('Pair review refers to stale problem text')
            action, basis = decision['action'], 'reviewed_instance_decision'
            if action not in {'join_instances','remove_protected_match','retain_distinct','retain_template_relative','quarantine'}:
                raise ValueError('Unknown pair action')
            if protected and action == 'join_instances': raise ValueError('Protected match must exclude training groups')
            if not protected and action == 'remove_protected_match': raise ValueError('Removal needs a protected question identity')
        if action == 'join_instances': groups.join(left, right)
        elif action == 'remove_protected_match': protected_ids.add(left if left in lookup else right)
        elif action == 'quarantine': protected_ids.update(item for item in [left, right] if item in lookup)
        elif action == 'unresolved': unresolved.append(key)
        records.append(dict(pair, kind=kind, action=action, decision_basis=basis))
    # Reviews discovered outside LSH candidates are authoritative too.
    for key, decision in decision_lookup.items():
        if key in used: continue
        left, right = decision['row_id_a'], decision['row_id_b']
        combined = dict(lookup, **(protected_lookup or {}))
        if left not in combined or right not in combined: raise ValueError('Unretrieved decision needs checked candidate/protected IDs')
        is_protected = left not in lookup or right not in lookup
        extra = dict(row_id_a=left, row_id_b=right, problem_a=combined[left], problem_b=combined[right], similarity=None, retrieval='review_discovered')
        subframe, subreport, _ = apply_instance_decisions(frame, [] if is_protected else [extra], [extra] if is_protected else [], [decision])
        action = subreport.iloc[0]['action']
        if action == 'join_instances': groups.join(left, right)
        elif action in {'quarantine','remove_protected_match'}: protected_ids.update(item for item in [left,right] if item in lookup)
        records.extend(subreport.to_dict('records'))
    frame['problem_group_id'] = frame['row_id'].map(lambda value: 'instance_' + sha256_text(groups.find(value))[:24])
    excluded_groups = set(frame.loc[frame['row_id'].isin(protected_ids), 'problem_group_id'])
    frame['group_excluded'] = frame['problem_group_id'].isin(excluded_groups)
    frame['group_representative_id'] = frame.groupby('problem_group_id')['row_id'].transform('min')
    report = pd.DataFrame(records)
    return frame, report, {'unresolved_pairs': len(unresolved), 'excluded_groups': sorted(excluded_groups),
                           'policy_version': MATCHING_VERSION, 'approximate_recall': True}


def retrieve_pairs(frame, protected, threshold=.65, permutations=128, progress=lambda message: None):
    from datasketch import MinHash, MinHashLSH
    index = MinHashLSH(threshold=threshold, num_perm=permutations)
    pairs, protected_pairs = [], []
    def fingerprint(shingles):
        value = MinHash(num_perm=permutations, seed=1729)
        value.update_batch([' '.join(shingle).encode('utf-8') for shingle in sorted(shingles)])
        return value
    lookup = frame.set_index('row_id')['problem'].to_dict()
    for position,row in enumerate(frame.itertuples(index=False)):
        shingles = word_shingles(row.problem)
        value = fingerprint(shingles)
        for other in sorted(index.query(value)):
            a, b = shingles, word_shingles(lookup[other])
            score = len(a & b)/len(a | b) if a | b else 1
            pairs.append(dict(row_id_a=other, row_id_b=row.row_id, problem_a=lookup[other], problem_b=row.problem,
                              similarity=score, retrieval='minhash_lsh'))
        index.insert(row.row_id, value)
        if position % 2000 == 0: progress(f'Near retrieval: indexed {position:,}/{len(frame):,}; {len(pairs):,} candidate pairs')
    for row in protected.itertuples(index=False):
        shingles = word_shingles(row.problem)
        for other in sorted(index.query(fingerprint(shingles))):
            a, b = shingles, word_shingles(lookup[other])
            score = len(a & b)/len(a | b) if a | b else 1
            protected_pairs.append(dict(row_id_a=other, row_id_b='protected:' + row.set_id + ':' + row.problem_id,
                problem_a=lookup[other], problem_b=row.problem, similarity=score, retrieval='protected_minhash_lsh'))
    return pairs, protected_pairs
