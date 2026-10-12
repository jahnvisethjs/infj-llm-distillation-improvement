"""Deterministic three-bank allocation using one set of source-specific token-bin edges."""
import hashlib
import json

import numpy as np
import pandas as pd


def length_bin_edges(frame, bins=10):
    return {source: sorted(set(float(value) for value in np.quantile(rows['supervised_tokens'], np.linspace(0, 1, bins+1))))
            for source, rows in frame.groupby('dataset_id')}


def assign_bins(frame, edges):
    frame = frame.copy()
    frame['length_bin'] = [int(np.searchsorted(edges[row.dataset_id][1:-1], row.supervised_tokens, side='right'))
                           for row in frame.itertuples(index=False)]
    return frame


def stable_order(frame, seed):
    frame = frame.copy()
    frame['_key'] = frame['row_id'].map(lambda value: hashlib.sha256(f'{seed}|{value}'.encode()).hexdigest())
    buckets = [list(rows.sort_values('_key').index) for _, rows in frame.groupby(['dataset_id','length_bin'], sort=True)]
    order = [bucket[index] for index in range(max(map(len, buckets), default=0)) for bucket in buckets if index < len(bucket)]
    return frame.loc[order].drop(columns='_key').reset_index(drop=True)


def allocate_banks(frame, *, train_per_source=20000, selection_per_source=5000, loss_per_source=500, seed=1729, order_seed=42):
    if frame['row_id'].duplicated().any() or frame['problem_group_id'].duplicated().any():
        raise ValueError('Allocation requires unique rows and final problem groups')
    if set(frame['dataset_id']) != {'orca', 'numina'}: raise ValueError('Both configured sources are required')
    quotas = {'train_bank': train_per_source, 'selection_bank': selection_per_source, 'loss_dev': loss_per_source}
    if any(type(value) is not int or value < 1 for value in quotas.values()): raise ValueError('Positive integer quotas required')
    required = sum(quotas.values())
    counts = frame['dataset_id'].value_counts().to_dict()
    if any(counts.get(source, 0) < required for source in ['orca','numina']):
        raise ValueError(f'Insufficient checked unique groups: need {required}/source, available {counts}')
    edges = length_bin_edges(frame)
    binned = assign_bins(frame, edges)
    banks = {name: [] for name in quotas}
    # Round-robin assignments consume the same source/bin order and retain length coverage.
    for source, rows in binned.groupby('dataset_id', sort=True):
        ordered = stable_order(rows, seed)
        remaining = dict(quotas)
        chosen = {name: [] for name in quotas}
        cursor = 0
        while any(remaining.values()):
            # Weighted interleaving distributes each bank across the same population.
            for name, quota in quotas.items():
                take = min(max(1, quota // min(quotas.values())), remaining[name])
                chosen[name].extend(range(cursor, cursor + take))
                cursor += take
                remaining[name] -= take
        for name in quotas: banks[name].append(ordered.iloc[chosen[name]])
    banks = {name: stable_order(pd.concat(parts, ignore_index=True), order_seed) for name, parts in banks.items()}
    group_sets = {name: set(rows['problem_group_id']) for name, rows in banks.items()}
    for left in banks:
        for right in banks:
            if left < right and group_sets[left] & group_sets[right]: raise AssertionError('Bank groups intersect')
    return banks, {'reference_population': 'all checked encodable unique candidates before allocation',
                   'source_edges': edges, 'endpoint_rule': 'searchsorted(inner_edges, side=right)',
                   'split_seed': seed, 'ordering_seed': order_seed, 'quotas_per_source': quotas}
