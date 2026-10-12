import json
import random
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from capstone_person1.pipeline_controls import file_sha256
from capstone_person1.snapshot_stream import SnapshotCanonicalizer, canonicalize_snapshot, stage_batches

ROOT = Path(__file__).resolve().parents[2]


class SnapshotStreamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mapper = SnapshotCanonicalizer(json.loads((ROOT / 'Person1_Full_Run_v2.ipynb').read_text(encoding='utf-8')))
        self.rows = [dict(original_row_id=str(i), dataset_revision=self.mapper.config['ORCA_REVISION'],
                          problem_raw='Find the number.', solution_raw=f'Add 6 to both sides to solve the equation.\n\nThe number is {i+32}.',
                          final_answer='2', answer_parse_status='answer_phrase') for i in range(12)]
        self.input = self.root / 'input.parquet'
        pq.write_table(pa.Table.from_pylist(self.rows), self.input)

    def test_resume_does_not_reextract_committed_batches(self):
        calls = []
        def failing(source, row, position):
            if position == 8: raise RuntimeError('simulated interruption')
            return self.mapper(source, row, position)
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            canonicalize_snapshot('orca', self.input, self.root / 'stage', failing, 'fixed', batch_size=4)
        def resumed(source, row, position):
            calls.append(position)
            return self.mapper(source, row, position)
        state = canonicalize_snapshot('orca', self.input, self.root / 'stage', resumed, 'fixed', batch_size=4)
        self.assertEqual(calls, [8,9,10,11])
        self.assertEqual(state['reused_rows'], 8)
        self.assertTrue(state['complete'])
        actual = [row for batch in stage_batches(self.root / 'stage') for row in batch.to_pylist()]
        self.assertEqual([r['final_answer'] for r in actual], [str(i+32) for i in range(12)])

    def test_changed_controls_cannot_resume(self):
        canonicalize_snapshot('orca', self.input, self.root / 'stage', self.mapper, 'old', batch_size=4)
        with self.assertRaisesRegex(ValueError, 'changed'):
            canonicalize_snapshot('orca', self.input, self.root / 'stage', self.mapper, 'new', batch_size=4)

    def test_modified_committed_shard_cannot_resume(self):
        canonicalize_snapshot('orca', self.input, self.root / 'stage', self.mapper, 'fixed', batch_size=4)
        (self.root / 'stage/batch_000000000.parquet').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'shard changed'):
            canonicalize_snapshot('orca', self.input, self.root / 'stage', self.mapper, 'fixed', batch_size=4)

    def test_terminal_partial_batch_and_full_reuse(self):
        state = canonicalize_snapshot('orca', self.input, self.root / 'stage', self.mapper, 'fixed', limit=10, batch_size=4)
        self.assertEqual([s['rows'] for s in state['shards']], [4,4,2])
        def must_not_run(*args): raise AssertionError('completed source was reprocessed')
        reused = canonicalize_snapshot('orca', self.input, self.root / 'stage', must_not_run, 'fixed', limit=10, batch_size=4)
        self.assertEqual(reused['reused_rows'], 10)

    def test_changed_input_cannot_resume_even_with_same_caller_fingerprint(self):
        canonicalize_snapshot('orca', self.input, self.root / 'stage', self.mapper, 'fixed', batch_size=4)
        self.rows[0]['solution_raw'] = 'A different source solution.'
        pq.write_table(pa.Table.from_pylist(self.rows), self.input)
        with self.assertRaisesRegex(ValueError, 'changed'):
            canonicalize_snapshot('orca', self.input, self.root / 'stage', self.mapper, 'fixed', batch_size=4)

    def test_random_sampling_preserves_original_ids_and_resumes(self):
        expected = sorted(random.Random(1729).sample(range(12), 7))
        state = canonicalize_snapshot('orca', self.input, self.root / 'stage', self.mapper, 'fixed',
                                      limit=7, batch_size=4, sampling='random', seed=1729)
        actual = [row for batch in stage_batches(self.root / 'stage') for row in batch.to_pylist()]
        self.assertEqual([int(row['original_row_id']) for row in actual], expected)
        self.assertEqual([shard['rows'] for shard in state['shards']], [4, 3])
        reused = canonicalize_snapshot('orca', self.input, self.root / 'stage', self.mapper, 'fixed',
                                       limit=7, batch_size=4, sampling='random', seed=1729)
        self.assertEqual(reused['reused_rows'], 7)
        with self.assertRaisesRegex(ValueError, 'changed'):
            canonicalize_snapshot('orca', self.input, self.root / 'stage', self.mapper, 'fixed',
                                  limit=7, batch_size=4, sampling='random', seed=42)

    def test_random_sampling_interruption_reuses_committed_samples(self):
        expected = sorted(random.Random(42).sample(range(12), 9))
        def failing(source, row, position):
            if position == 4: raise RuntimeError('simulated interruption')
            return self.mapper(source, row, position)
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            canonicalize_snapshot('orca', self.input, self.root / 'stage', failing, 'fixed',
                                  limit=9, batch_size=4, sampling='random', seed=42)
        calls = []
        def resumed(source, row, position):
            calls.append(position)
            return self.mapper(source, row, position)
        state = canonicalize_snapshot('orca', self.input, self.root / 'stage', resumed, 'fixed',
                                      limit=9, batch_size=4, sampling='random', seed=42)
        self.assertEqual(state['reused_rows'], 4)
        self.assertEqual(calls, [4,5,6,7,8])
        actual = [int(row['original_row_id']) for batch in stage_batches(self.root / 'stage') for row in batch.to_pylist()]
        self.assertEqual(actual, expected)


if __name__ == '__main__': unittest.main()
