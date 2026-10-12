import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from capstone_person1.snapshot_stream import SnapshotCanonicalizer, canonicalize_snapshot
from capstone_person1.snapshot_catalog import open_catalog, prepare_candidates

ROOT = Path(__file__).resolve().parents[2]


class SnapshotCatalogTests(unittest.TestCase):
    def test_exact_equivalence_conflicts_and_catalog_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mapper = SnapshotCanonicalizer(json.loads((ROOT / 'Person1_Full_Run_v2.ipynb').read_text(encoding='utf-8')))
            stages = {}
            problems = ['Shared equivalent instance?', 'Shared conflict instance?', 'Unique instance?']
            for source in ['orca','numina']:
                answers = ['3','1','4'] if source == 'orca' else ['9/3','2','5']
                rows = []
                for i, (problem, answer) in enumerate(zip(problems, answers)):
                    if i == 2: problem = source + ' ' + problem
                    numeric = '3' if answer == '9/3' else answer
                    rows.append(dict(original_row_id=str(i), dataset_revision=mapper.config[source.upper() + '_REVISION'],
                                     problem_raw=problem, solution_raw=f'Use the given relation and simplify the expression to determine the requested value.\n\nThe value is {numeric}.',
                                     final_answer=answer, answer_parse_status='parsed', original_source='other',
                                     question_type='math-word-problem', problem_type='', synthetic=False, quality_flags=[]))
                path = root / f'{source}.parquet'
                pq.write_table(pa.Table.from_pylist(rows), path)
                stages[source] = root / source
                canonicalize_snapshot(source, path, stages[source], mapper, 'fixture', batch_size=2)
            reports = root / 'reports'
            reports.mkdir()
            conn = open_catalog(stages, root / 'catalog.sqlite', 'fixture')
            ns = {'REPORT_DIR': reports, 'CANDIDATE_UNIVERSE_PER_SOURCE': 20, 'SELECTION_SEED': 42}
            prepare_candidates(conn, stages, ns)
            self.assertEqual(len(ns['conflicting_row_ids']), 2)
            self.assertEqual(len(ns['exact_alternative_ids']), 1)
            self.assertEqual(len(ns['exact_unique']), 3)
            self.assertEqual(len(ns['candidate_df']), 2)
            self.assertIn('problem_raw', ns['candidate_df'].columns)
            self.assertIn('solution_raw', ns['candidate_df'].columns)
            self.assertTrue(ns['candidate_df']['solution_raw'].str.len().gt(0).all())
            self.assertEqual(ns['candidate_df']['dataset_id'].value_counts().to_dict(), {'orca':1,'numina':1})
            conn.close()
            conn = open_catalog(stages, root / 'catalog.sqlite', 'fixture')
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM records').fetchone()[0], 6)
            conn.close()


if __name__ == '__main__': unittest.main()
