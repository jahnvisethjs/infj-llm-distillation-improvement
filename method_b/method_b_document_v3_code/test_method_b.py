"""Acceptance checks; run with --require-solver on Sol before selecting real data."""
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from fractions import Fraction

from select_b import (bounds, cell, edge_report, parent_edges, quotas, rebin,
                      solve, tie_utility, utility, validate, write)


def example(identity, tokens, source='orca', length_bin=0):
    return {'row_id':identity,'problem_group_id':'group_'+identity,'dataset_id':source,
            'length_bin':length_bin,'supervised_tokens':tokens,'completion_tokens':tokens-1,
            'prompt_tokens':5,'total_tokens':5+tokens,'eos_tokens':1,
            'problem':identity,'completion':'solution','final_answer':'1'}


def fixture():
    parent=[example(f'p_{source}_{i}',t,source) for source in ['orca','numina']
            for i,t in enumerate([10,12])]
    bank=[example(f'b_{source}_{i}',t,source) for source in ['orca','numina']
          for i,t in enumerate([10,12,10,12,10,12])]
    return parent,bank


class Accounting(unittest.TestCase):
    def test_linear_quantiles(self):
        p=[example(str(i),10*(i+1)) for i in range(10)]
        self.assertEqual(parent_edges(p)['orca'],[Fraction(19+9*i) for i in range(9)])
    def test_right_closed_equal_edge(self):
        data=[example('x',10),example('y',11)]
        self.assertEqual([r['length_bin'] for r in rebin(data,{'orca':[Fraction(10)]})],[0,1])
    def test_duplicate_edges_removed(self):
        self.assertEqual(parent_edges([example(str(i),7) for i in range(10)])['orca'],[Fraction(7)])
    def test_bins_are_source_specific(self):
        p=[example('a',10,'orca'),example('b',100,'numina')]
        self.assertEqual(parent_edges(p),{'numina':[Fraction(100)],'orca':[Fraction(10)]})
    def test_report_has_outer_infinities_as_text(self):
        report=edge_report({'orca':[Fraction(3,2)]})
        self.assertIn('right-closed',report['endpoint_rule'])
        json.dumps(report,allow_nan=False)
    def test_fractional_bounds(self):
        self.assertEqual(bounds(Fraction(318845,2),'0.005'),(158626,160219))
    def test_integer_bounds(self):
        self.assertEqual(bounds(318845,'0.005'),(317251,320439))
    def test_stratum_bounds(self):
        self.assertEqual(bounds(22,'0.05'),(21,23))
    def test_exact_primary_counts(self):
        p,_=fixture();self.assertEqual(sum(quotas(p,1).values()),4)
    def test_exact_fourfold_counts(self):
        p,_=fixture();self.assertEqual(sum(quotas(p,4).values()),16)
    def test_half_extension_total_count(self):
        p,_=fixture();self.assertEqual(sum(quotas(p,0.5).values()),2)
    def test_hash_utility_formula(self):
        raw=int(hashlib.sha256(b'42|random|abc').hexdigest()[:8],16)
        self.assertEqual(utility('abc'),math.floor(Fraction(raw*1000000,2**32)+Fraction(1,2)))
    def test_tie_hash_formula(self):
        self.assertEqual(tie_utility('abc'),int(hashlib.sha256(b'42|tie|abc').hexdigest()[:8],16))
    def test_duplicate_group_rejected(self):
        a,b=example('a',10),example('b',10);b['problem_group_id']=a['problem_group_id']
        with self.assertRaises(ValueError):validate([a,b])
    def test_overlength_rejected(self):
        with self.assertRaises(ValueError):validate([example('a',2050)])
    def test_wrong_eos_rejected(self):
        a=example('a',10);a['eos_tokens']=2
        with self.assertRaises(ValueError):validate([a])
    def test_write_preserves_other_version(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';write(p,{'a':1});write(p,{'a':1})
            with self.assertRaises(FileExistsError):write(p,{'a':2})
            self.assertEqual(json.loads(p.read_text()),{'a':1})


HAS_SOLVER=bool(importlib.util.find_spec('ortools'))


@unittest.skipUnless(HAS_SOLVER,'OR-Tools not installed in this execution environment')
class Solver(unittest.TestCase):
    def test_global_and_stratum_controls(self):
        p,b=fixture();chosen,r=solve(p,b,1,deterministic_limit=2,tie_limit=1)
        self.assertTrue(r['passed']);self.assertEqual(len(chosen),4)
        self.assertEqual(sum(x['supervised_tokens'] for x in chosen),44)
        self.assertTrue(all(s['actual_rows']==s['required_rows'] for s in r['strata']))
    def test_objective_matches_brute_force(self):
        p,b=fixture();chosen,r=solve(p,b,1,deterministic_limit=2,tie_limit=1)
        feasible=[]
        for subset in itertools.combinations(b,4):
            if sum(x['supervised_tokens'] for x in subset)!=44:continue
            if any(sum(x['dataset_id']==s for x in subset)!=2 or
                   sum(x['supervised_tokens'] for x in subset if x['dataset_id']==s)!=22
                   for s in ('orca','numina')):continue
            feasible.append(sum(utility(x['row_id']) for x in subset))
        self.assertEqual(sum(utility(x['row_id']) for x in chosen),max(feasible))
    def test_order_independent(self):
        p,b=fixture();a,_=solve(p,b,1,deterministic_limit=2,tie_limit=1)
        random.Random(91).shuffle(b);c,_=solve(p,b,1,deterministic_limit=2,tie_limit=1)
        self.assertEqual([x['row_id'] for x in a],[x['row_id'] for x in c])
    def test_capacity_failure(self):
        p,b=fixture();chosen,r=solve(p,b,4,deterministic_limit=2,tie_limit=1)
        self.assertIsNone(chosen);self.assertTrue(r['infeasibility_proven'])
    def test_stratum_failure_does_not_relax(self):
        p,b=fixture();chosen,r=solve(p,b,0.5,deterministic_limit=2,tie_limit=1)
        self.assertIsNone(chosen)
        self.assertIn(r['status'],['INFEASIBLE','INFEASIBLE_STRATUM_TOKEN_BOUNDS'])
        self.assertEqual(r['global_tolerance'],'0.005');self.assertEqual(r['stratum_tolerance'],'0.05')
    def test_timeout_not_called_infeasible(self):
        p,b=fixture();chosen,r=solve(p,b,1,deterministic_limit=0,tie_limit=0)
        self.assertIsNone(chosen);self.assertFalse(r['infeasibility_proven'])
        self.assertEqual(r['status'],'UNKNOWN')
    def test_parent_group_overlap_rejected(self):
        p,b=fixture();b[0]['problem_group_id']=p[0]['problem_group_id']
        with self.assertRaises(ValueError):solve(p,b,1,deterministic_limit=1)


if __name__=='__main__':
    if '--require-solver' in sys.argv:
        sys.argv.remove('--require-solver')
        if not HAS_SOLVER:raise SystemExit('The real solver is required for Sol acceptance checks')
    unittest.main(verbosity=2)
