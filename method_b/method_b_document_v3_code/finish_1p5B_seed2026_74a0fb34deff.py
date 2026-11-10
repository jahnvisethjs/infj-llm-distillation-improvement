"""Finish only mixed 1x, Qwen 1.5B, continuation seed 2026, parents 17 and 123."""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import time
import traceback

EXPECTED_BACKEND_SHA = 'd8101467d2037095bffdfb851959c2a2e2405daeb2de377b703ba2bac5d335a4'
SCOPE = 'Only mixed 1x / Qwen 1.5B / B training seed 2026 / parent seeds 17 and 123'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_backend(root):
    path = Path(root)/'Method_B_Overnight.py'
    if sha(path) != EXPECTED_BACKEND_SHA:
        raise ValueError('The existing Method_B_Overnight.py differs from the reviewed version; it was preserved.')
    spec = importlib.util.spec_from_file_location('finish_2026_backend', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def target_tasks(backend):
    tasks = [t for t in backend.make_tasks() if t['setting'] == 'mixed' and t['budget'] == 1
             and (t['stage'] == 'prepare' or
                  (t['model'] == '1.5B' and t['training_seed'] == 2026 and t['parent_seed'] in (17, 123)))]
    ids = {t['id'] for t in tasks}
    if len(tasks) != 7 or any(not set(t['dependencies']) <= ids for t in tasks):
        raise ValueError('Unexpected task graph; no experiment started.')
    if Counter(t['stage'] for t in tasks) != {'prepare': 1, 'smoke': 2, 'train': 2, 'evaluate': 2}:
        raise ValueError('Only two target training/evaluation cases are permitted.')
    return tasks


def controller_class(n):
    class FinishTwo(n.Controller):
        def __init__(self, root, folder, hours):
            super().__init__(root, folder, hours)
            self.tasks = target_tasks(n)
            self.area = root/'outputs'/n.STUDY/'finish_1p5B_seed2026'
            self.meta.update(scope=SCOPE, expected_training_cases=2, expected_evaluation_cases=2,
                external_requirements=['Shared evaluator/protection clearance', 'Compatible A/C comparisons',
                    'Transfer and other budgets are outside this two-run notebook'])

        def save(self, status='running'):
            self.meta['updated_utc'] = datetime.now(timezone.utc).isoformat()
            value = {'status': status, 'metadata': self.meta, 'gpu': self.gpu,
                'running_task': self.running, 'tasks': self.tasks,
                'capacity_checks': list(self.capacities.values())}
            n.atomic_json(self.folder/'status.json', value)
            n.atomic_json(self.area/'latest_status.json', value)
            self.report(status)

        def report(self, status):
            counts = Counter(t['status'] for t in self.tasks)
            train = sum(t['stage'] == 'train' and t['status'] == 'completed' for t in self.tasks)
            evaluation = sum(t['stage'] == 'evaluate' and t['status'] == 'completed' for t in self.tasks)
            lines = ['METHOD B - TWO REMAINING MAIN RUNS', SCOPE, 'Status: ' + status,
                f'Completed target training cases: {train}/2',
                f'Completed target evaluation cases: {evaluation}/2', 'Task counts: ' + json.dumps(counts),
                'Results remain preliminary development evidence.', '']
            for t in self.tasks:
                lines.append(t['id'] + ' | ' + t['status'] + ' | ' + t.get('reason', ''))
            if self.meta.get('fatal_error'):
                lines += ['', 'ERROR: ' + self.meta['fatal_error']]
            lines += ['', 'Logs and target results: ' + str(self.folder),
                'All saved development results: ' + str(self.root/'outputs'/n.STUDY/'development_results.csv')]
            n.atomic_text(self.folder/'FINISH_REPORT.txt', '\n'.join(lines) + '\n')
            n.atomic_text(self.area/'FINISH_REPORT.txt', '\n'.join(lines) + '\n')

        def preflight(self):
            self.code = n.bootstrap(self.root)
            reference = n.load_module('finish_reference', self.code/'run_method_b.py')
            reference.source(self.root)  # Existing pinned code and parent-reference checks.
            work = self.root/'outputs'/n.STUDY/'mixed/budget_1x'
            shared = self.root/'outputs'/n.STUDY/'shared'
            bank_audit = n.read(shared/'bank_audit.json')
            for name, key in [('mixed_candidate_bank.jsonl', 'mixed_bank_sha256'),
                              ('clean_development_reference.jsonl', 'clean_development_sha256'),
                              ('parent_length_bins.json', 'length_bins_sha256')]:
                if n.sha(shared/name) != bank_audit[key]:
                    raise ValueError('Existing primary-study input changed: ' + name)
            cfg = n.read(self.root/'b_workbench_500/config.json')
            if n.sha(self.root/'data/train_n1000.jsonl') != cfg['parent_manifest_sha256']:
                raise ValueError('Parent manifest changed')
            prepare = next(t for t in self.tasks if t['stage'] == 'prepare')
            if not n.completed(self.root, prepare, {}):
                raise ValueError('Existing 1x selection/label verification is missing or stale; no data were rebuilt.')
            registry = n.read(work/'runtime/parent_registry.json')
            original_registry = n.read(self.root/'b_workbench_500/parent_registry.json')
            if registry != original_registry:
                raise ValueError('Runtime parent registry differs from the pinned original')
            for parent in (17, 123):
                entry = registry[f'1.5B_seed{parent}']
                if n.sha(self.root/entry['adapter_relative_path']/'adapter_model.safetensors') != entry['expected_adapter_sha256']:
                    raise ValueError('Original parent adapter changed: ' + str(parent))
            self.gpu = n.gpu_probe(self.root/'venv_b_training_py312/bin/python')
            active = n.legacy_workers(self.root)
            if active:
                raise RuntimeError('Existing project training/evaluation workers are active: ' + json.dumps(active))
            self.capacities = {('mixed', 1): n.capacity(self.root, 'mixed', 1)}
            if self.capacities[('mixed', 1)]['infeasibility_proven']:
                raise ValueError('The existing 1x bank no longer satisfies necessary matching bounds')
            if self.work_deadline - time.time() < 180:
                raise RuntimeError('Too little allocation time remains; use a fresh GPU session.')
            self.save('ready')
            print('STARTUP_PASSED. GPU:', self.gpu['gpu'], flush=True)
            print('Only two main cases are scheduled: 1.5B / seed 2026 / parents 17 and 123.', flush=True)
            print('Existing verified smoke checks, training and predictions will be reused.', flush=True)

        def finish(self):
            self.running = None
            self.save('saving_results')
            if time.time() + 20 < self.hard_deadline:
                result = n.supervise([self.root/'venv_b_training_py312/bin/python', '-u',
                    self.code/'run_method_b_eval_unattended.py', 'summary', '--root', self.root],
                    self.folder/'logs/final_summary.log', self.hard_deadline - 5, stall_seconds=240)
            else:
                result = {'return_code': None, 'stop_reason': 'Insufficient time for final summary'}
            self.meta['summary_result'] = result
            target_results = []
            for task in self.tasks:
                if task['stage'] == 'evaluate' and task['status'] == 'completed':
                    if not n.completed(self.root, task, self.gpu):
                        task.update(status='failed', reason='Final canonical evidence recheck failed')
                        continue
                    run = n.run_directory(self.root, task)
                    saved = n.read(self.root/'outputs'/n.STUDY/'mixed/budget_1x/evaluation'/run.name/'B_vs_parent.json')
                    if len(saved['comparisons']) != 2:
                        task.update(status='failed', reason='Expected two benchmark comparisons')
                        continue
                    target_results.extend(saved['comparisons'])
            n.atomic_json(self.folder/'TARGET_RESULTS.json', target_results)
            if target_results:
                with (self.folder/'TARGET_RESULTS.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(target_results[0]))
                    writer.writeheader()
                    writer.writerows(target_results)
            passed = all(t['status'] == 'completed' for t in self.tasks) and len(target_results) == 4
            summary_ok = result['return_code'] == 0
            receipt = {'passed': passed and summary_ok, 'target_training_and_evaluation_passed': passed,
                'summary_updated': summary_ok, 'scope': SCOPE, 'target_comparisons': len(target_results),
                'new_transfer_or_extra_budget_runs': False, 'final_test_used': False,
                'results_remain_preliminary': True, 'session': str(self.folder)}
            n.atomic_json(self.folder/'COMPLETION.json', receipt)
            self.save('completed' if receipt['passed'] else 'finished_with_remaining_work')
            print('TWO_RUN_FINISH_REPORT:', json.dumps(receipt), flush=True)
            print('TARGET_RESULTS:', self.folder/'TARGET_RESULTS.csv', flush=True)
            print('ALL_DEVELOPMENT_RESULTS:', self.root/'outputs'/n.STUDY/'development_results.csv', flush=True)

    return FinishTwo


def run(root, folder, hours):
    root, folder = Path(root).resolve(), Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    n = load_backend(root)
    n.STOP_REQUESTED = False
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, n.signal_stop)
    controller = controller_class(n)(root, folder, hours)
    try:
        code = n.bootstrap(root)
        recovery = n.load_module('finish_recovery', code/'evaluation_recovery.py')
        lock = root/'outputs'/n.STUDY/'overnight/worker.lock'
        lock.parent.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            handle = stack.enter_context(lock.open('a+'))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another overnight/finish worker is already running; no second worker started.')
            stack.enter_context(recovery.single_evaluation(root))
            controller.preflight()
            n.atomic_json(folder/'startup.json', {'ready': True, 'pid': controller.meta['worker_pid'],
                'gpu': controller.gpu, 'scope': SCOPE})
            controller.queue()
            controller.finish()
        return 0 if n.read(folder/'COMPLETION.json')['passed'] else 2
    except BaseException as exc:
        controller.meta.update(fatal_error=str(exc), traceback=traceback.format_exc())
        controller.save('stopped_for_required_repair')
        n.atomic_json(folder/'startup.json', {'ready': False, 'error': str(exc)})
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-folder', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=2.75)
    args = parser.parse_args()
    if not 0 < args.hours <= 3:
        raise ValueError('This targeted runner accepts at most three worker hours.')
    raise SystemExit(run(args.root, args.run_folder, args.hours))
