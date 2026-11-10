"""Preserve invalid evaluation files and prevent overlapping recovery launches.

No rows are deduplicated or selected by correctness. Invalid/unvalidated outputs
are archived whole, and the unchanged evaluator regenerates that output.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid

STUDY = 'B_method_B_document_v3'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def inspect_predictions(path):
    path=Path(path)
    if not path.is_file():return {'exists':False,'path':str(path)}
    counts=Counter();variants=defaultdict(set);invalid=[];ids=[]
    differing_fields=Counter();by_id={}
    for line_no,line in enumerate(path.read_text().splitlines(),1):
        if not line.strip():continue
        try:
            record=json.loads(line)
            identity=record['problem_id']
            hash(identity)
        except (ValueError,KeyError,TypeError) as error:
            invalid.append({'line':line_no,'error':type(error).__name__});continue
        counts[identity]+=1;ids.append(identity)
        variants[identity].add(json.dumps(record,sort_keys=True,ensure_ascii=False))
        if identity in by_id:
            old=by_id[identity]
            for key in set(old)|set(record):
                if old.get(key)!=record.get(key):differing_fields[key]+=1
        else:by_id[identity]=record
    halfway=len(ids)//2
    return {'exists':True,'path':str(path),'sha256':sha(path),'valid_json_records':len(ids),
            'unique_question_ids':len(counts),'duplicated_question_ids':sum(n>1 for n in counts.values()),
            'conflicting_question_ids':sum(len(v)>1 for v in variants.values()),
            'different_record_fields':dict(differing_fields),
            'invalid_lines':invalid,'matching_two_halves':bool(ids) and len(ids)%2==0 and ids[:halfway]==ids[halfway:]}


def evaluation_processes(root,proc_root=Path('/proc')):
    """Read only our own project workers; no process is killed."""
    root=str(Path(root).resolve());found=[]
    for path in Path(proc_root).glob('[0-9]*'):
        try:
            pid=int(path.name)
            if pid==os.getpid() or path.stat().st_uid!=os.getuid():continue
            argv=[x.decode('utf-8',errors='replace') for x in (path/'cmdline').read_bytes().split(b'\0') if x]
            if not any(root in arg for arg in argv):continue
            names={Path(arg).name for arg in argv}
            is_evaluator='evaluate.py' in names
            is_runner=bool(names & {'run_method_b.py','run_method_b_eval_recovery.py'})
            is_evaluation_stage='evaluate' in argv or ('--action' in argv and
                    argv.index('--action')+1<len(argv) and argv[argv.index('--action')+1]=='evaluate')
            if is_evaluator or (is_runner and is_evaluation_stage):
                found.append({'pid':pid,'program':'evaluate.py' if is_evaluator else 'Method B evaluation runner'})
        except (OSError,ValueError):continue
    return found


@contextmanager
def single_evaluation(root):
    folder=Path(root)/'outputs'/STUDY
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'evaluation_recovery.lock'
    with path.open('a+') as handle:
        try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another recovery evaluation is running. Leave it to finish; no files were changed.')
        active=evaluation_processes(root)
        if active:
            raise RuntimeError('Existing evaluation workers are still active: '+json.dumps(active)+
                '. No files were changed. Wait for those workers to exit before starting recovery.')
        try:yield
        finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def archive_prediction_bundle(root,prediction,reason='Existing output lacks a matching verified completion record'):
    root=Path(root).resolve();base=root/'outputs'/STUDY
    prediction=Path(prediction)
    if prediction.is_symlink():raise ValueError('Refusing to move a symlink prediction: '+str(prediction))
    relative=prediction.resolve().relative_to(base.resolve())
    if not ('parent_evaluation_cache' in relative.parts or 'evaluation' in relative.parts):
        raise ValueError('Recovery is restricted to evaluation outputs: '+str(prediction))
    if prediction.suffix!='.jsonl':raise ValueError('Expected a prediction JSONL file')
    paths=[prediction,prediction.with_suffix('.config.json'),prediction.with_suffix('.validated.json')]
    existing=[p for p in paths if p.exists()]
    if not existing:return None
    audit=inspect_predictions(prediction)
    folder=base/'evaluation_recovery_archives'/(
           datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8])/relative.parent
    folder.mkdir(parents=True,exist_ok=False)
    moved=[]
    for path in existing:
        if path.is_symlink():raise ValueError('Refusing symlink metadata: '+str(path))
        identity=sha(path)
        target=folder/path.name
        path.rename(target)
        if sha(target)!=identity:raise RuntimeError('Archived file checksum differs: '+str(target))
        moved.append({'original':str(path),'archived':str(target),'sha256':identity})
    report={'reason':reason,'inspection':audit,'moved':moved,
            'records_cherry_picked':False,'training_changed':False,
            'next_action':'Regenerate this output using the unchanged pinned evaluator'}
    (folder/'recovery_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Archived previous evaluation output:',folder,flush=True)
    print('Regenerating this prediction file. No training is repeated.',flush=True)
    return report


def enter(main):
    import sys
    args=sys.argv[1:]
    if not args:raise ValueError('Specify evaluate, summary, or export')
    if '--root' not in args:raise ValueError('Expected --root PROJECT')
    root=Path(args[args.index('--root')+1])
    if args[0]=='evaluate':
        with single_evaluation(root):main()
    elif args[0]=='engine' and '--action' in args and args[args.index('--action')+1]=='evaluate':
        main()
    elif args[0] in ('summary','export'):main()
    else:raise ValueError('This recovery runner only permits evaluation, summary, and export')
