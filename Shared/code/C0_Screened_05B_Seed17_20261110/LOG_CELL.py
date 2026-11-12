from pathlib import Path
import json
import subprocess

WORK = Path('/scratch/sshind38/capstone_c0_c1_work')
location = json.loads((WORK / 'C0_Screened_05B_Seed17_20261110' / 'RUN_LOCATION.json').read_text())
submissions = sorted((Path(location['output_directory']) / 'submissions' / 'pilot01').glob('*.json'))
if not submissions:
    raise FileNotFoundError('No submitted job receipt exists yet')
job = json.loads(submissions[-1].read_text())
print('Job:', job['job_id'])
subprocess.run(['squeue', '-j', job['job_id']], check=False)
for label in ['stdout', 'stderr']:
    path = Path(job[label])
    print(f'\n{label}: {path}')
    if path.is_file():
        print('\n'.join(path.read_text(errors='replace').splitlines()[-35:]))
