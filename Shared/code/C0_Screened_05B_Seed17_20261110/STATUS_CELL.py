from pathlib import Path
import subprocess

WORK = Path('/scratch/sshind38/capstone_c0_c1_work')
NAME = 'C0_Screened_05B_Seed17_20261110'
TRAIN_PYTHON = '/scratch/sshind38/capstone_approach_b/venv_b_training_py312/bin/python'
subprocess.run([TRAIN_PYTHON, '-B', str(WORK / NAME / 'screened_c0.py'),
                'status', '--work', str(WORK), '--attempt', 'pilot01'],
               cwd=WORK, check=True)
