from pathlib import Path
import subprocess
import zipfile

WORK = Path('/scratch/sshind38/capstone_c0_c1_work')
NAME = 'C0_Screened_05B_Seed17_20261110'
TRAIN_PYTHON = Path('/scratch/sshind38/capstone_approach_b/venv_b_training_py312/bin/python')
ATTEMPT = 'pilot01'
archive = WORK / f'{NAME}.zip'
if not archive.is_file():
    raise FileNotFoundError(f'Upload {archive.name} into {WORK} first')
if not TRAIN_PYTHON.is_file():
    raise FileNotFoundError(f'Existing training Python is missing: {TRAIN_PYTHON}')
with zipfile.ZipFile(archive) as package:
    bad_member = package.testzip()
    if bad_member:
        raise ValueError(f'Archive is damaged: {bad_member}')
    for member in package.infolist():
        target = (WORK / member.filename).resolve()
        if (not target.is_relative_to(WORK.resolve())
                or Path(member.filename).parts[0] != NAME):
            raise ValueError(f'Unexpected archive path: {member.filename}')
        if member.is_dir():
            continue
        content = package.read(member)
        if target.exists():
            if target.read_bytes() != content:
                raise FileExistsError(f'Different existing file preserved: {target}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(content)
runner = WORK / NAME / 'screened_c0.py'
for action in ['prepare', 'submit']:
    subprocess.run([str(TRAIN_PYTHON), '-B', str(runner), action,
                    '--work', str(WORK), '--attempt', ATTEMPT],
                   cwd=WORK, check=True)
