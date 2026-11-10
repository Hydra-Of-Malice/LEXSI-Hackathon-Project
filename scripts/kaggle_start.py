"""Start (or resume) the NF4 run in a Kaggle notebook. `remote_nf4.py bundle` puts this file at the root of the private
bundle repo, so the notebook cell only has to fetch and run it (see docs/REMOTE_NF4.md):

    import os,sys,subprocess;from kaggle_secrets import UserSecretsClient as U;os.environ['HF_TOKEN']=U().get_secret('HF_TOKEN');subprocess.run([sys.executable,'-m','pip','install','-q','huggingface_hub==1.33.0'],check=True);from huggingface_hub import hf_hub_download as d;exec(open(d('HydraOfMalice/lexsi-nf4-private','kaggle_start.py',repo_type='dataset')).read())

It needs a T4 accelerator, internet on, and a Kaggle secret HF_TOKEN that can read and write the private repo.
"""
import os
import subprocess
import sys
import tarfile

from huggingface_hub import hf_hub_download

REPO = 'HydraOfMalice/lexsi-nf4-private'
WORK = '/tmp/lexsi'
if not os.environ.get('HF_TOKEN'):
    raise SystemExit('Add a Kaggle secret named HF_TOKEN first (Add-ons > Secrets)')
# Kaggle stops a session at 12 hours; stop cleanly at 11.5, sync, and let the next session resume.
os.environ.update(BUNDLE_REPO=REPO, PLATFORM='kaggle-t4', WORK=WORK, DEADLINE_MINUTES='690', CUDA_VISIBLE_DEVICES='0',
                  PATH=os.environ.get('PATH', '') + ':/usr/local/cuda/bin')  # nvcc, for the llama.cpp build
os.makedirs(WORK, exist_ok=True)
with tarfile.open(hf_hub_download(REPO, 'code.tar', repo_type='dataset', local_dir=WORK)) as archive:
    archive.extractall(os.path.join(WORK, 'code'), filter='data')
sys.path.insert(0, os.path.join(WORK, 'code', 'scripts'))
import remote_nf4  # noqa: E402

result = subprocess.run(['bash', '-c', remote_nf4.BOOTSTRAP])
print('NF4 run exit code', result.returncode)
