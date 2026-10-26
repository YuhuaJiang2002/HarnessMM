#!/usr/bin/env python3
"""Run unchanged upstream metrics on three pilot outputs, on a named server."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--hostname', required=True)
    p.add_argument('--clip-weight', type=Path, required=True)
    p.add_argument('--physical-gpu', type=int, required=True)
    a = p.parse_args()
    if platform.system() != 'Linux' or socket.gethostname() != a.hostname:
        raise RuntimeError('Remote-only pilot evaluation')
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = dict(status='PREPARING_RUNTIME', hostname=socket.gethostname(), argv=sys.argv,
                 scope='integration_pilot', case_count=1, candidate_count=3,
                 full_benchmark_result=False, seed=0, started_at=time.time())
    shutil.copy2(__file__, out / 'runner.py')
    shutil.copy2(a.stage / 'source-provenance.json', out / 'source-provenance.json')
    shutil.copy2(a.stage / 'generation-execution.json', out / 'generation-execution.json')

    def save():
        temp = out / 'execution.json.tmp'
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(out / 'execution.json')

    def run(argv, name, timeout=900, env=None, cwd=None):
        state['active_command'] = argv
        save()
        with (out / (name + '.log')).open('w') as log:
            proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout, env=env, cwd=cwd)
        if proc.returncode:
            raise RuntimeError(f'{name}: exit {proc.returncode}; see log')

    save()
    lease = None
    try:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', PIP_DISABLE_PIP_VERSION_CHECK='1',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
        venv = out / 'venv'
        run(['python3', '-m', 'venv', str(venv)], 'create-venv', env=env)
        python = str(venv / 'bin/python')
        run([python, '-m', 'pip', 'install', '--timeout', '25', '--retries', '2',
             'pip==25.1.1', 'setuptools==75.6.0', 'wheel', 'numpy==1.26.4', 'Cython==3.0.12'],
            'bootstrap', env=env)
        run([python, '-m', 'pip', 'install', '--timeout', '25', '--retries', '2',
             'torch==2.4.0', 'torchvision==0.19.0', 'transformers==4.51.3',
             'torchmetrics==1.7.1', 'av==13.1.0', 'decord==0.6.0',
             'omegaconf==2.3.0', 'opencv-python-headless==4.11.0.86',
             'json_repair==0.44.1', 'ipython==8.12.3', 'PyYAML==6.0.2',
             'tqdm==4.65.2', 'scipy==1.15.3', 'scikit-image==0.25.2', 'accelerate==1.7.0',
             'ftfy==6.3.1', 'regex', 'sentencepiece', 'einops', 'Pillow'], 'dependencies', env=env)
        run([python, '-m', 'pip', 'install', '--no-build-isolation', 'fastdtw==0.3.4',
             str(a.stage / 'clip-source')], 'clip-and-fastdtw', env=env)
        run([python, '-m', 'pip', 'freeze'], 'packages', env=env)
        source = a.stage / 'upstream'
        generation = json.loads((a.stage / 'generation-execution.json').read_text())
        manifest = json.loads((a.stage / 'source-provenance.json').read_text())
        for rel, expected in manifest['upstream_files'].items():
            if hashlib.sha256((source / rel).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Upstream source changed: ' + rel)
        for row in generation['completed']:
            i = row['repeat']
            video = a.stage / f'candidate-{i}.mp4'
            if hashlib.sha256(video.read_bytes()).hexdigest() != row['video_sha256']:
                raise RuntimeError('Input video hash mismatch')
            target = out / 'minimax_dataset/367/649559' / str(i) / 'video'
            target.mkdir(parents=True)
            run(['ffmpeg', '-v', 'error', '-threads', '4', '-i', str(video),
                 '-vf', 'fps=30,scale=640:480:flags=bicubic', '-q:v', '2',
                 str(target / 'frame_%05d.jpg')], f'normalize-{i}', 120, env)
        if hashlib.sha256(a.clip_weight.read_bytes()).hexdigest() != '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af':
            raise RuntimeError('Official CLIP weight hash mismatch')
        config = dict(model_name='minimax', data=dict(val_base=str(out / 'minimax_dataset'),
                      gt_path=str(out / 'UNUSED_GT')), save_path=str(out / 'metrics'),
                      ckpt=dict(diversity=str(a.clip_weight)))
        (out / 'config.json').write_text(json.dumps(config, indent=2))
        # Hold a cooperative physical-GPU lease and recheck other processes.
        locks = Path('/tmp/phiagent-gpu-leases')
        locks.mkdir(exist_ok=True)
        lease = (locks / f'physical-gpu-{a.physical_gpu}.lock').open('a+')
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inv = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], text=True)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'], text=True)
        (out / 'gpu-inventory.txt').write_text(inv)
        (out / 'gpu-processes.txt').write_text(apps)
        fields = next(line.split(',') for line in inv.splitlines() if int(line.split(',')[0]) == a.physical_gpu)
        uuid = fields[1].strip()
        if uuid in apps or int(fields[3].strip().split()[0]) > 256:
            raise RuntimeError('Selected GPU became occupied; no preemption')
        env['CUDA_VISIBLE_DEVICES'] = uuid
        (out / 'gpu-selection.json').write_text(json.dumps(dict(physical_gpu=a.physical_gpu, uuid=uuid,
            CUDA_VISIBLE_DEVICES=uuid, selected_at=time.time()), indent=2))
        state['status'] = 'EVALUATING_PILOT_DIVERSITY'
        # Preserve official metric code, with an explicitly saved fixed seed.
        entry = out / 'seeded-entry.py'
        entry.write_text("import random,numpy as np,torch,runpy,sys\nrandom.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)\nsys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[1:];sys.argv[0]+='/evaluate.py';runpy.run_path(sys.argv[0],run_name='__main__')\n")
        run([python, str(entry), str(source), '--dimension', 'diversity', '--config_path',
             str(out / 'config.json')], 'official-diversity', 900, env, source)
        state.update(status='PILOT_DIVERSITY_COMPLETE', finished_at=time.time(),
                     metric_file=str(out / 'metrics/minimax_results.json'))
    except Exception as exc:
        state.update(status='BLOCKED', error=f'{type(exc).__name__}: {exc}', finished_at=time.time())
        raise
    finally:
        save()
        if lease:
            lease.close()
        print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
