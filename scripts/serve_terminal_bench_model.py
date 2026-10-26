#!/usr/bin/env python3
"""Own one pinned, physical-GPU-bound service for the Terminal-Bench campaign."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from urllib.request import ProxyHandler, build_opener


ASSETS = Path('/opt/phiagent/work/same-model-services/20260831T151445Z-h2001-qwen38-27b-cu129-v14-textonly-torchcodecfix')
MODEL = 'qwen38-27b-dense-ms-e823e888'
GPU_UUID = 'GPU-fff23e0e-9414-4319-0726-0dd6c627992b'
MANIFEST_SHA = 'f2b4dcccf9d0204a393caa7903f768fd5043441f74497707250f7442d1fd7fbd'
PREINFERENCE_SOURCE_SHA256 = {
    'renderers/params.py': '3a551bcbda5d18c3258ab4a4c17db953e97c733badab9bd2157ff62726c73442',
    'entrypoints/openai/chat_completion/serving.py': '6239fae503211193942d0e2037f7b0edcf71ed70271604701283bfc0453d202b',
    'renderers/online_renderer.py': '0bf1a5620fe090a305ac21f7987a77f41a591c24186e5fe521abed432e44cded',
}


def service_profile(name):
    if name in ('h200-1', 'h200-1-current'):
        hostname = 'harnessmm-worker-1' if name == 'h200-1-current' else 'harnessmm-worker-1'
        return {'hostname': hostname, 'assets': ASSETS,
                'leases': Path('/opt/phiagent/gpu-leases'), 'gpu_name': 'H200',
                'flash_attention_version': 3, 'memory_utilization': '0.8'}
    if name == 'gcr-a100':
        return {'hostname': 'GCRAZGDL3008',
                'assets': Path('/path/to/user/phiagent-runs/tb0921s13-assets'),
                'leases': Path('/path/to/user/phiagent-runs/gpu-leases'),
                'additional_leases': (Path('/path/to/user/phiagent-work/searchqa9b/gpu-leases'),),
            'cuda_home': Path('/path/to/user/phiagent-runs/tb0921s13-assets/cuda-12.9'),
                'gpu_name': 'A100', 'flash_attention_version': 2, 'memory_utilization': '0.9'}
    raise ValueError('unknown authorized model service profile')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(target):
    value = hashlib.sha256()
    with Path(target).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write(root, name, value):
    target = root / name
    temporary = target.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(target)


def runtime_environment(root: Path, python: Path, *, gpu_uuid: str = GPU_UUID, cuda_home: Path | None = None):
    work = root / 'w'
    executable_path = str(python.parent) + ':'
    if cuda_home is not None:
        executable_path += str(cuda_home / 'bin') + ':'
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu_uuid, 'TMPDIR': str(work),
                   'TMP': str(work), 'TEMP': str(work), 'PYTHONPATH': '', 'PYTHONNOUSERSITE': '1',
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                   'CUDA_HOME': str(cuda_home or '/usr/local/cuda'), 'PATH': executable_path + os.environ.get('PATH', ''),
                   'HF_HOME': str(root / 'hf-cache'), 'VLLM_CACHE_ROOT': str(root / 'vllm-cache'),
                   'VLLM_CONFIG_ROOT': str(root / 'vllm-config'), 'VLLM_RPC_BASE_PATH': str(work),
                   'VLLM_BATCH_INVARIANT': '0', 'VLLM_NO_USAGE_STATS': '1',
                   'TRITON_CACHE_DIR': str(root / 'triton-cache'), 'TORCHINDUCTOR_CACHE_DIR': str(root / 'torch-cache'),
                   'NO_PROXY': 'localhost,127.0.0.1,::1', 'no_proxy': 'localhost,127.0.0.1,::1'}
    environment.pop('LD_PRELOAD', None)
    return environment


def stop_owned(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)


def gpu_state(root, name, *, idle, physical_index=0, gpu_uuid=GPU_UUID, gpu_name='H200'):
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True)
    compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                       '--format=csv,noheader,nounits'], text=True)
    rows = [[field.strip() for field in line.split(',')] for line in devices.splitlines()]
    selected = [row for row in rows if row[0] == str(physical_index)]
    if len(selected) != 1 or selected[0][1] != gpu_uuid or gpu_name not in selected[0][2]:
        raise RuntimeError('physical GPU identity changed')
    if idle and (int(selected[0][3]) > 128 or gpu_uuid in compute):
        raise RuntimeError('selected GPU is occupied')
    result = {'at_utc': now(), 'hostname': socket.gethostname(), 'devices_csv': devices,
              'compute_csv': compute, 'physical_gpu_index': physical_index, 'gpu_uuid': gpu_uuid,
              'CUDA_VISIBLE_DEVICES': gpu_uuid}
    write(root, name, result)
    return result


def acquire_gpu_leases(profile, physical_index, gpu_uuid):
    with ExitStack() as stack:
        for leases in (profile['leases'], *profile.get('additional_leases', ())):
            if not leases.is_dir():
                raise RuntimeError('shared physical GPU lease directory is unavailable')
            for filename in (f'physical-gpu-{physical_index}.lock', f'physical-{gpu_uuid}.lock',
                             f'physical-GPU-{gpu_uuid}.lock'):
                handle = stack.enter_context((leases / filename).open('a+'))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return stack.pop_all()


def wait_for_gpu_leases(root, profile, physical_index, gpu_uuid, stopped, deadline=None):
    while not stopped.is_set():
        if deadline is not None and datetime.now(timezone.utc) >= deadline:
            raise TimeoutError('original campaign deadline reached while awaiting a GPU lease')
        try:
            return acquire_gpu_leases(profile, physical_index, gpu_uuid)
        except BlockingIOError:
            if deadline is None:
                raise
            write(root, 'WAITING_FOR_GPU.json', {'at_utc': now(), 'status': 'WAITING_FOR_SHARED_GPU_LEASE',
                  'physical_gpu_index': physical_index, 'gpu_uuid': gpu_uuid,
                  'campaign_deadline_utc': deadline.isoformat(), 'other_processes_modified': False})
            stopped.wait(min(30, max(0, (deadline - datetime.now(timezone.utc)).total_seconds())))
    raise InterruptedError('service stopped while awaiting a GPU lease')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--profile', choices=('h200-1', 'h200-1-current', 'gcr-a100'), default='h200-1')
    parser.add_argument('--port', default=19991, type=int)
    parser.add_argument('--capture-port', type=int)
    parser.add_argument('--gpu-index', default=0, type=int)
    parser.add_argument('--gpu-uuid', default=GPU_UUID)
    parser.add_argument('--maximum-seconds', default=172800, type=int)
    parser.add_argument('--wait-for-gpu-until', type=datetime.fromisoformat)
    args = parser.parse_args()
    profile = service_profile(args.profile)
    if socket.gethostname() != profile['hostname']:
        parser.error('this model service is restricted to its authorized remote host')
    if not 900 <= args.maximum_seconds <= 172800:
        parser.error('the service allocation must be bounded to at most 48 hours')
    if args.wait_for_gpu_until is not None and args.wait_for_gpu_until.tzinfo is None:
        parser.error('the GPU wait deadline must include its timezone')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    assets = profile['assets']
    python = assets / 'runtime/bin/python'
    gpu_arguments = {'physical_index': args.gpu_index, 'gpu_uuid': args.gpu_uuid,
                     'gpu_name': profile['gpu_name']}
    stopped = threading.Event()
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, lambda *_: stopped.set())
    process = None
    capture = None
    allocated = None
    gpu_leases = ExitStack()
    try:
        gpu_state(root, 'GPU_BEFORE_LEASE.json', idle=False, **gpu_arguments)
        gpu_leases.enter_context(wait_for_gpu_leases(root, profile, args.gpu_index, args.gpu_uuid,
                                                     stopped, args.wait_for_gpu_until))
        gpu_state(root, 'GPU_BEFORE_ASSET_CHECK.json', idle=True, **gpu_arguments)
        manifest_path = assets / 'configs/qwen38_dense_cross_hub_manifest_v1.json'
        if sha256(manifest_path) != MANIFEST_SHA:
            raise RuntimeError('model identity manifest changed')
        manifest = json.loads(manifest_path.read_text())
        verified = []
        for row in manifest['files']:
            if stopped.is_set():
                raise InterruptedError('service stopped during CPU asset validation')
            target = assets / 'model' / row['path']
            if target.stat().st_size != row['size'] or sha256(target) != row['sha256']:
                raise RuntimeError(f"checkpoint identity mismatch: {row['path']}")
            verified.append(row)
        write(root, 'MODEL_IDENTITY.json', {'manifest_sha256': MANIFEST_SHA, 'files': verified,
              'canonical_revision': manifest['canonical_huggingface_revision'], 'python_sha256': sha256(python)})
        packages = subprocess.check_output([str(python), '-c',
            'import importlib.metadata as metadata,json;print(json.dumps({name:metadata.version(name) for name in ("torch","vllm","transformers","tokenizers")}))'], text=True)
        write(root, 'PACKAGES.json', json.loads(packages))
        runtime_packages = list((assets / 'runtime/lib').glob('python*/site-packages/vllm'))
        if len(runtime_packages) != 1:
            raise RuntimeError('pinned vLLM package path is ambiguous')
        validation_sources = {name: sha256(runtime_packages[0] / name) for name in PREINFERENCE_SOURCE_SHA256}
        if validation_sources != PREINFERENCE_SOURCE_SHA256:
            raise RuntimeError('pinned pre-inference validation source changed')
        write(root, 'PREINFERENCE_VALIDATION.json', {'source_sha256': validation_sources,
              'classification': 'context input validation precedes engine.generate'})
        with ExitStack() as stack:
            if args.wait_for_gpu_until is not None and datetime.now(timezone.utc) >= args.wait_for_gpu_until:
                raise TimeoutError('original campaign deadline reached during asset validation')
            gpu_state(root, 'GPU_SELECTION.json', idle=True, **gpu_arguments)
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', args.port))
            if args.capture_port is not None:
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1', args.capture_port))
            work = root / 'w'
            work.mkdir()
            command = [str(python), '-m', 'vllm.entrypoints.openai.api_server', '--model', str(assets / 'model'),
                       '--served-model-name', MODEL, '--host', '127.0.0.1', '--port', str(args.port),
                       '--dtype', 'bfloat16', '--tensor-parallel-size', '1', '--max-model-len', '32768',
                       '--max-num-seqs', '8', '--gpu-memory-utilization', profile['memory_utilization'], '--generation-config', 'vllm',
                       '--max-num-batched-tokens', '4096', '--gdn-prefill-backend', 'triton',
                       '--default-chat-template-kwargs', '{"enable_thinking":false}',
                       '--attention-config', json.dumps({'backend': 'FLASH_ATTN',
                                                        'flash_attn_version': profile['flash_attention_version']}),
                       '--language-model-only', '--enforce-eager', '--seed', '0',
                       '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder']
            environment = runtime_environment(root, python, gpu_uuid=args.gpu_uuid,
                                              cuda_home=profile.get('cuda_home'))
            if args.profile == 'gcr-a100':
                curand = assets / 'runtime/lib/python3.12/site-packages/nvidia/curand/include'
                environment['NVCC_PREPEND_FLAGS'] = '-I' + str(curand)
                environment['FLASHINFER_WORKSPACE_BASE'] = str(root / 'flashinfer-workspace')
            write(root, 'FROZEN_LAUNCH.json', {'command': command, 'argv': sys.argv, 'hostname': socket.gethostname(),
                  'seed': 0, 'maximum_seconds': args.maximum_seconds, 'source_sha256': sha256(__file__),
                  'profile': args.profile, 'hardware': profile['gpu_name'],
                  'environment': {key: environment[key] for key in ('CUDA_VISIBLE_DEVICES', 'TMPDIR', 'PYTHONPATH', 'HF_HUB_OFFLINE', 'PATH', 'CUDA_HOME',
                      'NVCC_PREPEND_FLAGS', 'FLASHINFER_WORKSPACE_BASE') if key in environment}})
            allocated = time.monotonic()
            with (root / 'service.log').open('x') as log:
                process = subprocess.Popen(command, env=environment, cwd=root, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                stack.callback(stop_owned, process)
                write(root, 'OWNER.json', {'pid': os.getpid(), 'server_pid': process.pid, 'gpu_uuid': args.gpu_uuid,
                                         'proc_start_ticks': Path(f'/proc/{process.pid}/stat').read_text().split()[21]})
                opener = build_opener(ProxyHandler({}))
                ready = False
                while not stopped.is_set() and time.monotonic() - allocated < args.maximum_seconds:
                    if (root / 'STOP').exists():
                        break
                    if process.poll() is not None:
                        raise RuntimeError(f'model service exited: {process.returncode}')
                    if not ready:
                        try:
                            with opener.open(f'http://127.0.0.1:{args.port}/v1/models', timeout=3) as response:
                                models = json.load(response)
                            if len(models['data']) != 1 or models['data'][0]['id'] != MODEL:
                                raise RuntimeError('live model catalog identity differs')
                            capture_receipt = None
                            if args.capture_port is not None:
                                from terminal_provider_capture import SCHEMA, start_capture

                                capture = start_capture(root / 'provider-capture', args.capture_port,
                                                        f'http://127.0.0.1:{args.port}', MODEL)
                                capture_receipt = {'schema': SCHEMA, 'port': args.capture_port,
                                    'response_saved_before_delivery': True,
                                    'source_sha256': sha256(Path(__file__).with_name('terminal_provider_capture.py'))}
                            gpu_state(root, 'GPU_READY.json', idle=False, **gpu_arguments)
                            write(root, 'READY.json', {'status': 'READY', 'at_utc': now(), 'model': MODEL,
                                  'revision': manifest['canonical_huggingface_revision'], 'models': models,
                                  'endpoint': f'http://127.0.0.1:{args.capture_port or args.port}/v1', 'gpu_uuid': args.gpu_uuid,
                                  'durable_capture': capture_receipt,
                                  'hardware': profile['gpu_name'], 'service_profile': args.profile,
                                  'flash_attention_version': profile['flash_attention_version'],
                                  'preinference_validation_source_sha256': validation_sources,
                                  'server_pid': process.pid, 'maximum_seconds': args.maximum_seconds})
                            print('MODEL_READY', flush=True)
                            ready = True
                        except (OSError, ValueError):
                            if time.monotonic() - allocated > 900:
                                raise RuntimeError('model readiness deadline exceeded')
                    write(root, 'HEARTBEAT.json', {'at_utc': now(), 'ready': ready, 'server_pid': process.pid,
                          'elapsed_seconds': round(time.monotonic() - allocated, 1)})
                    stopped.wait(5)
    except Exception as error:
        write(root, 'FAILURE.json', {'at_utc': now(), 'type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        if capture is not None:
            capture.shutdown()
            capture.server_close()
        stop_owned(process)
        gpu_leases.close()
        write(root, 'RETIREMENT.json', {'at_utc': now(), 'allocated_seconds': time.monotonic() - allocated if allocated else 0,
              'server_returncode': process.returncode if process else None, 'owned_service_stopped': True})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
