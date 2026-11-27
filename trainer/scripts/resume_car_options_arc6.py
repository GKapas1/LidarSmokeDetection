"""Benchmark concurrent training and resume the saved car-options experiment."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile

import run_car_options_arc6 as experiment

OUT = experiment.OUT
ROOT = experiment.ROOT
PY = sys.executable
evaluation_lock = threading.Lock()
status_lock = threading.Lock()
active = {}


def write_status(state, runs):
    completed = sum((OUT / row['run_name'] / 'arc6_test_metrics.json').exists()
                    and (OUT / row['run_name'] / 'car_test_metrics.json').exists() for row in runs)
    temporary = OUT / 'status.next.json'
    temporary.write_text(json.dumps({'status': state, 'completed': completed, 'total': len(runs),
                                    'current': ', '.join(active), 'active': dict(active)}, indent=2), encoding='utf-8')
    os.replace(temporary, OUT / 'status.json')


def command(args, log):
    experiment.command(args, log)


def evaluate(row):
    directory = OUT / row['run_name']
    for name, manifest, split in [
        ('car_test_metrics.json', experiment.DATA / 'dataset_manifest.json', row['split']),
        ('arc6_test_metrics.json', experiment.ARC / 'dataset_manifest.json', experiment.SPLIT),
    ]:
        if not (directory / name).exists():
            command([PY, '-m', 'smoke_trainer.cli', 'evaluate', '--bundle', directory / 'bundle.pt',
                     '--manifest', manifest, '--split', split, '--partition', 'test',
                     '--output', directory / name, '--device', 'cuda'], directory / 'execution.log')


def run_one(row, runs):
    directory = OUT / row['run_name']
    with status_lock:
        active[row['run_name']] = 'training'
        write_status('running', runs)
    if not (directory / 'bundle.pt').exists():
        args = [PY, '-m', 'smoke_trainer.cli', 'train', '--config', row['config'], '--device', 'cuda']
        if directory.exists():
            if not (directory / 'last.pt').exists():
                raise RuntimeError(f'Incomplete run without checkpoint: {directory}')
            args += ['--resume-run', directory]
        else:
            args += ['--run-name', row['run_name']]
        command(args, OUT / (row['run_name'] + '.log'))
    with status_lock:
        active[row['run_name']] = 'waiting for evaluation'
        write_status('running', runs)
    with evaluation_lock:
        with status_lock:
            active[row['run_name']] = 'evaluating'
            write_status('running', runs)
        evaluate(row)
    with status_lock:
        del active[row['run_name']]
        experiment.summarize(runs)
        write_status('running', runs)


def benchmark_worker(config_path, name):
    from smoke_trainer.config import load_config
    from smoke_trainer.train import train_model
    config = load_config(config_path)
    config['output']['root'] = str(OUT / 'throughput-benchmark')
    config['training']['epochs'] = 1
    train_model(config, run_name=name, max_train_frames=256)


def benchmark():
    directory = OUT / 'throughput-benchmark'
    directory.mkdir(exist_ok=True)
    configs = [OUT / 'setup' / f'{model}-d_both_with_arc5.toml' for model in ('gt', 'l1', 'g1', 'voxel')]
    timings = {'trials': []}
    attempt = str(time.time_ns())
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32)] + [
            (name, ctypes.c_uint64) for name in ('total_phys', 'avail_phys', 'total_commit',
                'avail_commit', 'total_virtual', 'avail_virtual', 'avail_extended')]

    def memory():
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError()
        return {'available_ram_gib': status.avail_phys / 2**30,
                'available_commit_gib': status.avail_commit / 2**30}

    for workers in (2, 3, 4):
        initial = memory()
        if workers > 2 and initial['available_commit_gib'] < 2.0 + (workers - 2) * 2.0:
            timings['skipped_workers'] = workers
            timings['skip_reason'] = 'Insufficient Windows commit headroom for additional workers'
            break
        samples = [initial]
        stop_sampling = threading.Event()
        def sample():
            while not stop_sampling.wait(0.5):
                samples.append(memory())
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        started = time.perf_counter()
        def launch(index):
            command([PY, Path(__file__).resolve(), '--benchmark-worker', configs[index],
                     '--name', f'scaling-{workers}-{attempt}-{index}'], directory / f'scaling-{workers}-{index}.log')
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(launch, range(4)))
        finally:
            stop_sampling.set()
            sampler.join()
        timings['trials'].append({'workers': workers, 'seconds': time.perf_counter() - started,
            'minimum_available_ram_gib': min(row['available_ram_gib'] for row in samples),
            'minimum_available_commit_gib': min(row['available_commit_gib'] for row in samples)})
        (directory / 'scaling-timings.json').write_text(json.dumps(timings, indent=2), encoding='utf-8')
    safe = [row for row in timings['trials'] if row['minimum_available_commit_gib'] >= 2.0
            and row['minimum_available_ram_gib'] >= 3.0]
    best = min(safe or timings['trials'][:1], key=lambda row: row['seconds'])
    # Prefer fewer workers when the measured benefit is under ten percent.
    chosen = next(row for row in safe or timings['trials'][:1] if row['seconds'] <= best['seconds'] * 1.10)
    timings['workers'] = chosen['workers']
    timings['speedup_over_two_workers'] = timings['trials'][0]['seconds'] / chosen['seconds']
    timings['protocol'] = 'GT, L1, G1 and voxel d_both_with_arc5, each 1 epoch, first 256 shuffled training frames, full selection and calibration'
    (directory / 'scaling-timings.json').write_text(json.dumps(timings, indent=2), encoding='utf-8')
    return timings['workers']


def wait_for_process(pid):
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x100000, False, pid)
    if handle:
        try:
            while kernel.WaitForSingleObject(handle, 1000) == 258:
                pass
        finally:
            kernel.CloseHandle(handle)


def main(wait_pids):
    runs = json.loads((OUT / 'setup/runs.json').read_text(encoding='utf-8'))
    runs.sort(key=lambda row: (row['recipe'], ['gt', 'l1', 'g1', 'voxel'].index(row['model'])))
    try:
        if wait_pids:
            write_status('finishing active training', runs)
            for pid in wait_pids:
                wait_for_process(pid)
        for row in runs:
            directory = OUT / row['run_name']
            if directory.exists() and not (directory / 'arc6_test_metrics.json').exists():
                run_one(row, runs)
        write_status('benchmarking', runs)
        workers = benchmark()
        shutil.copy2(Path(__file__), OUT / Path(__file__).name)
        (OUT / 'scheduling.json').write_text(json.dumps({'workers': workers,
            'evaluation_workers': 1, 'training_configuration_changed': False,
            'note': 'Independent processes share GPU. Wall-clock four-hour limits include contention.'}, indent=2), encoding='utf-8')
        pending = [row for row in runs if not (OUT / row['run_name'] / 'arc6_test_metrics.json').exists()]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_one, row, runs) for row in pending]
            for future in futures:
                future.result()
        write_status('packaging', runs)
        archive_path = OUT.with_suffix('.zip')
        with zipfile.ZipFile(archive_path.with_suffix('.zip.partial'), 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path in OUT.rglob('*'):
                if path.is_file():
                    archive.write(path, path.relative_to(OUT.parent))
        os.replace(archive_path.with_suffix('.zip.partial'), archive_path)
        write_status('complete', runs)
    except Exception:
        (OUT / 'resume-error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        write_status('failed', runs)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, nargs='+')
    parser.add_argument('--benchmark-worker', type=Path)
    parser.add_argument('--name')
    args = parser.parse_args()
    if args.benchmark_worker:
        benchmark_worker(args.benchmark_worker, args.name)
    else:
        main(args.wait_pid)
