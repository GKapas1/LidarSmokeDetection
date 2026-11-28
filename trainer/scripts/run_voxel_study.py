"""Run the nine voxel ablations on the existing car/ARC-5 dataset."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from smoke_trainer.config import load_config
from smoke_trainer.data import UnifiedDataset, load_split_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default='voxel-study-v1')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not args.prefix or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.prefix):
        raise ValueError('Prefix must contain only letters, digits, hyphens and underscores')
    repo = Path(__file__).resolve().parents[2]
    dataset_root = repo / 'data/training/car_training_options_v1'
    manifest = dataset_root / 'dataset_manifest.json'
    stationary_manifest = repo / 'data/training/unified_v1/dataset_manifest.json'
    dataset = UnifiedDataset(manifest, verify_checksums=True)
    load_split_plan(dataset, dataset_root / 'split_with_arc5.json')
    UnifiedDataset(stationary_manifest, verify_checksums=True)
    base = load_config(dataset_root / 'configs/a_baseline_with_arc5.toml')
    setup = repo / 'runs' / f'{args.prefix}-setup'
    setup.mkdir(parents=True, exist_ok=True)
    jobs = []
    for variant in ['baseline', 'interleaved', 'context']:
        for seed in [20260920, 20260921, 20260922]:
            config = deepcopy(base)
            config['data']['voxel_size_m'] = 0.1
            config['model'] = dict(name='voxel_pointnet', adapter='voxel_points', point_widths=[64,128], head_widths=[64])
            if variant == 'context':
                config['model'].update(name='context_voxel_pointnet', adapter='context_voxel_points')
            config['training'].update(seed=seed, device=args.device,
                frame_sampling='chunk_shuffle' if variant == 'baseline' else 'interleaved')
            config['output']['root'] = str(repo / 'runs')
            name = f'{args.prefix}-{variant}-{seed}'
            path = setup / f'{variant}-{seed}.toml'
            text = ''
            for section in ['data', 'model', 'training', 'output']:
                text += f'\n[{section}]\n'
                for key,value in config[section].items():
                    text += f'{key} = {json.dumps(value)}\n'
            # Paths serialized by JSON are valid TOML basic strings on Windows.
            if path.exists() and path.read_text(encoding='utf-8') != text:
                raise ValueError(f'Existing study configuration differs: {path}')
            path.write_text(text, encoding='utf-8')
            jobs.append(dict(name=name, config=str(path)))
    (setup / 'jobs.json').write_text(json.dumps(jobs, indent=2), encoding='utf-8')
    print(f'Prepared {len(jobs)} runs in {setup}', flush=True)
    if args.prepare_only:
        return
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
    for job in jobs:
        run = repo / 'runs' / job['name']
        if not (run / 'bundle.pt').exists():
            command = [sys.executable, '-m', 'smoke_trainer.cli', 'train', '--config', job['config']]
            command += ['--resume-run', str(run)] if (run / 'last.pt').exists() else ['--run-name', job['name']]
            subprocess.run(command, check=True, cwd=repo)
        if not (run / 'car_test_metrics.json').exists():
            subprocess.run([sys.executable, '-m', 'smoke_trainer.cli', 'evaluate',
                '--bundle', str(run/'bundle.pt'), '--manifest', str(manifest),
                '--split', str(dataset_root/'split_with_arc5.json'),
                '--output', str(run/'car_test_metrics.json'), '--device', args.device], check=True, cwd=repo)
        if not (run / 'stationary_diagnostic.json').exists():
            subprocess.run([sys.executable, str(repo/'trainer/scripts/voxel_stationary_diagnostic.py'),
                '--run', str(run), '--device', args.device], check=True, cwd=repo)
    rows = []
    for job in jobs:
        run = repo/'runs'/job['name']
        rows.append(dict(name=job['name'], car=json.loads((run/'car_test_metrics.json').read_text()),
            stationary=json.loads((run/'stationary_diagnostic.json').read_text())))
    (setup/'comparison.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
