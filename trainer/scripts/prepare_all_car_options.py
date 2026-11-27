"""Create model/recipe configurations from the existing car-options dataset."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from smoke_trainer.config import load_config
from smoke_trainer.data import UnifiedDataset, load_split_plan, split_summary

MODELS = {
    "l1": ('local_mlp', 'local_voxel', 'hidden_widths = [64, 32]'),
    "g1": ('global_pointnet', 'global_points', 'point_widths = [64, 128]\nhead_widths = [64]\ninput_alignment = "tnet"\nalignment_widths = [32, 64]\nalignment_regularization = 0.001'),
    "gt": ('global_temporal_pointnet', 'global_temporal', 'history_length = 5\ngru_hidden_width = 32\npoint_widths = [64, 128]\nhead_widths = [64]\ninput_alignment = "tnet"\nalignment_widths = [32, 64]\nalignment_regularization = 0.001'),
    "voxel": ('voxel_pointnet', 'voxel_points', 'point_widths = [64, 128]\nhead_widths = [64]'),
}


def temporal_split(dataset: UnifiedDataset, original: dict, output: Path) -> None:
    """Select 254 consecutive clean frames, giving 250 five-frame target windows."""
    plan = deepcopy(original)
    arc_indices = {c.index for c in dataset.chunks if c.source_domain == 'grandtour_ros1'}
    if any(row['chunk_index'] in arc_indices for role in ('selection', 'calibration', 'test') for row in plan['roles'][role]):
        raise ValueError('ARC-5 must be training-only')
    eligible = [c for c in dataset.chunks if c.index in arc_indices and c.frames >= 254]
    if not eligible:
        raise ValueError('Need an ARC-5 chunk containing at least 254 consecutive frames')
    chosen = eligible[0]
    arrays = dataset.load_chunk(chosen.index)
    times = arrays['frame_time_s'][:254]
    if np.any(np.diff(times) <= 0) or np.any(np.diff(times) > 0.25):
        raise ValueError('Selected ARC-5 frames do not form contiguous temporal windows')
    if np.any(arrays['label'][:int(arrays['frame_ptr'][254])] == 1):
        raise ValueError('Selected ARC-5 frames contain smoke labels')
    plan['roles']['train'] = [r for r in plan['roles']['train'] if r['chunk_index'] not in arc_indices]
    plan['roles']['train'].append({'chunk_index': chosen.index, 'ranges': [[0, 254]]})
    plan['excluded'] = [r for r in plan.get('excluded', []) if r['chunk_index'] not in arc_indices]
    for chunk in dataset.chunks:
        if chunk.index in arc_indices:
            start = 254 if chunk.index == chosen.index else 0
            if start < chunk.frames:
                plan['excluded'].append({'chunk_index': chunk.index, 'ranges': [[start, chunk.frames]]})
    plan['protocol'] = 'car_options_v1_gt_with_arc5'
    plan['arc5_training_policy'] = 'First 254 consecutive clean frames; 250 target windows at five-frame history'
    plan['summary'] = split_summary(dataset, plan)
    plan.pop('split_sha256', None)
    plan['split_sha256'] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    output.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    load_split_plan(dataset, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-prefix', required=True)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--recipes', nargs='+')
    args = parser.parse_args()
    root = args.dataset.resolve()
    output = args.output.resolve()
    inventory = json.loads((root / 'experiments.json').read_text(encoding='utf-8'))
    recipes = args.recipes or inventory['recipes']
    if len(set(recipes)) != len(recipes) or any(r not in inventory['recipes'] for r in recipes):
        raise ValueError('Unknown or duplicate recipe')
    if len(set(args.models)) != len(args.models):
        raise ValueError('Duplicate model')
    if not args.run_prefix or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.run_prefix):
        raise ValueError('Run prefix must contain only letters, numbers, hyphens or underscores')
    runs_root = output.parent
    for model in args.models:
        for recipe in recipes:
            if (runs_root / f'{args.run_prefix}-{model}-{recipe}').exists():
                raise FileExistsError(f'Run already exists: {args.run_prefix}-{model}-{recipe}')
    if output.exists():
        raise FileExistsError(f'Setup already exists: {output}')
    dataset = UnifiedDataset(root / 'dataset_manifest.json')
    for source in ('car_only', 'with_arc5'):
        load_split_plan(dataset, root / f'split_{source}.json')
    output.mkdir(parents=True)
    if 'gt' in args.models and any(r.endswith('_with_arc5') for r in recipes):
        temporal_split(dataset, load_split_plan(dataset, root / 'split_with_arc5.json'), output / 'split_gt_with_arc5.json')
    rows = []
    for model in args.models:
        name, adapter, parameters = MODELS[model]
        for recipe in recipes:
            source_config = root / 'configs' / f'{recipe}.toml'
            base = load_config(source_config)
            source = 'with_arc5' if recipe.endswith('_with_arc5') else 'car_only'
            split = output / 'split_gt_with_arc5.json' if model == 'gt' and source == 'with_arc5' else root / f'split_{source}.json'
            run_name = f'{args.run_prefix}-{model}-{recipe}'
            config_path = output / f'{model}-{recipe}.toml'
            text = f'''[data]
manifest = {json.dumps(dataset.manifest_path.as_posix())}
split = {json.dumps(split.as_posix())}
verify_checksums = false
voxel_size_m = 0.10
history_length = 5
maximum_time_gap_s = 0.25

[model]
name = "{name}"
adapter = "{adapter}"
{parameters}

[training]
seed = 20260920
device = "cuda"
epochs = 30
learning_rate = 0.001
weight_decay = 0.0001
positive_weight = 8.0
patience = 5
max_hours = 4.0
max_optimizer_steps_per_epoch = 1400
'''
            if model in ('g1', 'gt', 'voxel'):
                text += 'yaw_augmentation_deg = 180.0\npitch_augmentation_deg = 0.0\nroll_augmentation_deg = 0.0\n'
            text += f'\n[output]\nroot = {json.dumps(runs_root.as_posix())}\n'
            for section, values in base.get('augmentation', {}).items():
                text += f'\n[augmentation.{section}]\n'
                for key, value in values.items():
                    text += f'{key} = {json.dumps(value)}\n'
            config_path.write_text(text, encoding='utf-8')
            load_config(config_path)
            rows.append({'model': model, 'recipe': recipe, 'run_name': run_name,
                         'config': str(config_path), 'split': str(split)})
    (output / 'runs.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
    print(f'Prepared {len(rows)} runs in {output}')


if __name__ == '__main__':
    main()
