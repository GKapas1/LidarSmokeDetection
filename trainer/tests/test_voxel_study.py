import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from smoke_trainer.adapters import ContextVoxelAdapter
from smoke_trainer.data import UnifiedDataset, create_split_plan, iter_interleaved_frames
from smoke_trainer.models import ContextVoxelPointNet
from smoke_trainer.predict import SmokePredictor
from smoke_trainer.train import train_model
from helpers import make_dataset


def test_context_is_local_permutation_equivariant_and_differentiable():
    torch.manual_seed(5)
    model = ContextVoxelPointNet(7, [8], [4])
    features = torch.tensor([[0,0,0,1,0,0,0], [0,0,0,2,1,0,0], [0,0,0,3,20,0,0]], dtype=torch.float32)
    original = model(features)
    changed = features.clone()
    changed[2,3] = 100
    torch.testing.assert_close(model(changed)[:2], original[:2])
    shifted = features.clone()
    shifted[:,4:] += 100
    torch.testing.assert_close(model(shifted), original)
    order = torch.tensor([2,0,1])
    torch.testing.assert_close(model(features[order]), original[order])
    original.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert model(features[:0]).shape == (0,)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1)
    changed_neighbor = features.clone()
    changed_neighbor[1,3] = 20
    assert model(changed_neighbor)[0] > model(features)[0]


def test_context_trains_and_bundles_with_interleaved_sampling():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        dataset = UnifiedDataset(make_dataset(root/'data'))
        split = root/'split.json'
        plan = create_split_plan(dataset, split, purge_frames=1)
        refs = list(iter_interleaved_frames(dataset, plan, seed=3, steps=7))
        assert len(refs) == 7
        assert [f.ref for f in refs] == [f.ref for f in iter_interleaved_frames(dataset, plan, seed=3, steps=7)]
        frame = refs[0]
        adapter = ContextVoxelAdapter()
        normalizer = adapter.fit_normalizer([frame], {'voxel_size_m': .1})
        prepared = adapter.prepare(frame, {'voxel_size_m': .1})
        np.testing.assert_array_equal(normalizer.transform(prepared.features)[:,4:], prepared.features[:,4:])
        config = dict(config_path='test',
            data=dict(manifest=str(dataset.manifest_path), split=str(split), voxel_size_m=.1),
            model=dict(name='context_voxel_pointnet', adapter='context_voxel_points', point_widths=[8],head_widths=[4]),
            training=dict(seed=3,device='cpu',epochs=1,learning_rate=.001,weight_decay=0,
                positive_weight=1,patience=1,max_hours=.1,max_optimizer_steps_per_epoch=2,frame_sampling='interleaved'),
            output=dict(root=str(root/'runs')))
        run = train_model(config,run_name='context')
        probabilities,valid = SmokePredictor.load(run/'bundle.pt',device='cpu').predict(frame)
        assert probabilities.shape == frame.label.shape
        assert valid.all()
        assert json.loads((run/'history.json').read_text())[0]['training_frames'] == 2
