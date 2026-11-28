"""Measure stationary range recall at the saved and car-selection FPR thresholds."""
import argparse
import json
from pathlib import Path

import numpy as np

from smoke_trainer.data import UnifiedDataset, iter_frames, load_split_plan
from smoke_trainer.evaluate import collect_predictions
from smoke_trainer.predict import SmokePredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    predictor = SmokePredictor.load(args.run/'bundle.pt', device=args.device)
    car = UnifiedDataset(repo/'data/training/car_training_options_v1/dataset_manifest.json')
    split = load_split_plan(car, args.run/'split.json')
    selection = collect_predictions(predictor.model, iter_frames(car, split, 'selection'),
        predictor.normalizer, predictor.adapter, predictor.preprocessing, predictor.device)
    probabilities = predictor.calibration.apply(selection.logits)
    negatives = probabilities[selection.labels == 0]
    # Strictly exceed the quantile to handle tied scores conservatively.
    threshold = float(np.nextafter(np.quantile(negatives, .99, method='higher'), np.inf))
    thresholds = [predictor.threshold, threshold]
    achieved = float(np.mean(negatives.astype(np.float64) >= threshold))
    del selection, probabilities, negatives
    dataset = UnifiedDataset(repo/'data/training/unified_v1/dataset_manifest.json')
    counts = np.zeros((2, 5, 4), dtype=np.int64)
    frames = 0
    for chunk in dataset.chunks:
        if chunk.source_domain != 'stationary_ros2':
            continue
        arrays = dataset.load_chunk(chunk.index)
        for index in range(chunk.frames):
            frame = dataset.frame(chunk, arrays, index)
            probabilities, valid = predictor.predict(frame)
            valid &= frame.label != 255
            bands = np.searchsorted([3,5,8,12], np.linalg.norm(frame.xyz[valid], axis=1), side='right')
            for i, threshold in enumerate(thresholds):
                codes = bands*4 + 2*frame.label[valid] + (probabilities[valid].astype(np.float64) >= threshold)
                counts[i] += np.bincount(codes, minlength=20).reshape(5,4)
            frames += 1
    reports = []
    for name, threshold, cells in zip(['saved', 'car_selection_fpr_1pct'], thresholds, counts):
        bands = []
        for band, (tn,fp,fn,tp) in zip(['0-3','3-5','5-8','8-12','12+'], cells):
            bands.append(dict(range_m=band, positives=int(fn+tp), negatives=int(tn+fp),
                true_positive=int(tp), false_positive=int(fp),
                recall=float(tp/(tp+fn)) if tp+fn else None,
                false_positive_rate=float(fp/(fp+tn)) if fp+tn else None))
        reports.append(dict(operating_point=name, threshold=threshold, ranges=bands))
    result = dict(frames=frames, car_selection_achieved_fpr=achieved, operating_points=reports,
        usage='Stationary development diagnostic; thresholds derived exclusively from car selection')
    (args.run/'stationary_diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
