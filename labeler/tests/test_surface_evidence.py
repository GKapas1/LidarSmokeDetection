import tempfile
import unittest
from pathlib import Path
import numpy as np
from smoke_labeler.surface_evidence import SurfaceEvidence


class SurfaceEvidenceTests(unittest.TestCase):
    def test_near_surface_smoke_and_clean_uncertainty(self):
        axis=np.linspace(-1,1,81)
        x,y=np.meshgrid(axis,axis)
        reference=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
        model=SurfaceEvidence(reference)
        clean=reference.copy()
        clean[:,2]=.004*np.sin(np.arange(len(clean)))
        model.calibrate(clean,minimum_samples=20)
        points=np.array([[.1,.1,.004],[.1,.1,.06],[.1,.1,.15]])
        result=model.query(points)
        np.testing.assert_array_equal(result['allowed'],[False,True,True])
        self.assertTrue(np.all(result['distance']<.18))
        margin=model.query(np.array([[.1,.1,0.]]))['margin'][0]
        borderline=np.array([[.1,.1,.975*margin]])
        self.assertFalse(model.query(borderline)['allowed'][0])
        reduced=model.query(borderline,margin_scale=.95)
        self.assertTrue(reduced['allowed'][0])
        self.assertAlmostEqual(reduced['margin'][0],.95*margin)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.npz';model.save(path)
            loaded=SurfaceEvidence.load(path).query(points)
            np.testing.assert_array_equal(loaded['allowed'],result['allowed'])

    def test_uncalibrated_nearby_surface_stays_unresolved(self):
        axis=np.linspace(-1,1,81)
        x,y=np.meshgrid(axis,axis)
        model=SurfaceEvidence(np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size))))
        result=model.query(np.array([[0,0,.05],[0,0,.5]]))
        np.testing.assert_array_equal(result['allowed'],[False,True])
        # Distant allowed points still need the caller's measured ray evidence.

    def test_wider_fallback_recovers_a_noisy_plane(self):
        axis=np.arange(-.6,.61,.05)
        x,y=np.meshgrid(axis,axis)
        grid=np.indices(x.shape).sum(axis=0)
        z=np.where(grid%2, .025, -.025)
        reference=np.column_stack((x.ravel(),y.ravel(),z.ravel()))
        small=SurfaceEvidence(reference)
        adaptive=SurfaceEvidence(reference,fallback_neighbors=96)
        _,center=small.tree.query([0,0,0])
        self.assertFalse(small.planar[center])
        self.assertTrue(adaptive.planar[center])
        np.testing.assert_allclose(adaptive.centers[small.planar],small.centers[small.planar])
        np.testing.assert_allclose(adaptive.normals[small.planar],small.normals[small.planar])


if __name__=='__main__':unittest.main()
