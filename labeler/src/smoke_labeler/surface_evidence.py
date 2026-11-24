"""Local clean-surface ambiguity tests for offline smoke proposals.

Margins describe empirical clean replay discrepancies, not certified pose bounds.
"""
import numpy as np
from scipy.spatial import cKDTree


class SurfaceEvidence:
    def __init__(self, reference, neighbors=24, patch_radius=.30, cell_size=.50,
                 fallback_neighbors=None):
        self.reference = np.asarray(reference, dtype=float)
        self.tree = cKDTree(self.reference)
        self.cell_size = cell_size
        self.patch_radius = patch_radius
        (self.centers, self.normals, self.thickness,
         self.planar) = self._fit_patches(self.reference, neighbors)
        if fallback_neighbors is not None:
            unresolved = np.flatnonzero(~self.planar)
            centers, normals, thickness, planar = self._fit_patches(
                self.reference[unresolved], fallback_neighbors)
            accepted = unresolved[planar]
            self.centers[accepted] = centers[planar]
            self.normals[accepted] = normals[planar]
            self.thickness[accepted] = thickness[planar]
            self.planar[accepted] = True
        self.cell_keys, self.cell_ids = np.unique(
            np.floor(self.centers / cell_size).astype(np.int32), axis=0,
            return_inverse=True)
        self.margins = np.full(len(self.cell_keys), np.nan)

    def _fit_patches(self, points, neighbors):
        distance, indices = self.tree.query(points, k=neighbors)
        patches = self.reference[indices]
        centers = patches.mean(axis=1)
        centered = patches - centers[:, None, :]
        covariance = np.einsum('nki,nkj->nij', centered, centered) / neighbors
        eigenvalues, vectors = np.linalg.eigh(covariance)
        normals = vectors[:, :, 0]
        thickness = np.sqrt(np.maximum(eigenvalues[:, 0], 0))
        planar = ((distance[:, -1] <= self.patch_radius)
                  & (eigenvalues[:, 1] > 1e-5)
                  & (eigenvalues[:, 0] < .08 * eigenvalues[:, 1]))
        return centers, normals, thickness, planar

    def residuals(self, points):
        distance, index = self.tree.query(points)
        residual = np.abs(np.einsum('ij,ij->i', points - self.centers[index],
                                    self.normals[index]))
        return distance, index, residual

    def calibrate(self, points, minimum_samples=50):
        """Use one development clean drive exclusively to set local margins."""
        distance, index, residual = self.residuals(points)
        valid = self.planar[index] & (distance <= self.patch_radius)
        groups = self.cell_ids[index[valid]]
        values = residual[valid]
        order = np.argsort(groups)
        groups, values = groups[order], values[order]
        keys, starts, counts = np.unique(groups, return_index=True, return_counts=True)
        for key, start, count in zip(keys, starts, counts):
            if count >= minimum_samples:
                # Extra margin for variation beyond the sampled clean residuals.
                self.margins[key] = .01 + 1.5 * np.quantile(
                    values[start:start + count], .995)

    def query(self, points, margin_scale=1.0):
        if not np.isfinite(margin_scale) or margin_scale < 0:
            raise ValueError('margin_scale must be finite and nonnegative')
        distance, index, residual = self.residuals(points)
        margin = margin_scale * (
            self.margins[self.cell_ids[index]] + 2 * self.thickness[index])
        local = distance <= self.patch_radius
        modeled = self.planar[index] & np.isfinite(margin)
        separated = residual > margin
        # Nearby poorly observed edges remain unresolved. Distant points still
        # require measured free-space evidence in the caller.
        allowed = ~local | (modeled & separated)
        return dict(allowed=allowed, distance=distance, residual=residual,
                    margin=margin, modeled=modeled, local=local)

    def save(self, path):
        np.savez_compressed(path, reference=self.reference, centers=self.centers,
                            normals=self.normals, thickness=self.thickness,
                            planar=self.planar, cell_ids=self.cell_ids,
                            cell_keys=self.cell_keys, margins=self.margins,
                            cell_size=self.cell_size, patch_radius=self.patch_radius)

    @classmethod
    def load(cls, path):
        result = cls.__new__(cls)
        with np.load(path) as data:
            for key in data.files:
                setattr(result, key, data[key])
        result.tree = cKDTree(result.reference)
        return result
