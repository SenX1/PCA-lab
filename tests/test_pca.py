import unittest
import sys
import os
from pca import (
    gauss_solver, center_data, covariance_matrix,
    find_eigenvalues, find_eigenvectors, explained_variance_ratio,
    pca, reconstruction_error, auto_select_k, handle_missing_values
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from matrix import Matrix

class TestPCA(unittest.TestCase):
    def test_gauss_solver(self):
        A = Matrix([[2, 1], [1, 2]])
        b = Matrix([[3], [3]])
        solution = gauss_solver(A, b)
        self.assertAlmostEqual(solution[0].data[0][0], 1.0, places=6)
        self.assertAlmostEqual(solution[0].data[1][0], 1.0, places=6)

    def test_center_data(self):
        data = [[1, 2], [3, 4], [5, 6]]
        X = Matrix(data)
        centered = center_data(X)
        expected = [[-2, -2], [0, 0], [2, 2]]
        self.assertEqual(centered.data, expected)

    def test_covariance_matrix(self):
        X = Matrix([[1, 2], [3, 4], [5, 6]])
        X_centered = center_data(X)
        C = covariance_matrix(X_centered)
        self.assertAlmostEqual(C.data[0][0], 4.0, places=2)
        self.assertAlmostEqual(C.data[1][1], 4.0, places=2)

    def test_find_eigenvalues(self):
        C = Matrix([[4, 0], [0, 4]])
        eigenvalues = find_eigenvalues(C)
        self.assertAlmostEqual(eigenvalues[0], 4.0, places=2)
        self.assertAlmostEqual(eigenvalues[1], 4.0, places=2)

    def test_explained_variance_ratio(self):
        eigenvalues = [4.0, 2.0, 1.0]
        ratio = explained_variance_ratio(eigenvalues, k=2)
        self.assertAlmostEqual(ratio, 6/7, places=2)

    def test_pca(self):
        data = [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2]]
        X = Matrix(data)
        X_proj, ratio = pca(X, k=1)
        self.assertEqual(X_proj.rows, 4)
        self.assertGreater(ratio, 0.4)

    def test_auto_select_k(self):
        eigenvalues = [4.0, 3.0, 2.0, 1.0]
        k = auto_select_k(eigenvalues, threshold=0.7)
        self.assertEqual(k, 2)

    def test_handle_missing_values(self):
        data = [[1, float('nan')], [3, 4], [float('nan'), 6]]
        X = Matrix(data)
        X_filled = handle_missing_values(X)
        self.assertEqual(X_filled.data[0], [1, 5.0])  # mean(4, 6) = 5.0
        self.assertEqual(X_filled.data[1], [3, 4])
        self.assertEqual(X_filled.data[2], [2.0, 6])  # mean(1, 3) = 2.0

if __name__ == "__main__":
    unittest.main()