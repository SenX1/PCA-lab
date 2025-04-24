import unittest
import sys
import os

sys.path.append(r'C:/Users/arseni/all/git/repa/linalg_pca/PCA-lab/src')

from matrix import Matrix

class TestMatrix(unittest.TestCase):
    def test_init(self):
        data = [[1, 2], [3, 4]]
        matrix = Matrix(data)
        self.assertEqual(matrix.data, data)
        self.assertEqual(matrix.rows, 2)
        self.assertEqual(matrix.cols, 2)

    def test_transpose(self):
        data = [[1, 2, 3], [4, 5, 6]]
        matrix = Matrix(data)
        transposed = matrix.transpose()
        self.assertEqual(transposed.data, [[1, 4], [2, 5], [3, 6]])

    def test_sub(self):
        A = Matrix([[5, 3], [2, 1]])
        B = Matrix([[1, 1], [1, 1]])
        result = A - B
        self.assertEqual(result.data, [[4, 2], [1, 0]])

    def test_mul_scalar(self):
        A = Matrix([[1, 2], [3, 4]])
        result = A * 2
        self.assertEqual(result.data, [[2, 4], [6, 8]])

    def test_mul_matrix(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        result = A * B
        self.assertEqual(result.data, [[19, 22], [43, 50]])

    def test_determinant(self):
        A = Matrix([[2, 1], [1, 2]])
        self.assertEqual(A.determinant(), 3)

    def test_identity(self):
        I = Matrix.identity(3)
        expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.assertEqual(I.data, expected)