import sys
import os
import math
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from matrix import Matrix
from pca import (
    center_data,
    covariance_matrix,
    find_eigenvalues,
    find_eigenvectors,
    pca,
    plot_pca_projection,
    auto_select_k,
    handle_missing_values,
    reconstruction_error
)

def main():
    # --------------------------
    # 1. Пример данных (Easy)
    # --------------------------
    print("\n1. Пример данных (Easy):")
    data = [
        [2.5, 2.4, 1.7],
        [0.5, 0.7, 0.4],
        [2.2, 2.9, 1.9],
        [1.9, 2.2, 1.5]
    ]
    X = Matrix(data)
    print("Исходные данные:")
    for row in X.data:
        print(row)

    # --------------------------
    # 2. Центрирование данных (Easy)
    # --------------------------
    X_centered = center_data(X)
    print("\n2. Центрированные данные:")
    for row in X_centered.data:
        print([round(x, 2) for x in row])

    # --------------------------
    # 3. Ковариационная матрица (Easy)
    # --------------------------
    C = covariance_matrix(X_centered)
    print("\n3. Ковариационная матрица:")
    for row in C.data:
        print([round(x, 3) for x in row])

    # --------------------------
    # 4. Собственные значения (Normal)
    # --------------------------
    eigenvalues = find_eigenvalues(C)
    print("\n4. Собственные значения:", [round(x, 3) for x in eigenvalues])

    # --------------------------
    # 5. Собственные векторы (Normal)
    # --------------------------
    eigenvectors = find_eigenvectors(C, eigenvalues)
    print("\n5. Первый собственный вектор:", [round(x[0], 3) for x in eigenvectors[0].data])

    # --------------------------
    # 6. PCA проекция (Hard)
    # --------------------------
    X_proj, ratio = pca(X, k=2)
    print(f"\n6. Доля объяснённой дисперсии: {ratio:.3f}")

    # --------------------------
    # 7. Визуализация (Hard)
    # --------------------------
    fig = plot_pca_projection(X_proj)
    plt.show()

    # --------------------------
    # 8. Обработка пропусков (Expert)
    # --------------------------
    data_with_nan = [
        [1.0, 2.0, math.nan],
        [3.0, math.nan, 4.0],
        [math.nan, 5.0, 6.0]
    ]
    X_missing = Matrix(data_with_nan)
    X_filled = handle_missing_values(X_missing)
    print("\n8. Данные после обработки пропусков:")
    for row in X_filled.data:
        print([round(x, 2) if not math.isnan(x) else "NaN" for x in row])

    # --------------------------
    # 9. Автовыбор компонент (Expert)
    # --------------------------
    k = auto_select_k(eigenvalues, threshold=0.95)
    print(f"\n9. Оптимальное число компонент: {k}")

    # --------------------------
    # 10. Пример с реальным датасетом (Expert)
    # --------------------------
    from sklearn.datasets import load_iris
    iris = load_iris()
    X_iris = Matrix(iris.data.tolist())
    X_iris_proj, ratio_iris = pca(X_iris, k=2)
    print(f"\n10. Пример с Iris: доля дисперсии = {ratio_iris:.3f}")
    plot_pca_projection(X_iris_proj)
    plt.show()

if __name__ == "__main__":
    main()