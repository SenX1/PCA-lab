from typing import List, Tuple
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matrix import Matrix
import math
import random

# ----------------------------- EASY LEVEL -----------------------------
def gauss_solver(A: Matrix, b: Matrix) -> List[Matrix]:
    """Решение СЛАУ методом Гаусса"""
    n = A.rows
    augmented = [A.data[i] + b.data[i] for i in range(n)]
    
    for i in range(n):
        # Выбор ведущего элемента
        pivot = max(range(i, n), key=lambda r: abs(augmented[r][i]))
        augmented[i], augmented[pivot] = augmented[pivot], augmented[i]
        
        if abs(augmented[i][i]) < 1e-10:
            raise ValueError("Система несовместна")
            
        # Нормировка строки
        divisor = augmented[i][i]
        augmented[i] = [elem / divisor for elem in augmented[i]]
        
        # Исключение переменной
        for j in range(n):
            if j != i and abs(augmented[j][i]) > 1e-10:
                factor = augmented[j][i]
                augmented[j] = [augmented[j][k] - factor * augmented[i][k] 
                              for k in range(len(augmented[j]))]
    
    solution = [row[-1] for row in augmented]
    return [Matrix([[val] for val in solution])]

def center_data(X: Matrix) -> Matrix:
    """Центрирование данных"""
    means = [sum(row[col] for row in X.data) / X.rows for col in range(X.cols)]
    return Matrix([[x - means[col] for col, x in enumerate(row)] for row in X.data])

def covariance_matrix(X_centered: Matrix) -> Matrix:
    """Ковариационная матрица"""
    XT = X_centered.transpose()
    return (XT * X_centered) * (1 / (X_centered.rows - 1))

# ---------------------------- NORMAL LEVEL ----------------------------
def find_eigenvalues(C: Matrix, tol: float = 1e-6) -> List[float]:
    """Поиск собственных значений методом бисекции"""
    def f(lam):
        return (C - Matrix.identity(C.rows) * lam).determinant()
    
    eigenvalues = []
    # Для диагональной матрицы собственные значения - диагональные элементы
    if all(abs(C.data[i][j]) < tol for i in range(C.rows) for j in range(C.cols) if i != j):
        return sorted([C.data[i][i] for i in range(C.rows)], reverse=True)
    
    for _ in range(C.rows):
        a, b = -1e6, 1e6
        while b - a > tol:
            c = (a + b) / 2
            if f(a) * f(c) < 0:
                b = c
            else:
                a = c
        eigenvalue = round((a + b)/2, int(math.log10(1/tol)))
        eigenvalues.append(eigenvalue)
    return sorted(eigenvalues, reverse=True)

def find_eigenvectors(C: Matrix, eigenvalues: List[float]) -> List[Matrix]:
    """Поиск собственных векторов"""
    eigenvectors = []
    for lambda_ in eigenvalues:
        system = C - Matrix.identity(C.rows) * lambda_
        try:
            solution = gauss_solver(system, Matrix([[0] for _ in range(system.rows)]))
            eigenvectors.append(solution[0])
        except ValueError:
            continue
    return eigenvectors

def explained_variance_ratio(eigenvalues: List[float], k: int) -> float:
    """Доля объяснённой дисперсии"""
    total = sum(eigenvalues)
    return sum(eigenvalues[:k]) / total

# ----------------------------- HARD LEVEL -----------------------------
def pca(X: Matrix, k: int) -> Tuple[Matrix, float]:
    """Полный алгоритм PCA"""
    X_centered = center_data(X)
    C = covariance_matrix(X_centered)
    eigenvalues = find_eigenvalues(C)
    eigenvectors = find_eigenvectors(C, eigenvalues)
    
    # Сортировка по убыванию собственных значений
    sorted_pairs = sorted(zip(eigenvalues, eigenvectors), key=lambda x: -x[0])
    eigenvalues = [pair[0] for pair in sorted_pairs]
    eigenvectors = [pair[1] for pair in sorted_pairs]
    
    eigenvectors_flat = []
    for vec in eigenvectors:
        flat_vec = [elem[0] for elem in vec.data]
        eigenvectors_flat.append(flat_vec)
    
    Vk = Matrix(eigenvectors_flat[:k]).transpose()
    
    X_proj = X_centered * Vk
    gamma = explained_variance_ratio(eigenvalues, k)
    
    return X_proj, gamma

def plot_pca_projection(X_proj: Matrix) -> Figure:
    """
    Визуализирует проекцию данных на первые две главные компоненты.
    
    Параметры:
        X_proj (Matrix): Проекция данных размерности n×2
    
    Возвращает:
        Figure: Объект графика Matplotlib
    """
    x = [row[0] for row in X_proj.data]
    y = [row[1] for row in X_proj.data]
    
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    ax.scatter(x, y, alpha=0.7, c='blue', edgecolors='w', s=40)
    
    ax.set_title('Проекция данных на первые две главные компоненты', pad=15)
    ax.set_xlabel('Главная компонента 1 (PC1)')
    ax.set_ylabel('Главная компонента 2 (PC2)')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig

def reconstruction_error(X_orig: Matrix, X_recon: Matrix) -> float:
    """Среднеквадратическая ошибка восстановления"""
    error = 0.0
    for i in range(X_orig.rows):
        for j in range(X_orig.cols):
            error += (X_orig.data[i][j] - X_recon.data[i][j]) ** 2
    return error / (X_orig.rows * X_orig.cols)

# ---------------------------- EXPERT LEVEL ----------------------------
def auto_select_k(eigenvalues: List[float], threshold: float = 0.95) -> int:
    """Автоматический выбор числа компонент"""
    total = sum(eigenvalues)
    cumulative = 0.0
    for k, val in enumerate(eigenvalues, 1):
        cumulative += val
        if cumulative / total >= threshold:
            return k
    return len(eigenvalues)

def handle_missing_values(X: Matrix) -> Matrix:
    """Обработка пропущенных значений"""
    filled = []
    for col in range(X.cols):
        values = [row[col] for row in X.data if not math.isnan(row[col])]
        mean = sum(values) / len(values) if values else 0
        filled_col = [mean if math.isnan(x) else x for x in [row[col] for row in X.data]]
        filled.append(filled_col)
    return Matrix([list(row) for row in zip(*filled)])

def add_noise_and_compare(X: Matrix, noise_level: float = 0.1):
    """Исследование влияния шума"""
    noise = Matrix([[random.gauss(0, noise_level) for _ in range(X.cols)] for _ in range(X.rows)])
    X_noisy = X + noise
    X_proj_orig, _ = pca(X, 2)
    X_proj_noisy, _ = pca(X_noisy, 2)
    return X_proj_orig, X_proj_noisy

def apply_pca_to_dataset(dataset_name: str, k: int) -> Tuple[Matrix, float]:
    """Применение PCA к датасету"""
    from sklearn.datasets import load_iris
    data = load_iris()
    X = Matrix(data.data.tolist())
    X_proj, ratio = pca(X, k)
    return X_proj, ratio