class Matrix:
    def __init__(self, data):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0

    def transpose(self):
        return Matrix([[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)])

    def __sub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Матрицы разного размера")
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Matrix([[x * other for x in row] for row in self.data])
        elif isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Несовместимые размеры")
            result = [
                [sum(a * b for a, b in zip(row, col)) for col in zip(*other.data)]
                for row in self.data
            ]
            return Matrix(result)
        else:
            raise TypeError("Неподдерживаемый тип")

    def determinant(self):
        if self.rows != self.cols:
            raise ValueError("Матрица не квадратная")
        if self.rows == 1:
            return self.data[0][0]
        det = 0
        for col in range(self.cols):
            minor = [row[:col] + row[col+1:] for row in self.data[1:]]
            det += ((-1) ** col) * self.data[0][col] * Matrix(minor).determinant()
        return det

    @classmethod
    def identity(cls, size):
        return cls([[1 if i == j else 0 for j in range(size)] for i in range(size)])