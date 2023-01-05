class MyVector:

    def __init__(self, origin: list):
        self.array = origin

    def __mul__(self, other):  # перегрузка оператора умножения как векторное произведение
        return (sum([self.array[i] * other[i] for i in range(0, min(self.len, other.len))]))

    def __plus__(self, other):
        result = MyVector(self.array)
        for index in range(min(self.len, other.len)):
            result[index] += other.array[index]

        return result

    def __sub__(self, other):
        result = MyVector(self.array)
        for index in range(min(self.len, other.len)):
            result[index] -= other.array[index]

        return result

    def concatenate(self, other):
        result = MyVector(self.array + other.array)
        return result

    def concatenate_list(self, other):
        return self.array + other.array

    def __getitem__(self, index):
        self.__check_indexes(index)
        return self.array[index]

    def __setitem__(self, index, value):
        self.__check_indexes(index)
        self.array[index] = value

    def __check_indexes(self, index):
        assert isinstance(index, int)  # мы умеем индексироваться только используя указатели целочисленного типа
        assert 0 <= index and index < self.len  # проверяем что сможем достать элемент по индексу

    @property
    def len(self):
        return len(self.array)


class MyMatrix:

    @property
    def shape(self):
        return (len(self.array), self.array[0].len)

    def __init__(self, origin: list()):
        for i in range(len(origin)):
            assert isinstance(origin[i], list)
            assert len(origin[i]) == len(origin[0])

        for i in range(len(origin)):
            for j in range(len(origin[i])):
                assert isinstance(origin[i][j], float) or isinstance(origin[i][j], int)

        self.array = [MyVector(origin[i]) for i in
                      range(len(origin))]  # храню так чтобы удобно делать матричное умножение

    def __getitem__(self, index):
        self.__check_indexes(index)
        return self.array[index[0]][index[1]]

    def __setitem__(self, index, value):
        self.__check_indexes(index)
        assert isinstance(value, int) or isinstance(value, float)
        self.array[index[0]][index[1]] = value

    def scalars(n: int, k: int, scalar: float):
        return MyMatrix([[scalar for j in range(k)] for i in range(n)])

    def transpose(self):
        new_array = [[self.array[i][j] for i in range(self.shape[0])] for j in range(self.shape[1])]
        return MyMatrix(new_array)

    def concatenate(self, other):
        assert len(self.array) == len(other.array)
        return_matrix = MyMatrix([self.array[i].concatenate_list(other.array[i]) for i in range(self.shape[0])])
        return return_matrix

    def __check_indexes(self, index):
        assert (isinstance(index, list) or isinstance(index, tuple)) and len(
            index) == 2  # проверяем что индекс -- это лист размером 2 или кортеж размером 2
        assert isinstance(index[0], int) and isinstance(index[1], int)  # проверяем что это индексы типа int
        assert min(index[0], index[1]) >= 0 and index[0] < self.shape[0] and index[1] < self.shape[
            1]  # проверяем что умеем обращаться по индексам

    def __mul__(self, other):
        assert self.shape[1] == other.shape[0]

        return_matrix = MyMatrix.scalars(self.shape[0], other.shape[1], 0)
        other_t = other.transpose()

        for i in range(return_matrix.shape[0]):
            for j in range(return_matrix.shape[1]):
                return_matrix[(i, j)] = self.array[i] * other_t.array[j]

        return return_matrix

    def __plus__(self, other):
        assert self.shape[0] == other.shape[0] and self.shape[1] == other.shape[1]

        result = MyMatrix.scalars(self.shape[0], self.shape[1])

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[(i, j)] = self.array[i][j] + other.array[i][j]

        return result

    def __sub__(self, other):
        assert self.shape[0] == other.shape[0] and self.shape[1] == other.shape[1]

        result = MyMatrix.scalars(self.shape[0], self.shape[1], 0)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[(i, j)] = self.array[i][j] - other.array[i][j]

        return result

    def __str__(self):
        result = ""
        result += f"Размерность матрицы: {self.shape[0]}, {self.shape[1]}\n"

        for i in range(0, self.shape[0]):
            for j in range(0, self.shape[1]):
                result += f"{self.array[i][j]} "
            result += "\n"

        return result