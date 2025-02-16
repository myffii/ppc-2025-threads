#include "seq/nasedkin_e_strassen_algorithm/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>  // Для size_t
#include <vector>
#include <algorithm> // Для std::transform

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::PreProcessingImpl() {
    // Инициализация входных данных
    unsigned int input_size = task_data->inputs_count[0];
    auto* in_ptr_a = reinterpret_cast<int*>(task_data->inputs[0]);  // Первая матрица
    auto* in_ptr_b = reinterpret_cast<int*>(task_data->inputs[1]);  // Вторая матрица

    // Преобразуем входные данные в одномерные векторы
    matrix_size_ = static_cast<int>(std::sqrt(input_size));
    input_matrix_a_.resize(matrix_size_ * matrix_size_);
    input_matrix_b_.resize(matrix_size_ * matrix_size_);

    for (int i = 0; i < matrix_size_ * matrix_size_; ++i) {
        input_matrix_a_[i] = in_ptr_a[i];
        input_matrix_b_[i] = in_ptr_b[i];
    }

    // Проверяем, нужно ли дополнять матрицы до степени двойки
    if ((matrix_size_ & (matrix_size_ - 1)) != 0) {  // Если размер не является степенью двойки
        original_size_ = matrix_size_;
        input_matrix_a_ = PadMatrixToPowerOfTwo(input_matrix_a_);
        input_matrix_b_ = PadMatrixToPowerOfTwo(input_matrix_b_);
        matrix_size_ = static_cast<int>(input_matrix_a_.size());
    } else {
        original_size_ = matrix_size_;
    }

    // Инициализация выходной матрицы
    output_matrix_.resize(matrix_size_ * matrix_size_, 0);
    return true;
}

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::ValidationImpl() {
    unsigned int input_size_a = task_data->inputs_count[0];
    unsigned int input_size_b = task_data->inputs_count[1];
    unsigned int output_size = task_data->outputs_count[0];

    int size_a = static_cast<int>(std::sqrt(input_size_a));
    int size_b = static_cast<int>(std::sqrt(input_size_b));
    int size_output = static_cast<int>(std::sqrt(output_size));

    return (size_a == size_b) && (size_a == size_output);
}

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::RunImpl() {
    // Выполнение алгоритма Штрассена
    output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_);
    return true;
}

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::PostProcessingImpl() {
    // Если матрицы были дополнены, обрезаем результат до исходного размера
    if (original_size_ != matrix_size_) {
        output_matrix_ = TrimMatrixToOriginalSize(output_matrix_, original_size_);
    }

    // Сохранение результата в выходные данные
    auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
    for (size_t i = 0; i < original_size_ * original_size_; ++i) {
        out_ptr[i] = output_matrix_[i];
    }
    return true;
}

std::vector<int> nasedkin_e_strassen_algorithm_seq::StrassenSequential::AddMatrices(
    const std::vector<int>& a, const std::vector<int>& b) {
    int n = static_cast<int>(std::sqrt(a.size()));
    std::vector<int> result(n * n);
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<int>());
    return result;
}

std::vector<int> nasedkin_e_strassen_algorithm_seq::StrassenSequential::SubtractMatrices(
    const std::vector<int>& a, const std::vector<int>& b) {
    int n = static_cast<int>(std::sqrt(a.size()));
    std::vector<int> result(n * n);
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::minus<int>());
    return result;
}

void nasedkin_e_strassen_algorithm_seq::StrassenSequential::SplitMatrix(const std::vector<int>& parent,
                                                                        std::vector<int>& child,
                                                                        int row_start, int col_start) {
    int n = static_cast<int>(std::sqrt(child.size()));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            child[i * n + j] = parent[(row_start + i) * matrix_size_ + (col_start + j)];
        }
    }
}

void nasedkin_e_strassen_algorithm_seq::StrassenSequential::MergeMatrix(std::vector<int>& parent,
                                                                        const std::vector<int>& child,
                                                                        int row_start, int col_start) {
    int n = static_cast<int>(std::sqrt(child.size()));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            parent[(row_start + i) * matrix_size_ + (col_start + j)] = child[i * n + j];
        }
    }
}

std::vector<int> nasedkin_e_strassen_algorithm_seq::StrassenSequential::PadMatrixToPowerOfTwo(
    const std::vector<int>& matrix) {
    size_t original_size = static_cast<size_t>(std::sqrt(matrix.size()));
    size_t new_size = 1;
    while (new_size < original_size) {
        new_size *= 2;
    }

    std::vector<int> padded_matrix(new_size * new_size, 0);
    std::copy(matrix.begin(), matrix.begin() + original_size * original_size, padded_matrix.begin());
    return padded_matrix;
}

std::vector<int> nasedkin_e_strassen_algorithm_seq::StrassenSequential::TrimMatrixToOriginalSize(
    const std::vector<int>& matrix, size_t original_size) {
    return std::vector<int>(matrix.begin(), matrix.begin() + original_size * original_size);
}

std::vector<int> nasedkin_e_strassen_algorithm_seq::StrassenSequential::StandardMultiply(
    const std::vector<int>& a, const std::vector<int>& b) {
    int n = static_cast<int>(std::sqrt(a.size()));
    std::vector<int> result(n * n, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                result[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
    return result;
}

std::vector<int> nasedkin_e_strassen_algorithm_seq::StrassenSequential::StrassenMultiply(
    const std::vector<int>& a, const std::vector<int>& b) {
    int n = static_cast<int>(std::sqrt(a.size()));

    if (n <= 32) {
        return StandardMultiply(a, b);
    }

    int half_size = n / 2;
    std::vector<int> a11(half_size * half_size), a12(half_size * half_size);
    std::vector<int> a21(half_size * half_size), a22(half_size * half_size);
    SplitMatrix(a, a11, 0, 0);
    SplitMatrix(a, a12, 0, half_size);
    SplitMatrix(a, a21, half_size, 0);
    SplitMatrix(a, a22, half_size, half_size);

    std::vector<int> b11(half_size * half_size), b12(half_size * half_size);
    std::vector<int> b21(half_size * half_size), b22(half_size * half_size);
    SplitMatrix(b, b11, 0, 0);
    SplitMatrix(b, b12, 0, half_size);
    SplitMatrix(b, b21, half_size, 0);
    SplitMatrix(b, b22, half_size, half_size);

    std::vector<int> p1 = StrassenMultiply(AddMatrices(a11, a22), AddMatrices(b11, b22));
    std::vector<int> p2 = StrassenMultiply(AddMatrices(a21, a22), b11);
    std::vector<int> p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22));
    std::vector<int> p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11));
    std::vector<int> p5 = StrassenMultiply(AddMatrices(a11, a12), b22);
    std::vector<int> p6 = StrassenMultiply(SubtractMatrices(a21, a11), AddMatrices(b11, b12));
    std::vector<int> p7 = StrassenMultiply(SubtractMatrices(a12, a22), AddMatrices(b21, b22));

    std::vector<int> c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4), p5), p7);
    std::vector<int> c12 = AddMatrices(p3, p5);
    std::vector<int> c21 = AddMatrices(p2, p4);
    std::vector<int> c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3), p2), p6);

    std::vector<int> result(n * n);
    MergeMatrix(result, c11, 0, 0);
    MergeMatrix(result, c12, 0, half_size);
    MergeMatrix(result, c21, half_size, 0);
    MergeMatrix(result, c22, half_size, half_size);

    return result;
}