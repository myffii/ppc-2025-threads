#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace nasedkin_e_strassen_algorithm_seq {

    class StrassenSequential : public ppc::core::Task {
    public:
        explicit StrassenSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
        bool PreProcessingImpl() override;
        bool ValidationImpl() override;
        bool RunImpl() override;
        bool PostProcessingImpl() override;

    private:
        // Вспомогательные методы для алгоритма Штрассена
        static std::vector<std::vector<int>> AddMatrices(const std::vector<std::vector<int>>& a,
                                                         const std::vector<std::vector<int>>& b);
        static std::vector<std::vector<int>> SubtractMatrices(const std::vector<std::vector<int>>& a,
                                                              const std::vector<std::vector<int>>& b);
        static void SplitMatrix(const std::vector<std::vector<int>>& parent, std::vector<std::vector<int>>& child,
                                int row_start, int col_start);
        static void MergeMatrix(std::vector<std::vector<int>>& parent, const std::vector<std::vector<int>>& child,
                                int row_start, int col_start);
        static std::vector<std::vector<int>> StandardMultiply(const std::vector<std::vector<int>>& a,
                                                              const std::vector<std::vector<int>>& b);
        static std::vector<std::vector<int>> StrassenMultiply(const std::vector<std::vector<int>>& a,
                                                              const std::vector<std::vector<int>>& b);

        // Данные задачи
        std::vector<std::vector<int>> input_matrix_a_, input_matrix_b_;
        std::vector<std::vector<int>> output_matrix_;
        int matrix_size_{};
    };

}  // namespace nasedkin_e_strassen_algorithm_seq