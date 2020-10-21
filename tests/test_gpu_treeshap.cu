/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <GPUTreeShap/gpu_treeshap.h>
#include <cooperative_groups.h>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <vector>
#include <numeric>
#include "../GPUTreeShap/gpu_treeshap.h"

using namespace gpu_treeshap;  // NOLINT

class DenseDatasetWrapper {
  const float* data;
  int num_rows;
  int num_cols;

 public:
  DenseDatasetWrapper() = default;
  DenseDatasetWrapper(const float* data, int num_rows, int num_cols)
      : data(data), num_rows(num_rows), num_cols(num_cols) {}
  __device__ float GetElement(size_t row_idx, size_t col_idx) const {
    return data[row_idx * num_cols + col_idx];
  }
  __host__ __device__ size_t NumRows() const { return num_rows; }
  __host__ __device__ size_t NumCols() const { return num_cols; }
};

class TestDataset {
 public:
  std::vector<float> host_data;
  thrust::device_vector<float> device_data;
  size_t num_rows;
  size_t num_cols;
  TestDataset(size_t num_rows, size_t num_cols, size_t seed,
              float missing_fraction = 0.25)
      : num_rows(num_rows), num_cols(num_cols) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis;
    std::bernoulli_distribution bern(missing_fraction);
    host_data.resize(num_rows * num_cols);
    for (auto& e : host_data) {
      e = bern(gen) ? std::numeric_limits<float>::quiet_NaN() : dis(gen);
    }
    device_data = host_data;
  }
  DenseDatasetWrapper GetDeviceWrapper() {
    return DenseDatasetWrapper(device_data.data().get(), num_rows, num_cols);
  }
};

void GenerateModel(std::vector<PathElement>* model, int group, size_t max_depth,
                   size_t num_features, size_t num_paths, std::mt19937* gen,
                   float max_v) {
  std::uniform_real_distribution<float> value_dis(-max_v, max_v);
  std::uniform_int_distribution<int64_t> feature_dis(0, num_features - 1);
  std::bernoulli_distribution bern_dis;
  const float inf = std::numeric_limits<float>::infinity();
  size_t base_path_idx = model->empty() ? 0 : model->back().path_idx + 1;
  float z = std::pow(0.5, 1.0 / max_depth);
  for (auto i = 0ull; i < num_paths; i++) {
    float v = value_dis(*gen);
    model->emplace_back(
        PathElement{base_path_idx + i, -1, group, -inf, inf, false, 1.0, v});
    for (auto j = 0ull; j < max_depth; j++) {
      float lower_bound = -inf;
      float upper_bound = inf;
      // If the input feature value x_i is a uniform rv in [0,1)
      // We want a 50% chance of it reaching the end of this path
      // Each test should succeed with probability 0.5^(1/max_depth)
      std::uniform_real_distribution<float> bound_dis(0.0, 2.0 - 2 * z);
      if (bern_dis(*gen)) {
        lower_bound = bound_dis(*gen);
      } else {
        upper_bound = 1.0f - bound_dis(*gen);
      }
      // Don't make the zero fraction too small
      std::uniform_real_distribution<float> zero_fraction_dis(0.05, 1.0);
      model->emplace_back(
          PathElement{base_path_idx + i, feature_dis(*gen), group, lower_bound,
                      upper_bound, bern_dis(*gen), zero_fraction_dis(*gen), v});
    }
  }
}

std::vector<PathElement> GenerateEnsembleModel(size_t num_groups,
                                               size_t max_depth,
                                               size_t num_features,
                                               size_t num_paths, size_t seed,
                                               float max_v = 1.0f) {
  std::mt19937 gen(seed);
  std::vector<PathElement> model;
  for (auto group = 0llu; group < num_groups; group++) {
    GenerateModel(&model, group, max_depth, num_features, num_paths, &gen,
                  max_v);
  }
  return model;
}

std::vector<float> Predict(const std::vector<PathElement>& model,
                           const TestDataset& X, size_t num_groups) {
  std::vector<float> predictions(X.num_rows * num_groups);
  for (auto i = 0ull; i < X.num_rows; i++) {
    const float* row = X.host_data.data() + i * X.num_cols;
    float current_v = model.front().v;
    size_t current_path_idx = model.front().path_idx;
    int current_group = model.front().group;
    bool valid = true;
    for (const auto& e : model) {
      if (e.path_idx != current_path_idx) {
        if (valid) {
          predictions[i * num_groups + current_group] += current_v;
        }
        current_v = e.v;
        current_path_idx = e.path_idx;
        current_group = e.group;
        valid = true;
      }

      if (e.feature_idx != -1) {
        float fval = row[e.feature_idx];
        if (std::isnan(fval)) {
          valid = valid && e.is_missing_branch;
        } else if (fval < e.feature_lower_bound ||
                   fval >= e.feature_upper_bound) {
          valid = false;
        }
      }
    }
    if (valid) {
      predictions[i * num_groups + current_group] += current_v;
    }
  }

  return predictions;
}

class ShapSumTest : public ::testing::TestWithParam<
                        std::tuple<size_t, size_t, size_t, size_t, size_t>> {};

TEST_P(ShapSumTest, ShapSum) {
  size_t num_rows, num_features, num_groups, max_depth, num_paths;
  std::tie(num_rows, num_features, num_groups, max_depth, num_paths) =
      GetParam();
  auto model =
      GenerateEnsembleModel(num_groups, max_depth, num_features, num_paths, 78);
  TestDataset test_data(num_rows, num_features, 22);
  auto margin = Predict(model, test_data, num_groups);

  auto X = test_data.GetDeviceWrapper();

  thrust::device_vector<float> phis(X.NumRows() * (X.NumCols() + 1) *
                                    num_groups);
  GPUTreeShap(X, model.begin(), model.end(), num_groups, phis.data().get(),
              phis.size());
  thrust::host_vector<float> result(phis);
  std::vector<float > tmp(result.begin(), result.end());
  std::vector<float> sum(num_rows * num_groups);
  for (auto i = 0ull; i < num_rows; i++) {
    for (auto j = 0ull; j < num_features + 1; j++) {
      for (auto group = 0ull; group < num_groups; group++) {
        size_t result_index = IndexPhi(i, num_groups, group, num_features, j);
        sum[i * num_groups + group] += result[result_index];
      }
    }
  }
  for (auto i = 0ull; i < sum.size(); i++) {
    ASSERT_NEAR(sum[i], margin[i], 1e-3);
  }
}

class ShapInteractionsSumTest : public ::testing::TestWithParam<
                        std::tuple<size_t, size_t, size_t, size_t, size_t>> {};
TEST_P(ShapInteractionsSumTest, ShapInteractionsSum) {
  size_t num_rows, num_features, num_groups, max_depth, num_paths;
  std::tie(num_rows, num_features, num_groups, max_depth, num_paths) =
      GetParam();
  auto model =
      GenerateEnsembleModel(num_groups, max_depth, num_features, num_paths, 78);
  TestDataset test_data(num_rows, num_features, 22);

  auto X = test_data.GetDeviceWrapper();

  thrust::device_vector<float> phis(X.NumRows() * (X.NumCols() + 1) *
                                    num_groups);
  thrust::device_vector<float> phis_interactions(
      X.NumRows() * (X.NumCols() + 1) * (X.NumCols() + 1) * num_groups);
  GPUTreeShap(X, model.begin(), model.end(), num_groups, phis.data().get(),
              phis.size());
  GPUTreeShapInteractions(X, model.begin(), model.end(), num_groups,
                          phis_interactions.data().get(),
                          phis_interactions.size());
  thrust::host_vector<float> interactions_result(phis_interactions);
  std::vector<float> sum(phis.size());
  for (auto row_idx = 0ull; row_idx < num_rows; row_idx++) {
    for (auto group = 0ull; group < num_groups; group++) {
      for (auto i = 0ull; i < num_features + 1; i++) {
        for (auto j = 0ull; j < num_features + 1; j++) {
          size_t result_index = IndexPhiInteractions(row_idx, num_groups, group,
                                                     num_features, i, j);
          sum[IndexPhi(row_idx, num_groups, group, num_features, i)] +=
              interactions_result[result_index];
        }
      }
    }
  }

  thrust::host_vector<float> phis_host(phis);
  for (auto i = 0ull; i < sum.size(); i++) {
    ASSERT_NEAR(sum[i], phis_host[i], 1e-3);
  }
}

class ShapTaylorInteractionsSumTest : public ::testing::TestWithParam<
                        std::tuple<size_t, size_t, size_t, size_t, size_t>> {};
TEST_P(ShapTaylorInteractionsSumTest, ShapTaylorInteractionsSum) {
  size_t num_rows, num_features, num_groups, max_depth, num_paths;
  std::tie(num_rows, num_features, num_groups, max_depth, num_paths) =
    GetParam();
  auto model =
    GenerateEnsembleModel(num_groups, max_depth, num_features, num_paths, 78);
  TestDataset test_data(num_rows, num_features, 22);

  auto X = test_data.GetDeviceWrapper();

  auto margin = Predict(model, test_data, num_groups);

  thrust::device_vector<float> phis_interactions(
    X.NumRows() * (X.NumCols() + 1) * (X.NumCols() + 1) * num_groups);
  GPUTreeShapTaylorInteractions(X, model.begin(), model.end(), num_groups,
    phis_interactions.data().get(),
    phis_interactions.size());
  thrust::host_vector<float> interactions_result(phis_interactions);
  std::vector<float> sum(margin.size());
  for (auto row_idx = 0ull; row_idx < num_rows; row_idx++) {
    for (auto group = 0ull; group < num_groups; group++) {
      for (auto i = 0ull; i < num_features + 1; i++) {
        for (auto j = 0ull; j < num_features + 1; j++) {
          size_t result_index = IndexPhiInteractions(row_idx, num_groups, group,
            num_features, i, j);
          sum[row_idx * num_groups + group] +=
            interactions_result[result_index];
        }
      }
    }
  }

  for (auto i = 0ull; i < sum.size(); i++) {
    ASSERT_NEAR(sum[i], margin[i], 1e-3);
  }
}

std::string PrintTestName(
    const testing::TestParamInfo<ShapSumTest::ParamType>& info) {
  std::string name = "nrow" + std::to_string(std::get<0>(info.param)) + "_";
  name += "nfeat" + std::to_string(std::get<1>(info.param)) + "_";
  name += "ngroup" + std::to_string(std::get<2>(info.param)) + "_";
  name += "mdepth" + std::to_string(std::get<3>(info.param)) + "_";
  name += "npaths" + std::to_string(std::get<4>(info.param));
  return name;
}

// Generate a bunch of random models and check the shap results sum up to the
// predictions
size_t test_num_rows[] = {1, 10, 100, 1000};
size_t test_num_features[] = {1, 5, 8, 31};
size_t test_num_groups[] = {1, 5};
size_t test_max_depth[] = {1, 8, 20};
size_t test_num_paths[] = {1, 10};
INSTANTIATE_TEST_CASE_P(ShapInstantiation, ShapSumTest,
                        testing::Combine(testing::ValuesIn(test_num_rows),
                                         testing::ValuesIn(test_num_features),
                                         testing::ValuesIn(test_num_groups),
                                         testing::ValuesIn(test_max_depth),
                                         testing::ValuesIn(test_num_paths)),
                        PrintTestName);
INSTANTIATE_TEST_CASE_P(ShapInteractionsInstantiation, ShapInteractionsSumTest,
                        testing::Combine(testing::ValuesIn(test_num_rows),
                                         testing::ValuesIn(test_num_features),
                                         testing::ValuesIn(test_num_groups),
                                         testing::ValuesIn(test_max_depth),
                                         testing::ValuesIn(test_num_paths)),
                        PrintTestName);

INSTANTIATE_TEST_CASE_P(ShapTaylorInteractionsInstantiation,
                        ShapTaylorInteractionsSumTest,
                        testing::Combine(testing::ValuesIn(test_num_rows),
                                         testing::ValuesIn(test_num_features),
                                         testing::ValuesIn(test_num_groups),
                                         testing::ValuesIn(test_max_depth),
                                         testing::ValuesIn(test_num_paths)),
                        PrintTestName);

TEST(GPUTreeShap, PathTooLong) {
  std::vector<PathElement> path(33);
  path[0] = PathElement(0, -1, 0, 0, 0, 0, 0, 0);
  for (auto i = 1ull; i < path.size(); i++) {
    path[i] = PathElement(0, i, 0, 0, 0, 0, 0, 0);
  }

  thrust::device_vector<float> data =
      std::vector<float>({1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f});
  DenseDatasetWrapper X(data.data().get(), 2, 3);
  thrust::device_vector<float> phis(X.NumRows() * (X.NumCols() + 1));
  EXPECT_THROW(
      {
        try {
          GPUTreeShap(X, path.begin(), path.end(), 1, phis.data().get(),
                      phis.size());
        } catch (const std::invalid_argument& e) {
          EXPECT_STREQ("Tree depth must be <= 32", e.what());
          throw;
        }
      },
      std::invalid_argument);

  phis.resize((X.NumRows() * (X.NumCols() + 1) * (X.NumCols() + 1)));
  EXPECT_THROW(
      {
        try {
          GPUTreeShapInteractions(X, path.begin(), path.end(), 1,
                                  phis.data().get(), phis.size());
        } catch (const std::invalid_argument& e) {
          EXPECT_STREQ("Tree depth must be <= 32", e.what());
          throw;
        }
      },
      std::invalid_argument);
}

TEST(GPUTreeShap, PathVIncorrect) {
  std::vector<PathElement> path = {
      PathElement(0, -1, 0, 0.0f, 0.0f, false, 0.0, 1.0f),
      {0, 0, 0, 0.0f, 0.0f, false, 0.0f, 0.5f}};

  thrust::device_vector<float> data = std::vector<float>({1.0f});
  DenseDatasetWrapper X(data.data().get(), 1, 1);
  thrust::device_vector<float> phis(X.NumRows() * (X.NumCols() + 1));
  EXPECT_THROW(
      {
        try {
          GPUTreeShap(X, path.begin(), path.end(), 1, phis.data().get(),
                      phis.size());
        } catch (const std::invalid_argument& e) {
          EXPECT_STREQ("Leaf value v should be the same across a single path",
                       e.what());
          throw;
        }
      },
      std::invalid_argument);

  phis.resize((X.NumRows() * (X.NumCols() + 1) * (X.NumCols() + 1)));
  EXPECT_THROW(
      {
        try {
          GPUTreeShapInteractions(X, path.begin(), path.end(), 1,
                                  phis.data().get(), phis.size());
        } catch (const std::invalid_argument& e) {
          EXPECT_STREQ("Leaf value v should be the same across a single path",
                       e.what());
          throw;
        }
      },
      std::invalid_argument);
}

TEST(GPUTreeShap, PhisIncorrectLength) {
  std::vector<PathElement> path = {
      PathElement(0, -1, 0, 0.0f, 0.0f, false, 0.0, 0.0f)};

  thrust::device_vector<float> data =
      std::vector<float>({1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f});
  DenseDatasetWrapper X(data.data().get(), 2, 3);
  thrust::device_vector<float> phis((X.NumRows() * (X.NumCols() + 1)) - 1);
  EXPECT_THROW(
      {
        try {
          GPUTreeShap(X, path.begin(), path.end(), 1, phis.data().get(),
                      phis.size());
        } catch (const std::invalid_argument& e) {
          EXPECT_STREQ(
              "phis_out must be at least of size X.NumRows() * (X.NumCols() + "
              "1) * num_groups",
              e.what());
          throw;
        }
      },
      std::invalid_argument);

  phis.resize((X.NumRows() * (X.NumCols() + 1) * (X.NumCols() + 1)) - 1);
  EXPECT_THROW(
      {
        try {
          GPUTreeShapInteractions(X, path.begin(), path.end(), 1,
                                  phis.data().get(), phis.size());
        } catch (const std::invalid_argument& e) {
          EXPECT_STREQ(
              "phis_out must be at least of size X.NumRows() * (X.NumCols() + "
              "1)  * (X.NumCols() + 1) * num_groups",
              e.what());
          throw;
        }
      },
      std::invalid_argument);
}

TEST(GPUTreeShap, PhisIncorrectMemory) {
  std::vector<PathElement> path = {
      PathElement(0, -1, 0, 0.0f, 0.0f, false, 0.0, 0.0f)};
  thrust::device_vector<float> data =
      std::vector<float>({1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f});
  DenseDatasetWrapper X(data.data().get(), 2, 3);
  std::vector<float> phis(X.NumRows() * (X.NumCols() + 1));
  EXPECT_THROW(
      {
        try {
          GPUTreeShap(X, path.begin(), path.end(), 1, phis.data(), phis.size());
        } catch (const std::invalid_argument& e) {
          EXPECT_STREQ("phis_out must be device accessible", e.what());
          throw;
        }
      },
      std::invalid_argument);
}

// Test a simple tree and compare output to xgb shap values
// 0:[f0<0.5] yes=1,no=2,missing=1,gain=1.63333321,cover=5
//  1:leaf=-1,cover=2
//  2:[f1<0.5] yes=3,no=4,missing=3,gain=2.04166675,cover=3
//    3:leaf=-1,cover=1
//    4:[f2<0.5] yes=5,no=6,missing=5,gain=0.125,cover=2
//      5:leaf=1,cover=1
//      6:leaf=0.5,cover=1
TEST(GPUTreeShap, BasicPaths) {
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<PathElement> path{
      PathElement{0, -1, 0, -inf, inf, false, 1.0f, 0.5f},
      {0, 0, 0, 0.5f, inf, false, 0.6f, 0.5f},
      {0, 1, 0, 0.5f, inf, false, 2.0f / 3, 0.5f},
      {0, 2, 0, 0.5f, inf, false, 0.5f, 0.5f},
      {1, -1, 0, -inf, 0.0f, false, 1.0f, 1.0f},
      {1, 0, 0, 0.5f, inf, false, 0.6f, 1.0f},
      {1, 1, 0, 0.5f, inf, false, 2.0f / 3, 1.0f},
      {1, 2, 0, -inf, 0.5f, false, 0.5f, 1.0f},
      {2, -1, 0, -inf, 0.0f, false, 1.0f, -1},
      {2, 0, 0, 0.5f, inf, false, 0.6f, -1.0f},
      {2, 1, 0, -inf, 0.5f, false, 1.0f / 3, -1.0f},
      {3, -1, 0, -inf, 0.0f, false, 1.0f, -1.0f},
      {3, 0, 0, -inf, 0.5f, false, 0.4f, -1.0f}};
  thrust::device_vector<float> data =
      std::vector<float>({1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f});
  DenseDatasetWrapper X(data.data().get(), 2, 3);
  size_t num_trees = 1;
  thrust::device_vector<float> phis(X.NumRows() * (X.NumCols() + 1));
  GPUTreeShap(X, path.begin(), path.end(), 1, phis.data().get(), phis.size());
  thrust::host_vector<float> result(phis);
  // First instance
  EXPECT_NEAR(result[0], 0.6277778f * num_trees, 1e-5);
  EXPECT_NEAR(result[1], 0.5027776f * num_trees, 1e-5);
  EXPECT_NEAR(result[2], 0.1694444f * num_trees, 1e-5);
  EXPECT_NEAR(result[3], -0.3f * num_trees, 1e-5);
  // Second instance
  EXPECT_NEAR(result[4], 0.24444449f * num_trees, 1e-5);
  EXPECT_NEAR(result[5], -1.005555f * num_trees, 1e-5);
  EXPECT_NEAR(result[6], 0.0611111f * num_trees, 1e-5);
  EXPECT_NEAR(result[7], -0.3f * num_trees, 1e-5);
}

TEST(GPUTreeShap, BasicPathsInteractions) {
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<PathElement> path{
      PathElement{0, -1, 0, -inf, inf, false, 1.0f, 0.5f},
      {0, 0, 0, 0.5f, inf, false, 0.6f, 0.5f},
      {0, 1, 0, 0.5f, inf, false, 2.0f / 3, 0.5f},
      {0, 2, 0, 0.5f, inf, false, 0.5f, 0.5f},
      {1, -1, 0, -inf, 0.0f, false, 1.0f, 1.0f},
      {1, 0, 0, 0.5f, inf, false, 0.6f, 1.0f},
      {1, 1, 0, 0.5f, inf, false, 2.0f / 3, 1.0f},
      {1, 2, 0, -inf, 0.5f, false, 0.5f, 1.0f},
      {2, -1, 0, -inf, 0.0f, false, 1.0f, -1},
      {2, 0, 0, 0.5f, inf, false, 0.6f, -1.0f},
      {2, 1, 0, -inf, 0.5f, false, 1.0f / 3, -1.0f},
      {3, -1, 0, -inf, 0.0f, false, 1.0f, -1.0f},
      {3, 0, 0, -inf, 0.5f, false, 0.4f, -1.0f}};
  thrust::device_vector<float> data =
      std::vector<float>({1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f});
  DenseDatasetWrapper X(data.data().get(), 2, 3);
  thrust::device_vector<float> phis(X.NumRows() * (X.NumCols() + 1) *
                                    (X.NumCols() + 1));
  GPUTreeShapInteractions(X, path.begin(), path.end(), 1, phis.data().get(),
                          phis.size());
  std::vector<float> result(phis.begin(), phis.end());
  std::vector<float> expected_result = {
      0.46111116,  0.125,       0.04166666,  0.,          0.125,
      0.34444442,  0.03333333,  0.,          0.04166666,  0.03333335,
      0.09444444,  0.,          0.,          0.,          0.,
      -0.3,        0.47222224,  0.1083333,   -0.04166666, 0.,
      0.10833332,  0.35555553,  -0.03333333, 0.,          -0.04166666,
      -0.03333332, -0.09444447, 0.,          0.,          0.,
      0.,          -0.3};
  for (auto i = 0ull; i < result.size(); i++) {
    EXPECT_NEAR(result[i], expected_result[i], 1e-5);
  }
}

// Test a tree with features occurring multiple times in a path
TEST(GPUTreeShap, BasicPathsWithDuplicates) {
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<PathElement> path{{0, -1, 0, -inf, 0.0f, false, 1.0f, 3.0f},
                                {0, 0, 0, 0.5f, inf, false, 2.0f / 3, 3.0f},
                                {0, 0, 0, 1.5f, inf, false, 0.5f, 3.0f},
                                {0, 0, 0, 2.5f, inf, false, 0.5f, 3.0f},
                                {1, -1, 0, -inf, 0.0f, false, 1.0f, 2.0f},
                                {1, 0, 0, 0.5f, inf, false, 2.0f / 3.0f, 2.0f},
                                {1, 0, 0, 1.5f, inf, false, 0.5f, 2.0f},
                                {1, 0, 0, -inf, 2.5f, false, 0.5f, 2.0f},
                                {2, -1, 0, -inf, 0.0f, false, 1.0f, 1.0f},
                                {2, 0, 0, 0.5f, inf, false, 2.0f / 3.0f, 1.0f},
                                {2, 0, 0, -inf, 1.5f, false, 0.5f, 1.0f},
                                {3, -1, 0, -inf, 0.0f, false, 1.0f, -1.0f},
                                {3, 0, 0, -inf, 0.5f, false, 1.0f / 3, -1.0f}};
  thrust::device_vector<float> data = std::vector<float>({2.0f});
  DenseDatasetWrapper X(data.data().get(), 1, 1);
  size_t num_trees = 1;
  thrust::device_vector<float> phis(X.NumRows() * (X.NumCols() + 1));
  GPUTreeShap(X, path.begin(), path.end(), 1, phis.data().get(), phis.size());
  thrust::host_vector<float> result(phis);
  // First instance
  EXPECT_FLOAT_EQ(result[0], 1.1666666f * num_trees);
  EXPECT_FLOAT_EQ(result[1], 0.83333337f * num_trees);
}

__device__ bool FloatApproximatelyEqual(float a, float b) {
  const float kEps = 1e-5;
  return fabs(a - b) < kEps;
}

// Expose pweight for testing
class TestGroupPath : public detail::GroupPath {
 public:
  __device__ TestGroupPath(const detail::ContiguousGroup& g,
                           float zero_fraction, float one_fraction)
      : detail::GroupPath(g, zero_fraction, one_fraction) {}
  using detail::GroupPath::pweight_;
  using detail::GroupPath::unique_depth_;
};

template <typename DatasetT>
__global__ void TestExtendKernel(DatasetT X, size_t num_path_elements,
                                 const PathElement* path_elements) {
  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();
  auto group =
      cooperative_groups::tiled_partition<32, cooperative_groups::thread_block>(
          block);
  if (group.thread_rank() >= num_path_elements) return;

  // Test first training instance
  cooperative_groups::coalesced_group active_group =
      cooperative_groups::coalesced_threads();
  PathElement e = path_elements[active_group.thread_rank()];
  float one_fraction = detail::GetOneFraction(e, X, 0);
  float zero_fraction = e.zero_fraction;
  auto labelled_group = detail::active_labeled_partition(0);
  TestGroupPath path(labelled_group, zero_fraction, one_fraction);
  path.Extend();
  assert(path.unique_depth_ == 1);
  if (active_group.thread_rank() == 0) {
    assert(FloatApproximatelyEqual(path.pweight_, 0.3f));
  } else if (active_group.thread_rank() == 1) {
    assert(FloatApproximatelyEqual(path.pweight_, 0.5f));
  } else {
    assert(FloatApproximatelyEqual(path.pweight_, 0.0f));
  }

  path.Extend();
  assert(path.unique_depth_ == 2);
  if (active_group.thread_rank() == 0) {
    assert(FloatApproximatelyEqual(path.pweight_, 0.133333f));
  } else if (active_group.thread_rank() == 1) {
    assert(FloatApproximatelyEqual(path.pweight_, 0.21111f));
  } else if (active_group.thread_rank() == 2) {
    assert(FloatApproximatelyEqual(path.pweight_, 0.33333f));
  } else {
    assert(FloatApproximatelyEqual(path.pweight_, 0.0f));
  }

  path.Extend();
  assert(path.unique_depth_ == 3);
  if (active_group.thread_rank() == 0) {
    assert(FloatApproximatelyEqual(path.pweight_, 0.05f));
  } else if (active_group.thread_rank() == 1) {
    assert(FloatApproximatelyEqual(path.pweight_, 0.086111f));
  } else if (active_group.thread_rank() == 2) {
    assert(FloatApproximatelyEqual(path.pweight_, 0.147222f));
  } else if (active_group.thread_rank() == 3) {
    assert(FloatApproximatelyEqual(path.pweight_, 0.25f));
  } else {
    assert(FloatApproximatelyEqual(path.pweight_, 0.0f));
  }

  float unwound_sum = path.UnwoundPathSum();

  if (active_group.thread_rank() == 1) {
    assert(FloatApproximatelyEqual(unwound_sum, 0.63888f));
  } else if (active_group.thread_rank() == 2) {
    assert(FloatApproximatelyEqual(unwound_sum, 0.61666f));
  } else if (active_group.thread_rank() == 3) {
    assert(FloatApproximatelyEqual(unwound_sum, 0.67777f));
  } else if (active_group.thread_rank() > 3) {
    assert(FloatApproximatelyEqual(unwound_sum, 0.0f));
  }

  // Test second training instance
  one_fraction = detail::GetOneFraction(e, X, 1);
  TestGroupPath path2(labelled_group, zero_fraction, one_fraction);
  path2.Extend();
  assert(path2.unique_depth_ == 1);
  if (active_group.thread_rank() == 0) {
    assert(FloatApproximatelyEqual(path2.pweight_, 0.3f));
  } else if (active_group.thread_rank() == 1) {
    assert(FloatApproximatelyEqual(path2.pweight_, 0.5f));
  } else {
    assert(FloatApproximatelyEqual(path2.pweight_, 0.0f));
  }

  path2.Extend();
  assert(path2.unique_depth_ == 2);
  if (active_group.thread_rank() == 0) {
    assert(FloatApproximatelyEqual(path2.pweight_, 0.133333f));
  } else if (active_group.thread_rank() == 1) {
    assert(FloatApproximatelyEqual(path2.pweight_, 0.11111f));
  } else if (active_group.thread_rank() == 2) {
    assert(FloatApproximatelyEqual(path2.pweight_, 0.0f));
  } else {
    assert(FloatApproximatelyEqual(path2.pweight_, 0.0f));
  }

  path2.Extend();
  assert(path2.unique_depth_ == 3);
  if (active_group.thread_rank() == 0) {
    assert(FloatApproximatelyEqual(path2.pweight_, 0.05f));
  } else if (active_group.thread_rank() == 1) {
    assert(FloatApproximatelyEqual(path2.pweight_, 0.06111f));
  } else if (active_group.thread_rank() == 2) {
    assert(FloatApproximatelyEqual(path2.pweight_, 0.05555f));
  } else if (active_group.thread_rank() == 3) {
    assert(FloatApproximatelyEqual(path2.pweight_, 0.0f));
  } else {
    assert(FloatApproximatelyEqual(path2.pweight_, 0.0f));
  }

  unwound_sum = path2.UnwoundPathSum();

  if (active_group.thread_rank() == 1) {
    assert(FloatApproximatelyEqual(unwound_sum, 0.22222f));
  } else if (active_group.thread_rank() == 2) {
    assert(FloatApproximatelyEqual(unwound_sum, 0.61666f));
  } else if (active_group.thread_rank() == 3) {
    assert(FloatApproximatelyEqual(unwound_sum, 0.244444f));
  } else if (active_group.thread_rank() > 3) {
    assert(FloatApproximatelyEqual(unwound_sum, 0.0f));
  }
}

TEST(GPUTreeShap, Extend) {
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<PathElement> path;
  path.emplace_back(PathElement{0, -1, 0, -inf, 0.0f, false, 1.0f, 1.0f});
  path.emplace_back(PathElement{0, 0, 0, 0.5f, inf, false, 3.0f / 5, 1.0f});
  path.emplace_back(PathElement{0, 1, 0, 0.5f, inf, false, 2.0f / 3, 1.0f});
  path.emplace_back(PathElement{0, 2, 0, -inf, 0.5f, false, 1.0f / 2, 1.0f});
  thrust::device_vector<PathElement> device_path(path);
  thrust::device_vector<float> data =
      std::vector<float>({1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f});
  DenseDatasetWrapper X(data.data().get(), 2, 3);
  TestExtendKernel<<<1, 32>>>(X, 4, device_path.data().get());
}
template <typename DatasetT>
__global__ void TestExtendMultipleKernel(DatasetT X, size_t n_first,
                                         size_t n_second,
                                         const PathElement* path_elements) {
  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();
  auto warp =
      cooperative_groups::tiled_partition<32, cooperative_groups::thread_block>(
          block);
  if (warp.thread_rank() >= n_first + n_second) return;
  cooperative_groups::coalesced_group active_group =
      cooperative_groups::coalesced_threads();
  int label = warp.thread_rank() >= n_first;
  auto labeled_group = detail::active_labeled_partition(label);
  PathElement e = path_elements[warp.thread_rank()];

  // Test first training instance
  float one_fraction = detail::GetOneFraction(e, X, 0);
  float zero_fraction = e.zero_fraction;
  TestGroupPath path(labeled_group, zero_fraction, one_fraction);
  assert(path.unique_depth_ == 0);
  if (labeled_group.thread_rank() == 0) {
    assert(FloatApproximatelyEqual(path.pweight_, 1.0f));
  } else {
    assert(FloatApproximatelyEqual(path.pweight_, 0.0f));
  }

  path.Extend();
  assert(path.unique_depth_ == 1);
  if (labeled_group.thread_rank() == 0) {
    assert(FloatApproximatelyEqual(path.pweight_, 0.3f));
  } else if (labeled_group.thread_rank() == 1) {
    assert(FloatApproximatelyEqual(path.pweight_, 0.5f));
  } else {
    assert(FloatApproximatelyEqual(path.pweight_, 0.0f));
  }

  path.Extend();
  assert(path.unique_depth_ == 2);
  if (labeled_group.thread_rank() == 0) {
    assert(FloatApproximatelyEqual(path.pweight_, 0.133333f));
  } else if (labeled_group.thread_rank() == 1) {
    assert(FloatApproximatelyEqual(path.pweight_, 0.21111f));
  } else if (labeled_group.thread_rank() == 2) {
    assert(FloatApproximatelyEqual(path.pweight_, 0.33333f));
  } else {
    assert(FloatApproximatelyEqual(path.pweight_, 0.0f));
  }

  // Extend the first group only
  if (label == 0) {
    path.Extend();
    assert(path.unique_depth_ == 3);
    if (labeled_group.thread_rank() == 0) {
      assert(FloatApproximatelyEqual(path.pweight_, 0.05f));
    } else if (labeled_group.thread_rank() == 1) {
      assert(FloatApproximatelyEqual(path.pweight_, 0.086111f));
    } else if (labeled_group.thread_rank() == 2) {
      assert(FloatApproximatelyEqual(path.pweight_, 0.147222f));
    } else if (labeled_group.thread_rank() == 3) {
      assert(FloatApproximatelyEqual(path.pweight_, 0.25f));
    } else {
      assert(FloatApproximatelyEqual(path.pweight_, 0.0f));
    }
  } else {
    assert(path.unique_depth_ == 2);
    if (labeled_group.thread_rank() == 0) {
      assert(FloatApproximatelyEqual(path.pweight_, 0.133333f));
    } else if (labeled_group.thread_rank() == 1) {
      assert(FloatApproximatelyEqual(path.pweight_, 0.21111f));
    } else if (labeled_group.thread_rank() == 2) {
      assert(FloatApproximatelyEqual(path.pweight_, 0.33333f));
    } else {
      assert(FloatApproximatelyEqual(path.pweight_, 0.0f));
    }
  }
  if (label == 0) {
    float unwound_sum = path.UnwoundPathSum();

    if (labeled_group.thread_rank() == 1) {
      assert(FloatApproximatelyEqual(unwound_sum, 0.63888f));
    } else if (labeled_group.thread_rank() == 2) {
      assert(FloatApproximatelyEqual(unwound_sum, 0.61666f));
    } else if (labeled_group.thread_rank() == 3) {
      assert(FloatApproximatelyEqual(unwound_sum, 0.67777f));
    } else if (labeled_group.thread_rank() > 3) {
      assert(FloatApproximatelyEqual(unwound_sum, 0.0f));
    }
  }
}

TEST(GPUTreeShap, ExtendMultiplePaths) {
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<PathElement> path;
  path.emplace_back(PathElement{0, -1, 0, -inf, 0.0f, false, 1.0f, 1.0f});
  path.emplace_back(PathElement{0, 0, 0, 0.5f, inf, false, 3.0f / 5, 1.0f});
  path.emplace_back(PathElement{0, 1, 0, 0.5f, inf, false, 2.0f / 3, 1.0f});
  path.emplace_back(PathElement{0, 2, 0, -inf, 0.5f, false, 1.0f / 2, 1.0f});
  // Add the first three elements again
  path.emplace_back(path[0]);
  path.emplace_back(path[1]);
  path.emplace_back(path[2]);

  thrust::device_vector<PathElement> device_path(path);
  thrust::device_vector<float> data =
      std::vector<float>({1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f});
  DenseDatasetWrapper X(data.data().get(), 2, 3);
  TestExtendMultipleKernel<<<1, 32>>>(X, 4, 3, device_path.data().get());
}

__global__ void TestActiveLabeledPartition() {
  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();
  auto warp =
      cooperative_groups::tiled_partition<32, cooperative_groups::thread_block>(
          block);
  int label = warp.thread_rank() < 5 ? 3 : 6;
  auto labelled_partition = detail::active_labeled_partition(label);

  if (label == 3) {
    assert(labelled_partition.size() == 5);
    assert(labelled_partition.thread_rank() == warp.thread_rank());
  } else if (label == 6) {
    assert(labelled_partition.size() == 32 - 5);
    assert(labelled_partition.thread_rank() == warp.thread_rank() - 5);
  }

  if (warp.thread_rank() % 2 == 1) {
    auto labelled_partition2 = detail::active_labeled_partition(label);
    if (label == 3) {
      assert(labelled_partition2.size() == 2);
      assert(labelled_partition2.thread_rank() == warp.thread_rank() / 2);
    } else if (label == 6) {
      assert(labelled_partition2.size() == 14);
      assert(labelled_partition2.thread_rank() == (warp.thread_rank() / 2) - 2);
    }
  } else {
    auto labelled_partition2 = detail::active_labeled_partition(label);
    if (label == 3) {
      assert(labelled_partition2.size() == 3);
      assert(labelled_partition2.thread_rank() == warp.thread_rank() / 2);
    } else if (label == 6) {
      assert(labelled_partition2.size() == 13);
      assert(labelled_partition2.thread_rank() == (warp.thread_rank() / 2) - 3);
    }
  }
}

TEST(GPUTreeShap, ActiveLabeledPartition) {
  TestActiveLabeledPartition<<<1, 32>>>();
  EXPECT_EQ(cudaDeviceSynchronize(), 0);
}

TEST(GPUTreeShap, BFDBinPacking) {
  thrust::device_vector<int> counts(3);
  counts[0] = 2;
  counts[1] = 2;
  counts[2] = 1;
  auto bin_packing = detail::BFDBinPacking(counts, 3);
  EXPECT_EQ(bin_packing[0], 0u);
  EXPECT_EQ(bin_packing[1], 1u);
  EXPECT_EQ(bin_packing[2], 0u);

  counts.clear();
  counts.resize(12);
  counts[0] = 3;
  counts[1] = 3;
  counts[2] = 3;
  counts[3] = 3;
  counts[4] = 3;
  counts[5] = 3;
  counts[6] = 2;
  counts[7] = 2;
  counts[8] = 2;
  counts[9] = 2;
  counts[10] = 2;
  counts[11] = 2;
  bin_packing = detail::BFDBinPacking(counts, 10);
  EXPECT_EQ(bin_packing[0], 0u);
  EXPECT_EQ(bin_packing[1], 0u);
  EXPECT_EQ(bin_packing[2], 0u);
  EXPECT_EQ(bin_packing[3], 1u);
  EXPECT_EQ(bin_packing[4], 1u);
  EXPECT_EQ(bin_packing[5], 1u);
  EXPECT_EQ(bin_packing[6], 2u);
  EXPECT_EQ(bin_packing[7], 2u);
  EXPECT_EQ(bin_packing[8], 2u);
  EXPECT_EQ(bin_packing[9], 2u);
  EXPECT_EQ(bin_packing[10], 2u);
  EXPECT_EQ(bin_packing[11], 3u);
}

TEST(GPUTreeShap, NFBinPacking) {
  thrust::device_vector<int> counts(4);
  counts[0] = 3;
  counts[1] = 3;
  counts[2] = 1;
  counts[3] = 2;
  auto bin_packing = detail::NFBinPacking(counts, 5);
  EXPECT_EQ(bin_packing[0], 0u);
  EXPECT_EQ(bin_packing[1], 1u);
  EXPECT_EQ(bin_packing[2], 1u);
  EXPECT_EQ(bin_packing[3], 2u);
}

TEST(GPUTreeShap, FFDBinPacking) {
  thrust::device_vector<int> counts(5);
  counts[0] = 3;
  counts[1] = 2;
  counts[2] = 3;
  counts[3] = 4;
  counts[4] = 1;
  auto bin_packing = detail::FFDBinPacking(counts, 5);
  EXPECT_EQ(bin_packing[0], 1u);
  EXPECT_EQ(bin_packing[1], 1u);
  EXPECT_EQ(bin_packing[2], 2u);
  EXPECT_EQ(bin_packing[3], 0u);
  EXPECT_EQ(bin_packing[4], 0u);
}

__global__ void TestContiguousGroup() {
  int label = threadIdx.x > 2 && threadIdx.x < 6 ? 1 : threadIdx.x >= 6 ? 2 : 0;

  auto group = detail::active_labeled_partition(label);

  if (label == 1) {
    assert(group.size() == 3);
    assert(group.thread_rank() == threadIdx.x - 3);
    int up = group.shfl_up(threadIdx.x, 1);
    if (group.thread_rank() > 0) {
      assert(up == threadIdx.x - 1);
    }
    assert(group.shfl(threadIdx.x, 2) == 5);
  }
}

TEST(GPUTreeShap, ContiguousGroup) {
  TestContiguousGroup<<<1, 32>>>();
  EXPECT_EQ(cudaDeviceSynchronize(), 0);
}

TEST(GPUTreeShap, ShapDeterminism) {
  size_t num_rows = 100;
  size_t num_features = 100;
  size_t num_groups = 1;
  size_t max_depth = 10;
  size_t num_paths = 1000;
  size_t samples = 100;
  auto model =
      GenerateEnsembleModel(num_groups, max_depth, num_features, num_paths, 78);
  TestDataset test_data(num_rows, num_features, 22, 1e15);

  auto X = test_data.GetDeviceWrapper();

  thrust::device_vector<float> reference_phis(X.NumRows() * (X.NumCols() + 1) *
                                              num_groups);
  GPUTreeShap(X, model.begin(), model.end(), num_groups,
              reference_phis.data().get(), reference_phis.size());

  for (auto i = 0ull; i < samples; i++) {
    thrust::device_vector<float> phis(reference_phis.size());
    GPUTreeShap(X, model.begin(), model.end(), num_groups,
                phis.data().get(), phis.size());
    ASSERT_TRUE(thrust::equal(reference_phis.begin(), reference_phis.end(),
                              phis.begin()));
  }
}

TEST(GPUTreeShap, ShapInteractionsDeterminism) {
  size_t num_rows = 100;
  size_t num_features = 10;
  size_t num_groups = 1;
  size_t max_depth = 3;
  size_t num_paths = 1000;
  size_t samples = 10;
  auto model =
    GenerateEnsembleModel(num_groups, max_depth, num_features, num_paths, 78);
  TestDataset test_data(num_rows, num_features, 22, 1e15);

  auto X = test_data.GetDeviceWrapper();

  thrust::device_vector<float> reference_phis(X.NumRows() * (X.NumCols() + 1) *
                                              (X.NumCols() + 1) * num_groups);
  GPUTreeShapInteractions(X, model.begin(), model.end(), num_groups,
    reference_phis.data().get(), reference_phis.size());

  for (auto i = 0ull; i < samples; i++) {
    thrust::device_vector<float> phis(reference_phis.size());
    GPUTreeShapInteractions(X, model.begin(), model.end(), num_groups,
      phis.data().get(), phis.size());
    ASSERT_TRUE(thrust::equal(reference_phis.begin(), reference_phis.end(),
      phis.begin()));
  }
}

// Example from page 10 section 4.1
// Dhamdhere, Kedar, Ashish Agarwal, and Mukund Sundararajan. "The Shapley
// Taylor Interaction Index." arXiv preprint arXiv:1902.05622 (2019).
TEST(GPUTreeShap, TaylorInteractionsPaperExample) {
  const float inf = std::numeric_limits<float>::infinity();
  float c = 3.0f;
  std::vector<PathElement> path{
      PathElement{0, -1, 0, -inf, inf, false, 1.0f, 1.0f},
      {0, 0, 0, 0.5f, inf, false, 0.0f, 1.0f},
      {1, -1, 0, -inf, inf, false, 1.0f, 1.0f},
      {1, 1, 0, 0.5f, inf, false, 0.0f, 1.0f},
      {2, -1, 0, -inf, inf, false, 1.0f, 1.0f},
      {2, 2, 0, 0.5f, inf, false, 0.0f, 1.0f},
      {3, -1, 0, -inf, inf, false, 1.0f, c},
      {3, 0, 0, 0.5f, inf, false, 0.0f, c},
      {3, 1, 0, 0.5f, inf, false, 0.0f, c},
      {3, 2, 0, 0.5f, inf, false, 0.0f, c},
  };
  thrust::device_vector<float> data = std::vector<float>({1.0f, 1.0f, 1.0f});
  DenseDatasetWrapper X(data.data().get(), 1, 3);
  thrust::device_vector<float> interaction_phis(
      X.NumRows() * (X.NumCols() + 1) * (X.NumCols() + 1));
  GPUTreeShapTaylorInteractions(X, path.begin(), path.end(), 1,
                                interaction_phis.data().get(),
                                interaction_phis.size());

  std::vector<float> interactions_result(interaction_phis.begin(),
                                         interaction_phis.end());
  std::vector<float> expected_result = {1.0, 0.5, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0,
                                        0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  ASSERT_EQ(interaction_phis, expected_result);
}

TEST(GPUTreeShap, TaylorInteractionsBasic) {
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<PathElement> path{
      PathElement{0, -1, 0, -inf, inf, false, 1.0f, 2.0f},
      {0, 0, 0, 0.5f, inf, false, 0.25f, 2.0f},
      {0, 1, 0, 0.5f, inf, false, 0.5f, 2.0f},
      {0, 2, 0, 0.5f, inf, false, 0.6f, 2.0f},
      {0, 3, 0, 0.5f, inf, false, 1.0f, 2.0f},
  };
  thrust::device_vector<float> data =
      std::vector<float>({1.0f, 1.0f, 1.0f, 1.0f});
  DenseDatasetWrapper X(data.data().get(), 1, 4);
  thrust::device_vector<float> interaction_phis(
      X.NumRows() * (X.NumCols() + 1) * (X.NumCols() + 1));
  GPUTreeShapTaylorInteractions(X, path.begin(), path.end(), 1,
                                interaction_phis.data().get(),
                                interaction_phis.size());

  thrust::host_vector<float> interactions_result(interaction_phis);
  float sum =
      std::accumulate(interaction_phis.begin(), interaction_phis.end(), 0.0f);

  ASSERT_FLOAT_EQ(sum, 2.0f);
}
