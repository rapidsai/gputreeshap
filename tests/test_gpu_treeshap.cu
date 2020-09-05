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
#include <gtest/gtest.h>
#include <limits>
#include <vector>
#include "../GPUTreeShap/gpu_treeshap.h"

using namespace gpu_treeshap;  // NOLINT

class DenseDatasetWrapper {
  const float* data;
  int num_rows;
  int num_cols;

 public:
  DenseDatasetWrapper(const float* data, int num_rows, int num_cols)
      : data(data), num_rows(num_rows), num_cols(num_cols) {}
  __device__ float GetElement(size_t row_idx, size_t col_idx) const {
    return data[row_idx * num_cols + col_idx];
  }
  __host__ __device__ size_t NumRows() const { return num_rows; }
  __host__ __device__ size_t NumCols() const { return num_cols; }
};

// Test a simple tree and compare output to hand computed values
TEST(GPUTreeShap, BasicPaths) {
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<gpu_treeshap::PathElement> path{
      gpu_treeshap::PathElement{0, -1, 0, -inf, inf, false, 1.0f, 0.5f},
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
  GPUTreeShap(X, path.begin(), path.end(), 1, phis.data().get());
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

// Test a tree with features occurring multiple times in a path
TEST(GPUTreeShap, BasicPathsWithDuplicates) {
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<gpu_treeshap::PathElement> path{
      {0, -1, 0, -inf, 0.0f, false, 1.0f, 3.0f},
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
  GPUTreeShap(X, path.begin(), path.end(), 1, phis.data().get());
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
class TestGroupPath : public gpu_treeshap::detail::GroupPath {
 public:
  __device__ TestGroupPath(const gpu_treeshap::detail::ContiguousGroup& g,
                           float zero_fraction, float one_fraction)
      : gpu_treeshap::detail::GroupPath(g, zero_fraction, one_fraction) {}
  using gpu_treeshap::detail::GroupPath::pweight_;
  using gpu_treeshap::detail::GroupPath::unique_depth_;
};

template <typename DatasetT>
__global__ void TestExtendKernel(
    DatasetT X, size_t num_path_elements,
    const gpu_treeshap::PathElement* path_elements) {
  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();
  auto group =
      cooperative_groups::tiled_partition<32, cooperative_groups::thread_block>(
          block);
  if (group.thread_rank() >= num_path_elements) return;

  // Test first training instance
  cooperative_groups::coalesced_group active_group =
      cooperative_groups::coalesced_threads();
  gpu_treeshap::PathElement e = path_elements[active_group.thread_rank()];
  float one_fraction = gpu_treeshap::detail::GetOneFraction(e, X, 0);
  float zero_fraction = e.zero_fraction;
  auto labelled_group = gpu_treeshap::detail::active_labeled_partition(0);
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
  one_fraction = gpu_treeshap::detail::GetOneFraction(e, X, 1);
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
  std::vector<gpu_treeshap::PathElement> path;
  path.emplace_back(
      gpu_treeshap::PathElement{0, -1, 0, -inf, 0.0f, false, 1.0f, 1.0f});
  path.emplace_back(
      gpu_treeshap::PathElement{0, 0, 0, 0.5f, inf, false, 3.0f / 5, 1.0f});
  path.emplace_back(
      gpu_treeshap::PathElement{0, 1, 0, 0.5f, inf, false, 2.0f / 3, 1.0f});
  path.emplace_back(
      gpu_treeshap::PathElement{0, 2, 0, -inf, 0.5f, false, 1.0f / 2, 1.0f});
  thrust::device_vector<gpu_treeshap::PathElement> device_path(path);
  thrust::device_vector<float> data =
      std::vector<float>({1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f});
  DenseDatasetWrapper X(data.data().get(), 2, 3);
  TestExtendKernel<<<1, 32>>>(X, 4, device_path.data().get());
}
template <typename DatasetT>
__global__ void TestExtendMultipleKernel(
    DatasetT X, size_t n_first, size_t n_second,
    const gpu_treeshap::PathElement* path_elements) {
  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();
  auto warp =
      cooperative_groups::tiled_partition<32, cooperative_groups::thread_block>(
          block);
  if (warp.thread_rank() >= n_first + n_second) return;
  cooperative_groups::coalesced_group active_group =
      cooperative_groups::coalesced_threads();
  int label = warp.thread_rank() >= n_first;
  auto labeled_group = gpu_treeshap::detail::active_labeled_partition(label);
  gpu_treeshap::PathElement e = path_elements[warp.thread_rank()];

  // Test first training instance
  float one_fraction = gpu_treeshap::detail::GetOneFraction(e, X, 0);
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
  std::vector<gpu_treeshap::PathElement> path;
  path.emplace_back(
      gpu_treeshap::PathElement{0, -1, 0, -inf, 0.0f, false, 1.0f, 1.0f});
  path.emplace_back(
      gpu_treeshap::PathElement{0, 0, 0, 0.5f, inf, false, 3.0f / 5, 1.0f});
  path.emplace_back(
      gpu_treeshap::PathElement{0, 1, 0, 0.5f, inf, false, 2.0f / 3, 1.0f});
  path.emplace_back(
      gpu_treeshap::PathElement{0, 2, 0, -inf, 0.5f, false, 1.0f / 2, 1.0f});
  // Add the first three elements again
  path.emplace_back(path[0]);
  path.emplace_back(path[1]);
  path.emplace_back(path[2]);

  thrust::device_vector<gpu_treeshap::PathElement> device_path(path);
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
  auto labelled_partition =
      gpu_treeshap::detail::active_labeled_partition(label);

  if (label == 3) {
    assert(labelled_partition.size() == 5);
    assert(labelled_partition.thread_rank() == warp.thread_rank());
  } else if (label == 6) {
    assert(labelled_partition.size() == 32 - 5);
    assert(labelled_partition.thread_rank() == warp.thread_rank() - 5);
  }

  if (warp.thread_rank() % 2 == 1) {
    auto labelled_partition2 =
        gpu_treeshap::detail::active_labeled_partition(label);
    if (label == 3) {
      assert(labelled_partition2.size() == 2);
      assert(labelled_partition2.thread_rank() == warp.thread_rank() / 2);
    } else if (label == 6) {
      assert(labelled_partition2.size() == 14);
      assert(labelled_partition2.thread_rank() == (warp.thread_rank() / 2) - 2);
    }
  } else {
    auto labelled_partition2 =
        gpu_treeshap::detail::active_labeled_partition(label);
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
  auto bin_packing = gpu_treeshap::detail::BFDBinPacking(counts, 3);
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
  bin_packing = gpu_treeshap::detail::BFDBinPacking(counts, 10);
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
  auto bin_packing = gpu_treeshap::detail::NFBinPacking(counts, 5);
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
  auto bin_packing = gpu_treeshap::detail::FFDBinPacking(counts, 5);
  EXPECT_EQ(bin_packing[0], 1u);
  EXPECT_EQ(bin_packing[1], 1u);
  EXPECT_EQ(bin_packing[2], 2u);
  EXPECT_EQ(bin_packing[3], 0u);
  EXPECT_EQ(bin_packing[4], 0u);
}

__global__ void TestContiguousGroup() {
  int label = threadIdx.x > 2 && threadIdx.x < 6 ? 1 : threadIdx.x >= 6 ? 2 : 0;

  auto group = gpu_treeshap::detail::active_labeled_partition(label);

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
