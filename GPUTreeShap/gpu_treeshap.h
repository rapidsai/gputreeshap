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

#pragma once
#include <cooperative_groups.h>
#include <thrust/device_allocator.h>
#include <thrust/device_vector.h>
#include <algorithm>
#include <functional>
#include <map>
#include <utility>
#include <vector>

namespace gpu_treeshap {
struct PathElement {
  PathElement(size_t path_idx, int64_t feature_idx, int group,
              float feature_lower_bound, float feature_upper_bound,
              bool is_missing_branch, float zero_fraction, float v)
      : path_idx(path_idx),
        feature_idx(feature_idx),
        group(group),
        feature_lower_bound(feature_lower_bound),
        feature_upper_bound(feature_upper_bound),
        is_missing_branch(is_missing_branch),
        zero_fraction(zero_fraction),
        v(v) {
    assert(feature_lower_bound <= feature_upper_bound);
  }

  PathElement() = default;

  size_t path_idx;      // Unique path index
  int64_t feature_idx;  // Feature of this split, -1 indicates bias term
  int group;            // Indicates class for multiclass problems
  // Feature values >= lower and < upper flow down this path
  float feature_lower_bound;
  float feature_upper_bound;
  bool is_missing_branch;  // Do missing values flow down this path?
  float zero_fraction;
  float v;  // Leaf weight at the end of the path
};

namespace detail {
__forceinline__ __device__ unsigned int lanemask32_lt() {
  unsigned int lanemask32_lt;
  asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask32_lt));
  return (lanemask32_lt);
}

// Like a coalesced group, except we can make the assumption that all threads in
// a group are next to each other. This makes shuffle operations much cheaper.
class ContiguousGroup {
 public:
  __device__ ContiguousGroup(uint32_t mask) : mask_(mask) {}

  __device__ uint32_t size() const { return __popc(mask_); }
  __device__ uint32_t thread_rank() const {
    return __popc(mask_ & lanemask32_lt());
  }
  template <typename T>
  __device__ T shfl(T val, uint32_t src) const {
    return __shfl_sync(mask_, val, src + __ffs(mask_) - 1);
  }
  template <typename T>
  __device__ T shfl_up(T val, uint32_t delta) const {
    return __shfl_up_sync(mask_, val, delta);
  }

  uint32_t mask_;
};

// Separate the active threads by labels
// This functionality is available in cuda 11.0 on cc >=7.0
// We reimplement for backwards compatibility
// Assumes partitions are contiguous
inline __device__ ContiguousGroup active_labeled_partition(int label) {
#if __CUDA_ARCH__ >= 700
  uint32_t subgroup_mask = __match_any_sync(__activemask(), label);
#else
  uint32_t subgroup_mask = 0;
  uint32_t active_mask = __activemask();
  for (int i = 0; i < 32;) {
    int current_label = __shfl_sync(active_mask, label, i);
    uint32_t ballot = __ballot_sync(active_mask, label == current_label);
    if (label == current_label) {
      subgroup_mask = ballot;
    }
    uint32_t completed_mask =
        (1 << (32 - __clz(ballot))) - 1;  // Threads that have finished
    // Find the start of the next group, mask off completed threads from active
    // threads Then use ffs - 1 to find the position of the next group
    int next_i = __ffs(active_mask & ~completed_mask) - 1;
    if (next_i == -1) break;  // -1 indicates all finished
    assert(next_i > i);  // Prevent infinite loops when the constraints not met
    i = next_i;
  }
#endif
  return ContiguousGroup(subgroup_mask);
}

template <typename DatasetT>
__device__ float GetOneFraction(const PathElement& e, DatasetT X,
                                size_t row_idx) {
  // First element in path (bias term) is always zero
  if (e.feature_idx == -1) return 0.0;
  // Test the split
  // Does the training instance continue down this path if the feature is
  // present?
  float val = X.GetElement(row_idx, e.feature_idx);
  if (isnan(val)) {
    return e.is_missing_branch;
  }
  return val >= e.feature_lower_bound && val < e.feature_upper_bound;
}

// Group of threads where each thread holds a path element
class GroupPath {
 protected:
  const ContiguousGroup& g_;
  // These are combined so we can communicate them in a single 64 bit shuffle
  // instruction
  float zero_one_fraction_[2];
  float pweight_;
  int unique_depth_;

 public:
  __device__ GroupPath(const ContiguousGroup& g, float zero_fraction,
                       float one_fraction)
      : g_(g),
        zero_one_fraction_{zero_fraction, one_fraction},
        pweight_(g.thread_rank() == 0 ? 1.0f : 0.0f),
        unique_depth_(0) {}

  // Cooperatively extend the path with a group of threads
  // Each thread maintains pweight for its path element in register
  __device__ void Extend() {
    unique_depth_++;

    // Broadcast the zero and one fraction from the newly added path element
    // Combine 2 shuffle operations into 64 bit word
    const size_t rank = g_.thread_rank();
    const float inv_unique_depth =
        __fdividef(1.0f, static_cast<float>(unique_depth_ + 1));
    uint64_t res = g_.shfl(*reinterpret_cast<uint64_t*>(&zero_one_fraction_),
                           unique_depth_);
    const float new_zero_fraction = reinterpret_cast<float*>(&res)[0];
    const float new_one_fraction = reinterpret_cast<float*>(&res)[1];
    float left_pweight = g_.shfl_up(pweight_, 1);

    // pweight of threads with rank < unique_depth_ is 0
    // We use max(x,0) to avoid using a branch
    // pweight_ *=
    // new_zero_fraction * max(unique_depth_ - rank, 0llu) * inv_unique_depth;
    pweight_ = __fmul_rn(
        __fmul_rn(pweight_, new_zero_fraction),
        __fmul_rn(max(unique_depth_ - rank, size_t(0)), inv_unique_depth));

    // pweight_  += new_one_fraction * left_pweight * rank * inv_unique_depth;
    pweight_ = __fmaf_rn(__fmul_rn(new_one_fraction, left_pweight),
                         __fmul_rn(rank, inv_unique_depth), pweight_);
  }

  // Each thread unwinds the path for its feature and returns the sum
  // rank 0 returns a bias term - the product of all pz
  __device__ float UnwoundPathSum() {
    float next_one_portion = g_.shfl(pweight_, unique_depth_);
    float total = 0.0f;
    const float zero_frac_div_unique_depth = __fdividef(
        zero_one_fraction_[0], static_cast<float>(unique_depth_ + 1));
    for (int i = unique_depth_ - 1; i >= 0; i--) {
      float ith_pweight = g_.shfl(pweight_, i);
      float precomputed =
          __fmul_rn((unique_depth_ - i), zero_frac_div_unique_depth);
      const float tmp =
          __fdividef(__fmul_rn(next_one_portion, unique_depth_ + 1), i + 1);
      total = __fmaf_rn(tmp, zero_one_fraction_[1], total);
      next_one_portion = __fmaf_rn(-tmp, precomputed, ith_pweight);
      float numerator =
          __fmul_rn(__fsub_rn(1.0f, zero_one_fraction_[1]), ith_pweight);
      total += __fdividef(numerator, precomputed);
    }

    if (g_.thread_rank() == 0) {
      return pweight_ * (unique_depth_ + 1);
    }
    return total;
  }
};

template <typename DatasetT>
__global__ void ShapKernel(DatasetT X, size_t warps_per_row,
                           const PathElement* path_elements,
                           const size_t* bin_segments, size_t num_groups,
                           float* phis) {
  // Partition work
  // Each warp processes a training instance applied to a path
  cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();
  auto warp =
      cooperative_groups::tiled_partition<32, cooperative_groups::thread_block>(
          block);
  size_t tid = block.size() * block.group_index().x + block.thread_rank();
  size_t warp_rank = tid / warp.size();
  if (warp_rank >= warps_per_row * X.NumRows()) return;
  size_t row_idx = warp_rank / warps_per_row;
  size_t bin_idx = warp_rank % warps_per_row;
  size_t path_start = bin_segments[bin_idx];
  size_t path_end = bin_segments[bin_idx + 1];
  assert(path_end - path_start <= warp.size());
  assert(num_groups > 0);
  if (warp.thread_rank() >= path_end - path_start) return;
  const PathElement& e = path_elements[path_start + warp.thread_rank()];
  auto labelled_group = active_labeled_partition(e.path_idx);
  size_t unique_path_length = labelled_group.size();
  float one_fraction = GetOneFraction(e, X, row_idx);
  float zero_fraction = e.zero_fraction;
  GroupPath path(labelled_group, zero_fraction, one_fraction);

  // Extend the path
  for (auto unique_depth = 1ull; unique_depth < unique_path_length;
       unique_depth++) {
    path.Extend();
  }

  float sum = path.UnwoundPathSum();
  float* phis_row = &phis[(row_idx * num_groups + e.group) * (X.NumCols() + 1)];

  if (e.feature_idx == -1) {
    // Bias term is the expected value of this path, given no data
    atomicAdd(phis_row + X.NumCols(), sum * e.v);
  } else {
    atomicAdd(phis_row + e.feature_idx,
              sum * (one_fraction - zero_fraction) * e.v);
  }
}

template <typename DatasetT, typename SizeTAllocatorT, typename PathAllocatorT>
void ComputeShap(
    DatasetT X,
    const thrust::device_vector<size_t, SizeTAllocatorT>& bin_segments,
    const thrust::device_vector<PathElement, PathAllocatorT>& path_elements,
    size_t num_groups, float* phis) {
  size_t warps_per_row = bin_segments.size() - 1;
  const int kBlockThreads = 512;
  const int warps_per_block = kBlockThreads / 32;
  const uint32_t grid_size = static_cast<uint32_t>(
      (warps_per_row * X.NumRows() + warps_per_block - 1) / warps_per_block);

  ShapKernel<<<grid_size, kBlockThreads>>>(
      X, warps_per_row, path_elements.data().get(), bin_segments.data().get(),
      num_groups, phis);
}

inline std::vector<size_t> GetBinSegments(
    const std::vector<PathElement>& paths,
    const std::map<size_t, size_t>& bin_map) {
  std::vector<size_t> sizes = {0};
  auto previous = paths.front();
  size_t size = 1;
  for (auto i = 1ull; i < paths.size(); i++) {
    auto& next = paths[i];
    if (bin_map.at(next.path_idx) != bin_map.at(previous.path_idx)) {
      sizes.push_back(size);
    }
    size++;
    previous = next;
  }
  sizes.push_back(size);

  return sizes;
}

inline std::vector<PathElement> DeduplicatePaths(
    const std::vector<PathElement>& paths) {
  std::vector<PathElement> sorted_paths(paths);
  // Sort by feature
  std::sort(sorted_paths.begin(), sorted_paths.end(),
            [&](const PathElement& a, const PathElement& b) {
              if (a.path_idx < b.path_idx) return true;
              if (b.path_idx < a.path_idx) return false;

              if (a.feature_idx < b.feature_idx) return true;
              if (b.feature_idx < a.feature_idx) return false;
              return false;
            });
  std::vector<PathElement> new_paths;
  new_paths.reserve(paths.size());
  auto e = sorted_paths.front();
  for (auto i = 1ull; i < sorted_paths.size(); i++) {
    auto next = sorted_paths[i];
    if (e.path_idx == next.path_idx && e.feature_idx == next.feature_idx) {
      // Combine duplicate features
      e.feature_lower_bound =
          std::max(e.feature_lower_bound, next.feature_lower_bound);
      e.feature_upper_bound =
          std::min(e.feature_upper_bound, next.feature_upper_bound);
      e.is_missing_branch = e.is_missing_branch && next.is_missing_branch;
      e.zero_fraction *= next.zero_fraction;
    } else {
      new_paths.emplace_back(e);
      e = next;
    }
  }
  new_paths.emplace_back(e);
  return new_paths;
}

inline std::vector<PathElement> SortPaths(
    const std::vector<PathElement>& paths,
    const std::map<size_t, size_t>& bin_map) {
  std::vector<PathElement> sorted_paths(paths);
  std::sort(sorted_paths.begin(), sorted_paths.end(),
            [&](const PathElement& a, const PathElement& b) {
              size_t a_bin = bin_map.at(a.path_idx);
              size_t b_bin = bin_map.at(b.path_idx);
              if (a_bin < b_bin) return true;
              if (b_bin < a_bin) return false;

              if (a.path_idx < b.path_idx) return true;
              if (b.path_idx < a.path_idx) return false;

              if (a.feature_idx < b.feature_idx) return true;
              if (b.feature_idx < a.feature_idx) return false;
              return false;
            });
  return sorted_paths;
}

inline std::map<size_t, size_t> FFDBinPacking(
    const std::map<size_t, int>& counts, int bin_limit = 32) {
  using kv = std::pair<size_t, int>;
  std::vector<kv> path_lengths(counts.begin(), counts.end());
  std::sort(path_lengths.begin(), path_lengths.end(),
            [&](const kv& a, const kv& b) {
              std::greater<> op;
              return op(a.second, b.second);
            });

  // map unique_id -> bin
  std::map<size_t, size_t> bin_map;
  std::vector<int> bin_capacities(path_lengths.size(), bin_limit);
  for (auto pair : path_lengths) {
    int new_size = pair.second;
    for (auto j = 0ull; j < bin_capacities.size(); j++) {
      int& capacity = bin_capacities[j];

      if (capacity >= new_size) {
        capacity -= new_size;
        bin_map[pair.first] = j;
        break;
      }
    }
  }

  return bin_map;
}

};  // namespace detail

template <typename DeviceAllocatorT = thrust::device_allocator<int>,
          typename DatasetT>
void GPUTreeShap(DatasetT X, const std::vector<PathElement>& paths,
                 size_t num_groups, float* phis_out) {
  if (X.NumRows() == 0 || X.NumCols() == 0 || paths.empty()) return;
  // Sort paths by length and feature
  auto deduplicated_paths = detail::DeduplicatePaths(paths);
  std::map<size_t, int> counts;
  for (auto& p : paths) {
    counts[p.path_idx]++;
  }
  auto bin_map = detail::FFDBinPacking(counts);
  auto sorted_paths = detail::SortPaths(deduplicated_paths, bin_map);
  auto segments = detail::GetBinSegments(sorted_paths, bin_map);
  // Create allocators for our internal types
  thrust::device_vector<PathElement, typename DeviceAllocatorT::template rebind<
                                         PathElement>::other>
      device_paths(sorted_paths);
  thrust::device_vector<
      size_t, typename DeviceAllocatorT::template rebind<size_t>::other>
      bin_segments(segments);
  detail::ComputeShap(X, bin_segments, device_paths, num_groups, phis_out);
}

};  // namespace gpu_treeshap
