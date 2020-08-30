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
#include <set>

namespace gpu_treeshap {
/*! An element of a unique path through a decision tree. */
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

  /*! Unique path index. */
  size_t path_idx;
  /*! Feature of this split, -1 indicates bias term. */
  int64_t feature_idx;
  /*! Indicates class for multiclass problems. */
  int group;
  /*! Feature values >= lower and < upper flow down this path. */
  float feature_lower_bound;
  float feature_upper_bound;
  /*! Do missing values flow down this path? */
  bool is_missing_branch;
  /*! Probability of following this path when feature_idx is not in the active
   * set. */
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
  size_t tid =
      size_t(block.size()) * block.group_index().x + block.thread_rank();
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

template <typename PathVectorT, typename SizeVectorT, typename DeviceAllocatorT>
void GetBinSegmentsDevice(const PathVectorT& paths, const SizeVectorT& bin_map,
                          SizeVectorT* bin_segments) {
  bin_segments->resize(bin_map.size() + 1);
  auto counting = thrust::make_counting_iterator(0llu);
  auto d_paths = paths.data().get();
  auto d_bin_segments = bin_segments->data().get();
  auto d_bin_map = bin_map.data();
  thrust::for_each_n(counting, paths.size(), [=] __device__(size_t idx) {
    auto path_idx = d_paths[idx].path_idx;
    atomicAdd(reinterpret_cast<unsigned long long*>(d_bin_segments) +  // NOLINT
                  d_bin_map[path_idx],
              1);
  });
  DeviceAllocatorT alloc;
  thrust::exclusive_scan(thrust::cuda::par(alloc), bin_segments->begin(),
                         bin_segments->end(), bin_segments->begin());
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


template <typename PathVectorT, typename SizeVectorT, typename DeviceAllocatorT>
void SortPathsDevice(PathVectorT* paths, const SizeVectorT& bin_map) {
  auto d_bin_map = bin_map.data();
  DeviceAllocatorT alloc;
  thrust::sort(thrust::cuda::par(alloc), paths->begin(), paths->end(),
               [=] __device__(const PathElement& a, const PathElement& b) {
                 size_t a_bin = d_bin_map[a.path_idx];
                 size_t b_bin = d_bin_map[b.path_idx];
                 if (a_bin < b_bin) return true;
                 if (b_bin < a_bin) return false;

                 if (a.path_idx < b.path_idx) return true;
                 if (b.path_idx < a.path_idx) return false;

                 if (a.feature_idx < b.feature_idx) return true;
                 if (b.feature_idx < a.feature_idx) return false;
                 return false;
               });
}

using kv = std::pair<size_t, int>;

struct BFDCompare {
  bool operator()(const kv& lhs, const kv& rhs) const {
    if (lhs.second == rhs.second) {
      return lhs.first < rhs.first;
    }
    return lhs.second < rhs.second;
  }
};

// Best Fit Decreasing bin packing
// Efficient O(nlogn) implementation with balanced tree using std::set
inline std::vector<size_t> BFDBinPacking(
    const std::map<size_t, int>& counts, int bin_limit = 32) {
  std::vector<kv> path_lengths(counts.begin(), counts.end());
  std::sort(path_lengths.begin(), path_lengths.end(),
            [&](const kv& a, const kv& b) {
              std::greater<> op;
              return op(a.second, b.second);
            });

  // map unique_id -> bin
  std::vector<size_t> bin_map((--counts.end())->first + 1);
  std::set<kv, BFDCompare> bin_capacities;
  bin_capacities.insert({bin_capacities.size(), bin_limit});
  for (auto pair : path_lengths) {
    int new_size = pair.second;
    auto itr = bin_capacities.lower_bound({0, new_size});
    // Does not fit in any bin
    if (itr == bin_capacities.end()) {
      size_t new_bin_idx = bin_capacities.size();
      bin_capacities.insert({new_bin_idx, bin_limit - new_size});
      bin_map[pair.first] = new_bin_idx;
    } else {
      kv entry = *itr;
      entry.second -= new_size;
      bin_map[pair.first] = entry.first;
      bin_capacities.erase(itr);
      bin_capacities.insert(entry);
    }
  }

  return bin_map;
}

// First Fit Decreasing bin packing
// Inefficient O(n^2) implementation
inline std::vector<size_t> FFDBinPacking(
  const std::map<size_t, int>& counts, int bin_limit = 32) {
  using kv = std::pair<size_t, int>;
  std::vector<kv> path_lengths(counts.begin(), counts.end());
  std::sort(path_lengths.begin(), path_lengths.end(),
    [&](const kv& a, const kv& b) {
    std::greater<> op;
    return op(a.second, b.second);
  });

  // map unique_id -> bin
  std::vector<size_t> bin_map((--counts.end())->first + 1);
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

// Next Fit bin packing
// O(n) implementation
inline std::vector<size_t> NFBinPacking(
    const std::map<size_t, int>& counts, int bin_limit = 32) {
  std::vector<size_t> bin_map((--counts.end())->first + 1);
  size_t current_bin = 0;
  size_t current_capacity = bin_limit;
  for (auto pair : counts) {
    int new_size = pair.second;
    if (new_size <= current_capacity) {
      current_capacity -= new_size;
      bin_map[pair.first] = current_bin;
    } else {
      current_capacity = bin_limit - new_size;
      bin_map[pair.first] = ++current_bin;
    }
  }
  return bin_map;
}

inline std::map<size_t, int> GetPathLengths(
    const std::vector<PathElement>& paths) {
  std::map<size_t, int> lengths;
  int count = 0;
  size_t current_path = 0;
  for (const PathElement& e : paths) {
    if (e.path_idx == current_path) {
      count++;
    } else {
      lengths[current_path] = count;
      count = 1;
      current_path = e.path_idx;
    }
  }

  lengths[current_path] = count;
  return lengths;
}
};  // namespace detail

/*!
 * Compute feature contributions on the GPU given a set of unique paths through a tree ensemble
 * and a dataset. Uses device memory proportional to the tree ensemble size.
 *
 * \tparam  DeviceAllocatorT  Optional thrust style allocator.
 *
 * \param           X           Thin wrapper over a dataset allocated in device memory. X should be
 *                              trivially copyable as a kernel parameter (i.e. contain only pointers
 *                              to actual data) and must implement the methods
 *                              NumRows()/NumCols()/GetElement(size_t row_idx, size_t col_idx) as
 *                              __device__ functions. GetElement may return NaN where the feature
 *                              value is missing.
 * \param           paths       Vector of paths, where separate paths are delineated by
 *                              PathElement.path_idx. Each unique path should contain 1 root with
 *                              feature_idx = -1 and zero_fraction = 1.0. The ordering of path
 *                              elements inside a unique path does not matter - the result will be
 *                              the same. Paths may contain duplicate features. See the PathElement
 *                              class for more information.
 * \param           num_groups  Number of output groups. In multiclass classification the algorithm
 *                              outputs feature contributions per output class.
 * \param [in,out]  phis_out    Device memory buffer for returning the feature contributions. Must
 *                              be of size X.NumRows() * (X.NumCols() + 1) * num_groups. The last
 *                              feature column contains the bias term. Feature contributions can be
 *                              retrieved by phis_out[(row_idx * num_groups + group) * (X.NumCols() +
 *                              1) + feature_idx]. Results are added to the input buffer without
 *                              zeroing memory - do not pass uninitialised memory.
 *
 * \tparam  DatasetT  User-specified dataset container.
 */
template <typename DeviceAllocatorT = thrust::device_allocator<int>,
          typename DatasetT>
void GPUTreeShap(DatasetT X, const std::vector<PathElement>& paths,
                 size_t num_groups, float* phis_out) {
  if (X.NumRows() == 0 || X.NumCols() == 0 || paths.empty()) return;

  // Sort paths by length and feature
  auto deduplicated_paths = detail::DeduplicatePaths(paths);
  std::map<size_t, int> counts = detail::GetPathLengths(paths);
  auto bin_map = detail::BFDBinPacking(counts);
  using size_vector = thrust::device_vector<
      size_t, typename DeviceAllocatorT::template rebind<size_t>::other>;
  using path_vector = thrust::device_vector<
      PathElement,
      typename DeviceAllocatorT::template rebind<PathElement>::other>;
  size_vector device_bin_map(bin_map);
  path_vector device_paths(deduplicated_paths);
  detail::SortPathsDevice<path_vector, size_vector, DeviceAllocatorT>(
      &device_paths, device_bin_map);
  size_vector device_bin_segments;
  detail::GetBinSegmentsDevice<path_vector, size_vector, DeviceAllocatorT>(
      device_paths, device_bin_map, &device_bin_segments);
  detail::ComputeShap(X, device_bin_segments, device_paths, num_groups,
                      phis_out);
}

};  // namespace gpu_treeshap
