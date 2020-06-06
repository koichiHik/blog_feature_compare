/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#ifndef K_OPENCV_FLANN_KDTREE_INDEX_H_
#define K_OPENCV_FLANN_KDTREE_INDEX_H_

// STL
#include <algorithm>
#include <cassert>
#include <cstring>
#include <map>
#include <memory>

// OpenCV
#include <opencv2/flann/allocator.h>
#include <opencv2/flann/dynamic_bitset.h>
#include <opencv2/flann/general.h>
#include <opencv2/flann/matrix.h>
#include <opencv2/flann/random.h>

// Original
#include "heap.h"
#include "nn_index.h"
#include "params.h"
#include "result_set.h"

namespace kcvflann {

/**
 * Randomized kd-tree index
 *
 * Contains the k-d trees and other information for indexing a set of points
 * for nearest-neighbor matching.
 */
template <typename Distance>
class KDTreeIndex : public NNIndex<Distance> {
 public:
  typedef typename Distance::ElementType ElementType;
  typedef typename Distance::ResultType DistanceType;

  KDTreeIndex(const cvflann::Matrix<ElementType>& inputData,
              const NNIndexParams& params = KDTreeIndexParams(), Distance d = Distance())
      : dataset_(inputData), index_params_(params), distance_(d) {
    node_count_ = dataset_.rows;
    histogram_bins_ = dataset_.cols;

    // X. Number of tree. This parameter is set at "params.h"
    trees_ = kcvflann::get_param(index_params_, "trees", 4);
    // tree_roots_ = new NodePtr[trees_];
    // tree_roots_.resize(trees_);
    tree_roots_.resize(trees_);

    // Create a permutable array of indices to the input vectors.
    sorted_node_indices_.resize(node_count_);
    for (size_t i = 0; i < node_count_; ++i) {
      sorted_node_indices_[i] = int(i);
    }
  }

  KDTreeIndex(const KDTreeIndex&);
  KDTreeIndex& operator=(const KDTreeIndex&);

  void BuildIndex() CV_OVERRIDE {
    LOG(INFO) << "Build Index for KDTreeIndex";
    for (int i = 0; i < trees_; i++) {
      // X. Randomize the order of vectors to allow for unbiased sampling.
      cv::randShuffle(sorted_node_indices_);

      // X. Create randomized kd tree.
      int cur_idx = 0;
      tree_roots_[i] = DevideTree(cur_idx, sorted_node_indices_, int(node_count_));
    }
    LOG(INFO) << "End of BuildIndex()";
  }

  size_t GetNodeCount() const CV_OVERRIDE { return node_count_; }

  size_t GetVectorLength() const CV_OVERRIDE { return histogram_bins_; }

  int UsedMemory() const CV_OVERRIDE {
    return int(pool_.usedMemory + pool_.wastedMemory +
               dataset_.rows * sizeof(int));  // pool memory and vind array memory
  }

  flann_algorithm_t GetAlgoType() const CV_OVERRIDE { return FLANN_INDEX_KDTREE; }

  NNIndexParams GetParameters() const CV_OVERRIDE { return index_params_; }

  /**
   * Find set of nearest neighbors to vec. Their indices are stored inside
   * the result object.
   *
   * Params:
   *     result = the result object in which the indices of the nearest-neighbors are stored
   *     vec = the vector for which to search the nearest neighbors
   *     maxCheck = the maximum number of restarts (in a best-bin-first manner)
   */
  void FindNeighbors(kcvflann::ResultSet<DistanceType>& result, const ElementType* vec,
                     const SearchParams& searchParams) CV_OVERRIDE {
    int max_checks = get_param(searchParams, "checks", 32);
    float eps_error = 1 + get_param(searchParams, "eps", 0.0f);
    GetNeighbors(vec, max_checks, eps_error, result);
  }

 private:
  /*--------------------- Internal Data Structures --------------------------*/
  struct Node {
    // Feature index.
    int feature_idx;
    // Dimension used for subdivision.
    int div_dim_idx;
    // Threshold values used for subdivision.
    DistanceType div_dim_thr;
    // Child nodes.
    Node *child1, *child2;
  };
  typedef Node* NodePtr;
  typedef cvflann::BranchStruct<NodePtr, DistanceType> BranchSt;
  typedef BranchSt* Branch;

  NodePtr DevideTree(int cur_idx, std::vector<int>& ordered_node_indices, int num_elements) {
    NodePtr node = pool_.allocate<Node>();  // allocate memory

    // X. Last node.
    if (num_elements == 1) {
      // X. Store index of this vector.
      node->feature_idx = ordered_node_indices[cur_idx];

      // X. Mark as leaf node.
      node->div_dim_idx = -1;
      node->child1 = nullptr;
      node->child2 = nullptr;

    } else {
      // X. Set feature_idx to -1 since this is not leaf node.
      node->feature_idx = -1;

      // X. Split elements wrt mean value of index with the biggest variance.
      int split_idx;
      MeanSplit(cur_idx, num_elements, ordered_node_indices, split_idx, node->div_dim_idx,
                node->div_dim_thr);

      // X. Recursively devide tree.
      node->child1 = DevideTree(cur_idx, ordered_node_indices, split_idx);
      node->child2 =
          DevideTree(cur_idx + split_idx, ordered_node_indices, num_elements - split_idx);
    }

    return node;
  }

  void MeanSplit(int cur_idx, int element_count, std::vector<int>& ordered_node_indices,
                 int& split_idx, int& div_dim_idx, DistanceType& div_dim_thr) {
    // X. Search idx of dimension with highest variance and its mean.
    {
      std::vector<DistanceType> mean_vector(histogram_bins_, 0);
      std::vector<DistanceType> variance(histogram_bins_, 0);

      // X. Compute mean vector (i.e. mean histogram).
      int count_for_mean_computation = std::min((int)SAMPLE_MEAN + 1, element_count);
      {
        for (int j = 0; j < count_for_mean_computation; ++j) {
          // X. Add up values for each dimension.
          ElementType* v = dataset_[ordered_node_indices[cur_idx + j]];
          for (size_t k = 0; k < histogram_bins_; ++k) {
            mean_vector[k] += v[k];
          }
        }
        // X. Compute mean by division.
        for (size_t k = 0; k < histogram_bins_; ++k) {
          mean_vector[k] /= count_for_mean_computation;
        }
      }

      // X. Compute variances of sampled element for each dimension.
      {
        for (int j = 0; j < count_for_mean_computation; ++j) {
          ElementType* v = dataset_[ordered_node_indices[cur_idx + j]];
          // X. Compute variance for each dimension.
          for (size_t k = 0; k < histogram_bins_; ++k) {
            DistanceType dist = v[k] - mean_vector[k];
            variance[k] += dist * dist;
          }
        }
      }

      // X. Select dimension with the highest variance.
      div_dim_idx = SelectDividingDimension(variance);
      div_dim_thr = mean_vector[div_dim_idx];
    }

    // X. Compute splitting index based on the threshold computed above.
    {
      int equal_bound, higher_bound;
      PlaneSplit(cur_idx, element_count, div_dim_idx, div_dim_thr, ordered_node_indices,
                 equal_bound, higher_bound);

      // If # of less element is more than half of total.
      if (element_count / 2 < equal_bound) {
        split_idx = equal_bound;
        // If # of higher element is more than half of total.
      } else if (higher_bound < element_count / 2) {
        split_idx = higher_bound;
      } else {
        split_idx = element_count / 2;
      }

      // If either list is empty, it means that all remaining features are identical.
      // If this case is true, split in the middle.
      if ((equal_bound == element_count) || (higher_bound == 0)) {
        split_idx = element_count / 2;
      }
    }
  }

  // Select the top RAND_DIM largest values from v and return the index of
  // one of these selected at random.
  int SelectDividingDimension(const std::vector<DistanceType>& variance) {
    int num = 0;
    std::vector<size_t> indices_of_high_variances(RAND_DIM, 0);

    // Create a list of the indices of the top RAND_DIM values.
    for (size_t i = 0; i < histogram_bins_; ++i) {
      // X. Store index with higher variance.
      if ((num < RAND_DIM) || (variance[indices_of_high_variances[num - 1]] < variance[i])) {
        // X. Just add to list.
        if (num < RAND_DIM) {
          indices_of_high_variances[num] = i;
          num++;
          // X. Replace the last element.
        } else {
          indices_of_high_variances[num - 1] = i; /* Replace last element. */
        }
        // X. Keep the list in the order of decreasing.
        int j = num - 1;
        while (0 < j && variance[indices_of_high_variances[j - 1]] <
                            variance[indices_of_high_variances[j]]) {
          std::swap(indices_of_high_variances[j - 1], indices_of_high_variances[j]);
          --j;
        }
      }
    }
    // X. Select a random integer in range [0,num-1], and return that index.
    int rnd = cvflann::rand_int(num);
    return (int)indices_of_high_variances[rnd];
  }

  void PlaneSplit(int cur_idx, int element_count, int div_dim_idx, DistanceType div_dim_thr,
                  std::vector<int>& ordered_node_indices, int& equal_bound, int& higher_bound) {
    int left = 0;
    int right = element_count - 1;
    // X. Find the leftmost index.
    {
      for (;;) {
        // X. Find the leftmost index whose value exceeds threshold.
        while (left <= right &&
               dataset_[ordered_node_indices[cur_idx + left]][div_dim_idx] < div_dim_thr) {
          ++left;
        }
        // X. Find the rightmost index whose value deceeds threshold.
        while (left <= right &&
               div_dim_thr <= dataset_[ordered_node_indices[cur_idx + right]][div_dim_idx]) {
          --right;
        }
        if (left > right) {
          break;
        }
        // X. Swap indices.
        std::swap(ordered_node_indices[cur_idx + left], ordered_node_indices[cur_idx + right]);
        ++left;
        --right;
      }
      equal_bound = left;
    }

    // X. Find the second leftmost index.
    {
      right = element_count - 1;
      for (;;) {
        // X. Find the leftmost index whose value is bigger than thresh.
        while (left <= right &&
               dataset_[ordered_node_indices[cur_idx + left]][div_dim_idx] <= div_dim_thr) {
          ++left;
        }
        // X. Find the rightmost index whose value is less than or equal to thresh.
        while (left <= right &&
               div_dim_thr < dataset_[ordered_node_indices[cur_idx + right]][div_dim_idx]) {
          --right;
        }
        if (left > right) {
          break;
        }
        // X. Swap indices.
        std::swap(ordered_node_indices[cur_idx + left], ordered_node_indices[cur_idx + right]);
        ++left;
        --right;
      }
      higher_bound = left;
    }
  }

  /**
   * Performs an exact nearest neighbor search. The exact search performs a full
   * traversal of the tree.
   */
  void getExactNeighbors(kcvflann::ResultSet<DistanceType>& result, const ElementType* vec,
                         float epsError) {
    //		checkID -= 1;  /* Set a different unique ID for each search. */

    if (trees_ > 1) {
      fprintf(stderr, "It doesn't make any sense to use more than one tree for exact search");
    }
    if (trees_ > 0) {
      searchLevelExact(result, vec, tree_roots_[0], 0.0, epsError);
    }
    assert(result.full());
  }

  // Performs the approximate nearest-neighbor search. The search is approximate
  // because the tree traversal is abondoned after a given number of descends in the tree.
  void GetNeighbors(const ElementType* vec, int maxCheck, float epsError,
                    kcvflann::ResultSet<DistanceType>& result) {
    int checkCount = 0;
    std::unique_ptr<kcvflann::Heap<BranchSt>> priority_queue_for_branch(
        new kcvflann::Heap<BranchSt>((int)node_count_));
    cvflann::DynamicBitset checked_node(node_count_);

    // X. Tree Traversing Stage. Find the terminal node that this "vec" belongs.
    for (int i = 0; i < trees_; ++i) {
      bool debug = false;
      DistanceType minimum_dist = 0;
      int level = 0;
      SearchTreeRecursively(result, vec, tree_roots_[i], minimum_dist, checkCount, maxCheck,
                            epsError, priority_queue_for_branch, checked_node, debug, level);
    }

    // X. Back Tracking Stage.
    while (true) {
      // X. Extract branch with minimum distance.
      BranchSt branch;
      bool branch_exist = priority_queue_for_branch->popMin(branch);

      // X. Termination condition.
      if (!branch_exist || (maxCheck <= checkCount && result.full())) {
        break;
      }

      // X. Search branch starting from "node = branch.node" and "distance = branch.mindist".
      SearchTreeRecursively(result, vec, branch.node, branch.mindist, checkCount, maxCheck,
                            epsError, priority_queue_for_branch, checked_node);
    }

    // delete priority_queue_for_branch;
    assert(result.full());
  }

  /**
   *  Search starting from a given node of the tree.  Based on any mismatches at
   *  higher levels, all exemplars below this level must have a distance of
   *  at least "mindistsq".
   */
  void SearchTreeRecursively(kcvflann::ResultSet<DistanceType>& result_set, const ElementType* vec,
                             NodePtr node, DistanceType minimum_dist, int& checkCount, int maxCheck,
                             float epsError, std::unique_ptr<kcvflann::Heap<BranchSt>>& heap,
                             cvflann::DynamicBitset& checked_node, bool debug = false,
                             int level = 0) {
    // X. Ignoring branch, because we have already found better one.
    if (result_set.worstDist() < minimum_dist) {
      return;
    }

    // X. When node is terminal.
    if ((node->child1 == NULL) && (node->child2 == NULL)) {
      // X. Do not check same node more than once, when searching multiple trees.
      if (checked_node.test(node->feature_idx)) {
        return;
      }

      // X. If check count exceeds threshold.
      if (((checkCount >= maxCheck) && result_set.full())) {
        return;
      }

      checked_node.set(node->feature_idx);
      checkCount++;

      // X. Compute L2 distance in this case.
      DistanceType dist = distance_(dataset_[node->feature_idx], vec, histogram_bins_);
      // X. Try to add this point.
      result_set.addPoint(dist, node->feature_idx);
      return;
    }

    // X. Determine which child to select.
    ElementType val = vec[node->div_dim_idx];
    DistanceType diff = val - node->div_dim_thr;
    NodePtr bestChild = (diff < 0) ? node->child1 : node->child2;
    NodePtr otherChild = (diff < 0) ? node->child2 : node->child1;

    // X. Compute updated "distance".
    DistanceType new_distsq =
        minimum_dist + distance_.accum_dist(val, node->div_dim_thr, node->div_dim_idx);

    // X. Register "unexplored branch" for future possible candidate.
    if ((new_distsq * epsError < result_set.worstDist()) || !result_set.full()) {
      heap->insert(BranchSt(otherChild, new_distsq));
    }

    // X. Call search funciton for "bestChild".
    SearchTreeRecursively(result_set, vec, bestChild, minimum_dist, checkCount, maxCheck, epsError,
                          heap, checked_node, debug, level + 1);
  }

  /**
   * Performs an exact search in the tree starting from a node.
   */
  void searchLevelExact(kcvflann::ResultSet<DistanceType>& result_set, const ElementType* vec,
                        const NodePtr node, DistanceType mindist, const float epsError) {
    /* If this is a leaf node, then do check and return. */
    if ((node->child1 == NULL) && (node->child2 == NULL)) {
      int index = node->div_dim_idx;
      DistanceType dist = distance_(dataset_[index], vec, histogram_bins_);
      result_set.addPoint(dist, index);
      return;
    }

    /* Which child branch should be taken first? */
    ElementType val = vec[node->div_dim_idx];
    DistanceType diff = val - node->div_dim_thr;
    NodePtr bestChild = (diff < 0) ? node->child1 : node->child2;
    NodePtr otherChild = (diff < 0) ? node->child2 : node->child1;

    /* Create a branch record for the branch not taken.  Add distance
        of this feature boundary (we don't attempt to correct for any
        use of this feature in a parent node, which is unlikely to
        happen and would have only a small effect).  Don't bother
        adding more branches to heap after halfway point, as cost of
        adding exceeds their value.
     */

    DistanceType new_distsq = mindist + distance_.accum_dist(val, node->divval, node->divfeat);

    /* Call recursively to search next level down. */
    searchLevelExact(result_set, vec, bestChild, mindist, epsError);

    if (new_distsq * epsError <= result_set.worstDist()) {
      searchLevelExact(result_set, vec, otherChild, new_distsq, epsError);
    }
  }

 private:
  enum {
    /**
     * To improve efficiency, only SAMPLE_MEAN random values are used to
     * compute the mean and variance at each level when building a tree.
     * A value of 100 seems to perform as well as using all values.
     */
    SAMPLE_MEAN = 100,
    /**
     * Top random dimensions to consider
     *
     * When creating random trees, the dimension on which to subdivide is
     * selected at random from among the top RAND_DIM dimensions with the
     * highest variance.  A value of 5 works well.
     */
    RAND_DIM = 5
  };

  /**
   * Number of randomized trees that are used
   */
  int trees_;

  /**
   *  Array of indices to vectors in the dataset.
   */
  std::vector<int> sorted_node_indices_;

  /**
   * The dataset used by this index
   */
  const cvflann::Matrix<ElementType> dataset_;

  NNIndexParams index_params_;

  size_t node_count_;
  size_t histogram_bins_;

  /**
   * Array of k-d trees used to find neighbours.
   */
  // NodePtr* tree_roots_;
  std::vector<NodePtr> tree_roots_;

  /**
   * Pooled memory allocator.
   *
   * Using a pooled memory allocator is more efficient
   * than allocating memory directly when there is a large
   * number small of memory allocations.
   */
  cvflann::PooledAllocator pool_;

  Distance distance_;

};  // class KDTreeForest

}  // namespace kcvflann

#endif  // OPENCV_FLANN_KDTREE_INDEX_H_
