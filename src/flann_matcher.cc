
// Glog
#include <glog/logging.h>

// OpenCV
#include <opencv2/core/base.hpp>
#include <opencv2/core/core.hpp>

// Original
#include "matcher.hpp"
#include "miniflann.hpp"
#include "params.h"

using namespace cv_copy;

cv::Ptr<FlannBasedMatcher> FlannBasedMatcher::create() { return cv::makePtr<FlannBasedMatcher>(); }

void FlannBasedMatcher::ConvertToDMatches(const DescriptorCollection& collection,
                                          const cv::Mat& matches_indices,
                                          const cv::Mat& matches_distances,
                                          std::vector<std::vector<cv::DMatch>>& matches) {
  int num_query = matches_indices.rows;
  int num_knn = matches_indices.cols;
  matches.resize(num_query);
  // X. Loop for each query vector.
  for (int query_idx = 0; query_idx < num_query; query_idx++) {
    // X. Loop for nearest neighbor.
    for (int j = 0; j < num_knn; j++) {
      int global_idx = matches_indices.at<int>(query_idx, j);
      if (global_idx >= 0) {
        // X. Convert global to local.
        int image_idx, train_idx;
        collection.GetDescriptorIdxInOriginalImage(global_idx, image_idx, train_idx);

        // X. Extract distance.
        float dist = std::sqrt(matches_distances.at<float>(query_idx, j));
        matches[query_idx].push_back(cv::DMatch(query_idx, train_idx, image_idx, dist));
      }
    }
  }
}

void FlannBasedMatcher::knnMatch(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors,
                                 std::vector<std::vector<cv::DMatch>>& matches, int knn,
                                 cv::InputArrayOfArrays masks, bool compactResult) {
  CHECK(!query_descriptors.empty());
  CHECK(knn > 0);

  // X. Update descriptor collection for later search.
  { train_descriptor_collection.Set(std::vector<cv::Mat>{train_descriptors.clone()}); }

  // X. Create KDTreeIndex for knn search.
  cv::Ptr<kcv::flann::NNIndexAdapter> nn_index_adapter;
  cv::Ptr<::kcvflann::NNIndexParams> kdtree_index_params =
      cv::makePtr<::kcvflann::KDTreeIndexParams>();
  {
    // KDTree indices is built here.
    nn_index_adapter = cv::makePtr<kcv::flann::NNIndexAdapter>(
        train_descriptor_collection.GetDescriptors(), *kdtree_index_params);
  }

  // X. Knn search.
  cv::Mat matched_indices(query_descriptors.rows, knn, CV_32SC1);
  cv::Mat matched_distances(query_descriptors.rows, knn, CV_32FC1);
  {
    cv::Ptr<::kcvflann::SearchParams> search_params = cv::makePtr<::kcvflann::SearchParams>();
    nn_index_adapter->KNNSearch(query_descriptors, matched_indices, matched_distances, knn,
                                *search_params);
  }

  // X. Convert matched indices and distances into DMatches.
  { ConvertToDMatches(train_descriptor_collection, matched_indices, matched_distances, matches); }
}

///////////////////////////////////////////////////
// Definition of "DescriptorCollection".         //
///////////////////////////////////////////////////
void FlannBasedMatcher::DescriptorCollection::Set(const std::vector<cv::Mat>& descriptors) {
  // X. Clear
  {
    start_indices_for_images_.clear();
    descriptor_collection_.release();
  }

  size_t image_count = descriptors.size();
  CV_Assert(image_count > 0);

  // X. Allocate memory for starting index of descriptors in each image.
  start_indices_for_images_.resize(image_count);

  int dim = -1;
  int type = -1;
  start_indices_for_images_[0] = 0;

  // X. Creating starting indices of descriptors for each images.
  {
    for (size_t i = 1; i < image_count; i++) {
      int num_desc = 0;
      if (!descriptors[i - 1].empty()) {
        // Dimension of descriptor is along column direction.
        dim = descriptors[i - 1].cols;
        type = descriptors[i - 1].type();
        num_desc = descriptors[i - 1].rows;
      }
      start_indices_for_images_[i] = start_indices_for_images_[i - 1] + num_desc;
    }
  }

  if (image_count == 1) {
    // X. No descriptor exists.
    if (descriptors[0].empty()) {
      return;
    }
    dim = descriptors[0].cols;
    type = descriptors[0].type();
  }
  CV_Assert(dim > 0);

  int total_desc_count =
      start_indices_for_images_[image_count - 1] + descriptors[image_count - 1].rows;
  if (total_desc_count > 0) {
    // X. Create buffer for storing all descriptors.
    descriptor_collection_.create(total_desc_count, dim, type);

    // X. Copy descriptors of each image into buffer.
    for (size_t i = 0; i < image_count; i++) {
      if (!descriptors[i].empty()) {
        CV_Assert(descriptors[i].cols == dim && descriptors[i].type() == type);

        // X. Copy descriptors of image i into collection.
        cv::Mat m = descriptor_collection_.rowRange(
            start_indices_for_images_[i], start_indices_for_images_[i] + descriptors[i].rows);
        descriptors[i].copyTo(m);
      }
    }
  }
}

const cv::Mat& FlannBasedMatcher::DescriptorCollection::GetDescriptors() const {
  return descriptor_collection_;
}

void FlannBasedMatcher::DescriptorCollection::GetDescriptorIdxInOriginalImage(
    int global_desc_idx, int& image_idx, int& desc_idx_in_image) const {
  CV_Assert((global_desc_idx >= 0) && (global_desc_idx < descriptor_collection_.rows));

  // X. Compute starting index of descriptor for that image.
  std::vector<int>::const_iterator start_idx_for_img = std::upper_bound(
      start_indices_for_images_.begin(), start_indices_for_images_.end(), global_desc_idx);
  --start_idx_for_img;

  // X.
  image_idx = (int)(start_idx_for_img - start_indices_for_images_.begin());
  desc_idx_in_image = global_desc_idx - (*start_idx_for_img);
}