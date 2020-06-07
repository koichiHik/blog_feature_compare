

// Glog
#include <glog/logging.h>

// Original
#include "matcher.hpp"

using namespace cv_copy;

namespace {

const unsigned seed = 0;

const double kMean = 0.0;

const double kStdDev = 1.0;

void GetZeroMeanDescriptor(const std::vector<Eigen::VectorXf>& sift_desc, Eigen::VectorXf* mean) {
  mean->setZero(sift_desc[0].size());
  for (int i = 0; i < sift_desc.size(); i++) {
    *mean += sift_desc[i];
  }
  *mean /= static_cast<double>(sift_desc.size());
}

void ConvertCvMatToVectors(const cv::Mat& mat_descriptor,
                           std::vector<Eigen::VectorXf>& descriptor) {
  descriptor.clear();
  descriptor.resize(mat_descriptor.rows, Eigen::VectorXf(mat_descriptor.cols));

  for (int i = 0; i < mat_descriptor.rows; i++) {
    for (int j = 0; j < mat_descriptor.cols; j++) {
      descriptor[i](j) = mat_descriptor.at<float>(i, j);
    }
  }
}

}  // namespace

CascadeHashingMatcher::CascadeHashingMatcher()
    : random_generator_(seed), normal_distribution_(kMean, kStdDev) {}

bool CascadeHashingMatcher::Initialize(int num_dimensions_of_descriptor) {
  num_dimensions_of_descriptor_ = num_dimensions_of_descriptor;
  primary_hash_projection_.resize(kHashCodeSize, num_dimensions_of_descriptor_);

  // Initialize primary hash projection.
  for (int i = 0; i < kHashCodeSize; i++) {
    for (int j = 0; j < num_dimensions_of_descriptor_; j++) {
      primary_hash_projection_(i, j) = normal_distribution_(random_generator_);
    }
  }

  // Initialize secondary hash projection.
  for (int i = 0; i < kNumBucketGroups; i++) {
    secondary_hash_projection_[i].resize(kNumBucketBits, num_dimensions_of_descriptor_);
    for (int j = 0; j < kNumBucketBits; j++) {
      for (int k = 0; k < num_dimensions_of_descriptor_; k++) {
        secondary_hash_projection_[i](j, k) = normal_distribution_(random_generator_);
      }
    }
  }

  return true;
}

HashedImage CascadeHashingMatcher::CreateHashedSiftDescriptors(const cv::Mat& mat_sift_desc) const {
  // X. Convert cv::Mat to std::vector
  std::vector<Eigen::VectorXf> sift_desc;
  ConvertCvMatToVectors(mat_sift_desc, sift_desc);

  HashedImage hashed_image;

  // Allocate the buckets even if no descriptors exist to fill them.
  hashed_image.buckets.resize(kNumBucketGroups);
  for (int i = 0; i < kNumBucketGroups; i++) {
    hashed_image.buckets[i].resize(kNumBucketPerGroups);
  }

  if (sift_desc.size() == 0) {
    return hashed_image;
  }

  GetZeroMeanDescriptor(sift_desc, &hashed_image.mean_descriptor);

  // Allocate space for hash codes and bucket ids.
  hashed_image.hashed_desc.resize(sift_desc.size());

  // Allocate space for each bucket id.
  for (int i = 0; i < sift_desc.size(); i++) {
    hashed_image.hashed_desc[i].bucket_ids.resize(kNumBucketGroups);
  }

  // Create hash codes for each feature.
  CreateHashedDescriptors(sift_desc, hashed_image);

  // Build the buckets.
  BuildBuckets(hashed_image);

  return hashed_image;
}

void CascadeHashingMatcher::MatchImages(const HashedImage& hashed_image1,
                                        const cv::Mat& mat_descriptors1,
                                        const HashedImage& hashed_image2,
                                        const cv::Mat& mat_descriptors2, const double lowes_ratio,
                                        std::vector<cv::DMatch>& matches) const {
  if (mat_descriptors1.rows == 0 || mat_descriptors2.rows == 0) {
    return;
  }

  // X. Convert cv::Mat to std::vector.
  std::vector<Eigen::VectorXf> descriptors1, descriptors2;
  {
    ConvertCvMatToVectors(mat_descriptors1, descriptors1);
    ConvertCvMatToVectors(mat_descriptors2, descriptors2);
  }

  static const int kNumTopCandidates = 10;
  const double sq_lowes_ratio = lowes_ratio * lowes_ratio;

  // Reserve space for the matches.
  matches.reserve(static_cast<int>(std::min(descriptors1.size(), descriptors2.size())));

  // Preallocated hamming distances. Each column indicates the hamming distance
  // and the rows collect the descriptor ids with that
  // distance. num_descriptors_with_hamming_distance keeps track of how many
  // descriptors have that distance.
  Eigen::MatrixXi candidate_hamming_distances(descriptors2.size(), kHashCodeSize + 1);
  Eigen::VectorXi num_descriptors_with_hamming_distance(kHashCodeSize + 1);

  // Preallocate the container for keeping euclidean distances.
  std::vector<std::pair<float, int> > candidate_euclidean_distances;
  candidate_euclidean_distances.reserve(kNumTopCandidates);

  // X. Loop for descriptors from image 1.
  {
    // X. Preallocate the candidate descriptors container.
    std::vector<int> candidate_descriptors_of_img2;
    candidate_descriptors_of_img2.reserve(descriptors2.size());

    // A preallocated vector to determine if we have already used a particular
    // feature for matching (i.e., prevents duplicates).
    std::vector<bool> used_descriptor(descriptors2.size());
    for (int i = 0; i < hashed_image1.hashed_desc.size(); i++) {
      candidate_descriptors_of_img2.clear();
      num_descriptors_with_hamming_distance.setZero();
      candidate_euclidean_distances.clear();

      const auto& hashed_desc_of_img1 = hashed_image1.hashed_desc[i];

      // Accumulate all descriptors in each bucket group that are in the same
      // bucket id as the query descriptor.
      // X. Loop for Bucket of image 1.
      for (int j = 0; j < kNumBucketGroups; j++) {
        const uint16_t bucket_id = hashed_desc_of_img1.bucket_ids[j];
        // X. Collect candidate descriptor from img2.
        for (const auto& feature_id_of_img2 : hashed_image2.buckets[j][bucket_id]) {
          candidate_descriptors_of_img2.emplace_back(feature_id_of_img2);
          used_descriptor[feature_id_of_img2] = false;
        }
      }

      // X. Skip matching this descriptor if there are not at least 2 candidates.
      if (candidate_descriptors_of_img2.size() <= kNumTopCandidates) {
        continue;
      }

      // X. Compute the hamming distance of all candidates based on the comp hash code.
      for (const int candidate_id_of_img2 : candidate_descriptors_of_img2) {
        if (used_descriptor[candidate_id_of_img2]) {
          continue;
        }
        used_descriptor[candidate_id_of_img2] = true;
        const uint8_t hamming_distance = (hashed_desc_of_img1.hash_code ^
                                          hashed_image2.hashed_desc[candidate_id_of_img2].hash_code)
                                             .count();

        // X. Put the descripitors into buckets corresponding to their hamming distanch.
        candidate_hamming_distances(num_descriptors_with_hamming_distance(hamming_distance)++,
                                    hamming_distance) = candidate_id_of_img2;
      }

      // X. Compute the euclidean distance of the k descriptors with the best hamming distance.
      candidate_euclidean_distances.reserve(kNumTopCandidates);

      // X. Loop increasing order of hamming distance.
      for (int j = 0; j < candidate_hamming_distances.cols(); j++) {
        // X. Loop for candidate with same hamming distance.
        for (int k = 0; k < num_descriptors_with_hamming_distance(j); k++) {
          const int candidate_id = candidate_hamming_distances(k, j);
          const float distance = (descriptors2[candidate_id] - descriptors1[i]).squaredNorm();
          candidate_euclidean_distances.emplace_back(distance, candidate_id);

          // X. Break loop if we find candidate more than "kNumTopCandidates".
          if (candidate_euclidean_distances.size() > kNumTopCandidates) {
            break;
          }
        }

        if (candidate_euclidean_distances.size() > kNumTopCandidates) {
          break;
        }
      }

      // X. Find the top 2 candidates based on euclidean distance.
      std::partial_sort(candidate_euclidean_distances.begin(),
                        candidate_euclidean_distances.begin() + 2,
                        candidate_euclidean_distances.end());

      // X. Only add to output matches if it passes the ratio test.
      if (lowes_ratio > 0) {
        if (candidate_euclidean_distances[0].first >
            candidate_euclidean_distances[1].first * sq_lowes_ratio) {
          continue;
        }
      }

      matches.push_back(cv::DMatch(i, candidate_euclidean_distances[0].second,
                                   candidate_euclidean_distances[0].first));
    }
  }
}

void CascadeHashingMatcher::CreateHashedDescriptors(const std::vector<Eigen::VectorXf>& sift_desc,
                                                    HashedImage& hashed_image) const {
  for (int i = 0; i < sift_desc.size(); i++) {
    // Use the zero mean shifted descriptor.
    const auto descriptor = sift_desc[i] - hashed_image.mean_descriptor;
    auto& hash_code = hashed_image.hashed_desc[i].hash_code;

    // Compute Hash Code.
    const Eigen::VectorXf primary_projection = primary_hash_projection_ * descriptor;
    for (int j = 0; j < kHashCodeSize; j++) {
      hash_code[j] = primary_projection(j) > 0;
    }

    // Determine the bucket index for each group.
    for (int j = 0; j < kNumBucketGroups; j++) {
      uint16_t bucket_id = 0;
      const Eigen::VectorXf secondary_projection = secondary_hash_projection_[j] * descriptor;

      for (int k = 0; k < kNumBucketBits; k++) {
        bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
      }
      hashed_image.hashed_desc[i].bucket_ids[j] = bucket_id;
    }
  }
}

void CascadeHashingMatcher::BuildBuckets(HashedImage& hashed_image) const {
  for (int i = 0; i < kNumBucketGroups; i++) {
    // Add the descriptor ID to the proper bucket group and id.
    for (int j = 0; j < hashed_image.hashed_desc.size(); j++) {
      const uint16_t bucket_id = hashed_image.hashed_desc[j].bucket_ids[i];
      hashed_image.buckets[i][bucket_id].push_back(j);
    }
  }
}
