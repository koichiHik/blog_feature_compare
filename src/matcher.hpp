
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
// Copyright (C) 2014 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#ifndef _MATCHER_HPP_
#define _MATCHER_HPP_

// System
#include <bitset>
#include <random>

// Eigen
#include <eigen3/Eigen/Core>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>

namespace cv_copy {

/*********************************************/
/*     Definition of Brute Force Matcher     */
/*********************************************/
class BFMatcher {
 public:
  BFMatcher(){};

  static cv::Ptr<BFMatcher> create();

  void knnMatch(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors,
                std::vector<std::vector<cv::DMatch>>& matches, int k,
                cv::InputArray mask = cv::noArray(), bool compactResult = false,
                bool crossCheck = false);
};

/*********************************************/
/*     Definition of Flann Based Matcher     */
/*********************************************/
class FlannBasedMatcher {
 public:
  FlannBasedMatcher(){};

  static cv::Ptr<FlannBasedMatcher> create();

  void knnMatch(const cv::Mat& queryDescriptors, const cv::Mat& train_descriptors,
                std::vector<std::vector<cv::DMatch>>& matches, int knn,
                cv::InputArrayOfArrays masks, bool compactResult = false);

 protected:
  class DescriptorCollection {
   public:
    void Set(const std::vector<cv::Mat>& descriptors);

    const cv::Mat& GetDescriptors() const;

    void GetDescriptorIdxInOriginalImage(int globalDescIdx, int& imgIdx, int& localDescIdx) const;

   protected:
    cv::Mat descriptor_collection_;
    std::vector<int> start_indices_for_images_;
  };

 protected:
  static void ConvertToDMatches(const DescriptorCollection& descriptors, const cv::Mat& indices,
                                const cv::Mat& distances,
                                std::vector<std::vector<cv::DMatch>>& matches);

  DescriptorCollection train_descriptor_collection;
};

/*********************************************/
/*     Definition of Cascade Hasing Matcher  */
/*********************************************/

// The number of dimensions of the Hash code.
static const int kHashCodeSize = 128;
// The number of bucket bits.
static const int kNumBucketBits = 10;
// The number of bucket groups.
static const int kNumBucketGroups = 12;
// The number of buckets in each group.
static const int kNumBucketPerGroups = 1 << kNumBucketBits;

typedef std::vector<int> Bucket;

struct HashedSiftDescriptor {
  // Hash code generated by the primary hashing function.
  std::bitset<kHashCodeSize> hash_code;
  // Each bucket_ids[x] = y means the descriptor belongs to bucket y in bucket
  // group x.
  std::vector<uint16_t> bucket_ids;
};

struct HashedImage {
  HashedImage() {}

  // The mean of all descriptors (used for hashing).
  Eigen::VectorXf mean_descriptor;

  // The hash information.
  std::vector<HashedSiftDescriptor> hashed_desc;

  // buckets[bucket_group][bucket_id] = bucket (container of sift ids).
  std::vector<std::vector<Bucket>> buckets;
};

class CascadeHashingMatcher {
 public:
  CascadeHashingMatcher();

  bool Initialize(int num_dimensions_of_descriptor);

  HashedImage CreateHashedSiftDescriptors(const cv::Mat& mat_sift_desc) const;

  void MatchImages(const HashedImage& hashed_desc1, const cv::Mat& mat_descriptors1,
                   const HashedImage& hashed_desc2, const cv::Mat& mat_descriptors2,
                   const double lowes_ratio, std::vector<cv::DMatch>& matches) const;

 private:
  void CreateHashedDescriptors(const std::vector<Eigen::VectorXf>& sift_desc,
                               HashedImage& hashed_image) const;

  void BuildBuckets(HashedImage& hashed_image) const;

  int num_dimensions_of_descriptor_;

  std::mt19937 random_generator_;

  std::normal_distribution<double> normal_distribution_;

  Eigen::MatrixXf primary_hash_projection_;

  Eigen::MatrixXf secondary_hash_projection_[kNumBucketGroups];
};

}  // namespace cv_copy

#endif