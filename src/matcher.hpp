
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

#ifndef _MATCHER_HPP_
#define _MATCHER_HPP_

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>

namespace cv_copy {

class BFMatcher {
 public:
  BFMatcher(){};

  static cv::Ptr<BFMatcher> create();

  void knnMatch(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors,
                std::vector<std::vector<cv::DMatch> >& matches, int k,
                cv::InputArray mask = cv::noArray(), bool compactResult = false,
                bool crossCheck = false);
};

class FlannBasedMatcher {
 public:
  FlannBasedMatcher(){};

  static cv::Ptr<FlannBasedMatcher> create();

  void knnMatch(const cv::Mat& queryDescriptors, const cv::Mat& train_descriptors,
                std::vector<std::vector<cv::DMatch> >& matches, int knn,
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
                                std::vector<std::vector<cv::DMatch> >& matches);

  DescriptorCollection train_descriptor_collection;
};

}  // namespace cv_copy

#endif