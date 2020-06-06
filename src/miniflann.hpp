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

#ifndef K_OPENCV_MINIFLANN_HPP
#define K_OPENCV_MINIFLANN_HPP

//! @cond IGNORED

#include <opencv2/flann/defines.h>
#include <opencv2/flann/dist.h>
#include <opencv2/core/core.hpp>

// Original
#include "defines.h"
#include "nn_index.h"
#include "params.h"

namespace kcv {
namespace flann {

using namespace cv;

enum FlannIndexType {
  FLANN_INDEX_TYPE_8U = CV_8U,
  FLANN_INDEX_TYPE_8S = CV_8S,
  FLANN_INDEX_TYPE_16U = CV_16U,
  FLANN_INDEX_TYPE_16S = CV_16S,
  FLANN_INDEX_TYPE_32S = CV_32S,
  FLANN_INDEX_TYPE_32F = CV_32F,
  FLANN_INDEX_TYPE_64F = CV_64F,
  FLANN_INDEX_TYPE_STRING,
  FLANN_INDEX_TYPE_BOOL,
  FLANN_INDEX_TYPE_ALGORITHM,
  LAST_VALUE_FLANN_INDEX_TYPE = FLANN_INDEX_TYPE_ALGORITHM
};

class NNIndexAdapter {
  typedef float ElementType;
  typedef typename ::cvflann::L2<ElementType> DistanceType;
  typedef typename ::kcvflann::NNIndex<DistanceType> IndexType;

 public:
  // Constructor.
  NNIndexAdapter(const cv::Mat& train_descriptors, const ::kcvflann::NNIndexParams& params);

  // Search for "query" in stored data.
  void KNNSearch(InputArray query, OutputArray indices, OutputArray dists, int knn,
                 const ::kcvflann::SearchParams& params = ::kcvflann::SearchParams());

 protected:
  // Build index for the given train descriptors.
  void BuildIndex(const cv::Mat& train_descriptors, const ::kcvflann::NNIndexParams& params);

 protected:
  kcvflann::flann_algorithm_t algo;
  std::unique_ptr<IndexType> nn_index_;
};

}  // namespace flann
}  // namespace kcv

//! @endcond

#endif
