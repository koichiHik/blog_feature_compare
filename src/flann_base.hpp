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

#ifndef K_OPENCV_FLANN_BASE_HPP_
#define K_OPENCV_FLANN_BASE_HPP_

// STL
#include <cassert>
#include <cstdio>
#include <vector>

// Glog
#include <glog/logging.h>

// Boost
#include <boost/core/demangle.hpp>

// OpenCV
#include <opencv2/flann/general.h>
#include <opencv2/flann/matrix.h>

// Original
#include "kdtree_index.h"
#include "nn_index.h"
#include "params.h"

namespace kcvflann {

///////////////////////////////////////////////////////////////////////////////////
//   Definition of Index.
///////////////////////////////////////////////////////////////////////////////////

template <typename Distance>
class IndexFactory {
 public:
  typedef typename Distance::ElementType ElementType;
  typedef typename Distance::ResultType DistanceType;

  static std::unique_ptr<NNIndex<Distance>> CreateIndex(
      const cvflann::Matrix<ElementType>& features, const NNIndexParams& params,
      Distance distance = Distance()) {
    std::unique_ptr<NNIndex<Distance>> nnIndex;
    {
      flann_algorithm_t index_type = kcvflann::get_param<flann_algorithm_t>(params, "algorithm");
      switch (index_type) {
        case FLANN_INDEX_KDTREE:
          LOG(INFO) << "KDTreeIndex Created.";
          nnIndex.reset(new KDTreeIndex<Distance>(features, params, distance));
          break;
        default:
          throw cvflann::FLANNException("Unknown index type");
      }
    }
    return nnIndex;
  }
};

}  // namespace kcvflann

#endif /* OPENCV_FLANN_BASE_HPP_ */
