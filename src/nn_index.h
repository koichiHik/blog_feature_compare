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

#ifndef K_OPENCV_FLANN_NNINDEX_H
#define K_OPENCV_FLANN_NNINDEX_H

// Boost
#include <boost/core/demangle.hpp>

// Glog
#include <glog/logging.h>

// OpenCV Header
#include <opencv2/flann/general.h>
#include <opencv2/flann/matrix.h>

// Original
#include "defines.h"
#include "params.h"
#include "result_set.h"

//! @cond IGNORED

namespace kcvflann {

/**
 * Nearest-neighbour index base class
 */
template <typename Distance>
class NNIndex {
  typedef typename Distance::ElementType ElementType;
  typedef typename Distance::ResultType DistanceType;

 public:
  virtual ~NNIndex() {}

  virtual void BuildIndex() = 0;

  virtual void KNNSearch(const cvflann::Matrix<ElementType>& queries, cvflann::Matrix<int>& indices,
                         cvflann::Matrix<DistanceType>& dists, int knn,
                         const SearchParams& params) {
    {
      std::string distance = boost::core::demangle(typeid(Distance).name());
      LOG(INFO) << "kcvflann::NNIndex <" << distance << " >::knnSearch";
    }

    assert(queries.cols == GetVectorLength());
    assert(indices.rows >= queries.rows);
    assert(dists.rows >= queries.rows);
    assert(int(indices.cols) >= knn);
    assert(int(dists.cols) >= knn);

    kcvflann::KNNUniqueResultSet<DistanceType> resultSet(knn);
    for (size_t i = 0; i < queries.rows; i++) {
      // Clear result of last descriptor.
      resultSet.clear();
      FindNeighbors(resultSet, queries[i], params);
      if (get_param(params, "sorted", true)) {
        resultSet.sortAndCopy(indices[i], dists[i], knn);
      } else {
        resultSet.copy(indices[i], dists[i], knn);
      }
    }
  }

  virtual size_t GetNodeCount() const = 0;

  virtual size_t GetVectorLength() const = 0;

  virtual int UsedMemory() const = 0;

  virtual flann_algorithm_t GetAlgoType() const = 0;

  virtual NNIndexParams GetParameters() const = 0;

  virtual void FindNeighbors(kcvflann::ResultSet<DistanceType>& result, const ElementType* vec,
                             const SearchParams& searchParams) = 0;
};

}  // namespace kcvflann

#endif  // OPENCV_FLANN_NNINDEX_H
