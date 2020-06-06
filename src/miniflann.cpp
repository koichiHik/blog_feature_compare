
// STL
#include <cstdarg>
#include <cstdio>
#include <sstream>
#include <typeinfo>

// Glog
#include <glog/logging.h>

// Boost
#include <boost/core/demangle.hpp>

// OpenCV Core
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

// OpenCV Flann
#include <opencv2/flann/dist.h>
#include <opencv2/flann/general.h>
#include <opencv2/flann/index_testing.h>

// Original
#include "defines.h"
#include "flann_base.hpp"
#include "miniflann.hpp"
#include "params.h"

namespace kcv {
namespace flann {

using namespace cv;
using namespace kcvflann;

NNIndexAdapter::NNIndexAdapter(const cv::Mat& train_descriptors,
                               const ::kcvflann::NNIndexParams& params) {
  nn_index_ = nullptr;
  algo = FLANN_INDEX_LINEAR;
  BuildIndex(train_descriptors, params);
}

void NNIndexAdapter::BuildIndex(const cv::Mat& train_descriptors,
                                const ::kcvflann::NNIndexParams& params) {
  // X.
  if (DataType<ElementType>::type != train_descriptors.type()) {
    CV_Error_(Error::StsUnsupportedFormat, ("type=%d\n", train_descriptors.type()));
  }
  if (!train_descriptors.isContinuous()) {
    CV_Error(Error::StsBadArg, "Only continuous arrays are supported");
  }

  // X. Create Matrix<float> and kcvflann::Index<cvflann::L2<float> >
  {
    ::cvflann::Matrix<ElementType> dataset(reinterpret_cast<ElementType*>(train_descriptors.data),
                                           train_descriptors.rows, train_descriptors.cols);
    nn_index_ = IndexFactory<DistanceType>::CreateIndex(dataset, params, DistanceType());
  }

  // X. Build index.
  {
    try {
      nn_index_->BuildIndex();
    } catch (...) {
      throw;
    }
  }
}

static void PrepareBuffers(OutputArray _indices, OutputArray _dists, Mat& indices, Mat& dists,
                           int rows, int minCols, int maxCols, int dtype) {
  if (_indices.needed()) {
    indices = _indices.getMat();
    if (!indices.isContinuous() || indices.type() != CV_32S || indices.rows != rows ||
        indices.cols < minCols || indices.cols > maxCols) {
      if (!indices.isContinuous()) {
        _indices.release();
      }
      _indices.create(rows, minCols, CV_32S);
      indices = _indices.getMat();
    }
  } else {
    indices.create(rows, minCols, CV_32S);
  }

  if (_dists.needed()) {
    dists = _dists.getMat();
    if (!dists.isContinuous() || dists.type() != dtype || dists.rows != rows ||
        dists.cols < minCols || dists.cols > maxCols) {
      if (!_dists.isContinuous()) _dists.release();
      _dists.create(rows, minCols, dtype);
      dists = _dists.getMat();
    }
  } else {
    dists.create(rows, minCols, dtype);
  }
}

void NNIndexAdapter::KNNSearch(InputArray _query, OutputArray _indices, OutputArray _dists, int knn,
                               const SearchParams& params) {
  typedef typename DistanceType::ResultType DistanceResultType;
  int dtype = DataType<DistanceType::ResultType>::type;

  CV_Assert((size_t)knn <= nn_index_->GetNodeCount());
  CV_Assert(_query.type() == DataType<DistanceType::ElementType>::type &&
            _indices.type() == CV_32S && _dists.type() == DataType<DistanceType::ResultType>::type);
  CV_Assert(_query.isContinuous() && _indices.isContinuous() && _dists.isContinuous());

  Mat query = _query.getMat(), indices, dists;
  PrepareBuffers(_indices, _dists, indices, dists, query.rows, knn, knn, dtype);

  ::cvflann::Matrix<ElementType> tmp_query((ElementType*)query.data, query.rows, query.cols);
  ::cvflann::Matrix<int> tmp_indices(indices.ptr<int>(), indices.rows, indices.cols);
  ::cvflann::Matrix<DistanceResultType> tmp_dists(dists.ptr<DistanceResultType>(), dists.rows,
                                                  dists.cols);

  nn_index_->KNNSearch(tmp_query, tmp_indices, tmp_dists, knn, params);
}

}  // namespace flann
}  // namespace kcv
