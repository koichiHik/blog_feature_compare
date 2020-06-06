
// Glog
#include <glog/logging.h>

// OpenCV
#include <opencv2/core/hal/hal.hpp>

// Original
#include "matcher.hpp"

using namespace cv_copy;

typedef void (*BatchDistFunc)(const uchar* src1, const uchar* src2, size_t stride, int nvecs,
                              int num_hist_bins, uchar* dist, const uchar* mask);

void batchDistL2_(const float* query_descriptor, const float* train_descriptor, size_t stride,
                  int nvecs, int num_hist_bins, float* dist, const uchar* mask) {
  stride /= sizeof(train_descriptor[0]);
  if (!mask) {
    for (int i = 0; i < nvecs; i++) {
      dist[i] = std::sqrt(
          cv::hal::normL2Sqr_(query_descriptor, train_descriptor + stride * i, num_hist_bins));
    }
  } else {
    float val0 = std::numeric_limits<float>::max();
    for (int i = 0; i < nvecs; i++) {
      dist[i] = mask[i] ? std::sqrt(cv::hal::normL2Sqr_(
                              query_descriptor, train_descriptor + stride * i, num_hist_bins))
                        : val0;
    }
  }
}

static void batchDistL2_32f(const float* query_descriptor, const float* train_descriptor,
                            size_t step2, int nvecs, int len, float* dist, const uchar* mask) {
  batchDistL2_(query_descriptor, train_descriptor, step2, nvecs, len, dist, mask);
}

struct BatchDistInvoker : public cv::ParallelLoopBody {
  BatchDistInvoker(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors,
                   cv::Mat& distances, cv::Mat& num_idx, int _K, const cv::Mat& _mask, int _update,
                   BatchDistFunc _func)
      : num_idx_(num_idx), distances_(distances) {
    query_descriptors_ = &query_descriptors;
    train_descriptors_ = &train_descriptors;
    K = _K;
    mask = &_mask;
    update = _update;
    func = _func;
  }

  // This function will run for several query descriptor. (Will be parallelized by OpenCV.)
  void operator()(const cv::Range& range) const CV_OVERRIDE {
    cv::AutoBuffer<int> buf(train_descriptors_->rows);
    int* bufptr = buf.data();

    size_t stride = train_descriptors_->step;
    size_t num_train_descriptors = train_descriptors_->rows;
    size_t num_histogram_bins = train_descriptors_->cols;

    LOG(INFO) << "Step : " << stride;
    LOG(INFO) << "num_histogram_bins : " << num_histogram_bins;

    for (int i = range.start; i < range.end; i++) {
      // Compute distance from query_descriptors_->ptr(i) to all train_descriptors.
      func(query_descriptors_->ptr(i), train_descriptors_->ptr(), stride, num_train_descriptors,
           num_histogram_bins, (uchar*)bufptr, mask->data ? mask->ptr(i) : 0);

      // K is guaranteed to be greater than or equal to 1.
      if (K > 0) {
        // Index and distance of closest train descriptor.
        uint32_t* distptr = (uint32_t*)distances_.ptr<uint32_t>(i, 0);

        // Loop for all distance.
        for (int j = 0; j < train_descriptors_->rows; j++) {
          int d = bufptr[j];
          // if (d < distptr[K - 1]) {
          if (d < distances_.at<uint32_t>(i, K - 1)) {
            int k;

            // Replace position such that the increasing order is achieved.
            for (k = K - 2; k >= 0 && distptr[k] > d; k--) {
              num_idx_.at<uint32_t>(i, k + 1) = num_idx_.at<uint32_t>(i, k);
              distances_.at<uint32_t>(i, k + 1) = distances_.at<uint32_t>(i, k);
            }
            num_idx_.at<uint32_t>(i, k + 1) = j + update;
            distances_.at<uint32_t>(i, k + 1) = d;
          }
        }
      }
    }
  }

  const cv::Mat* query_descriptors_;
  const cv::Mat* train_descriptors_;
  cv::Mat& distances_;
  cv::Mat& num_idx_;
  const cv::Mat* mask;
  int K;
  int update;
  BatchDistFunc func;
};

static void batchDistance_L2(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors,
                             cv::Mat& distance, int dtype, cv::Mat& begin_idx, int K,
                             const cv::Mat& mask, int idx_offset_for_image, bool crosscheck) {
  // Row : array of descriptor.
  // Col : array of histogram elements. Meaning, cols is 128.
  int type = query_descriptors.type();
  CV_Assert(type == train_descriptors.type() && query_descriptors.cols == train_descriptors.cols &&
            (type == CV_32F || type == CV_8U));

  // X. If K > 0, _nidx will be used.
  // CV_Assert(num_idx.needed() == (K > 0));
  CV_Assert((type == CV_8U && dtype == CV_32S) || dtype == CV_32F);

  // X. Number of nearest neighbor.
  K = std::min(K, train_descriptors.rows);

  // X. Buffer size is "query_descriptor size X number of nearest neighbor."
  distance.create(query_descriptors.rows, (K > 0 ? K : train_descriptors.rows), dtype);
  begin_idx.create(distance.size(), CV_32S);

  // Intialize buffer.
  if (idx_offset_for_image == 0 && K > 0) {
    distance = cv::Scalar::all(dtype == CV_32S ? (double)INT_MAX : (double)FLT_MAX);
    begin_idx = cv::Scalar::all(-1);
  }

  if (crosscheck) {
    CV_Assert(K == 1 && idx_offset_for_image == 0 && mask.empty());
    CV_Assert(!begin_idx.empty());
    cv::Mat tdist, tidx, sdist, sidx;
    batchDistance_L2(train_descriptors, query_descriptors, tdist, dtype, tidx, K, mask, 0, false);
    batchDistance_L2(query_descriptors, train_descriptors, sdist, dtype, sidx, K, mask, 0, false);

    // if an idx-th element from src1 appeared to be the nearest to i-th element of src2,
    // we update the minimum mutual distance between idx-th element of src1 and the whole src2 set.
    // As a result, if nidx[idx] = i*, it means that idx-th element of src1 is the nearest
    // to i*-th element of src2 and i*-th element of src2 is the closest to idx-th element of src1.
    // If nidx[idx] = -1, it means that there is no such ideal couple for it in src2.
    // This O(2N) procedure is called cross-check and it helps to eliminate some false matches.
    if (dtype == CV_32S) {
      for (int i = 0; i < tdist.rows; i++) {
        int idx = tidx.at<int>(i);
        int d = tdist.at<int>(i), d0 = distance.at<int>(idx);
        if (d < d0) {
          distance.at<int>(idx) = d;
          begin_idx.at<int>(idx) = i + idx_offset_for_image;
        }
      }
    } else {
      for (int i = 0; i < tdist.rows; i++) {
        int idx = tidx.at<int>(i);
        float d = tdist.at<float>(i), d0 = distance.at<float>(idx);
        if (d < d0) {
          distance.at<float>(idx) = d;
          begin_idx.at<int>(idx) = i + idx_offset_for_image;
        }
      }
    }
    for (int i = 0; i < sdist.rows; i++) {
      if (tidx.at<int>(sidx.at<int>(i)) != i) {
        begin_idx.at<int>(i) = -1;
      }
    }
    return;
  }

  BatchDistFunc func = (BatchDistFunc)batchDistL2_32f;

  // BatchDistInvoker is invoked for each query_descriptors.
  cv::parallel_for_(cv::Range(0, query_descriptors.rows),
                    BatchDistInvoker(query_descriptors, train_descriptors, distance, begin_idx, K,
                                     mask, idx_offset_for_image, func));
}

cv::Ptr<BFMatcher> BFMatcher::create() { return cv::makePtr<BFMatcher>(); }

void BFMatcher::knnMatch(const cv::Mat& query_descriptors, const cv::Mat& train_descriptors,
                         std::vector<std::vector<cv::DMatch> >& matches, int knn,
                         cv::InputArray _masks, bool compactResult, bool crossCheck) {
  // X. Only available for CV_32FC1 type of cv::Mat.
  CHECK_EQ(query_descriptors.type(), CV_32FC1);
  CHECK_EQ(train_descriptors.type(), CV_32FC1);

  // X. Add train descriptors to the collections.
  std::vector<cv::Mat> train_descriptor_collection;
  {
    std::vector<cv::Mat> added_descriptors = std::vector<cv::Mat>(1, train_descriptors.clone());
    train_descriptor_collection.insert(train_descriptor_collection.end(), added_descriptors.begin(),
                                       added_descriptors.end());
  }

  // X. If necessary container is empty.
  if (query_descriptors.empty() || (train_descriptor_collection.empty())) {
    matches.clear();
    return;
  }

  std::vector<cv::Mat> masks;
  _masks.getMatVector(masks);
  int img_count = (int)train_descriptor_collection.size();

  const int IMGIDX_SHIFT = 18;
  const int MAX_DESCRIPTOR_IN_ONE_IMAGE = (1 << IMGIDX_SHIFT);
  CV_Assert((int64)img_count * MAX_DESCRIPTOR_IN_ONE_IMAGE < INT_MAX);

  // X. Fix distance type.
  int distance_type = CV_32FC1;

  // X. Compute distances.
  cv::Mat distances, num_idx;
  {
    // X. Loop for each images.
    int idx_offset_for_imgs = 0;
    for (int img_idx = 0; img_idx < img_count; img_idx++) {
      // X. Confirm descriptor in one image does not exceed maximum number.
      CV_Assert(train_descriptor_collection[img_idx].rows < MAX_DESCRIPTOR_IN_ONE_IMAGE);

      // X. Compute index offset for different image.
      idx_offset_for_imgs = img_idx * MAX_DESCRIPTOR_IN_ONE_IMAGE;

      // X. Compute distances.
      cv::Mat mask = masks.empty() ? cv::Mat() : masks[img_idx];
      batchDistance_L2(query_descriptors, train_descriptor_collection[img_idx], distances,
                       distance_type, num_idx, knn, mask, idx_offset_for_imgs, crossCheck);

      LOG(INFO) << "Size of computed distance matrix : " << distances.size();
    }
  }

  // X. Convert distance and fill into std::vector<std::vector<cv::DMatch> >
  {
    // X. Reserve memory for the given query descriptors.
    matches.reserve(query_descriptors.rows);

    // X. Loop for each query descriptor.
    for (int query_idx = 0; query_idx < query_descriptors.rows; query_idx++) {
      const float* distance_ptr = distances.ptr<float>(query_idx);
      const int* num_idx_ptr = num_idx.ptr<int>(query_idx);

      matches.push_back(std::vector<cv::DMatch>());
      std::vector<cv::DMatch>& single_query_match = matches.back();
      single_query_match.reserve(knn);

      // X. Loop for nearest neighbor of "This query idx.".
      for (int k = 0; k < num_idx.cols; k++) {
        if (num_idx_ptr[k] < 0) {
          break;
        }
        int train_idx = num_idx_ptr[k] & (MAX_DESCRIPTOR_IN_ONE_IMAGE - 1);
        int img_idx = num_idx_ptr[k] >> IMGIDX_SHIFT;
        single_query_match.push_back(cv::DMatch(query_idx, train_idx, img_idx, distance_ptr[k]));
      }

      if (single_query_match.empty() && compactResult) {
        matches.pop_back();
      }
    }
  }
}
