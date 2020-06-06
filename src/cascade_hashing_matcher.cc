

// Original
#include "matcher.hpp"

using namespace cv_copy;

bool CascadeHashingMatcher::Initialize(int num_dimensions_of_descriptor) { return true; }

void CascadeHashingMatcher::CreateHashedSiftDescriptors(
    const std::vector<Eigen::VectorXf>& sift_desc) const {}

void CascadeHashingMatcher::MatchKeyPoints(const cv::Mat& descriptors1, const cv::Mat& descriptors2,
                                           std::vector<cv::DMatch>& matches) const {}

void CascadeHashingMatcher::CreateHashedDescriptors() const {}

void CascadeHashingMatcher::BuildBuckets() const {}
