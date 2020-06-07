
// System
#include <chrono>
#include <iostream>
#include <random>

// GFLAGS
#include <gflags/gflags.h>

// GLOG
#include <glog/logging.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

// Original
#include "matcher.hpp"

DEFINE_string(image1_file_path, "", "Path to the image1 file.");

// Value defined in CMakeLists.txt file.
static const std::string project_folder_path = PRJ_FOLDER_PATH;

namespace {

const int LOOP_NO = 5;

void ExtractOnlyBestMatch(const std::vector<std::vector<cv::DMatch>>& src,
                          std::vector<cv::DMatch>& dst) {
  dst.clear();
  dst.reserve(src.size());
  for (auto match : src) {
    if (0 < match.size()) {
      dst.push_back(match[0]);
    }
  }
}

int CountFailedMatches(const std::vector<cv::DMatch>& matches,
                       const std::vector<int>& shuffled_indices) {
  int failed_count = 0;

  for (auto match : matches) {
    if (match.trainIdx != shuffled_indices[match.queryIdx]) {
      failed_count++;
    }
  }

  return failed_count;
}

void AddNoiseToDescriptors(const cv::Mat& descriptors, cv::Mat& noised_descriptors, double sigma) {
  std::mt19937 random_gen(0);
  std::normal_distribution<double> normal_dist(0.0, sigma);

  descriptors.copyTo(noised_descriptors);

  for (int i = 0; i < descriptors.rows; i++) {
    for (int j = 0; j < descriptors.cols; j++) {
      // float tmp = noised_descriptors.at<float>(i, j) + normal_dist(random_gen);
      // noised_descriptors.at<float>(i, j) = std::max<float>(std::min<float>(255.0, tmp), 0.0);
      noised_descriptors.at<float>(i, j) += normal_dist(random_gen);
    }
  }
}

void ComputeTimeAndSuccessRatioOf3Matchers(const std::vector<cv::KeyPoint>& keypoints1,
                                           const cv::Mat& descriptors1,
                                           const std::vector<cv::KeyPoint>& keypoints2,
                                           const cv::Mat& descriptors2,
                                           const std::vector<int>& shuffled_indices) {
  // X. Match based on Brute Force Matcher
  double time_bf;
  std::vector<cv::DMatch> bf_matches;
  {
    std::vector<std::vector<cv::DMatch>> tmp_matches;
    bool cross_check = true;
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, cross_check);
    matcher->add(descriptors2);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < LOOP_NO; i++) {
      LOG(INFO) << "Brute Force Matching : " << i;
      tmp_matches.clear();
      matcher->knnMatch(descriptors1, descriptors2, tmp_matches, 1, cv::noArray());
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    time_bf = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() /
              (static_cast<double>(LOOP_NO) * 1000.0);

    ExtractOnlyBestMatch(tmp_matches, bf_matches);
  }

  // X. Match based on Flann Based Matcher
  double time_kdtree;
  std::vector<cv::DMatch> kdtree_matches;
  {
    std::vector<std::vector<cv::DMatch>> tmp_matches;
    bool cross_check = true;
    cv::Ptr<cv::FlannBasedMatcher> matcher = cv::FlannBasedMatcher::create();
    matcher->add(descriptors2);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < LOOP_NO; i++) {
      LOG(INFO) << "KDTree Matching : " << i;
      tmp_matches.clear();
      matcher->knnMatch(descriptors1, tmp_matches, 1, cv::noArray());
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    time_kdtree = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() /
                  (static_cast<double>(LOOP_NO) * 1000.0);

    ExtractOnlyBestMatch(tmp_matches, kdtree_matches);
  }

  // X. Match based on Cascade Hashing Matcher.
  double time_hashed;
  std::vector<cv::DMatch> hashed_matches;
  {
    cv_copy::CascadeHashingMatcher matcher;
    matcher.Initialize(descriptors1.cols);

    cv_copy::HashedImage hashed_image1 = matcher.CreateHashedSiftDescriptors(descriptors1);
    cv_copy::HashedImage hashed_image2 = matcher.CreateHashedSiftDescriptors(descriptors2);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < LOOP_NO; i++) {
      LOG(INFO) << "Hashed Matching : " << i;
      hashed_matches.clear();
      matcher.MatchImages(hashed_image1, descriptors1, hashed_image2, descriptors2, -1,
                          hashed_matches);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    time_hashed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() /
                  (static_cast<double>(LOOP_NO) * 1000.0);
  }

  // 4. Result Confirmation.
  {
    int bf_failed_cnt = CountFailedMatches(bf_matches, shuffled_indices);
    int kdtree_failed_cnt = CountFailedMatches(kdtree_matches, shuffled_indices);
    int hashed_failed_cnt = CountFailedMatches(hashed_matches, shuffled_indices);
    double num_feature = static_cast<double>(keypoints1.size());

    LOG(INFO) << "################## Result Confirmation ####################";
    LOG(INFO) << "Total Point Compared                     : " << keypoints1.size();
    LOG(INFO) << "Computation Time (BF Match)              : " << time_bf;
    LOG(INFO) << "Computation Time (KdTree Match)          : " << time_kdtree;
    LOG(INFO) << "Computation Time (Hashed Match)          : " << time_hashed;
    LOG(INFO) << "Matching Succsess Ratio (BF Match)       : "
              << (keypoints1.size() - bf_failed_cnt) / num_feature;
    LOG(INFO) << "Matching Succsess Ratio (KdTree Match)   : "
              << (keypoints1.size() - kdtree_failed_cnt) / num_feature;
    LOG(INFO) << "Matching Succsess Ratio (Hashed Match)   : "
              << (keypoints1.size() - hashed_failed_cnt) / num_feature;
    LOG(INFO) << "###########################################################";
  }
}

}  // namespace

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = 1;
  FLAGS_stderrthreshold = google::GLOG_INFO;
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "Version of used OpenCV is : " << cv::getVersionString();
  LOG(INFO) << "Cascade Hashing Matcher Test";

  // Turn off multithread.
  cv::setNumThreads(0);

  // 0. Path adjustment.
  std::string homography_file_path, image1_file_path;
  {
    image1_file_path = FLAGS_image1_file_path == "" ? project_folder_path + "/data/img1.ppm"
                                                    : FLAGS_image1_file_path;
    std::cout << image1_file_path << std::endl;
  }

  // 1. Extract keypoint and descriptor.
  std::vector<cv::KeyPoint> keypoints1;
  cv::Mat descriptors1;
  cv::Mat img1 = cv::imread(image1_file_path);
  {
    cv::Ptr<cv::xfeatures2d::SIFT> descriptor = cv::xfeatures2d::SIFT::create();
    descriptor->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);

    LOG(INFO) << "############# Sift Feature Computation Done! ##############";
    LOG(INFO) << "Detected keypoints from image 1 : " << keypoints1.size();
    LOG(INFO) << "Number of descriptors           : " << descriptors1.rows;
    LOG(INFO) << "###########################################################";
  }

  // 2. Shuffle keypoints and descriptors.
  std::vector<cv::KeyPoint> keypoints2(keypoints1.size());
  cv::Mat descriptors2(keypoints1.size(), 128, CV_32F);
  std::vector<int> shuffled_indices(keypoints1.size(), 0);
  {
    for (int i = 0; i < keypoints1.size(); i++) {
      shuffled_indices[i] = i;
    }
    cv::setRNGSeed(0);
    cv::randShuffle(shuffled_indices);
    for (int i = 0; i < keypoints1.size(); i++) {
      int shuffled_idx = shuffled_indices[i];
      keypoints2[shuffled_idx] = keypoints1[i];
      descriptors1.row(i).copyTo(descriptors2.row(shuffled_idx));
    }
  }

  std::vector<double> sigmas{0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0};

  for (auto sigma : sigmas) {
    LOG(INFO) << "Test NN Matches with sigma = " << sigma;
    cv::Mat noised_descriptors;
    AddNoiseToDescriptors(descriptors2, noised_descriptors, sigma);
    ComputeTimeAndSuccessRatioOf3Matchers(keypoints1, descriptors1, keypoints2, noised_descriptors,
                                          shuffled_indices);
  }

  return 0;
}
