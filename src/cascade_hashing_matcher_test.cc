
// System
#include <iostream>

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

    // descriptors1 = descriptors1 / 255.0;

    for (int i = 0; i < keypoints1.size(); i++) {
      int shuffled_idx = shuffled_indices[i];
      keypoints2[shuffled_idx] = keypoints1[i];
      descriptors1.row(i).copyTo(descriptors2.row(shuffled_idx));
    }
  }

  /*
  // 3. Match based on Brute Force Matcher
  std::vector<std::vector<cv::DMatch>> org_matches;
  {
    bool cross_check = true;
    // cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, cross_check);
    cv::Ptr<cv::FlannBasedMatcher> matcher = cv::FlannBasedMatcher::create();
    matcher->knnMatch(descriptors1, descriptors2, org_matches, 1, cv::noArray());
  }
  */

  // X. Match based on Cascade Hashing Matcher.
  std::vector<cv::DMatch> hashed_match;
  {
    cv_copy::CascadeHashingMatcher matcher;
    matcher.Initialize(descriptors1.cols);
    cv_copy::HashedImage hashed_image1 = matcher.CreateHashedSiftDescriptors(descriptors1);
    cv_copy::HashedImage hashed_image2 = matcher.CreateHashedSiftDescriptors(descriptors2);
    matcher.MatchImages(hashed_image1, descriptors1, hashed_image2, descriptors2, -1, hashed_match);
  }

  // 4. Result Confirmation.
  {
    int match_failed_cnt = 0;
    for (auto match : hashed_match) {
      if (match.trainIdx != shuffled_indices[match.queryIdx]) {
        match_failed_cnt++;
      }
    }

    LOG(INFO) << "################## Result Confirmation ####################";
    LOG(INFO) << "Total Point Compared : " << keypoints1.size();
    LOG(INFO) << "Matching Success     : " << keypoints1.size() - match_failed_cnt;
    LOG(INFO) << "Matching Failed      : " << match_failed_cnt;
    LOG(INFO) << "###########################################################";
  }

  return 0;
}
