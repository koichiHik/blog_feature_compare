
// System
#include <iostream>
#include <map>
#include <string>

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

DEFINE_string(homography_file_path, "", "Path to the homography matrix from image1 to image2.");
DEFINE_string(image1_file_path, "", "Path to the image1 file.");
DEFINE_string(image2_file_path, "", "Path to the image2 file.");

// Value defined in CMakeLists.txt file.
static const std::string project_folder_path = PRJ_FOLDER_PATH;

static std::vector<std::vector<char>> ExtractTopXMatches(
    const std::vector<std::vector<cv::DMatch>>& matches, const int num,
    std::vector<std::vector<cv::DMatch>>& top_x_matches) {
  std::vector<std::vector<char>> mask(matches.size());
  std::map<float, int, std::less<float>> distance_map;

  for (int idx = 0; idx < matches.size(); idx++) {
    mask[idx].push_back(0);
    mask[idx].push_back(0);
    distance_map[matches[idx][0].distance] = idx;
  }

  int cnt = 0;
  for (auto const& entry : distance_map) {
    mask[entry.second][0] = 1;
    top_x_matches.push_back(matches[entry.second]);
    cnt++;
    if (num <= cnt) {
      break;
    }
  }
  return mask;
}

bool ResultIsSame(const std::vector<std::vector<cv::DMatch>>& org_matches,
                  const std::vector<std::vector<cv::DMatch>>& ref_matches) {
  if (org_matches.size() != ref_matches.size()) {
    LOG(INFO) << "Size is different!";
    return false;
  }
  for (int idx = 0; idx < org_matches.size(); idx++) {
    const std::vector<cv::DMatch>& org = org_matches[idx];
    const std::vector<cv::DMatch>& ref = ref_matches[idx];

    for (int idx_i = 0; idx_i < org.size(); idx_i++) {
      const cv::DMatch& org_match = org[idx_i];
      const cv::DMatch& ref_match = ref[idx_i];

      if (org_match.distance != ref_match.distance) {
        LOG(INFO) << "Distance is different!";
        return false;
      }
      if (org_match.imgIdx != ref_match.imgIdx) {
        LOG(INFO) << "imgIdx is different!";
        return false;
      }
      if (org_match.queryIdx != ref_match.queryIdx) {
        LOG(INFO) << "queryIdx is different!";
        return false;
      }
      if (org_match.trainIdx != ref_match.trainIdx) {
        LOG(INFO) << "trainIdx is different!";
        return false;
      }
    }
  }

  return true;
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = 1;
  FLAGS_stderrthreshold = google::GLOG_INFO;
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "Version of used OpenCV is : " << cv::getVersionString();

  // Turn off multithread.
  cv::setNumThreads(0);

  // 0. Path adjustment.
  std::string homography_file_path, image1_file_path, image2_file_path;
  {
    homography_file_path = FLAGS_homography_file_path == "" ? project_folder_path + "/data/H1to6p"
                                                            : FLAGS_homography_file_path;
    image1_file_path = FLAGS_image1_file_path == "" ? project_folder_path + "/data/img1.ppm"
                                                    : FLAGS_image1_file_path;
    image2_file_path = FLAGS_image2_file_path == "" ? project_folder_path + "/data/img2.ppm"
                                                    : FLAGS_image2_file_path;
    std::cout << image1_file_path << std::endl;
    std::cout << image2_file_path << std::endl;
  }

  // 1. Extract keypoint and descriptor.
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;
  cv::Mat img1 = cv::imread(image1_file_path);
  cv::Mat img2 = cv::imread(image2_file_path);

  {
    cv::Ptr<cv::xfeatures2d::SIFT> descriptor = cv::xfeatures2d::SIFT::create();
    descriptor->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    descriptor->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    LOG(INFO) << "";
    LOG(INFO) << "Detected keypoints from image 1 : " << keypoints1.size();
    LOG(INFO) << "Detected keypoints from image 2 : " << keypoints2.size();
    LOG(INFO) << "";
  }

  // 2. Key Point Compare with Brute Force Matcher.
  // std::vector<std::vector<cv::DMatch>> org_matches;
  std::vector<std::vector<cv::DMatch>> org_matches;
  {
    bool cross_check = false;
    cv::Ptr<cv::BFMatcher> bf_matcher = cv::BFMatcher::create(cv::NORM_L2, cross_check);
    bf_matcher->knnMatch(descriptors1, descriptors2, org_matches, 2, cv::noArray());
    LOG(INFO) << "Matched keypoints (BF Matcher) : " << org_matches.size();
  }

  // 3. Key Point Compare with Refactored Brute Force Matcher.
  std::vector<std::vector<cv::DMatch>> ref_matches;
  {
    cv::Ptr<cv_copy::BFMatcher> bf_matcher = cv_copy::BFMatcher::create();
    bf_matcher->knnMatch(descriptors1, descriptors2, ref_matches, 2, cv::noArray());
    LOG(INFO) << "Matched keypoints (BF Matcher) : " << ref_matches.size();
  }

  // 4. Compare output.
  {
    if (ResultIsSame(org_matches, ref_matches)) {
      LOG(INFO) << "Result is same.";
    }
  }

  return 0;
}
