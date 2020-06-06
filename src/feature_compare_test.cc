
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
  std::vector<std::vector<cv::DMatch>> matches;
  {
    int norm_type = cv::NORM_L2;
    bool cross_check = true;

    int repeat_num = 1;
    for (int i = 0; i < repeat_num; i++) {
      LOG(INFO) << "Loop No : " << i;

      cv::Ptr<cv::BFMatcher> bf_matcher = cv::BFMatcher::create(norm_type, cross_check);
      bf_matcher->knnMatch(descriptors1, descriptors2, matches, 1, cv::noArray());
      // bf_matcher->match(descriptors1, descriptors2, matches);
    }

    LOG(INFO) << "Matched keypoints (BF Matcher) : " << matches.size();
  }

  // X. Key Point Compare with FLANN based matcher.
  {
    //
    cv::Ptr<cv::flann::IndexParams> idx_params = cv::makePtr<cv::flann::KDTreeIndexParams>();
    cv::Ptr<cv::flann::SearchParams> search_params = cv::makePtr<cv::flann::SearchParams>();

    int repeat_num = 1;
    for (int i = 0; i < repeat_num; i++) {
      LOG(INFO) << "Loop No : " << i;
      cv::Ptr<cv::FlannBasedMatcher> flann_matcher = cv::FlannBasedMatcher::create();
      flann_matcher->knnMatch(descriptors1, descriptors2, matches, 1, cv::noArray());
      // flann_matcher->match(descriptors1, descriptors2, matches);
    }

    LOG(INFO) << "Matched keypoints (FLANN Matcher) : " << matches.size();
  }

  // X. Draw matches.
  {
    cv::Mat drawn_image;
    std::vector<std::vector<cv::DMatch>> top_x_matches;
    std::vector<std::vector<char>> mask = ExtractTopXMatches(matches, 20, top_x_matches);
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, drawn_image, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), mask, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow("Drawn Image", drawn_image);
    cv::waitKey(0);
  }

  return 0;
}
