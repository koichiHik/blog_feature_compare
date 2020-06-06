
// System
#include <iostream>

// GFLAGS
#include <gflags/gflags.h>

// GLOG
#include <glog/logging.h>

// Original
#include "matcher.hpp"

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = 1;
  FLAGS_stderrthreshold = google::GLOG_INFO;
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "Cascade Hashing Matcher Test";

  return 0;
}