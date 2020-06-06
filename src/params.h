/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2011  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2011  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
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

#ifndef K_OPENCV_FLANN_PARAMS_H_
#define K_OPENCV_FLANN_PARAMS_H_

// STL
#include <iostream>
#include <map>

// OpenCV
#include <opencv2/flann/any.h>
#include <opencv2/flann/general.h>

namespace kcvflann {

typedef std::map<cv::String, cvflann::any> NNIndexParams;

struct KDTreeIndexParams : public NNIndexParams {
  KDTreeIndexParams(int trees = 4) {
    (*this)["algorithm"] = FLANN_INDEX_KDTREE;
    (*this)["trees"] = trees;
  }
};

struct SearchParams : public NNIndexParams {
  SearchParams(int checks = 32, float eps = 0, bool sorted = true) {
    // how many leafs to visit when searching for neighbours (-1 for unlimited)
    (*this)["checks"] = checks;
    // search for eps-approximate neighbours (default: 0)
    (*this)["eps"] = eps;
    // only for radius search, require neighbours sorted by distance (default: true)
    (*this)["sorted"] = sorted;
  }
};

template <typename T>
T get_param(const NNIndexParams& params, cv::String name, const T& default_value) {
  NNIndexParams::const_iterator it = params.find(name);
  if (it != params.end()) {
    return it->second.cast<T>();
  } else {
    return default_value;
  }
}

template <typename T>
T get_param(const NNIndexParams& params, cv::String name) {
  NNIndexParams::const_iterator it = params.find(name);
  if (it != params.end()) {
    return it->second.cast<T>();
  } else {
    throw cvflann::FLANNException(cv::String("Missing parameter '") + name +
                                  cv::String("' in the parameters given"));
  }
}

inline void print_params(const NNIndexParams& params, std::ostream& stream) {
  NNIndexParams::const_iterator it;

  for (it = params.begin(); it != params.end(); ++it) {
    stream << it->first << " : " << it->second << std::endl;
  }
}

inline void print_params(const NNIndexParams& params) { kcvflann::print_params(params, std::cout); }

}  // namespace kcvflann

#endif /* OPENCV_FLANN_PARAMS_H_ */
