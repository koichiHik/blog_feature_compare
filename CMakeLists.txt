
cmake_minimum_required(VERSION 3.8)
project(vlfeat_sample)

# CXX Version.
set_property(GLOBAL
    PROPERTY 
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
)

# Additional modules.
list(APPEND CMAKE_MODULE_PATH 
    "${CMAKE_SOURCE_DIR}/${ADDITIONAL_CMAKE_MODULE_DIR}")

# Load external libraries.
FIND_PACKAGE(OpenCV)
FIND_PACKAGE(Glog)
FIND_PACKAGE(Gflags)
FIND_PACKAGE(Eigen3)
FIND_PACKAGE(Boost)

############# Test for refactored brute force macther. #################
# Source files.
set(TARGET "bf_matcher_refactor_test")
set(SRC src/bf_matcher_refactor_test.cc src/brute_force_matcher.cc)

# Generate Exe Files
add_executable(${TARGET} ${SRC})
target_link_libraries(${TARGET} ${GFLAGS_LIBRARIES})
target_link_libraries(${TARGET} ${GLOG_LIBRARIES})
target_link_libraries(${TARGET} ${OpenCV_LIBS})

# Macro Definition
target_compile_definitions(${TARGET}
    PRIVATE PRJ_FOLDER_PATH="${CMAKE_SOURCE_DIR}")

############# Test for refactored flann macther. #################
# Source files.
set(TARGET "flann_matcher_refactor_test")
set(SRC src/flann_matcher_refactor_test.cc src/flann_matcher.cc src/miniflann.cpp)

# Generate Exe Files
add_executable(${TARGET} ${SRC})
target_link_libraries(${TARGET} ${GFLAGS_LIBRARIES})
target_link_libraries(${TARGET} ${GLOG_LIBRARIES})
target_link_libraries(${TARGET} ${OpenCV_LIBS})

# Macro Definition
target_compile_definitions(${TARGET}
    PRIVATE PRJ_FOLDER_PATH="${CMAKE_SOURCE_DIR}")


############# Test for cascade hashing macther. #################
# Source files.
set(TARGET "cascade_hashing_matcher_test")
set(SRC src/cascade_hashing_matcher_test.cc src/cascade_hashing_matcher.cc)

# Generate Exe Files
add_executable(${TARGET} ${SRC})
target_link_libraries(${TARGET} ${GFLAGS_LIBRARIES})
target_link_libraries(${TARGET} ${GLOG_LIBRARIES})
target_link_libraries(${TARGET} ${OpenCV_LIBS})

# Macro Definition
target_compile_definitions(${TARGET}
    PRIVATE PRJ_FOLDER_PATH="${CMAKE_SOURCE_DIR}")

############# Computation time compare for each method. #################
# Source files.
set(TARGET "nearest_neighbor_perf_test")
set(SRC src/nearest_neighbor_perf_test.cc 
        src/cascade_hashing_matcher.cc)

# Generate Exe Files
add_executable(${TARGET} ${SRC})
target_link_libraries(${TARGET} ${GFLAGS_LIBRARIES})
target_link_libraries(${TARGET} ${GLOG_LIBRARIES})
target_link_libraries(${TARGET} ${OpenCV_LIBS})

# Macro Definition
target_compile_definitions(${TARGET}
    PRIVATE PRJ_FOLDER_PATH="${CMAKE_SOURCE_DIR}")
