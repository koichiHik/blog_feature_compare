#!/bin/bash
ADDITIONAL_CMAKE_MODULE_DIR=./cmake

# Configure Main Repository.
if [ ! -e ./build ]; then
  mkdir build
fi
cd build

cmake \
  -D ADDITIONAL_CMAKE_MODULE_DIR=${ADDITIONAL_CMAKE_MODULE_DIR} \
  -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
  ../

make

cd ../