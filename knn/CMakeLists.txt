cmake_minimum_required(VERSION 3.15)
project(knn VERSION 1.0.0 DESCRIPTION "machine learning library with bindings in Python" )

set(CMAKE_CXX_STANDARD 20)

add_library(knn SHARED Knn.cpp Knn.h)

install(TARGETS knn DESTINATION knn)
install(FILES Knn.h DESTINATION include/knn)