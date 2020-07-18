cmake_minimum_required(VERSION 3.15)
project(knn)

set(CMAKE_CXX_STANDARD 20)

add_library(knn SHARED Knn.cpp Knn.h)