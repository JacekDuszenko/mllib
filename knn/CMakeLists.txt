cmake_minimum_required(VERSION 3.15)
project(knn VERSION 1.0.0 DESCRIPTION "machine learning library with bindings in Python" )

set(CMAKE_CXX_STANDARD 20)

configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
        ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
endif()


add_library(knn SHARED Knn.cpp Knn.h)

install(TARGETS knn DESTINATION knn)
install(FILES Knn.h DESTINATION include/knn)

add_executable(runUnitTests test/knn_test.cpp)
target_include_directories(runUnitTests BEFORE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(runUnitTests gtest gtest_main knn)

add_test(runUnitTests runUnitTests)