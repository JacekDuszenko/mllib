//
// Created by Jacek Duszenko on 18/07/2020.
//

#include "gtest/gtest.h"
#include "Knn.h"
#include "knn_test_util.h"


TEST(KnnBasicTestSuite, ShouldIngestDataProperly_K1) {

    // GIVEN
    int K = 1;
    Mllib::Knn knn(STANDARD_VECTOR_DIM, K);
    knn.IngestData(createBasicTestData());

    // THEN NO ERROR THROWN
}

TEST(KnnBasicTestSuite, ShouldPickValidNeighbor_1) {

    // GIVEN
    int K = 1;
    Mllib::Knn knn(STANDARD_VECTOR_DIM, K);
    std::vector<double> datapoint = {3.15, 2.10};

    // WHEN
    knn.IngestData(createBasicTestData());
    double result = knn.FindKnn(datapoint);

    // THEN
    ASSERT_EQ(result, 13.0);
}

TEST(KnnBasicTestSuite, ShouldPickValidNeighbor_2) {

    // GIVEN
    int K = 1;
    Mllib::Knn knn(STANDARD_VECTOR_DIM, K);
    std::vector<double> datapoint = {100.0, 200.0};

    // WHEN
    knn.IngestData(createBasicTestData());
    double result = knn.FindKnn(datapoint);

    // THEN
    ASSERT_EQ(result, 300.0);
}

TEST(KnnBasicTestSuite, ShouldPickValidNeighbor_3) {

    // GIVEN
    int K = 1;
    Mllib::Knn knn(STANDARD_VECTOR_DIM, K);
    std::vector<double> datapoint = {12345678.0, 98765432.0};

    // WHEN
    knn.IngestData(createBasicTestData());
    double result = knn.FindKnn(datapoint);

    // THEN
    ASSERT_EQ(result, 800.133);
}