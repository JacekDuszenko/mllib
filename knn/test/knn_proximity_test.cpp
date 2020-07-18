//
// Created by Jacek Duszenko on 18/07/2020.
//

#include "gtest/gtest.h"
#include "Knn.h"
#include "knn_test_util.h"

TEST(KnnProximityTestSuite, KnnApproximatesWellLowK1) {

    // GIVEN
    Mllib::Knn knn(2, 1);
    LabeledData testData = createProximityTestData();
    knn.IngestData(testData);
    double lowerNeighborValue = 1.0;

    // WHEN
    std::vector datapoint = {10.0, 10.0};
    double result = knn.FindKnn(datapoint);

    // THEN
    ASSERT_EQ(result, lowerNeighborValue);
}

TEST(KnnProximityTestSuite, KnnApproximatesWellHighK1) {

    // GIVEN
    Mllib::Knn knn(2, 1);
    LabeledData testData = createProximityTestData();
    knn.IngestData(testData);
    double higherNeighborValue = 0.0;

    // WHEN
    std::vector datapoint = {200.0, 200.0};
    double result = knn.FindKnn(datapoint);

    // THEN
    ASSERT_EQ(result, higherNeighborValue);
}

TEST(KnnProximityTestSuite, KnnApproximatesWellAverageK2) {

    // GIVEN
    Mllib::Knn knn(2, 2);
    LabeledData testData = createProximityTestData();
    knn.IngestData(testData);
    double avgValue = 0.5;

    // WHEN
    std::vector datapoint = {200.0, 200.0};
    double result = knn.FindKnn(datapoint);

    // THEN
    ASSERT_EQ(result, avgValue);
}