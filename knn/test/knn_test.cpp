//
// Created by Jacek Duszenko on 18/07/2020.
//

#include "gtest/gtest.h"
#include "Knn.h"

const static int STANDARD_VECTOR_DIM = 2;

typedef std::vector<std::pair<std::vector<double>, double>> LabeledData;

LabeledData createTestData();

using namespace Mllib;

TEST(KnnTest, ShouldIngestDataProperly_K1) {

    // GIVEN
    int K = 1;
    Knn knn(STANDARD_VECTOR_DIM, K);
    knn.IngestData(createTestData());

    // THEN NO ERROR THROWN

}

TEST(KnnTest, ShouldPickValidNeighbor_1) {

    // GIVEN
    int K = 1;
    Knn knn(STANDARD_VECTOR_DIM, K);
    std::vector<double> datapoint = {3.15, 2.10};

    // WHEN
    knn.IngestData(createTestData());
    double result = knn.FindKnn(datapoint);

    // THEN
    ASSERT_EQ(result, 13.0);
}

TEST(KnnTest, ShouldPickValidNeighbor_2) {

    // GIVEN
    int K = 1;
    Knn knn(STANDARD_VECTOR_DIM, K);
    std::vector<double> datapoint = {100.0, 200.0};

    // WHEN
    knn.IngestData(createTestData());
    double result = knn.FindKnn(datapoint);

    // THEN
    ASSERT_EQ(result, 300.0);
}

TEST(KnnTest, ShouldPickValidNeighbor_3) {

    // GIVEN
    int K = 1;
    Knn knn(STANDARD_VECTOR_DIM, K);
    std::vector<double> datapoint = {12345678.0, 98765432.0};

    // WHEN
    knn.IngestData(createTestData());
    double result = knn.FindKnn(datapoint);

    // THEN
    ASSERT_EQ(result, 800.133);
}

LabeledData createTestData() {
    return LabeledData{
            {{3.15,       2.10},       13.0},
            {{100.0,      200.0},      300.0},
            {{-500.0,     -1337.3},    500.2},
            {{12345678.9, 98765432.1}, 800.133}
    };
}