//
// Created by Jacek Duszenko on 18/07/2020.
//


#ifndef KNN_KNN_TEST_UTIL_H
#define KNN_KNN_TEST_UTIL_H


#include<vector>

const static int STANDARD_VECTOR_DIM = 2;

typedef std::vector<std::pair<std::vector<double>, double>> LabeledData;

LabeledData createBasicTestData();

LabeledData createProximityTestData();

#endif //KNN_KNN_TEST_UTIL_H
