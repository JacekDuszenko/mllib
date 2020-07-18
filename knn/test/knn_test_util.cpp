//
// Created by Jacek Duszenko on 18/07/2020.
//


#include "knn_test_util.h"


LabeledData createBasicTestData() {
    return LabeledData{
            {{3.15,       2.10},       13.0},
            {{100.0,      200.0},      300.0},
            {{-500.0,     -1337.3},    500.2},
            {{12345678.9, 98765432.1}, 800.133}
    };
}

LabeledData createProximityTestData() {
    return LabeledData{
            {{33.3333333, 33.3333333}, 1.0},
            {{66.6666666, 66.6666666}, 0.0}
    };
}