#include "Knn.h"
#include "DataDimensionsMismatch.h"

#include <iostream>
#include <utility>
#include <numeric>
#include <cmath>

Mllib::Knn::Knn(int vectorDimension, int k) : vector_dimension(vectorDimension), k(k) {}

void Mllib::Knn::IngestData(Mllib::DataPoints dps, Mllib::Classes clss) {
    validateDimensions(dps, clss);
    this->data_points = std::move(dps);
    this->classes = std::move(clss);
}

void Mllib::Knn::validateDimensions(const Mllib::DataPoints& xs, const Mllib::Classes& ys) {
    if (xs.size() != ys.size()) throw DataDimensionsMismatch(xs.size(), ys.size());
}


int Mllib::Knn::FindKnn(std::vector<double> vec) {
    auto cmp = euclidianMetric(vec);
    std::sort(data_points.begin(),data_points.end(), cmp);
}

bool (*Mllib::Knn::euclidianMetric())(std::vector<double> vec) {

    return [vec](std::vector<double> fst, std::vector<double> snd) {
            double fst_d = 0.0;
            double snd_d = 0.0;
            for (int i =0;i<fst.size();++i) {
                fst_d += pow(vec.at(i) - fst.at(i), 2);
                snd_d += pow(vec.at(i) - snd.at(i), 2);
            }
            return sqrt(fst_d) > sqrt(snd_d);
        };
}

