#include "Knn.h"

#include <numeric>
#include <cmath>

#define min(A, B) (A > B ? A : B)

Mllib::Knn::Knn(int vectorDimension, int k) : vector_dimension(vectorDimension), k(k) {}

void Mllib::Knn::IngestData(Mllib::DataPoints dps) {
    this->data_points = std::move(dps);
}

std::function<bool(std::vector<double> &, std::vector<double> &)> euclid(const std::vector<double> &vec) {
    auto comparator_function = [&vec](std::vector<double> &fst, std::vector<double> &snd) {
        double fst_d = 0.0;
        double snd_d = 0.0;
        for (int i = 0; i < fst.size(); ++i) {
            fst_d += pow(vec.at(i) - fst.at(i), 2);
            snd_d += pow(vec.at(i) - snd.at(i), 2);
        }
        return sqrt(fst_d) > sqrt(snd_d);
    };
    return comparator_function;
}

std::vector<double> Mllib::Knn::FindKnn(std::vector<double> &vec) {
    auto cmp = euclid(vec);
    std::sort(data_points.begin(), data_points.end(), cmp);
    int iter = 0;
    std::vector<double> avg(data_points[0].size(), 0.0);
    int iterElems = min(data_points.size(), k);
    while (iter < iterElems) {
        auto &elem = data_points.at(iter);
        for (int i = 0; i < elem.size(); ++i) {
            avg[i] += elem[i];
        }
        ++iter;
    }
    std::transform(avg.begin(), avg.end(), avg.begin(), [iterElems](auto &e) { return e / iterElems; });
    return avg;
}