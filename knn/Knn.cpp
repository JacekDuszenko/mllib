#include "Knn.h"

#include <numeric>
#include <cmath>

#define min(A, B) (A > B ? B : A)

Mllib::Knn::Knn(int vector_dim, int k) : vector_dimension(vector_dim), k(k) {}

void Mllib::Knn::IngestData(Mllib::Dataset dps) {
    this->data_set = std::move(dps);
}

typedef std::pair<std::vector<double>, double> &DataPair;

std::function<bool(DataPair, DataPair)> euclid(const std::vector<double> &vec) {
    auto comparator_fun = [&vec](DataPair fst, DataPair snd) {
        double fst_d{0.0};
        double snd_d{0.0};
        for (int i = 0; i < fst.first.size(); ++i) {
            fst_d += pow(vec.at(i) - fst.first.at(i), 2);
            snd_d += pow(vec.at(i) - snd.first.at(i), 2);
        }
        return sqrt(fst_d) < sqrt(snd_d);
    };
    return comparator_fun;
}

double Mllib::Knn::FindKnn(std::vector<double> &vec) {
    auto cmp = euclid(vec);
    std::sort(data_set.begin(), data_set.end(), cmp);
    int iter = 0;
    double avg = 0;
    int iter_elems = min(data_set.size(), k);
    while (iter < iter_elems) {
        auto &elem = data_set.at(iter);
        avg += elem.second;
        ++iter;
    }
    avg /= iter_elems;
    return avg;
}