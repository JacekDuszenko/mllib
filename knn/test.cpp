//
// Created by Jacek Duszenko on 18/07/2020.
//
#include<iostream>
#include "Knn.h"

int main(int argc, char **argv) {
    for (int i = 0; i < 10; ++i) {
        std::cout << i << std::endl;
    }
    using namespace Mllib;
    Knn k = Knn(2, 1);
    std::vector<std::vector<double> > v;
    v.push_back({2.34, 50.3});
    v.push_back({10.0, 3.0});
    v.push_back({5.55, 6.66});
    v.push_back({13.37, 3.33});

    k.IngestData(v);
    std::vector<double> vek = {5.0, 6.0};
    auto vekz = k.FindKnn(vek);
    std::cout << vekz[0] << " " <<  vekz[1] << std::endl;


}