
#ifndef KNN_KNN_H
#define KNN_KNN_H

#include <vector>

namespace Mllib {
    typedef std::vector<std::pair<std::vector<double>, double>> Dataset;

    class Knn {
    private:
        int vector_dimension;
        int k;
        Dataset data_set;


    public:
        Knn(int vector_dim, int k);

        void IngestData(Dataset dps);

        double FindKnn(std::vector<double> &vec);

        virtual ~Knn();

    };

    Knn::~Knn() = default;
}
#endif //KNN_KNN_H
