
#ifndef KNN_KNN_H
#define KNN_KNN_H

#include <vector>

namespace Mllib {
    typedef std::vector<std::vector<double>> DataPoints;

    class Knn {
        int vector_dimension;
        int k;
        DataPoints data_points;


    public:
        Knn(int vector_dim, int k);

        void IngestData(DataPoints dps);

        std::vector<double> FindKnn(std::vector<double>& vec);

        virtual ~Knn();

    };

    Knn::~Knn() = default;
}
#endif //KNN_KNN_H
