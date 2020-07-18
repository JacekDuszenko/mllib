
#ifndef KNN_KNN_H
#define KNN_KNN_H

#include <vector>

namespace Mllib {
    typedef std::vector<std::vector<double>> DataPoints;
    typedef std::vector<int> Classes;

    class Knn {
        int vector_dimension;
        int k;
        DataPoints data_points;
        Classes classes;

        static void validateDimensions(const DataPoints &vector, const Classes &vector1);


    public:
        Knn(int vectorDimension, int k);

        void IngestData(DataPoints dps, Classes clss);

        int FindKnn(std::vector<double> vec);

        virtual ~Knn();

    };

    Knn::~Knn() = default;
}
#endif //KNN_KNN_H
