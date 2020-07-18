//
// Created by Jacek Duszenko on 18/07/2020.
//

#ifndef KNN_DATADIMENSIONSMISMATCH_H
#define KNN_DATADIMENSIONSMISMATCH_H

static const char exception_msg[] = "Features have different dimensions that labels";

class DataDimensionsMismatch : public std::exception {
    size_t xs_size;
    size_t ys_size;

public:
    DataDimensionsMismatch(size_t xs_size, size_t ys_size) : xs_size(xs_size), ys_size(ys_size) {}

    [[nodiscard]] const char *what() const noexcept override {
        return exception_msg;
    }
};


#endif //KNN_DATADIMENSIONSMISMATCH_H
