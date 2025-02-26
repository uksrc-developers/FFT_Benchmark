//
// Created by marcuskeil on 18/12/24.
//

#ifndef FFT_BENCH_ABSTRACT_FFT_HPP
#define FFT_BENCH_ABSTRACT_FFT_HPP
#include <chrono>
#include <fstream>
#include <iomanip>

class Abstract_FFT {
public:
    virtual inline std::string name() = 0;
    virtual void transform() = 0;
    virtual std::chrono::duration<double, std::milli> time_transform(int runs) = 0;
};


#endif //FFT_BENCH_ABSTRACT_FFT_HPP
