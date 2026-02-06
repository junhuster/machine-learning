#pragma once
#include <chrono>
#include <cuda_runtime.h>

bool compare_mat_diff(float* mat_a, float* mat_b, int row_num, int col_num);
bool compare_float_diff(float* left, float* right, int val_num);
float getRandomFloat();
uint64_t getRandomInt();
void constructInts(uint64_t* output_buffer, int val_num);
class MyTime {
public:
    MyTime() {}
    MyTime(MyTime& my_time) = delete;
    int64_t start() {
        using namespace std::chrono;
        start_us_ = end_us_ = duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
        return start_us_;
    }
    int64_t interval() {
        using namespace std::chrono;
        int64_t now = duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
        int64_t inter = now - end_us_;
        end_us_ = now;
        return inter;
    }
    int64_t stop() {
        using namespace std::chrono;
        int64_t total = duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count() - start_us_;
        start_us_ = end_us_ = 0;
        return total;
    }
private:
    int64_t start_us_ = 0;
    int64_t end_us_ = 0;
};

void getDeviceProp();
