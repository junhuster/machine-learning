#pragma once

#include <iostream>
const int N = 1000;
void lanuchMatAddGpu(float* mat_a, float* mat_b, float* mat_c, int row_num, int col_num);
void lanuchMatAddCpu(float* mat_a, float* mat_b, float* mat_c, int row_num, int col_num);
