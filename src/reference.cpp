#include "gemv.h"

// Serial reference implementation, matching the original course baseline.
// Single threaded, no vectorisation: this is the number the GPU result is
// compared against, so it is deliberately naive.
accum_t cpu_reference(const elem_t* a, const elem_t* b, int n) {
    accum_t total = 0;
    for (int row = 0; row < n; ++row) {
        accum_t temp = 0;
        for (int col = 0; col < n; ++col) {
            const accum_t bk = b[col];
            temp += static_cast<accum_t>(a[row * n + col]) * bk * bk * bk;
        }
        total += temp;
    }
    return total;
}
