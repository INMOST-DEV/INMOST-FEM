//
// Created by Liogky Alexey on 01.07.2022.
//

#ifndef CARNUM_TIMERWRAP_H
#define CARNUM_TIMERWRAP_H

#include "inmost.h"

/// Helper structure for measuring the running time of code blocks
struct TimerWrap{
    double time_point = 0;
    void reset(){ time_point = Timer(); }
    double elapsed() const { return Timer() - time_point; }
    double elapsed_and_reset() { double res = elapsed(); reset(); return res; }
};

#endif //CARNUM_TIMERWRAP_H