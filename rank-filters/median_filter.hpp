#ifndef MEDIAN_FILTER_H
#define MEDIAN_FILTER_H

#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "hls_vision_utils.hpp"


#define MAX_WIDTH 4096 
#define PPP 2

// Tipo de dato AXI Stream (32-bit: YUYV)
typedef ap_axiu<32,0,0,0> axis_t;

void median_filter_3x3(hls::stream<axis_t>& in_stream, 
                   hls::stream<axis_t>& out_stream,
                   int width, int height);

#endif
