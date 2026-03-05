#ifndef KERNELS_H
#define KERNELS_H

#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "hls_vision_utils.hpp"

#define MAX_WIDTH 4096 
#define PPP 2

// Tipo de dato AXI Stream (32-bit: YUYV)
typedef ap_axiu<32,0,0,0> axis_t;

void filter_y_3x3(hls::stream<axis_t>& in_stream, 
                   hls::stream<axis_t>& out_stream,
                   int width, int height,
                   short k00, short k01, short k02,
                   short k10, short k11, short k12,
                   short k20, short k21, short k22,
                   uint8_t shift_val);
                   

void stage_filter(hls::stream<ap_uint<PPP*8>>& y_in,
    hls::stream<ap_uint<PPP*8>>& y_out,
    int width, int height,
    short k[3][3],
    uint8_t shift_val);

#endif
