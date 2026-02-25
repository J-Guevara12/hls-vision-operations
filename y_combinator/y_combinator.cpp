#include <ap_int.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include "ap_axi_sdata.h"

typedef ap_axiu<32,0,0,0> axis_t;

uint8_t sat_add(uint8_t a, uint8_t b) {
    uint16_t sum = (uint16_t)a + b;
    return sum > 255 ? 255 : (uint8_t)sum;
}

void y_combinator(hls::stream<axis_t>& gx_stream,
                   hls::stream<axis_t>& gy_stream,
                   hls::stream<axis_t>& out_stream,
                   int width, int height) {
    #pragma HLS INTERFACE axis port=gx_stream
    #pragma HLS INTERFACE axis port=gy_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=width bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=height bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL


    for (int i = 0; i < width * height / 2; i++) {
        #pragma HLS PIPELINE II=1
        axis_t gx = gx_stream.read();
        axis_t gy = gy_stream.read();

        // Aproximación |Gx| + |Gy| para Y_par e Y_impar
        uint8_t y0 = sat_add((gx.data.range(7,0)), 
                             (gy.data.range(7,0)));
        uint8_t y1 = sat_add((gx.data.range(23,16)), (gy.data.range(23,16)));

        axis_t out;
        out.data.range(7,0)   = y0;
        out.data.range(23,16) = y1;
        // propagar croma de uno de los dos (son iguales)
        out.data.range(15,8)  = gx.data.range(15,8);
        out.data.range(31,24) = gx.data.range(31,24);
        out.last = gx.last;
        out.keep = gx.keep;
        out.strb = gx.strb;
        out_stream.write(out);
    }
}