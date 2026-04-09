#ifndef RESIZE2_HPP
#define RESIZE2_HPP

#include "hls_vision_utils.hpp"
#include "resizer.hpp"


template<int PPP_ = PPP>
void resize_half_read(hls::stream<axis_t>&          in,
                      hls::stream<ap_uint<PPP_*8>>& y_out,
                      hls::stream<uint16_t>&         uv_out,
                      int width, int height) {

    constexpr int EXP = log2_const(PPP_);
    const int GROUPS = width >> EXP;
    const int H_EVEN = (height >> 1) * 2;

    Row_Loop: for (int y = 0; y < height; y++) {
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT max=2160
        bool emit_row = (y % 2 == 0) && (y < H_EVEN);
        ap_uint<PPP_*8> accum = 0;

        Col_Loop: for (int g = 0; g < GROUPS; g++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT max=4096/PPP_

            axis_t pkt = in.read();

            uint8_t u = pkt.data.range(15, 8);
            uint8_t v = pkt.data.range(31, 24);

            ap_uint<PPP_*8> word = 0;
            for (int p = 0; p < PPP_; p++) {
                #pragma HLS UNROLL
                word |= (ap_uint<PPP_*8>)pkt.data.range((p*16)+7, p*16) << (p*8);
            }

            if (emit_row) {
                bool odd_group = (g % 2 == 1);
                if (!odd_group) {
                    for (int p = 0; p < PPP_/2; p++) {
                        #pragma HLS UNROLL
                        accum.range(p*8+7, p*8) = word.range(p*2*8+7, p*2*8);
                    }
                } else {
                    for (int p = 0; p < PPP_/2; p++) {
                        #pragma HLS UNROLL
                        accum.range((p+PPP_/2)*8+7, (p+PPP_/2)*8) = word.range(p*2*8+7, p*2*8);
                    }
                    y_out.write(accum);
                    uv_out.write(((uint16_t)v << 8) | u);
                }
            }
        }
    }
}

// ============================================================
// resize_half_2x — Top level para síntesis
//
// Reduce la resolución a la mitad en ambas dimensiones
// usando decimación 2:1 nearest neighbor.
//
// Entradas:
//   in_stream  : stream YUYV 32bpp
//   width      : ancho de entrada (debe ser múltiplo de PPP*2)
//   height     : alto de entrada  (debe ser múltiplo de 2)
//
// Salida:
//   out_stream : stream YUYV 32bpp
//                ancho  = width/2
//                alto   = height/2
// ============================================================
void resize_half_2x(hls::stream<axis_t>& in_stream,
                    hls::stream<axis_t>& out_stream,
                    int width, int height) {

    #pragma HLS INTERFACE axis        port=in_stream
    #pragma HLS INTERFACE axis        port=out_stream
    #pragma HLS INTERFACE s_axilite   port=width   bundle=CTRL
    #pragma HLS INTERFACE s_axilite   port=height  bundle=CTRL
    #pragma HLS INTERFACE s_axilite   port=return  bundle=CTRL

    #pragma HLS DATAFLOW disable_start_propagation 


    hls::stream<ap_uint<PPP*8>> y_resized("y_resized");
    #pragma HLS STREAM variable=y_resized depth=4
    hls::stream<uint16_t> uv_resized("uv_resized");
    #pragma HLS STREAM variable=uv_resized depth=4

    resize_half_read<PPP>(in_stream,  y_resized, uv_resized, width,   height);
    hls_lib::stage_write<PPP>(y_resized,  uv_resized, out_stream, width>>1, height>>1);
}

#endif // RESIZE2_HPP
