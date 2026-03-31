#ifndef RESIZE2_HPP
#define RESIZE2_HPP

#include "hls_vision_utils.hpp"
#include "resizer.hpp"

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

    hls::stream<ap_uint<PPP*8>> y_stream("y_stream");
    #pragma HLS STREAM variable=y_stream depth=2

    hls::stream<uint16_t> uv_stream("uv_stream");
    #pragma HLS STREAM variable=uv_stream depth=MAX_WIDTH/PPP * 2

    hls::stream<ap_uint<PPP*8>> y_resized("y_resized");
    #pragma HLS STREAM variable=y_resized depth=4

    hls::stream<uint16_t> uv_resized("uv_resized");
    #pragma HLS STREAM variable=uv_resized depth=MAX_WIDTH/PPP + 8

    hls_lib::stage_read<PPP> (in_stream,  y_stream,   uv_stream,  width,   height);
    resize_half_y<PPP>       (y_stream,   y_resized,  width,       height);
    resize_half_uv<PPP>      (uv_stream,  uv_resized, width,       height);
    hls_lib::stage_write<PPP>(y_resized,  uv_resized, out_stream, width>>1, height>>1);
}

#endif // RESIZE2_HPP
