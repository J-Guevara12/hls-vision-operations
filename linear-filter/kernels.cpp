#include "kernels.hpp"
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <iostream>
#define PPP 2


// ============================================================
// stage_filter
//
// Aplica la convolución 3x3 sobre el stream de píxeles Y.
// Procesa PPP píxeles por ciclo usando WindowManager<PPP>.
//
// Desfase de salida: (width + PPP) * (height + 1) iteraciones
// para drenar el pipeline al final del frame.
// ============================================================
void stage_filter(hls::stream<ap_uint<PPP*8>>& y_in,
                  hls::stream<ap_uint<PPP*8>>& y_out,
                  int width, int height,
                  short k[3][3], uint8_t shift_val) {
    hls_lib::WindowManager<uint8_t, 3, MAX_WIDTH, PPP> wm;
    const int GROUPS = width / PPP;

    for (int y = 0; y < height + 1; y++) {
        #pragma HLS LOOP_TRIPCOUNT max=2160
        for (int g = 0; g <= GROUPS + 1; g++) {
            #pragma HLS LOOP_TRIPCOUNT max=4096/PPP
            #pragma HLS PIPELINE II=1

            // Leer del stream solo cuando hay datos reales
            uint8_t pixels[PPP] = {};
            #pragma HLS ARRAY_PARTITION variable=pixels complete

            //std::cout << "( " << g*PPP << ", " << y << ")"<<std::endl;

            if (y < height && g < GROUPS) {
                //std::cout << "Leyendo pixeles: ";
                ap_uint<PPP*8> y_val = y_in.read();
                for (int p = 0; p < PPP; p++) {
                    #pragma HLS UNROLL
                    pixels[p] = (y_val >> (p<<3)) & 0xFF;
                    //std::cout << (int)pixels[p] << ", ";
                }
                //std::cout << std::endl;
            }

            uint8_t wins[PPP][3][3];
            #pragma HLS ARRAY_PARTITION variable=wins complete
            wm.shiftN(pixels, g, wins);

            // Emitir con desfase (1 fila, 1 grupo)
            if (y >= 1 && g >= 1 && g <= GROUPS) {
                ap_uint<PPP*8> packet_out = 0;
                for (int p = 0; p < PPP; p++) {
                    #pragma HLS UNROLL
                    int col = (g-1) * PPP + p;
                    bool is_border = (y == 1      || y == height ||
                                      col == 0 || col >= width - 1 );
                    uint8_t result = 0;
                    if (!is_border) {
                        int acc = 0;
                        for (int dy = 0; dy < 3; dy++){
                        #pragma HLS UNROLL
                            for (int dx = 0; dx < 3; dx++) {
                                #pragma HLS UNROLL
                                //std::cout << k[dy][dx] << "*" << (int) wins[p][dy][dx] << ", ";
                                acc += wins[p][dy][dx] * k[dy][dx];
                            }
                            //std::cout << std::endl;
                        }
                        //std::cout << "Resultado X=" << col << ", Y=" << y-1 << "======" << acc << std::endl << std::endl;
                        result = hls_lib::saturate_cast<8>(acc >> shift_val);
                    }
                    else {
                        //std::cout << "Borde X=" << col << ", Y=" << y-1 << "======" << std::endl << std::endl;
                    }
                    packet_out |= (ap_uint<PPP*8>) result << (p<<3) ;
                }
                y_out.write(packet_out);
            }
        }
    }
}

// ============================================================
// filter_y_3x3 — Top Level
//
// Pipeline de tres stages con DATAFLOW:
//   hls_lib::stage_read   → desempaqueta YUYV
//   stage_filter → convolución 3x3 con PPP píxeles/ciclo
//   hls_lib::stage_write  → reempaqueta YUYV con UV bypass
//
// Throughput: width*height/PPP ciclos por frame (vs width*height anterior)
// Para PPP=2: 2x mejora de throughput sobre implementación original
// ============================================================
void filter_y_3x3(hls::stream<axis_t>& in_stream,
                   hls::stream<axis_t>& out_stream,
                   int width, int height,
                   short k00, short k01, short k02,
                   short k10, short k11, short k12,
                   short k20, short k21, short k22,
                   uint8_t shift_val) {

    // ── Interfaces AXI ──────────────────────────────────────
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=width     bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=height    bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=shift_val bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=k00       bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=k01       bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=k02       bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=k10       bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=k11       bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=k12       bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=k20       bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=k21       bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=k22       bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return    bundle=CTRL

    // ── DATAFLOW ─────────────────────────────────────────────
    #pragma HLS DATAFLOW

    // Ensamblar kernel en array para pasarlo a stage_filter
    // ARRAY_PARTITION complete para que las 9 multiplicaciones
    // sean paralelas en stage_filter
    short k[3][3] = {{k00, k01, k02},
                     {k10, k11, k12},
                     {k20, k21, k22}};
    #pragma HLS ARRAY_PARTITION variable=k complete

    // ── Streams internos ────────────────────────────────────
    // y_stream: PPP píxeles Y por ciclo entre read y filter
    hls::stream<ap_uint<PPP*8>> y_stream("y_stream");
    #pragma HLS STREAM variable=y_stream depth=2

    // uv_stream: bypass de croma con profundidad suficiente para
    // absorber el desfase de 1 fila que introduce stage_filter
    // depth = width/PPP + margen para no bloquear hls_lib::stage_read
    hls::stream<uint16_t> uv_stream("uv_stream");
    #pragma HLS STREAM variable=uv_stream depth=MAX_WIDTH/PPP + 8

    // y_filt_stream: PPP píxeles Y filtrados por ciclo entre filter y write
    hls::stream<ap_uint<PPP*8>> y_filt_stream("y_filt_stream");
    #pragma HLS STREAM variable=y_filt_stream depth=2

    // ── Stages ──────────────────────────────────────────────
    hls_lib::stage_read  (in_stream,    y_stream,      uv_stream, width, height);
    stage_filter(y_stream,     y_filt_stream, width, height, k, shift_val);
    hls_lib::stage_write (y_filt_stream, uv_stream,    out_stream, width, height);
}
