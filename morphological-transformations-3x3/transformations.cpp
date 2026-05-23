#include "transformations.hpp"
#include <iostream>

uint8_t apply_morphology(uint8_t wm[3][3],
                         uint16_t mask, bool is_dilation) {
    uint8_t result = is_dilation ? 0 : 255;

    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            int bit = ky * 3 + kx;
            if ((mask >> bit) & 1) {
                uint8_t val = wm[ky][kx];
                if (is_dilation)
                    result = (val > result) ? val : result; // MAX
                else {
                    result = (val < result) ? val : result; // MIN
                }
            }
        }
    }
    return result;
}

void stage_filter(hls::stream<ap_uint<PPP*8>>& y_in,
                  hls::stream<ap_uint<PPP*8>>& y_out,
                  int width, int height, int mask, bool is_dilation) {
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
                        
                        result = apply_morphology(wins[p], mask, is_dilation);
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

void transformation_3x3(hls::stream<axis_t>& in_stream, 
                   hls::stream<axis_t>& out_stream,
                   int width, int height, int mask, bool is_dilation) {
    // Interfaces para conectar con Zynq (PS)
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=width bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=height bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=mask bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=is_dilation bundle=CTRL
    
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    #pragma HLS DATAFLOW

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
    stage_filter(y_stream,     y_filt_stream, width, height, mask, is_dilation);
    hls_lib::stage_write (y_filt_stream, uv_stream,    out_stream, width, height);
}
