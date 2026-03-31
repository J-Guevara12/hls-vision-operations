#ifndef RESIZER_HPP
#define RESIZER_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include "hls_vision_utils.hpp"

// ============================================================
// Configuración global del resizer
// ============================================================
#ifndef PPP
#define PPP 2
#endif

#ifndef MAX_WIDTH
#define MAX_WIDTH 4096
#endif

#ifndef MAX_HEIGHT
#define MAX_HEIGHT 2160
#endif

// ============================================================
// resize_half — Decimación 2:1 nearest neighbor
//
// Reduce la resolución a la mitad en ambas dimensiones
// tomando píxeles pares de filas pares.
//
// Restricciones:
//   - width  debe ser múltiplo de PPP*2
//   - height debe ser múltiplo de 2
//   - width  <= MAX_WIDTH
//
// Streams:
//   y_in  : ap_uint<PPP*8> — PPP píxeles Y por palabra
//   y_out : ap_uint<PPP*8> — PPP píxeles Y por palabra
//             out_width  = width/2
//             out_height = height/2
//
//   uv_in  : uint16_t — 1 par UV por cada PPP píxeles Y de entrada
//   uv_out : uint16_t — 1 par UV por cada PPP píxeles Y de salida
//             out_width  = width/2
//             out_height = height/2
// ============================================================
template<int PPP_=PPP>
void resize_half_y(hls::stream<ap_uint<PPP*8>>& y_in,
                   hls::stream<ap_uint<PPP*8>>& y_out,
                   int width, int height) {

    constexpr int EXP = log2_const(PPP);
    const int GROUPS = width >> EXP;


    Row_Loop: for (int y = 0; y < height; y++) {
        ap_uint<PPP*8> accum = 0;  // acumula PPP píxeles decimados
        bool emit_row = (y % 2 == 0) && (y < (height & ~1));
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT max=2160
        Col_Loop: for (int g = 0; g < GROUPS; g++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT max=4096/PPP

            ap_uint<PPP*8> word = y_in.read();
            bool odd_group = (g % 2 == 1);

            if (emit_row) {
                // Tomar píxeles pares de cada palabra: bits[7:0] de cada píxel
                // Grupo par: llenar mitad baja del acumulador
                // Grupo impar: llenar mitad alta y emitir
                if (!odd_group) {
                    // Píxeles p=0,2,4,... de la palabra actual van a accum[PPP/2-1:0]
                    for (int p = 0; p < PPP/2; p++) {
                        #pragma HLS UNROLL
                        accum.range(p*8+7, p*8) = word.range(p*2*8+7, p*2*8);
                    }
                } else {
                    // Píxeles pares de esta palabra van a accum[PPP-1:PPP/2]
                    for (int p = 0; p < PPP/2; p++) {
                        #pragma HLS UNROLL
                        accum.range((p+PPP/2)*8+7, (p+PPP/2)*8) =
                            word.range(p*2*8+7, p*2*8);
                    }
                    y_out.write(accum);
                }
            }
        }
    }
}

template<int PPP_=PPP>
void resize_half_uv(hls::stream<uint16_t>& uv_in,
                    hls::stream<uint16_t>& uv_out,
                    int width, int height) {

    constexpr int EXP = log2_const(PPP);
    const int GROUPS = width >> EXP;  // mismo ritmo que Y

    Row_Loop: for (int y = 0; y < height; y++) {
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT max=2160
        Col_Loop: for (int g = 0; g < GROUPS; g++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT max=4096/PPP

            uint16_t uv = uv_in.read();
            bool emit_row = (y % 2 == 0) && (y<((height>>1)<<1));
            bool even_group = (g % 2 == 0);

            if (emit_row && even_group)
                uv_out.write(uv);
        }
    }
}


// ============================================================
// [FUTURO] resize_integer — Decimación N:1 nearest neighbor
//
// Reduce la resolución por un factor entero arbitrario.
//
// Restricciones:
//   - width  debe ser múltiplo de PPP*factor_x
//   - height debe ser múltiplo de factor_y
//
// template<int PPP_ = PPP>
// void resize_integer_y(hls::stream<ap_uint<PPP_*8>>& y_in,
//                       hls::stream<ap_uint<PPP_*8>>& y_out,
//                       int width, int height,
//                       int factor_x, int factor_y);
//
// template<int PPP_ = PPP>
// void resize_integer_uv(hls::stream<uint16_t>& uv_in,
//                        hls::stream<uint16_t>& uv_out,
//                        int width, int height,
//                        int factor_x, int factor_y);
// ============================================================

// ============================================================
// [FUTURO] resize_bilinear — Interpolación bilineal 2:1
//
// Reduce la resolución a la mitad usando interpolación bilineal.
// Requiere line buffer de 2 filas para acceso a filas adyacentes.
//
// template<int PPP_ = PPP>
// void resize_bilinear_y(hls::stream<ap_uint<PPP_*8>>& y_in,
//                        hls::stream<ap_uint<PPP_*8>>& y_out,
//                        int width, int height);
//
// template<int PPP_ = PPP>
// void resize_bilinear_uv(hls::stream<uint16_t>& uv_in,
//                         hls::stream<uint16_t>& uv_out,
//                         int width, int height);
// ============================================================

// ============================================================
// [FUTURO] resize_rational — Decimación P:Q nearest neighbor
//
// Reduce la resolución por un factor racional P/Q usando
// acumulador de Bresenham. Soporta downscaling arbitrario.
//
// Restricciones:
//   - out_width  = width  * Q / P debe ser entero
//   - out_height = height * Q / P debe ser entero
//   - P > Q (solo downscaling)
//
// template<int PPP_ = PPP>
// void resize_rational_y(hls::stream<ap_uint<PPP_*8>>& y_in,
//                        hls::stream<ap_uint<PPP_*8>>& y_out,
//                        int in_width,  int in_height,
//                        int out_width, int out_height);
//
// template<int PPP_ = PPP>
// void resize_rational_uv(hls::stream<uint16_t>& uv_in,
//                         hls::stream<uint16_t>& uv_out,
//                         int in_width,  int in_height,
//                         int out_width, int out_height);
// ============================================================

void resize_half_2x(hls::stream<axis_t>& in_stream,
                    hls::stream<axis_t>& out_stream,
                    int width, int height);

#endif // RESIZER_HPP
