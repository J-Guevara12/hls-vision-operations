#ifndef HLS_VISION_UTILS_HPP
#define HLS_VISION_UTILS_HPP

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include "ap_axi_sdata.h"

// Tipo de dato AXI Stream (32-bit: YUYV)
typedef ap_axiu<32,0,0,0> axis_t;

constexpr int log2_const(int n) {
    return (n <= 1) ? 0 : 1 + log2_const(n / 2);
}

namespace hls_lib {

template<typename T, int WIN_SIZE, int MAX_W, int PPP = 1>
class WindowManager {
    static_assert(WIN_SIZE >= 3,     "WIN_SIZE debe ser >= 3");
    static_assert(WIN_SIZE % 2 == 1, "WIN_SIZE debe ser impar");
    static_assert(MAX_W % PPP == 0,  "MAX_W debe ser múltiplo de PPP");
    static_assert(PPP >= 1,          "PPP debe ser >= 1");

private:
    T line_buf[WIN_SIZE-1][MAX_W/PPP][PPP];

    T lb_cache_left[WIN_SIZE-1];

    T lb_cache_curr[WIN_SIZE-1][PPP];

    T prev_pixels_curr[PPP];

    T prev_cache_last;

public:
    WindowManager() {
        #pragma HLS ARRAY_PARTITION variable=prev_pixels_curr complete
        #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
        #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=3
        #pragma HLS ARRAY_PARTITION variable=lb_cache_left complete
        #pragma HLS ARRAY_PARTITION variable=lb_cache_curr complete
        for (int r = 0; r < WIN_SIZE-1; r++) {
            lb_cache_left[r] = (T)0;
            for (int p = 0; p < PPP; p++) {
                lb_cache_curr[r][p] = (T)0;
                for (int g = 0; g < MAX_W/PPP; g++)
                    line_buf[r][g][p] = (T)0;
            }
        }
        for (int p = 0; p < PPP; p++)
            prev_pixels_curr[p] = (T)0;
        prev_cache_last = (T)0;
    }

    void shiftN(T pixels[PPP],
                int col_group,
                T wins[PPP][WIN_SIZE][WIN_SIZE]) {
        #pragma HLS INLINE
        #pragma HLS ARRAY_PARTITION variable=pixels complete
        #pragma HLS ARRAY_PARTITION variable=wins   complete

        // Paso 1: leer lb_curr del grupo actual
        T lb_curr[WIN_SIZE-1][PPP];
        #pragma HLS ARRAY_PARTITION variable=lb_curr complete

        for (int r = 0; r < WIN_SIZE-1; r++) {
            #pragma HLS UNROLL
            for (int p = 0; p < PPP; p++) {
                #pragma HLS UNROLL
                lb_curr[r][p] = line_buf[r][col_group][p];
            }
        }

        // Paso 2: construir ventanas del grupo anterior
        for (int p = 0; p < PPP; p++) {
            #pragma HLS UNROLL
            for (int r = 0; r < WIN_SIZE-1; r++) {
                #pragma HLS UNROLL
                wins[p][r][0] = (p == 0)     ? lb_cache_left[r]
                                              : lb_cache_curr[r][p-1];
                wins[p][r][1] = lb_cache_curr[r][p];
                wins[p][r][2] = (p == PPP-1) ? lb_curr[r][0]
                                              : lb_cache_curr[r][p+1];
            }
            wins[p][WIN_SIZE-1][0] = (p == 0)     ? prev_cache_last
                                                   : prev_pixels_curr[p-1];
            wins[p][WIN_SIZE-1][1] = prev_pixels_curr[p];
            wins[p][WIN_SIZE-1][2] = (p == PPP-1) ? pixels[0]
                                                   : prev_pixels_curr[p+1];
        }

        // Paso 3: actualizar cachés
        for (int r = 0; r < WIN_SIZE-1; r++) {
            #pragma HLS UNROLL
            lb_cache_left[r] = lb_cache_curr[r][PPP-1];
            for (int p = 0; p < PPP; p++) {
                #pragma HLS UNROLL
                lb_cache_curr[r][p] = lb_curr[r][p];
            }
        }
        prev_cache_last = prev_pixels_curr[PPP-1];
        for (int p = 0; p < PPP; p++) {
            #pragma HLS UNROLL
            prev_pixels_curr[p] = pixels[p];
        }

        // Paso 4: actualizar line buffer
        for (int r = 0; r < WIN_SIZE-2; r++) {
            #pragma HLS UNROLL
            for (int p = 0; p < PPP; p++) {
                #pragma HLS UNROLL
                line_buf[r][col_group][p] = line_buf[r+1][col_group][p];
            }
        }
        for (int p = 0; p < PPP; p++) {
            #pragma HLS UNROLL
            line_buf[WIN_SIZE-2][col_group][p] = pixels[p];
        }
    }

    void shift(T new_pixel, int col, T win[WIN_SIZE][WIN_SIZE]) {
        #pragma HLS INLINE
        T pixels[1] = {new_pixel};
        T wins[1][WIN_SIZE][WIN_SIZE];
        #pragma HLS ARRAY_PARTITION variable=wins complete
        shiftN(pixels, col, wins);
        for (int r = 0; r < WIN_SIZE; r++) {
            #pragma HLS UNROLL
            for (int c = 0; c < WIN_SIZE; c++) {
                #pragma HLS UNROLL
                win[r][c] = wins[0][r][c];
            }
        }
    }
};

template<int BITS_OUT, typename T>
ap_uint<BITS_OUT> saturate_cast(T val) {
    #pragma HLS INLINE
    const int max_val = (1 << BITS_OUT) - 1;
    if      (val < 0)       return (ap_uint<BITS_OUT>)0;
    else if (val > max_val) return (ap_uint<BITS_OUT>)max_val;
    else                    return (ap_uint<BITS_OUT>)val;
}

// ============================================================
// stage_read
//
// Desempaqueta el stream YUYV en:
//   - y_stream  : píxeles Y individuales (PPP por ciclo)
//   - uv_stream : pares UV (1 por cada PPP píxeles Y)
//
// No tiene lógica de filtrado — solo desempaquetado.
// ============================================================
template<int PPP = 2>
void stage_read(hls::stream<axis_t>&   in,
                        hls::stream<ap_uint<PPP*8>>&  y_out,
                        hls::stream<uint16_t>& uv_out,
                        int width, int height) {

    constexpr int EXP = log2_const(PPP);
    const int GROUPS = width >> EXP;

    Row_Loop: for (int y = 0; y < height; y++) {
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT max=2160
        Col_Loop: for (int g = 0; g < GROUPS; g++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT max=4096/PPP

            axis_t pkt = in.read();

            // Extraer PPP píxeles Y y 1 par UV por paquete
            // Formato YUYV 32 bits (PPP=2):
            //   bits  7:0  = Y0
            //   bits 15:8  = U
            //   bits 23:16 = Y1
            //   bits 31:24 = V
            uint8_t u = pkt.data.range(15, 8);
            uint8_t v = pkt.data.range(31, 24);

            
            ap_uint<PPP*8> y_val=0;
            for (int p = 0; p < PPP; p++) {
                #pragma HLS UNROLL
                y_val |= (ap_uint<PPP*8>) pkt.data.range((p*16) + 7, p*16)<<(p<<3) ;
                //std::cout << (int) y_val << "(" << (int) pkt.data.range((p<<4) + 7, p<<4)<< ")" << ", ";
            }
            //std::cout << std::endl;
            y_out.write(y_val);

            // UV se envía una vez por grupo de PPP píxeles
            uv_out.write(((uint16_t)v << 8) | u);
        }
    }
}

// ============================================================
// stage_write
//
// Reempaqueta los píxeles Y filtrados con los UV del bypass
// en paquetes YUYV de salida.
//
// El UV llega con 1 fila de retraso natural respecto al Y
// filtrado porque stage_filter introduce ese desfase —
// la profundidad del uv_stream absorbe esa diferencia.
// ============================================================
template<int PPP = 2>
void stage_write(hls::stream<ap_uint<PPP*8>>&  y_in,
                         hls::stream<uint16_t>& uv_in,
                         hls::stream<axis_t>&   out,
                         int width, int height) {

    constexpr int EXP = log2_const(PPP);
    const int GROUPS = width >> EXP;

    Row_Loop: for (int y = 0; y < height; y++) {
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT max=2160
        Col_Loop: for (int g = 0; g < GROUPS; g++) {
            #pragma HLS LOOP_TRIPCOUNT max=4096/PPP
            #pragma HLS PIPELINE II=1

            bool last = (y == height-1) && (g == GROUPS-1);

            // Leer PPP píxeles Y filtrados
            uint8_t y_vals[PPP];
            #pragma HLS ARRAY_PARTITION variable=y_vals complete

            ap_uint<PPP*8> y_val = y_in.read();
            for (int p = 0; p < PPP; p++)
                #pragma HLS UNROLL
                y_vals[p] = (y_val >> (p<<3)) & 0xFF;

            // Leer par UV del bypass
            uint16_t uv = uv_in.read();
            uint8_t  u  = uv & 0xFF;
            uint8_t  v  = (uv >> 8) & 0xFF;

            // Empaquetar en YUYV
            axis_t pkt;
            pkt.data.range(7,  0)  = y_vals[0]; // Y0
            pkt.data.range(15, 8)  = u;
            pkt.data.range(23, 16) = y_vals[1]; // Y1
            pkt.data.range(31, 24) = v;
            pkt.keep = 0xF;
            pkt.strb = 0xF;
            pkt.last = last ? 1 : 0;

            out.write(pkt);
        }
    }
}

} // namespace hls_lib

#endif // HLS_VISION_UTILS_HPP
