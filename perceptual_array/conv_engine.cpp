#include "perceptual_array.hpp"

// ============================================================
// stage_conv_engine — WIN_SIZE×WIN_SIZE, II=4
//
// Para cada ventana calcula:
//   conv[c] = Σ(i,j) (win[i][j] == c) ? gauss[i][j] : 0
//
// Arquitectura: PIPELINE II=4 en el loop principal.
// La clase loop se despliega factor=4 → 4 clases por ciclo,
// 4 ciclos por ventana → (WIN_SIZE² × 4) ops paralelas/ciclo.
//
// Máximo acumulador: 49 × 63 = 3087 → 12 bits (conv_result_t)
// ============================================================

void stage_conv_engine(
    hls::stream<window_t>&    win_in,
    hls::stream<conv_vec_t>&  conv_out,
    uint8_t gauss[WIN_SIZE][WIN_SIZE],
    int width, int height) {

    #pragma HLS ARRAY_PARTITION variable=gauss complete dim=0

    const int TOTAL = width * height;

    Main_Loop: for (int i = 0; i < TOTAL; i++) {
        #pragma HLS PIPELINE II=4
        #pragma HLS LOOP_TRIPCOUNT max=MAX_WIDTH*2160

        window_t win = win_in.read();

        px_t pixels[WIN_SIZE][WIN_SIZE];
        #pragma HLS ARRAY_PARTITION variable=pixels complete dim=0

        for (int wy = 0; wy < WIN_SIZE; wy++) {
            #pragma HLS UNROLL
            for (int wx = 0; wx < WIN_SIZE; wx++) {
                #pragma HLS UNROLL
                int bit_idx = (wy * WIN_SIZE + wx) * BITS_CLASS;
                pixels[wy][wx] = win.range(bit_idx + BITS_CLASS-1, bit_idx);
            }
        }

        conv_vec_t result_vec = 0;

        for (int c = 0; c < NUM_CLASSES; c++) {
            #pragma HLS UNROLL factor=4

            conv_result_t acc = 0;

            for (int wy = 0; wy < WIN_SIZE; wy++) {
                #pragma HLS UNROLL
                for (int wx = 0; wx < WIN_SIZE; wx++) {
                    #pragma HLS UNROLL
                    ap_uint<BITS_GAUSS> weight = gauss[wy][wx] & 0x3F;
                    ap_uint<BITS_GAUSS> contribution =
                        (pixels[wy][wx] == (px_t)c) ? weight : (ap_uint<BITS_GAUSS>)0;
                    acc += contribution;
                }
            }

            result_vec.range(c*12+11, c*12) = acc;
        }

        conv_out.write(result_vec);
    }
}
