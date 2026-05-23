#include "perceptual_array.hpp"

// ============================================================
// stage_conv_engine
//
// Recibe 1 ventana WIN_SIZE×WIN_SIZE de píxeles de BITS_CLASS bits.
// Para cada una de las NUM_CLASSES clases calcula:
//
//   conv[clase] = Σ(i,j) (win[i][j] == clase) ? gauss[i][j] : 0
//
// La multiplicación desaparece porque el píxel es binario en
// representación one-hot — se reemplaza por AND de 1 bit con
// el peso gaussiano de BITS_GAUSS bits.
//
// Árbol de sumas: WIN_SIZE*WIN_SIZE = 169 entradas de BITS_GAUSS bits
// Máximo acumulado: 169 × (2^BITS_GAUSS - 1) = 169 × 63 = 10647
// Bits necesarios: ceil(log2(10647)) = 14 bits → conv_result_t
//
// Emite conv_vec_t: NUM_CLASSES resultados de 14 bits empaquetados
// ============================================================

void stage_conv_engine(
    hls::stream<window_t>&    win_in,
    hls::stream<conv_vec_t>&  conv_out,
    uint8_t gauss[WIN_SIZE][WIN_SIZE],
    int width, int height) {

    #pragma HLS ARRAY_PARTITION variable=gauss complete dim=0

    // Total de ventanas a procesar: width * height (PPP ventanas por grupo,
    // emitidas de a 1 por ciclo por el WM en opción B)
    const int TOTAL = width * height;

    Main_Loop: for (int i = 0; i < TOTAL; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT max=MAX_WIDTH*2160

        window_t win = win_in.read();

        // Extraer los WIN_SIZE*WIN_SIZE píxeles de la ventana
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

        // Para cada clase calcular la convolución
        conv_vec_t result_vec = 0;

        for (int c = 0; c < NUM_CLASSES; c++) {
            #pragma HLS UNROLL

            conv_result_t acc = 0;

            for (int wy = 0; wy < WIN_SIZE; wy++) {
                #pragma HLS UNROLL
                for (int wx = 0; wx < WIN_SIZE; wx++) {
                    #pragma HLS UNROLL
                    // AND: si el píxel pertenece a la clase c, sumar el peso
                    ap_uint<BITS_GAUSS> weight = gauss[wy][wx] & 0x3F;
                    ap_uint<BITS_GAUSS> contribution =
                        (pixels[wy][wx] == (px_t)c) ? weight : (ap_uint<BITS_GAUSS>)0;
                    acc += contribution;
                }
            }

            result_vec.range(c*14+13, c*14) = acc;
        }

        conv_out.write(result_vec);
    }
}