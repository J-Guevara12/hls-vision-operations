#include "perceptual_array.hpp"

// ============================================================
// stage_argmax_and_count
//
// Para cada píxel:
//   1. Encuentra la clase con mayor valor en blend_vec_t → ArgMax
//   2. Compara con Pk original → si difiere, incrementa changed_count
//
// ArgMax: árbol de comparadores de log2(16)=4 niveles en LUTs.
// Compara en pares (0vs1, 2vs3, ...) hasta encontrar el máximo.
//
// changed_count se resetea al inicio de cada frame (y==0, píxel 0).
// Al final del frame el valor está disponible en el registro AXI-Lite.
//
// Recibe pk_original_in a ritmo de PPP píxeles por ciclo (pgroup_t).
// Igual que en scale_and_blend, lee el grupo cuando i%PPP==0.
// ============================================================

void stage_argmax_and_count(
    hls::stream<blend_vec_t>&    blend_in,
    hls::stream<pgroup_t>&       pk_original_in,
    hls::stream<result_group_t>& result_out,
    int& changed_count,
    int width, int height) {

    const int TOTAL = width * height;
    int local_count = 0;

    // Buffer de resultados para acumular PPP índices antes de emitir
    result_group_t result_group = 0;

    pgroup_t pk_group = 0;

    Main_Loop: for (int i = 0; i < TOTAL; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT max=MAX_WIDTH*2160

        blend_vec_t blend_vec = blend_in.read();

        // Leer grupo Pk original cuando toca
        if (i % PPP == 0) {
            pk_group = pk_original_in.read();
        }
        px_t pk_orig = pk_group.range(
            (i % PPP)*BITS_CLASS + BITS_CLASS-1,
            (i % PPP)*BITS_CLASS);

        // ArgMax: árbol de comparadores de 4 niveles
        // Nivel 0: 16 valores → 8 ganadores
        ap_uint<8>  val_l0[8];
        class_idx_t idx_l0[8];
        #pragma HLS ARRAY_PARTITION variable=val_l0 complete
        #pragma HLS ARRAY_PARTITION variable=idx_l0 complete

        for (int c = 0; c < 8; c++) {
            #pragma HLS UNROLL
            ap_uint<8> v0 = blend_vec.range(c*2*8+7,    c*2*8);
            ap_uint<8> v1 = blend_vec.range(c*2*8+15,   c*2*8+8);
            if (v0 >= v1) { val_l0[c] = v0; idx_l0[c] = c*2;   }
            else          { val_l0[c] = v1; idx_l0[c] = c*2+1; }
        }

        // Nivel 1: 8 ganadores → 4 ganadores
        ap_uint<8>  val_l1[4];
        class_idx_t idx_l1[4];
        #pragma HLS ARRAY_PARTITION variable=val_l1 complete
        #pragma HLS ARRAY_PARTITION variable=idx_l1 complete

        for (int c = 0; c < 4; c++) {
            #pragma HLS UNROLL
            if (val_l0[c*2] >= val_l0[c*2+1]) {
                val_l1[c] = val_l0[c*2]; idx_l1[c] = idx_l0[c*2];
            } else {
                val_l1[c] = val_l0[c*2+1]; idx_l1[c] = idx_l0[c*2+1];
            }
        }

        // Nivel 2: 4 ganadores → 2 ganadores
        ap_uint<8>  val_l2[2];
        class_idx_t idx_l2[2];
        #pragma HLS ARRAY_PARTITION variable=val_l2 complete
        #pragma HLS ARRAY_PARTITION variable=idx_l2 complete

        for (int c = 0; c < 2; c++) {
            #pragma HLS UNROLL
            if (val_l1[c*2] >= val_l1[c*2+1]) {
                val_l2[c] = val_l1[c*2]; idx_l2[c] = idx_l1[c*2];
            } else {
                val_l2[c] = val_l1[c*2+1]; idx_l2[c] = idx_l1[c*2+1];
            }
        }

        // Nivel 3: 2 ganadores → 1 ganador
        class_idx_t winner;
        if (val_l2[0] >= val_l2[1]) winner = idx_l2[0];
        else                         winner = idx_l2[1];

        // Comparar con Pk original para contar cambios
        if (winner != pk_orig) local_count++;

        // Acumular PPP índices en result_group
        int p = i % PPP;
        result_group.range(p*BITS_CLASS + BITS_CLASS-1, p*BITS_CLASS) = winner;

        // Emitir cuando el grupo está completo
        if (p == PPP-1) {
            result_out.write(result_group);
            result_group = 0;
        }
    }

    changed_count = local_count;
}