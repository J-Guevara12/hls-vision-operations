#include "perceptual_array.hpp"

// ============================================================
// stage_argmax_and_count — PPP=1
//
// Para cada pixel:
//   1. ArgMax de blend_vec_t → clase ganadora (árbol de 4 niveles)
//   2. Compara con Pk original → incrementa changed_count si difiere
//   3. Emite 1 result_group_t (4 bits = 1 clase) por pixel
//
// Con PPP=1, Pk original se lee una vez por pixel.
// ============================================================

void stage_argmax_and_count(
    hls::stream<blend_vec_t>&    blend_in,
    hls::stream<pgroup_t>&       pk_original_in,
    hls::stream<result_group_t>& result_out,
    int& changed_count,
    int width, int height) {

    const int TOTAL = width * height;
    int local_count = 0;

    Main_Loop: for (int i = 0; i < TOTAL; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT max=MAX_WIDTH*2160

        blend_vec_t blend_vec = blend_in.read();
        px_t pk_orig = (px_t)pk_original_in.read();

        // Nivel 0: 16 → 8
        ap_uint<8>  val_l0[8];
        class_idx_t idx_l0[8];
        #pragma HLS ARRAY_PARTITION variable=val_l0 complete
        #pragma HLS ARRAY_PARTITION variable=idx_l0 complete

        for (int c = 0; c < 8; c++) {
            #pragma HLS UNROLL
            ap_uint<8> v0 = blend_vec.range(c*2*8+7,  c*2*8);
            ap_uint<8> v1 = blend_vec.range(c*2*8+15, c*2*8+8);
            if (v0 >= v1) { val_l0[c] = v0; idx_l0[c] = c*2;   }
            else          { val_l0[c] = v1; idx_l0[c] = c*2+1; }
        }

        // Nivel 1: 8 → 4
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

        // Nivel 2: 4 → 2
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

        // Nivel 3: 2 → 1
        class_idx_t winner;
        if (val_l2[0] >= val_l2[1]) winner = idx_l2[0];
        else                         winner = idx_l2[1];

        if (winner != pk_orig) local_count++;

        result_out.write((result_group_t)winner);
    }

    changed_count = local_count;
}
