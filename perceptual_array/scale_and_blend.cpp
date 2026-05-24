#include "perceptual_array.hpp"

// ============================================================
// stage_scale_and_blend — PPP=1
//
// Para cada pixel aplica:
//   scaled[c]  = clamp((conv[c] * inv_k_1malpha) >> SHIFT, 0, 255)
//   p0_term[c] = (P0 == c) ? alpha : 0
//   result[c]  = clamp(scaled[c] + p0_term[c], 0, 255)
//
// Con PPP=1, P0 se lee una vez por pixel (sin agrupación).
// ============================================================

void stage_scale_and_blend(
    hls::stream<conv_vec_t>&  conv_in,
    hls::stream<pgroup_t>&    p0_in,
    hls::stream<blend_vec_t>& blend_out,
    uint8_t  alpha,
    uint16_t inv_k_1malpha,
    int width, int height) {

    const int TOTAL = width * height;

    Main_Loop: for (int i = 0; i < TOTAL; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT max=MAX_WIDTH*2160

        conv_vec_t conv_vec = conv_in.read();
        px_t p0_val = (px_t)p0_in.read();

        blend_vec_t result_vec = 0;

        for (int c = 0; c < NUM_CLASSES; c++) {
            #pragma HLS UNROLL

            conv_result_t conv_val = conv_vec.range(c*12+11, c*12);

            ap_uint<30> scaled_full = (ap_uint<30>)conv_val * (ap_uint<16>)inv_k_1malpha;
            ap_uint<14> scaled_14   = (ap_uint<14>)(scaled_full >> SHIFT);
            ap_uint<8>  scaled      = (scaled_14 > 255) ? (ap_uint<8>)0xFF
                                                         : (ap_uint<8>)scaled_14;

            ap_uint<8> p0_term = (p0_val == (px_t)c) ? (ap_uint<8>)alpha
                                                       : (ap_uint<8>)0;

            ap_uint<9> blended = (ap_uint<9>)scaled + (ap_uint<9>)p0_term;
            ap_uint<8> result = blended[8] ? (ap_uint<8>)0xFF : (ap_uint<8>)blended;

            result_vec.range(c*8+7, c*8) = result;
        }

        blend_out.write(result_vec);
    }
}
