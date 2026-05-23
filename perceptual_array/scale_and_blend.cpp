#include "perceptual_array.hpp"

// ============================================================
// stage_scale_and_blend
//
// Para cada píxel y cada clase aplica:
//
//   scaled[c]  = (conv[c] * inv_k_1malpha) >> SHIFT
//   p0_term[c] = (P0 == c) ? alpha : 0
//   result[c]  = scaled[c] + p0_term[c]
//
// Donde:
//   inv_k_1malpha = Inv_K * (1-alpha), precomputado en PS (12 bits)
//   alpha         = 6 bits sin signo
//   SHIFT         = 16 fijo
//
// El término P0 no necesita DSP — es un mux: si la clase c
// coincide con P0, se toma alpha; si no, cero.
//
// Recibe p0_in sincronizado via FIFO de retardo desde stage_unpack.
// La profundidad del FIFO absorbe la latencia del WM + conv_engine.
//
// Emite blend_vec_t: NUM_CLASSES resultados de 8 bits empaquetados.
// ============================================================

void stage_scale_and_blend(
    hls::stream<conv_vec_t>&  conv_in,
    hls::stream<pgroup_t>&    p0_in,
    hls::stream<blend_vec_t>& blend_out,
    uint8_t  alpha,
    uint16_t inv_k_1malpha,
    int width, int height) {

    const int TOTAL = width * height;
    pgroup_t p0_group = 0;

    Main_Loop: for (int i = 0; i < TOTAL; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT max=MAX_WIDTH*2160

        conv_vec_t conv_vec = conv_in.read();

        // Leer P0 del grupo correspondiente cada PPP píxeles
        if (i % PPP == 0) {
            p0_group = p0_in.read();
        }
        px_t p0_val = p0_group.range(
            (i % PPP)*BITS_CLASS + BITS_CLASS-1,
            (i % PPP)*BITS_CLASS);

        blend_vec_t result_vec = 0;

        for (int c = 0; c < NUM_CLASSES; c++) {
            #pragma HLS UNROLL

            // Extraer resultado de convolución para esta clase
            conv_result_t conv_val = conv_vec.range(c*14+13, c*14);

            // Escalar: conv * C >> SHIFT (1 DSP)
            // conv_val: 14 bits, inv_k_1malpha: 16 bits → producto: 30 bits → resultado: 14 bits
            ap_uint<30> scaled_full = (ap_uint<30>)conv_val * (ap_uint<16>)inv_k_1malpha;
            ap_uint<14> scaled_14   = (ap_uint<14>)(scaled_full >> SHIFT);
            ap_uint<8>  scaled      = (scaled_14 > 255) ? (ap_uint<8>)0xFF
                                                         : (ap_uint<8>)scaled_14;

            // Término P0: mux sin DSP
            ap_uint<8> p0_term = (p0_val == (px_t)c) ? (ap_uint<8>)alpha
                                                       : (ap_uint<8>)0;

            // Blend
            ap_uint<9> blended = (ap_uint<9>)scaled + (ap_uint<9>)p0_term;
            // Saturar a 8 bits
            ap_uint<8> result = blended[8] ? (ap_uint<8>)0xFF : (ap_uint<8>)blended;

            result_vec.range(c*8+7, c*8) = result;
        }

        blend_out.write(result_vec);
    }
}