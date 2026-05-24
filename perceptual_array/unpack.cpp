#include "perceptual_array.hpp"

// ============================================================
// stage_unpack — PPP=1
//
// Lee 1 word de 32 bits por ciclo.
// Formato de entrada (32 bits, solo bits [7:0] usados):
//   bits [7:4] = P0   (clase prior)
//   bits [3:0] = Pk   (clase actual)
//   bits [31:8] = no usados (relleno del clasificador externo)
//
// Emite 4 streams, 1 elemento de BITS_CLASS bits por ciclo:
//   p0_blend_out : P0 → stage_scale_and_blend (FIFO retardo largo)
//   p0_pack_out  : P0 → stage_pack            (FIFO retardo largo)
//   pk_wm_out    : Pk → stage_window_manager
//   pk_argmax_out: Pk → stage_argmax_and_count (FIFO retardo largo)
// ============================================================

void stage_unpack(
    hls::stream<axis_t>&   in,
    hls::stream<pgroup_t>& p0_blend_out,
    hls::stream<pgroup_t>& p0_pack_out,
    hls::stream<pgroup_t>& pk_wm_out,
    hls::stream<pgroup_t>& pk_argmax_out,
    int width, int height) {

    Main_Loop: for (int i = 0; i < width * height; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT max=MAX_WIDTH*2160

        axis_t pkt = in.read();
        pgroup_t p0 = pkt.data.range(7, 4);
        pgroup_t pk = pkt.data.range(3, 0);

        p0_blend_out.write(p0);
        p0_pack_out.write(p0);
        pk_wm_out.write(pk);
        pk_argmax_out.write(pk);
    }
}
