#include "perceptual_array.hpp"

// ============================================================
// stage_unpack
//
// Lee 1 paquete de 32 bits por ciclo.
// Formato de entrada (32 bits = 4 píxeles × 8 bits):
//   bits [7:0]   = píxel 0: [7:4]=P0, [3:0]=Pk
//   bits [15:8]  = píxel 1: [15:12]=P0, [11:8]=Pk
//   bits [23:16] = píxel 2: [23:20]=P0, [19:16]=Pk
//   bits [31:24] = píxel 3: [31:28]=P0, [27:24]=Pk
//
// Emite 4 streams por ciclo:
//   p0_blend_out : P0 para stage_scale_and_blend (via FIFO retardo largo)
//   p0_pack_out  : P0 para stage_pack (via FIFO retardo largo)
//   pk_wm_out    : Pk para stage_window_manager
//   pk_argmax_out: Pk para stage_argmax_and_count (via FIFO retardo largo)
// ============================================================

void stage_unpack(
    hls::stream<axis_t>&   in,
    hls::stream<pgroup_t>& p0_blend_out,
    hls::stream<pgroup_t>& p0_pack_out,
    hls::stream<pgroup_t>& pk_wm_out,
    hls::stream<pgroup_t>& pk_argmax_out,
    int width, int height) {

    const int GROUPS = width / PPP;

    Row_Loop: for (int y = 0; y < height; y++) {
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT max=2160
        Col_Loop: for (int g = 0; g < GROUPS; g++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT max=MAX_WIDTH/PPP

            axis_t pkt = in.read();

            pgroup_t p0_group = 0;
            pgroup_t pk_group = 0;

            for (int p = 0; p < PPP; p++) {
                #pragma HLS UNROLL
                ap_uint<8>          raw = pkt.data.range(p*8+7, p*8);
                ap_uint<BITS_CLASS> p0  = raw.range(7, 4);
                ap_uint<BITS_CLASS> pk  = raw.range(3, 0);
                p0_group.range(p*BITS_CLASS+BITS_CLASS-1, p*BITS_CLASS) = p0;
                pk_group.range(p*BITS_CLASS+BITS_CLASS-1, p*BITS_CLASS) = pk;
            }

            p0_blend_out.write(p0_group);
            p0_pack_out.write(p0_group);
            pk_wm_out.write(pk_group);
            pk_argmax_out.write(pk_group);
        }
    }
}