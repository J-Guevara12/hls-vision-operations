#include "perceptual_array.hpp"

// ============================================================
// stage_pack — PPP=1
//
// Lee 1 resultado (Pk+1) y 1 valor P0 por ciclo.
// Formato de salida (32 bits, bits [7:0] usados):
//   bits [7:4] = P0
//   bits [3:0] = Pk+1
//   bits [31:8] = 0
//
// TLAST=1 en el último pixel del frame.
// ============================================================

void stage_pack(
    hls::stream<result_group_t>& result_in,
    hls::stream<pgroup_t>&       p0_pack_in,
    hls::stream<axis_t>&         out,
    int width, int height) {

    const int TOTAL = width * height;

    Main_Loop: for (int i = 0; i < TOTAL; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT max=MAX_WIDTH*2160

        result_group_t pk1 = result_in.read();
        pgroup_t       p0v = p0_pack_in.read();

        axis_t pkt;
        pkt.data = 0;
        pkt.data.range(7, 4) = p0v;
        pkt.data.range(3, 0) = pk1;
        pkt.keep = 0xF;
        pkt.strb = 0xF;
        pkt.last = (i == TOTAL - 1) ? 1 : 0;

        out.write(pkt);
    }
}
