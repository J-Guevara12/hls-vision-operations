#include "perceptual_array.hpp"

// ============================================================
// stage_pack
//
// Reagrupa PPP índices ganadores (Pk+1) + PPP valores P0
// en 1 paquete axis_t de 32 bits.
//
// Formato de salida (igual que entrada para encadenar iteraciones):
//   bits [7:0]   = píxel 0: [7:4]=P0, [3:0]=Pk+1
//   bits [15:8]  = píxel 1: [15:12]=P0, [11:8]=Pk+1
//   bits [23:16] = píxel 2: [23:20]=P0, [19:16]=Pk+1
//   bits [31:24] = píxel 3: [31:28]=P0, [27:24]=Pk+1
//
// p0_pack_in llega sincronizado — mismo P0 que viajó por el
// FIFO de retardo largo, ahora con un segundo tap para el pack.
// TLAST se genera en el último paquete del frame.
// ============================================================

void stage_pack(
    hls::stream<result_group_t>& result_in,
    hls::stream<pgroup_t>&       p0_pack_in,
    hls::stream<axis_t>&         out,
    int width, int height) {

    const int GROUPS = width / PPP;

    Row_Loop: for (int y = 0; y < height; y++) {
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT max=2160
        Col_Loop: for (int g = 0; g < GROUPS; g++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT max=MAX_WIDTH/PPP

            result_group_t result = result_in.read();
            pgroup_t       p0     = p0_pack_in.read();

            axis_t pkt;
            pkt.data = 0;

            for (int p = 0; p < PPP; p++) {
                #pragma HLS UNROLL
                ap_uint<BITS_CLASS> pk1 = result.range(p*BITS_CLASS+BITS_CLASS-1,
                                                        p*BITS_CLASS);
                ap_uint<BITS_CLASS> p0v = p0.range(p*BITS_CLASS+BITS_CLASS-1,
                                                    p*BITS_CLASS);
                // [7:4]=P0, [3:0]=Pk+1
                pkt.data.range(p*8+7, p*8+4) = p0v;
                pkt.data.range(p*8+3, p*8)   = pk1;
            }

            pkt.keep = 0xF;
            pkt.strb = 0xF;
            pkt.last = (y == height-1 && g == GROUPS-1) ? 1 : 0;

            out.write(pkt);
        }
    }
}