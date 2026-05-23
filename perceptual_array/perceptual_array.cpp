#include "perceptual_array.hpp"
#include "unpack.cpp"
#include "window_manager.cpp"
#include "conv_engine.cpp"
#include "scale_and_blend.cpp"
#include "max_and_count.cpp"
#include "pack.cpp"

// ============================================================
// perceptual_array — Top Level
//
// Pipeline DATAFLOW:
//
//   in_stream
//       │
//   stage_unpack ──────────────────────────────────────────┐
//       │pk_wm        │pk_argmax      │p0_blend  │p0_pack  │
//       │             │(FIFO retardo) │(FIFO)    │(FIFO)   │
//   stage_wm      pk_delayed      p0_delayed  p0_pack_d    │
//       │win                          │           │         │
//   stage_conv                        │           │         │
//       │conv                         │           │         │
//   stage_scale_and_blend ←──────────┘           │         │
//       │blend                                    │         │
//   stage_argmax_and_count ←── pk_delayed         │         │
//       │result          └──→ changed_count       │         │
//   stage_pack ←─────────────────────────────────┘         │
//       │                                                   │
//   out_stream                                              │
//                                                           │
// Nota: p0_pack_d es un segundo FIFO de retardo para P0    │
// que sincroniza con la latencia total del pipeline         │
// (WM + conv + blend + argmax)                             ─┘
//
// Latencia total estimada:
//   WM: HALF_WIN*(MAX_WIDTH/PPP) + HALF_WIN/PPP = 6*1024+2 = 6146 grupos
//   conv: 1 ciclo por ventana (PPP ventanas por grupo → PPP ciclos/grupo)
//   scale: 1 ciclo por ventana
//   argmax: 1 ciclo por ventana
//   Total en grupos: ≈ 6146 + PPP + PPP + PPP = 6158 grupos
//   Con 5% margen → DELAY_DEPTH = ceil(6158 * 1.05) = 6466 grupos
// ============================================================

#define DELAY_DEPTH 6466

void perceptual_array(
    hls::stream<axis_t>& in_stream,
    hls::stream<axis_t>& out_stream,
    int                  width,
    int                  height,
    uint32_t             gauss_packed[GAUSS_REGS],
    uint8_t              alpha,
    uint16_t             inv_k_1malpha,
    int&                 changed_count) {

    #pragma HLS INTERFACE axis      port=in_stream
    #pragma HLS INTERFACE axis      port=out_stream
    #pragma HLS INTERFACE s_axilite port=width          bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=height         bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=gauss_packed   bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=alpha          bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=inv_k_1malpha  bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=changed_count  bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return         bundle=CTRL

    #pragma HLS DATAFLOW

    // ── Desempaquetar kernel gaussiano (combinacional) ────────
    uint8_t gauss[WIN_SIZE][WIN_SIZE];
    #pragma HLS ARRAY_PARTITION variable=gauss complete dim=0

    for (int i = 0; i < WIN_SIZE * WIN_SIZE; i++) {
        #pragma HLS UNROLL
        gauss[i / WIN_SIZE][i % WIN_SIZE] =
            (gauss_packed[i / 4] >> ((i % 4) * 8)) & 0xFF;
    }

    // ── Streams internos ─────────────────────────────────────

    // unpack → wm y FIFOs de retardo
    hls::stream<pgroup_t> pk_wm_stream      ("pk_wm");
    hls::stream<pgroup_t> pk_argmax_stream  ("pk_argmax");
    hls::stream<pgroup_t> p0_blend_stream   ("p0_blend");
    hls::stream<pgroup_t> p0_pack_stream    ("p0_pack");
    #pragma HLS STREAM variable=pk_wm_stream    depth=2
    #pragma HLS STREAM variable=pk_argmax_stream depth=DELAY_DEPTH
    #pragma HLS STREAM variable=p0_blend_stream  depth=DELAY_DEPTH
    #pragma HLS STREAM variable=p0_pack_stream   depth=DELAY_DEPTH

    // wm → conv
    hls::stream<window_t> win_stream("win");
    #pragma HLS STREAM variable=win_stream depth=8

    // conv → scale_and_blend
    hls::stream<conv_vec_t> conv_stream("conv");
    #pragma HLS STREAM variable=conv_stream depth=4

    // scale_and_blend → argmax
    hls::stream<blend_vec_t> blend_stream("blend");
    #pragma HLS STREAM variable=blend_stream depth=4

    // argmax → pack
    hls::stream<result_group_t> result_stream("result");
    #pragma HLS STREAM variable=result_stream depth=4

    // ── Pipeline ─────────────────────────────────────────────
    stage_unpack(
        in_stream,
        p0_blend_stream, p0_pack_stream,
        pk_wm_stream, pk_argmax_stream,
        width, height);

    stage_window_manager(
        pk_wm_stream, win_stream,
        width, height);

    stage_conv_engine(
        win_stream, conv_stream,
        gauss, width, height);

    stage_scale_and_blend(
        conv_stream, p0_blend_stream, blend_stream,
        alpha, inv_k_1malpha, width, height);

    stage_argmax_and_count(
        blend_stream, pk_argmax_stream, result_stream,
        changed_count, width, height);

    stage_pack(
        result_stream, p0_pack_stream, out_stream,
        width, height);
}