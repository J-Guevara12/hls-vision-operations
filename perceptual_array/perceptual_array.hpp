#ifndef PERCEPTUAL_ARRAY_HPP
#define PERCEPTUAL_ARRAY_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// ============================================================
// Parámetros del sistema — modificar aquí para reconfigurar
// ============================================================
#ifndef WIN_SIZE
#define WIN_SIZE      7
#endif
#ifndef NUM_CLASSES
#define NUM_CLASSES   16
#endif
#ifndef MAX_WIDTH
#define MAX_WIDTH     4096
#endif
#ifndef PPP
#define PPP           1        // píxeles por ciclo (1 pixel = 8 bits de un word de 32)
#endif
#ifndef BITS_CLASS
#define BITS_CLASS    4        // log2(NUM_CLASSES)
#endif
#ifndef BITS_GAUSS
#define BITS_GAUSS    6        // bits efectivos de los coeficientes gaussianos
#endif
#ifndef BITS_ALPHA
#define BITS_ALPHA    6        // bits de alpha y C = Inv_K*(1-alpha)
#endif
#ifndef SHIFT
#define SHIFT         16       // shift fijo post-multiplicación
#endif
#define HALF_WIN      (WIN_SIZE / 2)                  // = 6
#define GAUSS_REGS    ((WIN_SIZE * WIN_SIZE + 3) / 4) // = 43 registros de 32 bits

// ============================================================
// Tipos de datos
// ============================================================

// Stream AXI de entrada/salida (32 bits: 4 píxeles × 8 bits)
typedef ap_axiu<32, 0, 0, 0> axis_t;

// Pixel individual: 4 bits P0 en MSB, 4 bits Pk en LSB
typedef ap_uint<8>  pixel_t;   // [7:4]=P0, [3:0]=Pk

// Grupo de PPP píxeles empaquetados
typedef ap_uint<PPP * BITS_CLASS> pgroup_t;   // 16 bits — solo Pk o solo P0

// Ventana de un solo píxel: WIN_SIZE×WIN_SIZE valores de BITS_CLASS bits
// Opción B: 1 ventana por ciclo entre WM y conv_engine
// Para migrar a opción A: cambiar a ap_uint<WIN_SIZE*WIN_SIZE*BITS_CLASS*PPP>
typedef ap_uint<WIN_SIZE * WIN_SIZE * BITS_CLASS> window_t;  // 676 bits

// Resultado de la convolución por clase para 1 píxel
// 49 entradas de 6 bits → máx 49*63 = 3087 → 12 bits
typedef ap_uint<12> conv_result_t;

// Vector de resultados de convolución para las NUM_CLASSES clases
// Empaquetado: 16 × 12 bits = 192 bits
typedef ap_uint<NUM_CLASSES * 12> conv_vec_t;

// Resultado del blend por clase para 1 píxel
// alpha*P0 + (1-alpha)*conv/K → cabe en 8 bits con los 6 bits de alpha
typedef ap_uint<8> blend_result_t;

// Vector de resultados del blend para las NUM_CLASSES clases
typedef ap_uint<NUM_CLASSES * 8> blend_vec_t;

// Índice ganador de ArgMax: 0..15 → 4 bits
typedef ap_uint<BITS_CLASS> class_idx_t;

// Grupo de PPP índices ganadores empaquetados
typedef ap_uint<PPP * BITS_CLASS> result_group_t;  // 16 bits

// Tipo interno para un píxel individual de la imagen
typedef ap_uint<BITS_CLASS> px_t;

// ============================================================
// Firmas de los stages internos
// ============================================================

// Stage 1 — Unpack
// Lee 1 paquete axis_t de 32 bits → emite PPP grupos separados P0 y Pk
// Emite P0 dos veces y Pk dos veces para satisfacer la restricción
// DATAFLOW de un único consumidor por stream.
void stage_unpack(
    hls::stream<axis_t>&    in,
    hls::stream<pgroup_t>&  p0_blend_out,   // → scale_and_blend (via FIFO retardo)
    hls::stream<pgroup_t>&  p0_pack_out,    // → pack (via FIFO retardo)
    hls::stream<pgroup_t>&  pk_wm_out,      // → window_manager
    hls::stream<pgroup_t>&  pk_argmax_out,  // → argmax_and_count (via FIFO retardo)
    int width, int height
);

// Stage 2 — Window Manager 13x13 custom
// Recibe pk_stream → emite 1 ventana por ciclo (PPP ventanas por grupo)
// Opción B: PPP writes por grupo al win_stream
void stage_window_manager(
    hls::stream<pgroup_t>&  pk_in,
    hls::stream<window_t>&  win_out,
    int width, int height
);

// Stage 3 — Motor de convolución
// Recibe 1 ventana → emite vector de conv_result_t para NUM_CLASSES clases
void stage_conv_engine(
    hls::stream<window_t>&   win_in,
    hls::stream<conv_vec_t>& conv_out,
    uint8_t gauss[WIN_SIZE][WIN_SIZE],
    int width, int height
);

// Stage 4 — Escala y ponderación α
// resultado[c] = alpha*(pk0==c) + C*conv[c] >> SHIFT
void stage_scale_and_blend(
    hls::stream<conv_vec_t>&  conv_in,
    hls::stream<pgroup_t>&    p0_in,       // P0 sincronizado via FIFO retardo
    hls::stream<blend_vec_t>& blend_out,
    uint8_t  alpha,
    uint16_t inv_k_1malpha,
    int width, int height
);

// Stage 5 — ArgMax y contador de cambios
// Encuentra la clase ganadora y cuenta píxeles que cambiaron Pk→Pk+1
void stage_argmax_and_count(
    hls::stream<blend_vec_t>&   blend_in,
    hls::stream<pgroup_t>&      pk_original_in,  // Pk original para comparar
    hls::stream<result_group_t>& result_out,
    int& changed_count,
    int width, int height
);

// Stage 6 — Pack
// Reagrupa PPP índices ganadores + PPP P0 en 1 paquete axis_t de 32 bits
void stage_pack(
    hls::stream<result_group_t>& result_in,
    hls::stream<pgroup_t>&       p0_pack_in,    // P0 para reempaquetar
    hls::stream<axis_t>&         out,
    int width, int height
);

// ============================================================
// Top level
// ============================================================
void perceptual_array(
    hls::stream<axis_t>& in_stream,
    hls::stream<axis_t>& out_stream,
    int                  width,
    int                  height,
    uint32_t             gauss_packed[GAUSS_REGS],  // 43 registros × 32 bits
    uint8_t              alpha,                      // 6 bits efectivos
    uint16_t             inv_k_1malpha,              // C = Inv_K*(1-alpha), hasta ~1569 → 11 bits
    int&                 changed_count
);

#endif // PERCEPTUAL_ARRAY_HPP
