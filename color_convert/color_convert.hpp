#ifndef COLOR_CONVERT_HPP
#define COLOR_CONVERT_HPP

#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"

// Definición de tipos para AXI-Stream (32 bits de ancho)
typedef ap_axiu<32, 0, 0, 0> axis_t;
typedef hls::stream<axis_t> axis_stream;
typedef ap_uint<8>  u8;
typedef ap_int<10>  coeff_t;
typedef ap_int<18>  calc_t; 

// Coeficientes en Punto Fijo (Escala 2^7 = 128)
// R = Y + 1.402*(V-128)  -> 1.402 * 128 = 179.45 -> 179
// B = Y + 1.772*(U-128)  -> 1.772 * 128 = 226.81 -> 227
// G = Y - 0.344*(U-128) - 0.714*(V-128)
//     0.344 * 128 = 44.03 -> 44
//     0.714 * 128 = 91.39 -> 91

#define C_RV 179
#define C_BU 227
#define C_GU 44
#define C_GV 91

// Declaración de la función principal
void yuyv2bgra(axis_stream& in_stream, axis_stream& out_stream, int width, int height);

#endif