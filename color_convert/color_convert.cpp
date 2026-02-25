#include "color_convert.hpp"

const coeff_t C_R_V = C_RV;
const coeff_t C_G_U = C_GU;
const coeff_t C_G_V = C_GV;
const coeff_t C_B_U = C_BU;

// FUNCIÓN DE SATURACIÓN POR BITS (La clave del Timing)
u8 saturate_cast(calc_t val) {
    #pragma HLS INLINE
    // 1. Bit 17 es el SIGNO (ap_int<18>). Si es 1, el número es negativo.
    bool is_negative = val[17]; 
    
    // 2. Bits 16 y 15 detectan OVERFLOW positivo.
    // El valor máximo válido es 255*128 = 32640 (0111 1111 1000 0000 en binario)
    // Si el bit 16 o 15 son '1' (y no es negativo), nos pasamos de 255.
    bool is_overflow = !is_negative && (val[16] | val[15]);

    // 3. Mux de salida (Lógica combinacional pura, muy rápida)
    if (is_negative) {
        return 0;      // Underflow -> Negro
    } else if (is_overflow) {
        return 255;    // Overflow -> Blanco
    } else {
        return val.range(14, 7); // Cortamos los bits decimales (>> 7)
    }
}

void yuyv2bgra(axis_stream& in_stream, axis_stream& out_stream, int width, int height) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=width bundle=control
    #pragma HLS INTERFACE s_axilite port=height bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // YUYV tiene 2 píxeles por cada 32 bits (4 bytes)
    // Total de lecturas = (Total Píxeles) / 2
    int num_pixels = width * height;
    int num_reads = num_pixels >> 1;
    

    loop_processing:
    for (int i = 0; i < num_reads; i++) {
        #pragma HLS PIPELINE II=2

        // 1. Lectura del Bus (32 bits)
        // Formato Little Endian: [V | Y1 | U | Y0] (Bits 31..0)
        axis_t in_val = in_stream.read();
        
        u8 y0 = in_val.data.range(7, 0);
        u8 u  = in_val.data.range(15, 8);
        u8 y1 = in_val.data.range(23, 16);
        u8 v  = in_val.data.range(31, 24);

        // 2. Preparación Aritmética
        // Restamos 128 a los cromas para centrarlos en 0.
        // Convertimos a 16 bits SIGNADOS para manejar negativos.
        calc_t u_diff = (calc_t)u - 128;
        calc_t v_diff = (calc_t)v - 128;

        // Escalado de Y para coincidir con el shift de 7 (Y * 128)
        calc_t y0_s = (calc_t)y0 << 7;
        calc_t y1_s = (calc_t)y1 << 7;

        // --- MATEMÁTICA DEL COLOR (Ahora en 18 bits) ---
        // Al usar variables intermedias, HLS puede balancear mejor los ciclos
        
        // Píxel 0
        calc_t r0_raw = y0_s + (C_R_V * v_diff);
        calc_t g0_raw = y0_s - (C_G_U * u_diff) - (C_G_V * v_diff);
        calc_t b0_raw = y0_s + (C_B_U * u_diff);

        // Píxel 1
        calc_t r1_raw = y1_s + (C_R_V * v_diff);
        calc_t g1_raw = y1_s - (C_G_U * u_diff) - (C_G_V * v_diff);
        calc_t b1_raw = y1_s + (C_B_U * u_diff);

        // --- EMPAQUETADO Y SALIDA ---
        axis_t out_val0;

        out_val0.data = (255, saturate_cast(r0_raw), saturate_cast(g0_raw), saturate_cast(b0_raw));
        // Manejo de señales de control (keep, strb, last)
        out_val0.keep = 0xF;
        out_val0.strb = 0xF;
        out_val0.last = 0; // El pixel 0 nunca es el último del par
        out_stream.write(out_val0);


        // 5. Salida Pixel 1 (BGRA)
        axis_t out_val1;
        out_val1.data = (255, saturate_cast(r1_raw), saturate_cast(g1_raw), saturate_cast(b1_raw));

        
        out_val1.keep = 0xF;
        out_val1.strb = 0xF;
        // Si la entrada tenía TLAST activo, lo pasamos en el segundo píxel de salida
        out_val1.last = in_val.last; 
        out_stream.write(out_val1);
    }
}