#include "kernels.hpp"
#include "hls_vision_utils.hpp"
#include <iostream>

void filter_y_3x3(hls::stream<axis_t>& in_stream, 
                   hls::stream<axis_t>& out_stream,
                   int width, int height,
                   short k00, short k01, short k02,
                   short k10, short k11, short k12,
                   short k20, short k21, short k22,
                   uint8_t shift_val) {

    // Interfaces para conectar con Zynq (PS)
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=width bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=height bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=shift_val bundle=CTRL
    // Coeficientes individuales
    #pragma HLS INTERFACE s_axilite port=k00 bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=k01 bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=k02 bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=k10 bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=k11 bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=k12 bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=k20 bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=k21 bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=k22 bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // 1. Line Buffers: Guardan Y de las filas anteriores
    // Usamos 'static' para que se mapee a BRAM
    static hls_lib::WindowManager<uint8_t, 3, MAX_WIDTH> wm;

    // 2. Chroma Buffer: Necesitamos guardar U/V para sincronizarlos
    // con el retraso que introduce el filtro (1 fila de latencia)
    static uint16_t chroma_queue[MAX_WIDTH>>1]; // Guarda pares UV
    #pragma HLS BIND_STORAGE variable=chroma_queue type=RAM_T2P impl=BRAM
    #pragma HLS DEPENDENCE variable=chroma_queue inter false

    // Registro para sostener el dato de salida durante los 2 ciclos (par/impar)
    uint16_t uv_output_reg[2];

    // Registros para mantener el estado entre iteraciones par/impar
    axis_t packet_in;
    uint8_t y_impar_reg;
    uint8_t y_filt_par_reg;
    

    Row_Loop: for (int y = 0; y < height + 1; y++) {
        #pragma HLS LOOP_TRIPCOUNT max=2160
        Col_Loop: for (int x = 0; x < width + 1; x++) {
            #pragma HLS LOOP_TRIPCOUNT max=4096
            #pragma HLS PIPELINE II=1 

            uint16_t uv_from_prev_row = chroma_queue[x>>1];
            uint8_t y_current = 0;
            uint8_t u_current, v_current;

            if (y != height && x != width){
                // 1. GESTIÓN DE ENTRADA (Lectura cada 2 ciclos)
                if ((x & 1) == 0) {
                    packet_in = in_stream.read();
                    y_current = packet_in.data.range(7, 0);   // Y_par
                    y_impar_reg = packet_in.data.range(23, 16);

                    u_current = packet_in.data.range(15, 8);  // U
                    v_current = packet_in.data.range(31, 24);

                    
                    
                    // Sincronización de croma (usando el buffer de línea)
                    
                    uv_output_reg[1] = uv_output_reg[00];
                    uv_output_reg[0] = uv_from_prev_row;

                    chroma_queue[x>>1] = (v_current << 8) | u_current;
                    

                } else {
                    y_current = y_impar_reg;
                }

                // 2. ACTUALIZAR VENTANA (Compartida)
                wm.shift(y_current, x);
                
            }
            

            if (y != 0 && x != 0){
                bool is_border = (y  == 1 || y == height || x == 1 || x == width);
                uint8_t y_filt;


                if (is_border) {
                    y_filt = 0; 
                } else {
                    
                    int acc = wm.get(0,0)*k00 + wm.get(0,1)*k01 + wm.get(0,2)*k02 +
                            wm.get(1,0)*k10 + wm.get(1,1)*k11 + wm.get(1,2)*k12 +
                            wm.get(2,0)*k20 + wm.get(2,1)*k21 + wm.get(2,2)*k22;
                    
                    y_filt = hls_lib::saturate_cast<8>(acc >> shift_val);
                }
                
                // 4. GESTIÓN DE SALIDA (Escritura cada 2 ciclos)
                if ((x & 1) == 1) {
                    y_filt_par_reg = y_filt;
                } else {
                    uint16_t uv_output = 0; 
                    if (y == height){
                        uv_output = chroma_queue[(x>>1)-1];
                    } else {
                        uv_output = x == width ? uv_output_reg[0]: uv_output_reg[1];
                    }
                    
                    axis_t packet_out;
                    packet_out.data.range(7, 0)   = y_filt_par_reg;
                    packet_out.data.range(15, 8)  = uv_output & 0xFF;        // U
                    packet_out.data.range(23, 16) = y_filt;                    // Y_impar filtrado
                    packet_out.data.range(31, 24) = (uv_output >> 8) & 0xFF; // V

                    
                    // Propagar señales de control
                    packet_out.keep = packet_in.keep;
                    packet_out.strb = packet_in.strb;
                    packet_out.last = (x == width) && (y == height) ? 1 : 0;   // EOL

                    out_stream.write(packet_out);
                }

            }
        }
    }
}
