#include <ap_axi_sdata.h>
#include <cstdint>
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>

typedef ap_axiu<32,0,0,0> axis_t;

#define MAX_WIDTH 4096 
#define PPP 2 // Pixels per package


void histogram_yuyv(hls::stream<axis_t>& in_stream, 
                   hls::stream<axis_t>& out_stream,
                   int width, int height) {
    // Interfaces para conectar con Zynq (PS)
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=width bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=height bundle=CTRL

    
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // 1. Line Buffers: Guardan Y de las filas anteriores
    // Usamos 'static' para que se mapee a BRAM

    // 2. Chroma Buffer: Necesitamos guardar U/V para sincronizarlos
    // con el retraso que introduce el filtro (1 fila de latencia)
    static uint32_t col_register[MAX_WIDTH]; // Guarda El acumulado de Y para las filas
    static uint32_t y_counter[PPP][256]; // Contador de los diferentes valores de Y
    

    #pragma HLS BIND_STORAGE variable=col_register type=RAM_T2P impl=BRAM
    #pragma HLS BIND_STORAGE variable=y_counter type=RAM_T2P impl=URAM
    #pragma HLS DEPENDENCE variable=y_counter inter false

    // Registros de Forwarding (uno por carril)
    static uint8_t prev_y[PPP] = {255, 255};
    static uint32_t prev_count[PPP] = {0, 0};


    // Reset al inicio de cada frame
    Reset_Loop:
    for (int i = 0; i < 256; i++) {
        #pragma HLS PIPELINE II=1
        for (int lane=0; lane < PPP; lane++) {
            #pragma HLS UNROLL
            y_counter[lane][i] = 0;
        }
        
    }

    
    Col_Reset:
    for (int x = 0; x < width; x++) {
        #pragma HLS PIPELINE II=1
        col_register[x] = 0;
    }

    // Registros para mantener el estado entre iteraciones par/impar
    axis_t packet_in;
    Row_Loop: for (int y = 0; y < height; y++) {
        uint32_t y_col = 0;
        Col_Loop: for (int x = 0; x < width; x+=PPP) {
            #pragma HLS PIPELINE II=1 

            // 1. GESTIÓN DE ENTRADA 
            packet_in = in_stream.read();


            for (int l = 0; l < PPP; l++) {
                #pragma HLS UNROLL
                uint8_t y_val = packet_in.data.range((l<<4)+7, l<<4);
                uint32_t count;

                // LÓGICA DE BYPASS: Si el píxel es igual al procesado 
                // por este carril en el ciclo anterior, usamos el registro.
                if (y_val == prev_y[l]) {
                    count = prev_count[l] + 1;
                } else {
                    count = y_counter[l][y_val] + 1;
                }

                // Guardamos en memoria y en registro de forwarding
                y_counter[l][y_val] = count;
                prev_y[l] = y_val;
                prev_count[l] = count;

                y_col += y_val;
                col_register[x+l] += y_val;
            }            
        }
        axis_t packet_out;
        packet_out.data  = y_col;
        

        
        // Propagar señales de control
        packet_out.keep = packet_in.keep;
        packet_out.strb = packet_in.strb;
        packet_out.last = 0;   // EOL

        out_stream.write(packet_out);
    }
    for (int x = 0; x < width; x++) {
        #pragma HLS PIPELINE II=1 
        axis_t packet_out;
        packet_out.data  = col_register[x];
        

        
        // Propagar señales de control
        packet_out.keep = packet_in.keep;
        packet_out.strb = packet_in.strb;
        packet_out.last = 0;   // EOL

        out_stream.write(packet_out);
    }
    for (int x = 0; x < 256; x++) {
        #pragma HLS PIPELINE II=1
        uint32_t sum = 0;
        axis_t packet_out;
        for (int lane=0; lane < PPP; lane++) {
            #pragma HLS UNROLL
            sum += y_counter[lane][x];
        }
        packet_out.data  = sum;
        

        
        // Propagar señales de control
        packet_out.keep = packet_in.keep;
        packet_out.strb = packet_in.strb;
        packet_out.last = x == 255? 1: 0;   // EOL

        out_stream.write(packet_out);
    }
}
