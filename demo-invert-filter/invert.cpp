#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"

#define BUS_WIDTH 32
#define PIXEL_SIZE 32
#define PPP BUS_WIDTH/PIXEL_SIZE

// Definimos un pixel estándar de 32 bits (RGBA o similar)
// ap_axiu<32,1,1,1> significa: 32 bits de datos, y bits extra de control
typedef ap_axiu<32,0,0,0> pixel_stream;

void image_invert(hls::stream<pixel_stream>& stream_in, hls::stream<pixel_stream>& stream_out, int width, int height) {
    // 1. INTERFACES (Los "Cables")
    #pragma HLS INTERFACE axis port=stream_in
    #pragma HLS INTERFACE axis port=stream_out
    #pragma HLS INTERFACE s_axilite port=width bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=height bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // Variables temporales
    pixel_stream pixel_in;
    pixel_stream pixel_out;
    

    // 2. EL BUCLE PRINCIPAL
    // Iteramos por todos los pixeles de la imagen
    Row_Loop: for (uint16_t y = 0; y < height; y++) {
        #pragma HLS LOOP_TRIPCOUNT max=2160
        #pragma HLS LOOP_FLATTEN off
        Col_Loop: for (uint16_t x = 0; x < width; x++) {
            #pragma HLS LOOP_TRIPCOUNT max=4096
            #pragma HLS PIPELINE II=1
            // Leer del stream (bloqueante si está vacío)
            stream_in >> pixel_in;

            pixel_out.data = pixel_in.data ^ 0x00FFFFFF;
            // Copiar las señales de control (Side-channels)
            // 'keep', 'strb', 'user', 'id', 'dest' deben pasar intactos
            pixel_out.keep = pixel_in.keep;
            pixel_out.strb = pixel_in.strb;

            // TLAST: Es CRÍTICO. Indica al DMA que la imagen terminó.
            // Si no se gestiona bien, el DMA se cuelga esperando el final.
            if (x == (width - 1) && y == (height - 1)) {
                pixel_out.last = 1; 
            } else {
                pixel_out.last = 0;
                }

            // Escribir a la salida
            stream_out << pixel_out;
        }
    }
}