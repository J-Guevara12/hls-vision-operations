#ifndef HLS_VISION_UTILS_HPP
#define HLS_VISION_UTILS_HPP

#include <ap_int.h>
#include <ap_fixed.h>

namespace hls_lib {

    // Clase para gestionar el flujo de pixeles 2D
    template<typename T, int WIN_SIZE, int MAX_W>
    class WindowManager {
    private:
        // Line Buffer: Almacena las líneas necesarias para la ventana
        // Se requieren (WIN_SIZE - 1) líneas en BRAM
        T line_buf[WIN_SIZE - 1][MAX_W];
        
        // Window Buffer: Registros de alta velocidad para el kernel
        T window[WIN_SIZE][WIN_SIZE];

    public:
        WindowManager() {
            #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
            #pragma HLS ARRAY_PARTITION variable=window complete dim=0
        }

        // Procesa un pixel y desliza la ventana
        // II=1 garantizado por la estructura de buffers
        void shift(T new_pixel, int col) {
            #pragma HLS INLINE

            // 1. Desplazamiento Horizontal de la Ventana
            for (int r = 0; r < WIN_SIZE; r++) {
                for (int c = 0; c < WIN_SIZE - 1; c++) {
                    window[r][c] = window[r][c + 1];
                }
            }

            // 2. Carga Vertical (Desde LineBuffers hacia Ventana)
            // El pixel actual entra en la última fila
            window[WIN_SIZE - 1][WIN_SIZE - 1] = new_pixel;

            // Movemos datos de los LineBuffers a la ventana y los actualizamos
            for (int r = 0; r < WIN_SIZE - 1; r++) {
                T val_from_buf = line_buf[r][col];
                window[r][WIN_SIZE - 1] = val_from_buf;
                
                // El pixel de la fila de abajo sube al buffer de arriba
                if (r < WIN_SIZE - 2) {
                    line_buf[r][col] = line_buf[r + 1][col];
                } else {
                    // La última fila de buffers recibe el pixel nuevo
                    line_buf[r][col] = new_pixel;
                }
            }
        }

        // Acceso rápido a cualquier pixel de la ventana
        T get(int r, int c) const {
            #pragma HLS INLINE
            return window[r][c];
        }
    };

    // Función de Clipping usando tipos nativos de HLS (Sin comparadores)
    // Se basa en el modo AP_SAT de ap_fixed
    template<int BITS_OUT, typename T>
    ap_uint<BITS_OUT> saturate_cast(T val) {
        #pragma HLS INLINE
        // Calculamos los límites dinámicamente
        const int max_val = (1 << BITS_OUT) - 1;
        
        if (val < 0) {
            return 0;
        } else if (val > max_val) {
            return max_val;
        } else {
            return (ap_uint<BITS_OUT>)val;
        }
    }
}

#endif