#include <iostream>
#include <iomanip>
#include "transformations.hpp"

// Función auxiliar para imprimir una matriz pequeña y verificar visualmente
void print_y_channel(const std::string& name, uint8_t* image, int w, int h) {
    std::cout << "--- " << name << " ---" << std::endl;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            // setfill y setw para que se vea alineado
            std::cout << std::setw(3) << std::setfill(' ') << (int)image[y*w + x] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Nueva función para verificar croma
void print_chroma_channels(const std::string& name, uint8_t* u_img, uint8_t* v_img, int w, int h) {
    std::cout << "--- " << name << " (U / V) ---" << std::endl;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w/2; x++) {
            std::cout << "[" << std::setw(3) << (int)u_img[y*(w/2) + x] 
                      << "," << std::setw(3) << (int)v_img[y*(w/2) + x] << "] ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    // 1. Configuración de la imagen de prueba
    const int TEST_WIDTH = 512;
    const int TEST_HEIGHT = 768;
    
    // Streams
    hls::stream<axis_t> src_stream("input_stream");
    hls::stream<axis_t> dst_stream("output_stream");

    // Buffers de software para verificación
    uint8_t input_img_y[TEST_WIDTH * TEST_HEIGHT];
    uint8_t input_img_u[TEST_WIDTH * TEST_HEIGHT / 2];
    uint8_t input_img_v[TEST_WIDTH * TEST_HEIGHT / 2];

    uint8_t out_y[TEST_WIDTH * TEST_HEIGHT];
    uint8_t out_u[TEST_WIDTH * TEST_HEIGHT / 2];
    uint8_t out_v[TEST_WIDTH * TEST_HEIGHT / 2];

    // 2. Generar Patrón de Prueba (Caja Blanca en fondo Gris Oscuro)
    // Esto es ideal para probar bordes.
    for (int i = 0; i < TEST_WIDTH * TEST_HEIGHT; i++) {
        int x = i % TEST_WIDTH;
        int y = i / TEST_WIDTH;
        
        // Fondo gris (50), Cuadro brillante en el centro (200)
        if (x >= 3 && x <= 6 && y >= 3 && y <= 6)
            input_img_y[i] = 50 + x;
        else
            input_img_y[i] = 200 + x;
    }
    for (int i = 0; i < (TEST_WIDTH * TEST_HEIGHT / 2); i++) {
        // U varía con la columna, V varía con la fila
        input_img_u[i] = (i % (TEST_WIDTH/2)) + 100; 
        input_img_v[i] = (i / (TEST_WIDTH/2)) + 50;
    }
    

    // 3. Empaquetar datos a Stream (Simulando YUYV 4:2:2)
    // Cada transacción envía 2 píxeles. Total transacciones = (W*H)/2
    for (int i = 0; i < (TEST_WIDTH * TEST_HEIGHT) / 2; i++) {
        axis_t temp;
        
        // Tomamos 2 píxeles de Y
        uint8_t y0 = input_img_y[i*2];
        uint8_t y1 = input_img_y[i*2 + 1];
        uint8_t u  = input_img_u[i];
        uint8_t v  = input_img_v[i];

        // Empaquetar: V | Y1 | U | Y0  (Little Endian)
        temp.data = (v << 24) | (y1 << 16) | (u << 8) | y0;
        
        // Señales de control AXI (importantes para que el IP no se cuelgue)
        temp.keep = -1; // Todos los bytes válidos
        temp.strb = -1;
        bool is_last = (i == ((TEST_WIDTH * TEST_HEIGHT) / 2) - 1);
        temp.last = is_last ? 1 : 0;

        src_stream.write(temp);
    }


    // 5. LLAMADA AL DUT (Device Under Test)
    transformation_3x3(src_stream, dst_stream, TEST_WIDTH, TEST_HEIGHT, 0xFFFF, false);

    std::cout << "Simulacion Terminada. Verificando resultados..." << std::endl;

    // 6. Desempaquetar y Validar
    int y_ptr = 0;
    int c_ptr = 0;
    bool chroma_error = false;

    while(!dst_stream.empty()) {
        axis_t res = dst_stream.read();
        
        out_y[y_ptr++] = res.data.range(7, 0);   // Y0
        uint8_t u_out  = res.data.range(15, 8);  // U
        out_y[y_ptr++] = res.data.range(23, 16); // Y1
        uint8_t v_out  = res.data.range(31, 24); // V

        out_u[c_ptr] = u_out;
        out_v[c_ptr] = v_out;

        // Validación automática: El croma debería estar retrasado 1 fila
        // Si el IP funciona bien, out_u[c_ptr] debe ser igual a input_img_u[c_ptr - width/2]
        if (c_ptr >= (TEST_WIDTH/2)) {
            if (u_out != input_img_u[c_ptr - (TEST_WIDTH/2)]) {
                std::cout << "Error en el valor de U" << std::endl;
            }

        }
        c_ptr++;
        if (res.last == 1) {
            std::cout << "Detectado LAST" << y_ptr << std::endl;
        }
    }

    // 7. Visualización
    print_y_channel("ENTRADA Y", input_img_y, TEST_WIDTH, TEST_HEIGHT);
    print_y_channel("SALIDA Y FILTRADA", out_y, TEST_WIDTH, TEST_HEIGHT);
     print_chroma_channels("CROMINANCIA Entrada", input_img_u, input_img_v, TEST_WIDTH, TEST_HEIGHT);
    print_chroma_channels("CROMINANCIA SALIDA", out_u, out_v, TEST_WIDTH, TEST_HEIGHT);

    return 0;
}
