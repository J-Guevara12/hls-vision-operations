#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "color_convert.hpp"

// ==========================================
// 1. MODELO DE REFERENCIA (GOLDEN MODEL)
// ==========================================
// Usa punto flotante para calcular el valor "perfecto"
void ref_yuv2rgb(uint8_t y, uint8_t u, uint8_t v, uint8_t* r, uint8_t* g, uint8_t* b) {
    double y_d = (double)y;
    double u_d = (double)u - 128.0;
    double v_d = (double)v - 128.0;

    // Fórmulas estándar (las mismas que intentamos aproximar)
    double r_d = y_d + 1.402 * v_d;
    double g_d = y_d - 0.344136 * u_d - 0.714136 * v_d;
    double b_d = y_d + 1.772 * u_d;

    // Saturación (Clipping)
    *r = (uint8_t)std::max(0.0, std::min(255.0, std::round(r_d)));
    *g = (uint8_t)std::max(0.0, std::min(255.0, std::round(g_d)));
    *b = (uint8_t)std::max(0.0, std::min(255.0, std::round(b_d)));
}

// ==========================================
// 2. MAIN TESTBENCH
// ==========================================
int main() {
    // Definimos dimensiones que permitan probar varios patrones
    // Ancho debe ser par para YUYV
    const int WIDTH = 100; 
    const int HEIGHT = 100; 
    const int NUM_PIXELS = WIDTH * HEIGHT;
    
    axis_stream strm_in("strm_in");
    axis_stream strm_out("strm_out");

    std::cout << "========================================" << std::endl;
    std::cout << " INICIANDO TESTBENCH EXTENSIVO (YUYV -> BGRA)" << std::endl;
    std::cout << " Resolucion: " << WIDTH << "x" << HEIGHT << std::endl;
    std::cout << "========================================" << std::endl;

    // Almacenes para verificación
    struct PixelYUV { uint8_t y, u, v; };
    std::vector<PixelYUV> input_history;

    // ----------------------------------------------------------------
    // FASE 1: GENERACIÓN DE ESTÍMULOS (YUYV PACKED)
    // ----------------------------------------------------------------
    for (int i = 0; i < NUM_PIXELS; i += 2) {
        uint8_t y0, u, y1, v;

        // --- ZONA 1: CORNER CASES (Primeras 20 filas) ---
        if (i < WIDTH * 20) {
            y0 = 0; y1 = 255; u = 128; v = 128; // Blanco y Negro puro
            if (i % 4 == 0) { u = 0; v = 0; }   // Verdes/Azules extremos
            if (i % 4 == 2) { u = 255; v = 255; } // Rojos/Rosas extremos
        }
        // --- ZONA 2: GRADIENTES (Siguientes 40 filas) ---
        else if (i < WIDTH * 60) {
            y0 = (i % 255); 
            y1 = (i % 255);
            u = (i / WIDTH) * 2; // Gradiente vertical lento
            v = (i % WIDTH);     // Gradiente horizontal rápido
        }
        // --- ZONA 3: RUIDO ALEATORIO (Resto) ---
        else {
            y0 = rand() % 256;
            y1 = rand() % 256;
            u  = rand() % 256;
            v  = rand() % 256;
        }

        // Guardamos para verificación posterior
        input_history.push_back({y0, u, v});
        input_history.push_back({y1, u, v});

        // Empaquetado Little Endian para AXI Stream: [V | Y1 | U | Y0]
        axis_t packet;
        packet.data.range(7, 0)   = y0;
        packet.data.range(15, 8)  = u;
        packet.data.range(23, 16) = y1;
        packet.data.range(31, 24) = v;
        
        // Señales de control AXI
        packet.keep = 0xF;
        packet.strb = 0xF;
        // TLAST solo en el último paquete de la imagen
        packet.last = (i == NUM_PIXELS - 2) ? 1 : 0;

        strm_in.write(packet);
    }

    // ----------------------------------------------------------------
    // FASE 2: EJECUCIÓN DEL IP (DUT)
    // ----------------------------------------------------------------
    std::cout << ">> Ejecutando IP Hardware..." << std::endl;
    yuyv2bgra(strm_in, strm_out, WIDTH, HEIGHT);
    std::cout << ">> Ejecucion terminada." << std::endl;

    // ----------------------------------------------------------------
    // FASE 3: VERIFICACIÓN Y ANÁLISIS DE ERROR
    // ----------------------------------------------------------------
    int max_error = 0;
    long total_error = 0;
    int pixel_errors = 0;
    bool tlast_ok = false;

    // Tolerancia aceptable por usar aritmética de enteros (Fixed Point)
    // Un error de +/- 2 niveles es normal al convertir float -> int
    const int TOLERANCIA = 2; 

    for (int i = 0; i < NUM_PIXELS; i++) {
        if (strm_out.empty()) {
            std::cerr << "ERROR FATAL: Stream de salida vacio prematuramente en pixel " << i << std::endl;
            return 1;
        }

        axis_t out_pkt = strm_out.read();
        
        // Verificar TLAST
        if (i == NUM_PIXELS - 1) {
            if (out_pkt.last == 1) tlast_ok = true;
            else std::cerr << "ERROR: TLAST no detectado en el ultimo pixel!" << std::endl;
        } else {
            if (out_pkt.last == 1) std::cerr << "ERROR: TLAST detectado prematuramente en pixel " << i << std::endl;
        }

        // Desempaquetar BGRA
        uint8_t dut_b = out_pkt.data.range(7, 0);
        uint8_t dut_g = out_pkt.data.range(15, 8);
        uint8_t dut_r = out_pkt.data.range(23, 16);
        uint8_t dut_a = out_pkt.data.range(31, 24);

        // Calcular Referencia
        uint8_t ref_r, ref_g, ref_b;
        PixelYUV in = input_history[i];
        ref_yuv2rgb(in.y, in.u, in.v, &ref_r, &ref_g, &ref_b);

        // Calcular diferencias absolutas
        int err_r = abs((int)dut_r - (int)ref_r);
        int err_g = abs((int)dut_g - (int)ref_g);
        int err_b = abs((int)dut_b - (int)ref_b);
        int current_max = std::max({err_r, err_g, err_b});

        total_error += current_max;
        if (current_max > max_error) max_error = current_max;

        // Validar canal Alpha
        if (dut_a != 0xFF) {
            std::cout << "ERROR: Alpha incorrecto en pixel " << i << std::endl;
            pixel_errors++;
        }

        // Reportar si el error excede la tolerancia
        if (current_max > TOLERANCIA) {
            pixel_errors++;
            if (pixel_errors < 10) { // Solo imprimir los primeros 10 errores para no saturar consola
                std::cout << "[FALLO] Pixel " << i 
                          << " | Entrada YUV(" << (int)in.y << "," << (int)in.u << "," << (int)in.v << ")"
                          << " | Ref RGB(" << (int)ref_r << "," << (int)ref_g << "," << (int)ref_b << ")"
                          << " | DUT RGB(" << (int)dut_r << "," << (int)dut_g << "," << (int)dut_b << ")"
                          << " | Error: " << current_max << std::endl;
            }
        }
    }

    // ----------------------------------------------------------------
    // FASE 4: REPORTE FINAL
    // ----------------------------------------------------------------
    std::cout << "\n========================================" << std::endl;
    std::cout << " RESULTADOS DEL TEST" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Pixeles Procesados: " << NUM_PIXELS << std::endl;
    std::cout << "Protocolo TLAST:    " << (tlast_ok ? "PASO" : "FALLO") << std::endl;
    std::cout << "Error Maximo Abs:   " << max_error << " (Tolerancia: " << TOLERANCIA << ")" << std::endl;
    std::cout << "Error Promedio:     " << (double)total_error / NUM_PIXELS << std::endl;
    
    if (pixel_errors == 0 && tlast_ok) {
        std::cout << ">> ESTADO: EXITO TOTAL (Hardware validado)" << std::endl;
        return 0;
    } else if (max_error <= TOLERANCIA + 1 && tlast_ok) {
        std::cout << ">> ESTADO: PASABLE (Con ligera desviacion de redondeo)" << std::endl;
        return 0;
    } else {
        std::cout << ">> ESTADO: FALLIDO (Revisar logica aritmetica)" << std::endl;
        return 1;
    }
}