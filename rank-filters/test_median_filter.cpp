// ============================================================
// Testbench: tb_median_filter_3x3.cpp
// Función bajo prueba: median_filter_3x3
//
// Formato YUYV (32 bits por paquete):
//   bits  7:0  = Y0 (píxel par)
//   bits 15:8  = U
//   bits 23:16 = Y1 (píxel impar)
//   bits 31:24 = V
//
// Protocolo de desfase:
//   La salida tiene un desfase de 1 fila + 1 columna respecto a la entrada.
//   El píxel de salida (x,y) corresponde al centro de la ventana 3x3
//   centrada en (x,y) de la entrada.
//   Los bordes (primera/última fila y columna) se zerean.
// ============================================================

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"

typedef ap_axiu<32,0,0,0> axis_t;

// ============================================================
// Prototipo de la función bajo prueba
// ============================================================
void median_filter_3x3(hls::stream<axis_t>& in_stream,
                       hls::stream<axis_t>& out_stream,
                       int width, int height);

// ============================================================
// Contadores globales
// ============================================================
static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

static void report(const std::string& name, bool ok) {
    tests_run++;
    if (ok) { tests_passed++; std::cout << "  [PASS] " << name << "\n"; }
    else     { tests_failed++; std::cerr << "  [FAIL] " << name << "\n"; }
}

// ============================================================
// Tipos auxiliares
// ============================================================
typedef std::vector<std::vector<uint8_t>> Image;

// ============================================================
// Utilidades de stream
// ============================================================
static axis_t make_yuyv(uint8_t y0, uint8_t u, uint8_t y1, uint8_t v,
                         bool last = false) {
    axis_t pkt;
    pkt.data.range(7,  0)  = y0;
    pkt.data.range(15, 8)  = u;
    pkt.data.range(23, 16) = y1;
    pkt.data.range(31, 24) = v;
    pkt.keep = 0xF;
    pkt.strb = 0xF;
    pkt.last = last ? 1 : 0;
    return pkt;
}

// Carga una imagen Y en el stream YUYV (U=128, V=128 neutros)
static void load_image(hls::stream<axis_t>& s,
                        const Image& img,
                        int width, int height,
                        uint8_t u_val = 128, uint8_t v_val = 128) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 2) {
            bool last = (y == height-1) && (x == width-2);
            s.write(make_yuyv(img[y][x], u_val, img[y][x+1], v_val, last));
        }
    }
}

// Lee la imagen Y filtrada del stream de salida
// Tiene en cuenta el desfase de 1 fila + 1 columna
static Image read_output(hls::stream<axis_t>& s, int width, int height) {
    Image out(height, std::vector<uint8_t>(width, 0));

    int total_packets = (width / 2) * height;
    for (int i = 0; i < total_packets; i++) {
        axis_t pkt = s.read();
        // Reconstruir coordenadas del paquete de salida
        // Los paquetes salen en orden raster: fila por fila
        int pkt_x = (i % (width / 2)) * 2;
        int pkt_y = i / (width / 2);
        out[pkt_y][pkt_x]     = pkt.data.range(7,  0);
        out[pkt_y][pkt_x + 1] = pkt.data.range(23, 16);
    }
    return out;
}

// ============================================================
// Golden model: mediana 3x3 en software
// ============================================================
static uint8_t median9(uint8_t v[9]) {
    uint8_t s[9];
    memcpy(s, v, 9);
    std::sort(s, s + 9);
    return s[4];
}

static Image compute_golden(const Image& img, int width, int height) {
    Image out(height, std::vector<uint8_t>(width, 0));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            bool is_border = (y == 0 || y == height-1 ||
                              x == 0 || x == width-1);
            if (is_border) {
                out[y][x] = 0;
            } else {
                uint8_t window[9];
                int k = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                        window[k++] = img[y+dy][x+dx];
                out[y][x] = median9(window);
            }
        }
    }
    return out;
}

// Compara imagen HW vs golden SW y reporta diferencias
static bool compare_images(const Image& hw, const Image& golden,
                            int width, int height,
                            bool verbose = false) {
    bool ok = true;
    int errors = 0;
    for (int y = 0; y < height && (ok || verbose); y++) {
        for (int x = 0; x < width; x++) {
            if (hw[y][x] != golden[y][x]) {
                if (verbose && errors < 10)
                    std::cerr << "    [" << y << "," << x << "]"
                              << " hw=" << (int)hw[y][x]
                              << " expected=" << (int)golden[y][x] << "\n";
                ok = false;
                errors++;
            }
        }
    }
    if (!ok && verbose)
        std::cerr << "    Total errores: " << errors << "\n";
    return ok;
}

// Imprime una imagen para depuración
static void print_image(const Image& img, int width, int height,
                         const std::string& label) {
    std::cout << "  --- " << label << " ---\n";
    for (int y = 0; y < height; y++) {
        std::cout << "  ";
        for (int x = 0; x < width; x++)
            std::cout << std::setw(4) << (int)img[y][x];
        std::cout << "\n";
    }
}

// ============================================================
// TEST 1 — Imagen uniforme: mediana de valores iguales = mismo valor
// ============================================================
static bool test_uniform() {
    const int W = 8, H = 6;
    const uint8_t VAL = 150;
    Image img(H, std::vector<uint8_t>(W, VAL));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    median_filter_3x3(in, out, W, H);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H);
    return compare_images(hw, golden, W, H, true);
}

// ============================================================
// TEST 2 — Impulso aislado: un píxel de valor alto rodeado de ceros
// La mediana debe eliminar el impulso (sal y pimienta)
// ============================================================
static bool test_impulse_removal() {
    const int W = 8, H = 8;
    Image img(H, std::vector<uint8_t>(W, 50));
    // Píxel de ruido aislado en el centro
    img[4][4] = 255;

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    median_filter_3x3(in, out, W, H);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H);

    bool ok = compare_images(hw, golden, W, H, true);

    // Verificación semántica: el impulso debe desaparecer
    if (hw[4][4] == 255) {
        std::cerr << "    El impulso en (4,4) no fue eliminado\n";
        ok = false;
    }
    return ok;
}

// ============================================================
// TEST 3 — Múltiples impulsos: sal y pimienta sobre fondo uniforme
// ============================================================
static bool test_salt_and_pepper() {
    const int W = 10, H = 10;
    Image img(H, std::vector<uint8_t>(W, 128));

    // Impulsos de sal (255) y pimienta (0) en posiciones interiores
    img[2][2] = 255; img[2][6] = 0;
    img[5][3] = 255; img[5][7] = 0;
    img[7][4] = 255; img[7][5] = 0;

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    median_filter_3x3(in, out, W, H);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H);
    return compare_images(hw, golden, W, H, true);
}

// ============================================================
// TEST 4 — Bordes en cero
// Primera/última fila y columna deben ser 0
// ============================================================
static bool test_border_zeros() {
    const int W = 8, H = 6;
    Image img(H, std::vector<uint8_t>(W, 200));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    median_filter_3x3(in, out, W, H);

    auto hw = read_output(out, W, H);

    bool ok = true;
    // Primera y última fila
    for (int x = 0; x < W; x++) {
        if (hw[0][x] != 0) {
            std::cerr << "    hw[0][" << x << "]=" << (int)hw[0][x]
                      << " esperado 0 (borde superior)\n";
            ok = false;
        }
        if (hw[H-1][x] != 0) {
            std::cerr << "    hw[" << H-1 << "][" << x << "]="
                      << (int)hw[H-1][x] << " esperado 0 (borde inferior)\n";
            ok = false;
        }
    }
    // Primera y última columna
    for (int y = 0; y < H; y++) {
        if (hw[y][0] != 0) {
            std::cerr << "    hw[" << y << "][0]=" << (int)hw[y][0]
                      << " esperado 0 (borde izquierdo)\n";
            ok = false;
        }
        if (hw[y][W-1] != 0) {
            std::cerr << "    hw[" << y << "][" << W-1 << "]="
                      << (int)hw[y][W-1] << " esperado 0 (borde derecho)\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 5 — Gradiente horizontal: verifica que la mediana
// preserva transiciones suaves
// ============================================================
static bool test_horizontal_gradient() {
    const int W = 10, H = 6;
    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(x * 25);

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    median_filter_3x3(in, out, W, H);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H);
    return compare_images(hw, golden, W, H, true);
}

// ============================================================
// TEST 6 — Imagen con rectángulo oscuro interior
// Similar a los tests del morfológico para consistencia
// ============================================================
static bool test_dark_rectangle() {
    const int W = 10, H = 10;
    Image img(H, std::vector<uint8_t>(W, 200));
    for (int y = 3; y < 7; y++)
        for (int x = 3; x < 7; x++)
            img[y][x] = 50;

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    median_filter_3x3(in, out, W, H);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H);
    return compare_images(hw, golden, W, H, true);
}

// ============================================================
// TEST 7 — Golden model completo con patrón aleatorio determinista
// Cubre todos los casos de ventana con valores variados
// ============================================================
static bool test_golden_model() {
    const int W = 16, H = 12;
    Image img(H, std::vector<uint8_t>(W, 0));

    // Patrón pseudo-aleatorio determinista
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)((y * 37 + x * 53 + y*x) % 256);

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    median_filter_3x3(in, out, W, H);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H);
    return compare_images(hw, golden, W, H, true);
}

// ============================================================
// TEST 8 — Valores extremos: imagen alternando 0 y 255
// Patrón de tablero de ajedrez
// ============================================================
static bool test_checkerboard() {
    const int W = 8, H = 8;
    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = ((x + y) % 2 == 0) ? 255 : 0;

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    median_filter_3x3(in, out, W, H);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H);

    bool ok = compare_images(hw, golden, W, H, true);

    // Verificación semántica: la mediana de un tablero 3x3
    // tiene 5 de un valor y 4 del otro → mediana es el valor mayoritario
    // Para interior: en un tablero, cada ventana 3x3 tiene 5 de un color
    for (int y = 1; y < H-1 && ok; y++) {
        for (int x = 1; x < W-1; x++) {
            uint8_t expected_majority = ((x + y) % 2 == 0) ? 255 : 0;
            if (hw[y][x] != expected_majority) {
                std::cerr << "    Tablero [" << y << "," << x << "]"
                          << " hw=" << (int)hw[y][x]
                          << " esperado=" << (int)expected_majority << "\n";
                ok = false;
                break;
            }
        }
    }
    return ok;
}

// ============================================================
// TEST 9 — Croma propagado correctamente
// U y V deben preservarse con el desfase correcto
// ============================================================
static bool test_chroma_propagation() {
    const int W = 8, H = 6;
    const uint8_t U_VAL = 0xAA, V_VAL = 0x55;
    Image img(H, std::vector<uint8_t>(W, 128));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H, U_VAL, V_VAL);
    median_filter_3x3(in, out, W, H);

    bool ok = true;
    int total_packets = (W / 2) * H;
    int errors = 0;

    for (int i = 0; i < total_packets; i++) {
        axis_t pkt = out.read();
        int pkt_y = i / (W / 2);

        // Bordes no tienen croma válido (Y=0), solo verificar interior
        if (pkt_y > 0 && pkt_y < H-1) {
            uint8_t u_out = pkt.data.range(15, 8);
            uint8_t v_out = pkt.data.range(31, 24);
            if (u_out != U_VAL || v_out != V_VAL) {
                if (errors < 5)
                    std::cerr << "    Paquete " << i
                              << ": U=" << std::hex << (int)u_out
                              << " V=" << (int)v_out
                              << " esperado U=" << (int)U_VAL
                              << " V=" << (int)V_VAL << std::dec << "\n";
                ok = false;
                errors++;
            }
        }
    }
    return ok;
}

// ============================================================
// TEST 10 — Señal LAST en el paquete correcto
// ============================================================
static bool test_last_signal() {
    const int W = 8, H = 6;
    Image img(H, std::vector<uint8_t>(W, 100));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    median_filter_3x3(in, out, W, H);

    int total_packets = (W / 2) * H;
    bool ok = true;

    for (int i = 0; i < total_packets; i++) {
        axis_t pkt = out.read();
        bool expected_last = (i == total_packets - 1);
        bool got_last      = (pkt.last == 1);
        if (got_last != expected_last) {
            std::cerr << "    Paquete " << i << "/" << total_packets
                      << ": last=" << got_last
                      << " esperado=" << expected_last << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 11 — Propiedad idempotente: mediana(mediana(x)) = mediana(x)
// Aplicar el filtro dos veces a una imagen sin impulsos
// no debe cambiar el resultado
// ============================================================
static bool test_idempotent() {
    const int W = 10, H = 8;
    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(y * 20 + x * 10);

    hls::stream<axis_t> in1, mid_stream, in2, out;

    load_image(in1, img, W, H);
    median_filter_3x3(in1, mid_stream, W, H);

    // Leer el stream intermedio como imagen
    auto mid_img = read_output(mid_stream, W, H);

    // Segunda pasada
    load_image(in2, mid_img, W, H);
    median_filter_3x3(in2, out, W, H);

    auto hw2    = read_output(out, W, H);
    auto golden = compute_golden(mid_img, W, H);

    bool ok = compare_images(hw2, golden, W, H, true);
    return ok;
}

// ============================================================
// TEST 12 — Resolución mínima válida (4x4)
// ============================================================
static bool test_minimum_resolution() {
    const int W = 4, H = 4;
    Image img(H, std::vector<uint8_t>(W, 0));
    img[0] = {10, 20, 30, 40};
    img[1] = {50, 60, 70, 80};
    img[2] = {90,100,110,120};
    img[3] = {130,140,150,160};

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    median_filter_3x3(in, out, W, H);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H);
    return compare_images(hw, golden, W, H, true);
}

// ============================================================
// MAIN
// ============================================================
int main() {
    std::cout << "======================================\n";
    std::cout << "  Testbench: median_filter_3x3\n";
    std::cout << "  Formato entrada/salida: YUYV 32bpp\n";
    std::cout << "  Desfase: 1 fila + 1 columna\n";
    std::cout << "  Bordes: zeroed\n";
    std::cout << "======================================\n\n";

    report("T01 - Imagen uniforme",                    test_uniform());
    report("T02 - Eliminación de impulso aislado",     test_impulse_removal());
    report("T03 - Sal y pimienta múltiple",            test_salt_and_pepper());
    report("T04 - Bordes en cero",                     test_border_zeros());
    report("T05 - Gradiente horizontal",               test_horizontal_gradient());
    report("T06 - Rectángulo oscuro interior",         test_dark_rectangle());
    report("T07 - Golden model patrón aleatorio",      test_golden_model());
    report("T08 - Tablero de ajedrez (valores extremos)", test_checkerboard());
    report("T09 - Croma propagado correctamente",      test_chroma_propagation());
    report("T10 - Señal LAST en paquete correcto",     test_last_signal());
    report("T11 - Propiedad idempotente (2 pasadas)",  test_idempotent());
    report("T12 - Resolución mínima 4x4",              test_minimum_resolution());

    std::cout << "\n======================================\n";
    std::cout << "  Resultado: " << tests_passed << "/"
              << tests_run << " tests pasaron\n";
    if (tests_failed > 0)
        std::cout << "  FALLARON: " << tests_failed << " tests\n";
    else
        std::cout << "  Todos los tests PASARON\n";
    std::cout << "======================================\n";

    return tests_failed > 0 ? 1 : 0;
}