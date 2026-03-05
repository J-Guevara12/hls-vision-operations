// ============================================================
// Testbench: tb_filter_y_3x3.cpp
// Función bajo prueba: filter_y_3x3
//
// Formato YUYV (32 bits por paquete):
//   bits  7:0  = Y0 (píxel par)
//   bits 15:8  = U
//   bits 23:16 = Y1 (píxel impar)
//   bits 31:24 = V
//
// Protocolo de desfase:
//   La salida tiene un desfase de 1 fila + 1 columna.
//   Los bordes (primera/última fila y columna) se zerean.
//
// Kernels de prueba:
//   Identidad, Gaussiano, Laplaciano, Sobel Gx, Sobel Gy
// ============================================================

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <climits>

#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "kernels.hpp"

// ============================================================
// Prototipo de la función bajo prueba
// ============================================================
void filter_y_3x3(hls::stream<axis_t>& in_stream,
                  hls::stream<axis_t>& out_stream,
                  int width, int height,
                  short k00, short k01, short k02,
                  short k10, short k11, short k12,
                  short k20, short k21, short k22,
                  uint8_t shift_val);

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
// Tipos y estructuras
// ============================================================
typedef std::vector<std::vector<uint8_t>> Image;

struct Kernel3x3 {
    short k[3][3];
    uint8_t shift;
    std::string name;
};

// Kernels de referencia
static const Kernel3x3 K_IDENTITY = {
    {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}, 0, "Identidad"
};
static const Kernel3x3 K_GAUSSIAN = {
    {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}}, 4, "Gaussiano (div 16)"
};
static const Kernel3x3 K_LAPLACIAN = {
    {{0, 1, 0}, {1, -4, 1}, {0, 1, 0}}, 0, "Laplaciano"
};
static const Kernel3x3 K_SOBEL_GX = {
    {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}, 0, "Sobel Gx"
};
static const Kernel3x3 K_SOBEL_GY = {
    {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}, 0, "Sobel Gy"
};
static const Kernel3x3 K_SHARPEN = {
    {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}}, 0, "Sharpening"
};
static const Kernel3x3 K_BOX = {
    {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, 3, "Box blur (div 8 aprox)"
};

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

static Image read_output(hls::stream<axis_t>& s, int width, int height) {
    Image out(height, std::vector<uint8_t>(width, 0));
    int total_packets = (width / 2) * height;
    for (int i = 0; i < total_packets; i++) {
        axis_t pkt = s.read();
        int pkt_x = (i % (width / 2)) * 2;
        int pkt_y = i / (width / 2);
        out[pkt_y][pkt_x]     = (uint8_t)pkt.data.range(7,  0);
        out[pkt_y][pkt_x + 1] = (uint8_t)pkt.data.range(23, 16);
    }
    return out;
}

// Wrapper que llama al IP con un kernel estructurado
static void run_filter(hls::stream<axis_t>& in, hls::stream<axis_t>& out,
                        int width, int height, const Kernel3x3& k) {
    filter_y_3x3(in, out, width, height,
                 k.k[0][0], k.k[0][1], k.k[0][2],
                 k.k[1][0], k.k[1][1], k.k[1][2],
                 k.k[2][0], k.k[2][1], k.k[2][2],
                 k.shift);
}

// ============================================================
// Golden model: convolución 3x3 en software
// ============================================================
static uint8_t saturate(int val) {
    if (val < 0)   return 0;
    if (val > 255) return 255;
    return (uint8_t)val;
}

static Image compute_golden(const Image& img, int width, int height,
                              const Kernel3x3& k) {
    Image out(height, std::vector<uint8_t>(width, 0));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            bool is_border = (y == 0 || y == height-1 ||
                              x == 0 || x == width-1);
            if (is_border) {
                out[y][x] = 0;
            } else {
                int acc = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                        acc += img[y+dy][x+dx] * k.k[dy+1][dx+1];
                out[y][x] = saturate(acc >> k.shift);
            }
        }
    }
    return out;
}

static bool compare_images(const Image& hw, const Image& golden,
                             int width, int height,
                             bool verbose = true) {
    bool ok = true;
    int errors = 0;
    for (int y = 0; y < height; y++) {
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
// TEST 1 — Kernel identidad: salida == entrada (interior)
// ============================================================
static bool test_identity() {
    const int W = 8, H = 6;
    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(y * 20 + x * 10);

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run_filter(in, out, W, H, K_IDENTITY);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H, K_IDENTITY);

    bool ok = compare_images(hw, golden, W, H, true);

    // Verificación semántica: interior debe ser idéntico a la entrada
    for (int y = 1; y < H-1 && ok; y++)
        for (int x = 1; x < W-1 && ok; x++)
            if (hw[y][x] != img[y][x]) {
                std::cerr << "    Identidad falló en [" << y << "," << x << "]"
                          << " hw=" << (int)hw[y][x]
                          << " original=" << (int)img[y][x] << "\n";
                ok = false;
            }
    return ok;
}

// ============================================================
// TEST 2 — Bordes en cero para cualquier kernel
// ============================================================
static bool test_border_zeros() {
    const int W = 8, H = 6;
    Image img(H, std::vector<uint8_t>(W, 200));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run_filter(in, out, W, H, K_GAUSSIAN);

    auto hw = read_output(out, W, H);
    bool ok = true;

    for (int x = 0; x < W; x++) {
        if (hw[0][x] != 0)   { std::cerr << "    hw[0][" << x << "]=" << (int)hw[0][x] << " esperado 0\n"; ok=false; }
        if (hw[H-1][x] != 0) { std::cerr << "    hw[" << H-1 << "][" << x << "]=" << (int)hw[H-1][x] << " esperado 0\n"; ok=false; }
    }
    for (int y = 0; y < H; y++) {
        if (hw[y][0] != 0)   { std::cerr << "    hw[" << y << "][0]=" << (int)hw[y][0] << " esperado 0\n"; ok=false; }
        if (hw[y][W-1] != 0) { std::cerr << "    hw[" << y << "][" << W-1 << "]=" << (int)hw[y][W-1] << " esperado 0\n"; ok=false; }
    }
    return ok;
}

// ============================================================
// TEST 3 — Gaussiano: imagen uniforme → misma imagen (sin bordes)
// El Gaussiano de una constante es la misma constante
// ============================================================
static bool test_gaussian_uniform() {
    const int W = 8, H = 6;
    const uint8_t VAL = 100;
    Image img(H, std::vector<uint8_t>(W, VAL));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run_filter(in, out, W, H, K_GAUSSIAN);

    auto hw = read_output(out, W, H);
    bool ok = true;

    // Interior debe ser VAL (1*100 + 2*100 + ... = 16*100, shift 4 = 100)
    for (int y = 1; y < H-1 && ok; y++)
        for (int x = 1; x < W-1 && ok; x++)
            if (hw[y][x] != VAL) {
                std::cerr << "    Gaussiano uniforme [" << y << "," << x << "]"
                          << " hw=" << (int)hw[y][x]
                          << " esperado=" << (int)VAL << "\n";
                ok = false;
            }
    return ok;
}

// ============================================================
// TEST 4 — Laplaciano: imagen uniforme → cero en interior
// El laplaciano de una constante es 0
// ============================================================
static bool test_laplacian_uniform() {
    const int W = 8, H = 6;
    Image img(H, std::vector<uint8_t>(W, 128));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run_filter(in, out, W, H, K_LAPLACIAN);

    auto hw = read_output(out, W, H);
    bool ok = true;

    for (int y = 1; y < H-1 && ok; y++)
        for (int x = 1; x < W-1 && ok; x++)
            if (hw[y][x] != 0) {
                std::cerr << "    Laplaciano uniforme [" << y << "," << x << "]"
                          << " hw=" << (int)hw[y][x] << " esperado=0\n";
                ok = false;
            }
    return ok;
}

// ============================================================
// TEST 5 — Sobel Gx: borde vertical debe tener respuesta máxima
// Imagen con mitad izquierda=0, mitad derecha=255
// ============================================================
static bool test_sobel_gx_vertical_edge() {
    const int W = 10, H = 8;
    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = W/2; x < W; x++)
            img[y][x] = 255;

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run_filter(in, out, W, H, K_SOBEL_GX);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H, K_SOBEL_GX);

    bool ok = compare_images(hw, golden, W, H, true);

    // Verificación semántica: columnas alejadas del borde deben ser 0
    for (int y = 1; y < H-1 && ok; y++) {
        if (hw[y][1] != 0) {
            std::cerr << "    Sobel Gx: hw[" << y << "][1]=" << (int)hw[y][1]
                      << " esperado=0 (zona uniforme izquierda)\n";
            ok = false;
        }
        if (hw[y][W-2] != 0) {
            std::cerr << "    Sobel Gx: hw[" << y << "][" << W-2 << "]="
                      << (int)hw[y][W-2]
                      << " esperado=0 (zona uniforme derecha)\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 6 — Sobel Gy: borde horizontal debe tener respuesta máxima
// Imagen con mitad superior=0, mitad inferior=255
// ============================================================
static bool test_sobel_gy_horizontal_edge() {
    const int W = 10, H = 10;
    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = H/2; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = 255;

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run_filter(in, out, W, H, K_SOBEL_GY);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H, K_SOBEL_GY);

    bool ok = compare_images(hw, golden, W, H, true);

    // Filas alejadas del borde deben ser 0
    for (int x = 1; x < W-1 && ok; x++) {
        if (hw[1][x] != 0) {
            std::cerr << "    Sobel Gy: hw[1][" << x << "]=" << (int)hw[1][x]
                      << " esperado=0\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 7 — Golden model completo con patrón pseudo-aleatorio
// Cubre todas las posiciones de la ventana con valores variados
// ============================================================
static bool test_golden_model(const Kernel3x3& k) {
    const int W = 16, H = 12;
    Image img(H, std::vector<uint8_t>(W, 0));

    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)((y * 37 + x * 53 + y*x*7) % 256);

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run_filter(in, out, W, H, k);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H, k);

    return compare_images(hw, golden, W, H, true);
}

// ============================================================
// TEST 8 — Saturación: resultado negativo → 0, resultado >255 → 255
// Laplaciano sobre impulso → valores negativos saturan a 0
// ============================================================
static bool test_saturation() {
    const int W = 8, H = 8;
    Image img(H, std::vector<uint8_t>(W, 128));
    img[4][4] = 255;  // impulso que genera negativos en vecinos con Laplaciano

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run_filter(in, out, W, H, K_LAPLACIAN);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H, K_LAPLACIAN);

    // Verificar que no hay wrap-around (que sería señal de bug en saturación)
    bool ok = compare_images(hw, golden, W, H, true);

    // Verificación adicional: ningún píxel puede tener wrap-around
    for (int y = 0; y < H && ok; y++)
        for (int x = 0; x < W && ok; x++)
            if (abs((int)hw[y][x] - (int)golden[y][x]) > 1) {
                std::cerr << "    Posible wrap-around en [" << y << "," << x << "]"
                          << " hw=" << (int)hw[y][x]
                          << " golden=" << (int)golden[y][x] << "\n";
                ok = false;
            }
    return ok;
}

// ============================================================
// TEST 9 — Shift_val correcto: Gaussiano con shift=4 vs shift=0
// Con shift=0 el resultado debería saturar a 255 para imagen media
// ============================================================
static bool test_shift_val() {
    const int W = 8, H = 6;
    Image img(H, std::vector<uint8_t>(W, 100));

    // Con shift=4: resultado = (16*100) >> 4 = 100
    hls::stream<axis_t> in1, out1;
    load_image(in1, img, W, H);
    run_filter(in1, out1, W, H, K_GAUSSIAN);  // shift=4
    auto hw_shift4 = read_output(out1, W, H);

    // Con shift=0: resultado = 16*100 = 1600 → saturado a 255
    Kernel3x3 k_no_shift = K_GAUSSIAN;
    k_no_shift.shift = 0;
    hls::stream<axis_t> in2, out2;
    load_image(in2, img, W, H);
    run_filter(in2, out2, W, H, k_no_shift);
    auto hw_shift0 = read_output(out2, W, H);

    bool ok = true;
    // Con shift=4 el interior debe ser 100
    if (hw_shift4[2][2] != 100) {
        std::cerr << "    Shift=4: hw[2][2]=" << (int)hw_shift4[2][2]
                  << " esperado=100\n";
        ok = false;
    }
    // Con shift=0 el interior debe saturar a 255
    if (hw_shift0[2][2] != 255) {
        std::cerr << "    Shift=0: hw[2][2]=" << (int)hw_shift0[2][2]
                  << " esperado=255 (saturado)\n";
        ok = false;
    }
    return ok;
}

// ============================================================
// TEST 10 — Croma propagado correctamente
// U y V deben preservarse con el desfase correcto
// ============================================================
static bool test_chroma_propagation() {
    const int W = 8, H = 6;
    const uint8_t U_VAL = 0xAA, V_VAL = 0x55;
    Image img(H, std::vector<uint8_t>(W, 128));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H, U_VAL, V_VAL);
    run_filter(in, out, W, H, K_IDENTITY);

    bool ok = true;
    int total_packets = (W / 2) * H;
    int errors = 0;

    for (int i = 0; i < total_packets; i++) {
        axis_t pkt = out.read();
        int pkt_y = i / (W / 2);

        if (pkt_y > 0 && pkt_y < H-1) {
            uint8_t u_out = (uint8_t)pkt.data.range(15, 8);
            uint8_t v_out = (uint8_t)pkt.data.range(31, 24);
            if (u_out != U_VAL || v_out != V_VAL) {
                if (errors < 5)
                    std::cerr << "    Paquete " << i
                              << ": U=0x" << std::hex << (int)u_out
                              << " V=0x" << (int)v_out
                              << " esperado U=0x" << (int)U_VAL
                              << " V=0x" << (int)V_VAL << std::dec << "\n";
                ok = false;
                errors++;
            }
        }
    }
    return ok;
}

// ============================================================
// TEST 11 — Señal LAST en el paquete correcto
// ============================================================
static bool test_last_signal() {
    const int W = 8, H = 6;
    Image img(H, std::vector<uint8_t>(W, 100));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run_filter(in, out, W, H, K_GAUSSIAN);

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
// TEST 12 — Kernel de todos ceros: salida debe ser cero en interior
// ============================================================
static bool test_zero_kernel() {
    const int W = 8, H = 6;
    Image img(H, std::vector<uint8_t>(W, 200));
    Kernel3x3 k_zero = {{{0,0,0},{0,0,0},{0,0,0}}, 0, "Cero"};

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run_filter(in, out, W, H, k_zero);

    auto hw = read_output(out, W, H);
    bool ok = true;

    for (int y = 0; y < H && ok; y++)
        for (int x = 0; x < W && ok; x++)
            if (hw[y][x] != 0) {
                std::cerr << "    Kernel cero: hw[" << y << "][" << x << "]="
                          << (int)hw[y][x] << " esperado=0\n";
                ok = false;
            }
    return ok;
}

// ============================================================
// TEST 13 — Conteo exacto de paquetes de salida
// ============================================================
static bool test_packet_count() {
    const int W = 8, H = 6;
    Image img(H, std::vector<uint8_t>(W, 100));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run_filter(in, out, W, H, K_GAUSSIAN);

    int count = 0;
    while (!out.empty()) { out.read(); count++; }

    int expected = (W / 2) * H;
    bool ok = (count == expected);
    if (!ok)
        std::cerr << "    Paquetes recibidos: " << count
                  << " esperados: " << expected << "\n";
    return ok;
}

// ============================================================
// TEST 14 — Resolución mínima válida (4x4)
// ============================================================
static bool test_minimum_resolution() {
    const int W = 4, H = 4;
    Image img = {
        {10, 20, 30, 40},
        {50, 60, 70, 80},
        {90,100,110,120},
        {130,140,150,160}
    };

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run_filter(in, out, W, H, K_GAUSSIAN);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H, K_GAUSSIAN);
    return compare_images(hw, golden, W, H, true);
}

// ============================================================
// TEST 15 — Linealidad: filter(A+B) == filter(A) + filter(B)
// Solo válido para kernels lineales sin saturación
// Se verifica con valores pequeños para evitar saturación
// ============================================================
static bool test_linearity() {
    const int W = 8, H = 6;

    Image imgA(H, std::vector<uint8_t>(W, 0));
    Image imgB(H, std::vector<uint8_t>(W, 0));
    Image imgAB(H, std::vector<uint8_t>(W, 0));

    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            imgA[y][x]  = (uint8_t)((y * 3 + x * 2) % 30);   // valores pequeños
            imgB[y][x]  = (uint8_t)((y * 2 + x * 3) % 30);
            imgAB[y][x] = imgA[y][x] + imgB[y][x];
        }

    hls::stream<axis_t> inA, inB, inAB, outA, outB, outAB;
    load_image(inA,  imgA,  W, H);
    load_image(inB,  imgB,  W, H);
    load_image(inAB, imgAB, W, H);

    // Usar kernel identidad para que la linealidad sea exacta
    run_filter(inA,  outA,  W, H, K_IDENTITY);
    run_filter(inB,  outB,  W, H, K_IDENTITY);
    run_filter(inAB, outAB, W, H, K_IDENTITY);

    auto hwA  = read_output(outA,  W, H);
    auto hwB  = read_output(outB,  W, H);
    auto hwAB = read_output(outAB, W, H);

    bool ok = true;
    for (int y = 1; y < H-1 && ok; y++)
        for (int x = 1; x < W-1 && ok; x++) {
            int sum_ab = (int)hwA[y][x] + (int)hwB[y][x];
            if ((int)hwAB[y][x] != sum_ab) {
                std::cerr << "    Linealidad falla en [" << y << "," << x << "]"
                          << " f(A+B)=" << (int)hwAB[y][x]
                          << " f(A)+f(B)=" << sum_ab << "\n";
                ok = false;
            }
        }
    return ok;
}
int main_filter();

// ============================================================
// MAIN
// ============================================================
int main() {
    if (main_filter()!=0){
        return 1;
    }
    std::cout << "======================================\n";
    std::cout << "  Testbench: filter_y_3x3\n";
    std::cout << "  Formato: YUYV 32bpp\n";
    std::cout << "  Desfase: 1 fila + 1 columna\n";
    std::cout << "  Bordes: zeroed\n";
    std::cout << "======================================\n\n";

    report("T01 - Kernel identidad",                   test_identity());
    report("T02 - Bordes en cero",                     test_border_zeros());
    report("T03 - Gaussiano imagen uniforme",          test_gaussian_uniform());
    report("T04 - Laplaciano imagen uniforme → cero",  test_laplacian_uniform());
    report("T05 - Sobel Gx borde vertical",            test_sobel_gx_vertical_edge());
    report("T06 - Sobel Gy borde horizontal",          test_sobel_gy_horizontal_edge());
    report("T07 - Golden model Gaussiano",             test_golden_model(K_GAUSSIAN));
    report("T07b- Golden model Laplaciano",            test_golden_model(K_LAPLACIAN));
    report("T07c- Golden model Sobel Gx",              test_golden_model(K_SOBEL_GX));
    report("T07d- Golden model Sobel Gy",              test_golden_model(K_SOBEL_GY));
    report("T07e- Golden model Sharpening",            test_golden_model(K_SHARPEN));
    report("T08 - Saturación correcta",                test_saturation());
    report("T09 - Shift_val correcto",                 test_shift_val());
    report("T10 - Croma propagado correctamente",      test_chroma_propagation());
    report("T11 - Señal LAST en paquete correcto",     test_last_signal());
    report("T12 - Kernel de todos ceros",              test_zero_kernel());
    report("T13 - Conteo exacto de paquetes",          test_packet_count());
    report("T14 - Resolución mínima 4x4",              test_minimum_resolution());
    report("T15 - Propiedad de linealidad",            test_linearity());

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
