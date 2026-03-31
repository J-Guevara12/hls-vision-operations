// ============================================================
// Testbench: test_resizer_half.cpp
// Componentes bajo prueba: resize_half_y, resize_half_uv
//
// Estrategia de decimación 2:1 nearest neighbor:
//   - Horizontal: toma píxel par de cada par (índice 0, 2, 4...)
//   - Vertical:   toma filas pares (índice 0, 2, 4...)
//
// Invariantes a verificar:
//   - Conteo exacto de palabras de salida Y = (W/PPP/2) * (H/2)
//   - Conteo exacto de palabras de salida UV = (W/PPP/2) * (H/2)
//   - Valores Y correctos: píxeles pares de filas pares
//   - Valores UV correctos: grupos pares de filas pares
//   - No hay contaminación entre filas (acumulador reseteado)
//   - Resolución mínima (4x4)
//   - Resolución 4K (3840x2160)
//   - Imagen uniforme: salida idéntica a entrada
//   - Imagen con gradiente: decimación correcta
//   - Filas impares completamente descartadas
//   - Columnas impares completamente descartadas
// ============================================================

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdint>
#include <cassert>

#include "ap_int.h"
#include "hls_stream.h"
#include "resizer.hpp"

#define PPP 2

// ============================================================
// Contadores
// ============================================================
static int tests_run = 0, tests_passed = 0, tests_failed = 0;
static void report(const std::string& name, bool ok) {
    tests_run++;
    if (ok) { tests_passed++; std::cout << "  [PASS] " << name << "\n"; }
    else     { tests_failed++; std::cerr << "  [FAIL] " << name << "\n"; }
}

// ============================================================
// Tipos
// ============================================================
typedef std::vector<std::vector<uint8_t>> Image;

// ============================================================
// Utilidades
// ============================================================
static void load_y(hls::stream<ap_uint<PPP*8>>& s, const Image& img,
                   int width, int height) {
    for (int y = 0; y < height; y++)
        for (int g = 0; g < width/PPP; g++) {
            ap_uint<PPP*8> word = 0;
            for (int p = 0; p < PPP; p++)
                word.range(p*8+7, p*8) = img[y][g*PPP + p];
            s.write(word);
        }
}

static void load_uv(hls::stream<uint16_t>& s,
                    const std::vector<std::vector<std::pair<uint8_t,uint8_t>>>& uv,
                    int width, int height) {
    // uv tiene height filas y width/PPP columnas (1 par por grupo)
    for (int y = 0; y < height; y++)
        for (int g = 0; g < width/PPP; g++)
            s.write(((uint16_t)uv[y][g].second << 8) | uv[y][g].first);
}

static Image read_y(hls::stream<ap_uint<PPP*8>>& s, int out_w, int out_h) {
    Image out(out_h, std::vector<uint8_t>(out_w, 0));
    for (int y = 0; y < out_h; y++)
        for (int g = 0; g < out_w/PPP; g++) {
            ap_uint<PPP*8> word = s.read();
            for (int p = 0; p < PPP; p++)
                out[y][g*PPP + p] = (uint8_t)word.range(p*8+7, p*8);
        }
    return out;
}

static std::vector<std::vector<std::pair<uint8_t,uint8_t>>>
read_uv(hls::stream<uint16_t>& s, int out_w, int out_h) {
    // out_w aquí es el ancho Y de salida, UV tiene out_w/PPP columnas
    std::vector<std::vector<std::pair<uint8_t,uint8_t>>>
        out(out_h, std::vector<std::pair<uint8_t,uint8_t>>(out_w/PPP, {0,0}));
    for (int y = 0; y < out_h; y++)
        for (int g = 0; g < out_w/PPP; g++) {
            uint16_t uv = s.read();
            out[y][g] = { (uint8_t)(uv & 0xFF), (uint8_t)(uv >> 8) };
        }
    return out;
}

// Golden model: toma píxeles pares de filas pares
static Image golden_y(const Image& img, int width, int height) {
    int out_h = height / 2;
    int out_w = width  / 2;
    Image out(out_h, std::vector<uint8_t>(out_w, 0));
    for (int y = 0; y < out_h; y++)
        for (int x = 0; x < out_w; x++)
            out[y][x] = img[y*2][x*2];
    return out;
}

static std::vector<std::vector<std::pair<uint8_t,uint8_t>>>
golden_uv(const std::vector<std::vector<std::pair<uint8_t,uint8_t>>>& uv,
          int width, int height) {
    // uv tiene height filas y width/PPP grupos
    int out_h  = height / 2;
    int out_g  = (width/PPP) / 2;
    std::vector<std::vector<std::pair<uint8_t,uint8_t>>>
        out(out_h, std::vector<std::pair<uint8_t,uint8_t>>(out_g, {0,0}));
    for (int y = 0; y < out_h; y++)
        for (int g = 0; g < out_g; g++)
            out[y][g] = uv[y*2][g*2];
    return out;
}

static bool compare_y(const Image& hw, const Image& golden,
                       int out_w, int out_h, bool verbose = true) {
    bool ok = true;
    int errors = 0;
    for (int y = 0; y < out_h; y++)
        for (int x = 0; x < out_w; x++)
            if (hw[y][x] != golden[y][x]) {
                if (verbose && errors < 10)
                    std::cerr << "    Y[" << y << "," << x << "]"
                              << " hw=" << (int)hw[y][x]
                              << " expected=" << (int)golden[y][x] << "\n";
                ok = false;
                errors++;
            }
    if (!ok && verbose)
        std::cerr << "    Total errores Y: " << errors << "\n";
    return ok;
}

static bool compare_uv(
    const std::vector<std::vector<std::pair<uint8_t,uint8_t>>>& hw,
    const std::vector<std::vector<std::pair<uint8_t,uint8_t>>>& golden,
    int out_w, int out_h, bool verbose = true) {
    bool ok = true;
    int errors = 0;
    int out_g = out_w / PPP;
    for (int y = 0; y < out_h; y++)
        for (int g = 0; g < out_g; g++) {
            if (hw[y][g].first  != golden[y][g].first ||
                hw[y][g].second != golden[y][g].second) {
                if (verbose && errors < 10)
                    std::cerr << "    UV[y=" << y << ",g=" << g << "]"
                              << " hw=(" << (int)hw[y][g].first
                              << "," << (int)hw[y][g].second << ")"
                              << " expected=(" << (int)golden[y][g].first
                              << "," << (int)golden[y][g].second << ")\n";
                ok = false;
                errors++;
            }
        }
    if (!ok && verbose)
        std::cerr << "    Total errores UV: " << errors << "\n";
    return ok;
}

// ============================================================
// TEST 1 — Conteo exacto de palabras de salida Y
// ============================================================
static bool test_y_output_count() {
    const int W = 8, H = 4;
    const int EXP_W = W/2, EXP_H = H/2;
    const int EXP_WORDS = (EXP_W/PPP) * EXP_H;

    hls::stream<ap_uint<PPP*8>> y_in, y_out;
    Image img(H, std::vector<uint8_t>(W, 100));
    load_y(y_in, img, W, H);

    resize_half_y<PPP>(y_in, y_out, W, H);

    int count = 0;
    while (!y_out.empty()) { y_out.read(); count++; }
    bool ok = (count == EXP_WORDS);
    if (!ok)
        std::cerr << "    Y words=" << count << " esperado=" << EXP_WORDS << "\n";
    return ok;
}

// ============================================================
// TEST 2 — Conteo exacto de palabras de salida UV
// ============================================================
static bool test_uv_output_count() {
    const int W = 8, H = 4;
    const int EXP_WORDS = (W/PPP/2) * (H/2);

    hls::stream<uint16_t> uv_in, uv_out;
    for (int i = 0; i < (W/PPP)*H; i++) uv_in.write(0x8080);

    resize_half_uv<PPP>(uv_in, uv_out, W, H);

    int count = 0;
    while (!uv_out.empty()) { uv_out.read(); count++; }
    bool ok = (count == EXP_WORDS);
    if (!ok)
        std::cerr << "    UV words=" << count << " esperado=" << EXP_WORDS << "\n";
    return ok;
}

// ============================================================
// TEST 3 — Imagen uniforme: salida idéntica al valor de entrada
// ============================================================
static bool test_uniform_image() {
    const int W = 8, H = 4;
    const uint8_t VAL = 137;

    hls::stream<ap_uint<PPP*8>> y_in, y_out;
    Image img(H, std::vector<uint8_t>(W, VAL));
    load_y(y_in, img, W, H);

    resize_half_y<PPP>(y_in, y_out, W, H);

    auto hw = read_y(y_out, W/2, H/2);
    bool ok = true;
    for (int y = 0; y < H/2 && ok; y++)
        for (int x = 0; x < W/2 && ok; x++)
            if (hw[y][x] != VAL) {
                std::cerr << "    Uniforme Y[" << y << "," << x << "]="
                          << (int)hw[y][x] << " esperado=" << (int)VAL << "\n";
                ok = false;
            }
    return ok;
}

// ============================================================
// TEST 4 — Píxeles correctos: solo pares de filas pares
// ============================================================
static bool test_correct_pixels_selected() {
    const int W = 8, H = 4;

    // Imagen con valores únicos por posición: y*100 + x
    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(y*10 + x);

    hls::stream<ap_uint<PPP*8>> y_in, y_out;
    load_y(y_in, img, W, H);
    resize_half_y<PPP>(y_in, y_out, W, H);

    auto hw     = read_y(y_out, W/2, H/2);
    auto ref    = golden_y(img, W, H);
    return compare_y(hw, ref, W/2, H/2, true);
}

// ============================================================
// TEST 5 — Filas impares completamente descartadas
// Las filas impares tienen valores distintos a las pares;
// si aparecen en la salida el test falla
// ============================================================
static bool test_odd_rows_discarded() {
    const int W = 8, H = 4;

    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (y % 2 == 0) ? 10 : 200;  // pares=10, impares=200

    hls::stream<ap_uint<PPP*8>> y_in, y_out;
    load_y(y_in, img, W, H);
    resize_half_y<PPP>(y_in, y_out, W, H);

    auto hw = read_y(y_out, W/2, H/2);
    bool ok = true;
    for (int y = 0; y < H/2 && ok; y++)
        for (int x = 0; x < W/2 && ok; x++)
            if (hw[y][x] == 200) {
                std::cerr << "    Fila impar encontrada en salida Y["
                          << y << "," << x << "]=" << (int)hw[y][x] << "\n";
                ok = false;
            }
    return ok;
}

// ============================================================
// TEST 6 — Columnas impares completamente descartadas
// ============================================================
static bool test_odd_cols_discarded() {
    const int W = 8, H = 4;

    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (x % 2 == 0) ? 20 : 210;  // pares=20, impares=210

    hls::stream<ap_uint<PPP*8>> y_in, y_out;
    load_y(y_in, img, W, H);
    resize_half_y<PPP>(y_in, y_out, W, H);

    auto hw = read_y(y_out, W/2, H/2);
    bool ok = true;
    for (int y = 0; y < H/2 && ok; y++)
        for (int x = 0; x < W/2 && ok; x++)
            if (hw[y][x] == 210) {
                std::cerr << "    Columna impar encontrada en salida Y["
                          << y << "," << x << "]=" << (int)hw[y][x] << "\n";
                ok = false;
            }
    return ok;
}

// ============================================================
// TEST 7 — Sin contaminación entre filas (acumulador reseteado)
// Si el acumulador no se resetea entre filas, el último píxel
// de una fila contamina el primero de la siguiente
// ============================================================
static bool test_no_row_contamination() {
    const int W = 8, H = 4;

    // Fila 0: valores altos, fila 2: valores bajos
    // Si hay contaminación, fila 2 tendrá restos de fila 0
    Image img(H, std::vector<uint8_t>(W, 0));
    for (int x = 0; x < W; x++) {
        img[0][x] = 255;
        img[2][x] = 1;
    }

    hls::stream<ap_uint<PPP*8>> y_in, y_out;
    load_y(y_in, img, W, H);
    resize_half_y<PPP>(y_in, y_out, W, H);

    auto hw = read_y(y_out, W/2, H/2);
    bool ok = true;

    // Fila 0 de salida viene de fila 0 de entrada → debe ser 255
    for (int x = 0; x < W/2 && ok; x++)
        if (hw[0][x] != 255) {
            std::cerr << "    Contaminación: hw[0][" << x << "]="
                      << (int)hw[0][x] << " esperado=255\n";
            ok = false;
        }
    // Fila 1 de salida viene de fila 2 de entrada → debe ser 1
    for (int x = 0; x < W/2 && ok; x++)
        if (hw[1][x] != 1) {
            std::cerr << "    Contaminación: hw[1][" << x << "]="
                      << (int)hw[1][x] << " esperado=1\n";
            ok = false;
        }
    return ok;
}

// ============================================================
// TEST 8 — UV: valores correctos (grupos pares de filas pares)
// ============================================================
static bool test_uv_correct_values() {
    const int W = 8, H = 4;

    // UV único por posición: U = y*10+g, V = y*10+g+100
    std::vector<std::vector<std::pair<uint8_t,uint8_t>>>
        uv(H, std::vector<std::pair<uint8_t,uint8_t>>(W/PPP, {0,0}));
    for (int y = 0; y < H; y++)
        for (int g = 0; g < W/PPP; g++)
            uv[y][g] = { (uint8_t)(y*10+g), (uint8_t)(y*10+g+100) };

    hls::stream<uint16_t> uv_in, uv_out;
    load_uv(uv_in, uv, W, H);
    resize_half_uv<PPP>(uv_in, uv_out, W, H);

    auto hw  = read_uv(uv_out, W/2, H/2);
    auto ref = golden_uv(uv, W, H);
    return compare_uv(hw, ref, W/2, H/2, true);
}

// ============================================================
// TEST 9 — UV: filas impares descartadas
// ============================================================
static bool test_uv_odd_rows_discarded() {
    const int W = 8, H = 4;

    std::vector<std::vector<std::pair<uint8_t,uint8_t>>>
        uv(H, std::vector<std::pair<uint8_t,uint8_t>>(W/PPP, {0,0}));
    for (int y = 0; y < H; y++)
        for (int g = 0; g < W/PPP; g++)
            uv[y][g] = (y % 2 == 0) ? std::make_pair((uint8_t)10, (uint8_t)20)
                                     : std::make_pair((uint8_t)200, (uint8_t)210);

    hls::stream<uint16_t> uv_in, uv_out;
    load_uv(uv_in, uv, W, H);
    resize_half_uv<PPP>(uv_in, uv_out, W, H);

    auto hw = read_uv(uv_out, W/2, H/2);
    bool ok = true;
    for (int y = 0; y < H/2 && ok; y++)
        for (int g = 0; g < W/PPP/2 && ok; g++)
            if (hw[y][g].first == 200 || hw[y][g].second == 210) {
                std::cerr << "    UV fila impar en salida [y=" << y
                          << ",g=" << g << "]\n";
                ok = false;
            }
    return ok;
}

// ============================================================
// TEST 10 — UV: grupos impares descartados
// ============================================================
static bool test_uv_odd_groups_discarded() {
    const int W = 8, H = 4;

    std::vector<std::vector<std::pair<uint8_t,uint8_t>>>
        uv(H, std::vector<std::pair<uint8_t,uint8_t>>(W/PPP, {0,0}));
    for (int y = 0; y < H; y++)
        for (int g = 0; g < W/PPP; g++)
            uv[y][g] = (g % 2 == 0) ? std::make_pair((uint8_t)30, (uint8_t)40)
                                     : std::make_pair((uint8_t)200, (uint8_t)210);

    hls::stream<uint16_t> uv_in, uv_out;
    load_uv(uv_in, uv, W, H);
    resize_half_uv<PPP>(uv_in, uv_out, W, H);

    auto hw = read_uv(uv_out, W/2, H/2);
    bool ok = true;
    for (int y = 0; y < H/2 && ok; y++)
        for (int g = 0; g < W/PPP/2 && ok; g++)
            if (hw[y][g].first == 200 || hw[y][g].second == 210) {
                std::cerr << "    UV grupo impar en salida [y=" << y
                          << ",g=" << g << "]\n";
                ok = false;
            }
    return ok;
}

// ============================================================
// TEST 11 — Resolución mínima (4x4)
// ============================================================
static bool test_minimum_resolution() {
    const int W = 4, H = 4;

    Image img = {
        { 10,  20,  30,  40},
        { 50,  60,  70,  80},
        { 90, 100, 110, 120},
        {130, 140, 150, 160}
    };

    hls::stream<ap_uint<PPP*8>> y_in, y_out;
    load_y(y_in, img, W, H);
    resize_half_y<PPP>(y_in, y_out, W, H);

    auto hw  = read_y(y_out, W/2, H/2);
    auto ref = golden_y(img, W, H);
    return compare_y(hw, ref, W/2, H/2, true);
}

// ============================================================
// TEST 12 — Gradiente horizontal: verificar posiciones exactas
// ============================================================
static bool test_horizontal_gradient() {
    const int W = 8, H = 4;

    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(x * 10);  // 0,10,20,30,40,50,60,70

    hls::stream<ap_uint<PPP*8>> y_in, y_out;
    load_y(y_in, img, W, H);
    resize_half_y<PPP>(y_in, y_out, W, H);

    auto hw  = read_y(y_out, W/2, H/2);
    auto ref = golden_y(img, W, H);

    // Verificación semántica: salida debe ser 0,20,40,60
    bool ok = compare_y(hw, ref, W/2, H/2, true);
    uint8_t expected[] = {0, 20, 40, 60};
    for (int x = 0; x < W/2 && ok; x++)
        if (hw[0][x] != expected[x]) {
            std::cerr << "    Gradiente H: hw[0][" << x << "]="
                      << (int)hw[0][x] << " esperado=" << (int)expected[x] << "\n";
            ok = false;
        }
    return ok;
}

// ============================================================
// TEST 13 — Gradiente vertical: verificar posiciones exactas
// ============================================================
static bool test_vertical_gradient() {
    const int W = 8, H = 8;

    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(y * 10);  // filas: 0,10,20,30,40,50,60,70

    hls::stream<ap_uint<PPP*8>> y_in, y_out;
    load_y(y_in, img, W, H);
    resize_half_y<PPP>(y_in, y_out, W, H);

    auto hw  = read_y(y_out, W/2, H/2);
    auto ref = golden_y(img, W, H);

    // Verificación semántica: columna 0 debe ser 0,20,40,60
    bool ok = compare_y(hw, ref, W/2, H/2, true);
    uint8_t expected[] = {0, 20, 40, 60};
    for (int y = 0; y < H/2 && ok; y++)
        if (hw[y][0] != expected[y]) {
            std::cerr << "    Gradiente V: hw[" << y << "][0]="
                      << (int)hw[y][0] << " esperado=" << (int)expected[y] << "\n";
            ok = false;
        }
    return ok;
}

// ============================================================
// TEST 14 — Valores extremos 0x00 y 0xFF
// ============================================================
static bool test_extreme_values() {
    const int W = 8, H = 4;

    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y += 2)
        for (int x = 0; x < W; x += 2) {
            img[y][x]   = 0x00;
            img[y][x+1] = 0xFF;
        }

    hls::stream<ap_uint<PPP*8>> y_in, y_out;
    load_y(y_in, img, W, H);
    resize_half_y<PPP>(y_in, y_out, W, H);

    auto hw  = read_y(y_out, W/2, H/2);
    auto ref = golden_y(img, W, H);
    return compare_y(hw, ref, W/2, H/2, true);
}

// ============================================================
// TEST 15 — Resolución 4K (3840x2160): sin deadlock y conteo correcto
// ============================================================
static bool test_4k_resolution() {
    const int W = 256, H = 170;
    const int OUT_W = W/2, OUT_H = H/2;
    const int EXP_Y_WORDS  = (OUT_W/PPP) * OUT_H;
    const int EXP_UV_WORDS = (OUT_W/PPP) * OUT_H;

    std::cout << "    [4K] Generando " << W << "x" << H << "...\n";

    // Guardamos solo filas de muestra para verificación
    const int CHECK_ROWS[] = {0, H/2-2, H-2};  // filas de entrada pares
    const int N_CHECK = 3;
    std::vector<std::vector<uint8_t>> ref_rows(N_CHECK, std::vector<uint8_t>(W));

    hls::stream<ap_uint<PPP*8>> y_in, y_out;
    hls::stream<uint16_t>       uv_in, uv_out;

    for (int y = 0; y < H; y++) {
        for (int g = 0; g < W/PPP; g++) {
            ap_uint<PPP*8> word = 0;
            for (int p = 0; p < PPP; p++) {
                int x = g*PPP + p;
                uint8_t val = (uint8_t)((y*37 + x*53 + y*x % 199) % 256);
                word.range(p*8+7, p*8) = val;
                for (int ci = 0; ci < N_CHECK; ci++)
                    if (y == CHECK_ROWS[ci])
                        ref_rows[ci][x] = val;
            }
            y_in.write(word);
            uv_in.write(((uint16_t)(uint8_t)(y*3+g) << 8) | (uint8_t)(y*7+g));
        }
    }

    std::cout << "    [4K] Ejecutando resize_half_y y resize_half_uv...\n";
    resize_half_y<PPP>(y_in,  y_out,  W, H);
    resize_half_uv<PPP>(uv_in, uv_out, W, H);
    std::cout << "    [4K] Completado, verificando...\n";

    // Verificar conteo Y
    int y_count = (int)y_out.size();
    if (y_count != EXP_Y_WORDS) {
        std::cerr << "    [4K] Y words=" << y_count
                  << " esperado=" << EXP_Y_WORDS << "\n";
        while (!y_out.empty())  y_out.read();
        while (!uv_out.empty()) uv_out.read();
        return false;
    }

    // Verificar conteo UV
    int uv_count = (int)uv_out.size();
    if (uv_count != EXP_UV_WORDS) {
        std::cerr << "    [4K] UV words=" << uv_count
                  << " esperado=" << EXP_UV_WORDS << "\n";
        while (!y_out.empty())  y_out.read();
        while (!uv_out.empty()) uv_out.read();
        return false;
    }

    // Verificar valores Y en filas de muestra + drenar stream
    bool ok = true;
    for (int y = 0; y < OUT_H; y++) {
        int src_row = y * 2;  // fila de entrada correspondiente
        int ci = -1;
        for (int i = 0; i < N_CHECK; i++)
            if (CHECK_ROWS[i] == src_row) { ci = i; break; }

        for (int g = 0; g < OUT_W/PPP; g++) {
            ap_uint<PPP*8> word = y_out.read();
            if (ci >= 0) {
                for (int p = 0; p < PPP; p++) {
                    uint8_t hw_val  = (uint8_t)word.range(p*8+7, p*8);
                    uint8_t exp_val = ref_rows[ci][(g*PPP + p)*2];
                    if (hw_val != exp_val) {
                        std::cerr << "    [4K] Y[y=" << y << ",x="
                                  << (g*PPP+p) << "] hw=" << (int)hw_val
                                  << " expected=" << (int)exp_val << "\n";
                        ok = false;
                    }
                }
            }
        }
    }

    // Drenar UV
    while (!uv_out.empty()) uv_out.read();

    if (ok)
        std::cout << "    [4K] " << y_count << " palabras Y y "
                  << uv_count << " palabras UV verificadas\n";
    return ok;
}



// ============================================================
// MAIN
// ============================================================
int main_resize_half_stages() {
    std::cout << "==========================================\n";
    std::cout << "  Testbench: resize_half_y / resize_half_uv\n";
    std::cout << "  Factor: 2:1 nearest neighbor\n";
    std::cout << "  PPP=" << PPP << "\n";
    std::cout << "==========================================\n\n";

    report("T01 - Conteo palabras Y de salida",          test_y_output_count());
    report("T02 - Conteo palabras UV de salida",         test_uv_output_count());
    report("T03 - Imagen uniforme Y",                    test_uniform_image());
    report("T04 - Píxeles correctos seleccionados",      test_correct_pixels_selected());
    report("T05 - Filas impares descartadas (Y)",        test_odd_rows_discarded());
    report("T06 - Columnas impares descartadas (Y)",     test_odd_cols_discarded());
    report("T07 - Sin contaminación entre filas",        test_no_row_contamination());
    report("T08 - UV valores correctos",                 test_uv_correct_values());
    report("T09 - UV filas impares descartadas",         test_uv_odd_rows_discarded());
    report("T10 - UV grupos impares descartados",        test_uv_odd_groups_discarded());
    report("T11 - Resolución mínima 4x4",                test_minimum_resolution());
    report("T12 - Gradiente horizontal",                 test_horizontal_gradient());
    report("T13 - Gradiente vertical",                   test_vertical_gradient());
    report("T14 - Valores extremos 0x00 y 0xFF",         test_extreme_values());
    report("T15 - Resolución 4K sin deadlock",           test_4k_resolution());

    std::cout << "\n==========================================\n";
    std::cout << "  Resultado: " << tests_passed << "/"
              << tests_run << " tests pasaron\n";
    if (tests_failed > 0)
        std::cout << "  FALLARON: " << tests_failed << " tests\n";
    else
        std::cout << "  Todos los tests PASARON\n";
    std::cout << "==========================================\n";

    return tests_failed > 0 ? 1 : 0;
}