// ============================================================
// Testbench: test_transformation_3x3.cpp
// Componente bajo prueba: transformation_3x3
//
// Operaciones morfológicas 3x3 sobre canal Y de stream YUYV:
//   - Dilatación: MAX de píxeles bajo la máscara
//   - Erosión:    MIN de píxeles bajo la máscara
//
// Invariantes a verificar:
//   - Conteo exacto de paquetes de salida
//   - Bordes en cero
//   - Dilatación imagen uniforme → misma imagen
//   - Erosión imagen uniforme → misma imagen
//   - Dilatación máscara completa = MAX local
//   - Erosión máscara completa = MIN local
//   - Máscara vacía → cero en interior
//   - Máscara solo centro → identidad
//   - Dilatación impulso → forma de máscara
//   - Erosión impulso → cero
//   - Croma propagado correctamente
//   - Señal LAST en paquete correcto
//   - Golden model completo con patrón pseudo-aleatorio
//   - Resolución mínima 4x4
//   - Resolución 4K sin deadlock
// ============================================================

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>

#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "transformations.hpp"

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
// Codificación de máscara 3x3 en uint16_t
// bit = ky*3 + kx, orden row-major
// ============================================================
static uint16_t encode_mask(const bool m[3][3]) {
    uint16_t mask = 0;
    for (int ky = 0; ky < 3; ky++)
        for (int kx = 0; kx < 3; kx++)
            if (m[ky][kx])
                mask |= (1 << (ky*3 + kx));
    return mask;
}

// Máscaras predefinidas
static const bool MASK_FULL[3][3] = {
    {true, true, true},
    {true, true, true},
    {true, true, true}
};
static const bool MASK_CROSS[3][3] = {
    {false, true,  false},
    {true,  true,  true },
    {false, true,  false}
};
static const bool MASK_CENTER[3][3] = {
    {false, false, false},
    {false, true,  false},
    {false, false, false}
};
static const bool MASK_EMPTY[3][3] = {
    {false, false, false},
    {false, false, false},
    {false, false, false}
};
static const bool MASK_CORNER[3][3] = {
    {true,  false, false},
    {false, false, false},
    {false, false, false}
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

static void load_image(hls::stream<axis_t>& s, const Image& img,
                        int width, int height,
                        uint8_t u_val = 128, uint8_t v_val = 128) {
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x += 2) {
            bool last = (y == height-1) && (x == width-2);
            s.write(make_yuyv(img[y][x], u_val, img[y][x+1], v_val, last));
        }
}

static Image read_output(hls::stream<axis_t>& s, int width, int height) {
    Image out(height, std::vector<uint8_t>(width, 0));
    int total = (width/2) * height;
    for (int i = 0; i < total; i++) {
        axis_t pkt = s.read();
        int px = (i % (width/2)) * 2;
        int py = i / (width/2);
        out[py][px]   = (uint8_t)pkt.data.range(7,  0);
        out[py][px+1] = (uint8_t)pkt.data.range(23, 16);
    }
    return out;
}

static void run(hls::stream<axis_t>& in, hls::stream<axis_t>& out,
                int width, int height, const bool mask[3][3], bool is_dilation) {
    transformation_3x3(in, out, width, height,
                        (int)encode_mask(mask), is_dilation);
}

// ============================================================
// Golden model
// ============================================================
static Image compute_golden(const Image& img, int width, int height,
                              const bool mask[3][3], bool is_dilation) {
    Image out(height, std::vector<uint8_t>(width, 0));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            bool is_border = (y == 0 || y == height-1 ||
                              x == 0 || x == width-1);
            if (is_border) {
                out[y][x] = 0;
                continue;
            }
            uint8_t result = is_dilation ? 0 : 255;
            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    if (!mask[ky][kx]) continue;
                    uint8_t val = img[y+ky-1][x+kx-1];
                    if (is_dilation)
                        result = std::max(result, val);
                    else
                        result = std::min(result, val);
                }
            }
            out[y][x] = result;
        }
    }
    return out;
}

static bool compare(const Image& hw, const Image& golden,
                     int width, int height, bool verbose = true) {
    bool ok = true;
    int errors = 0;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            if (hw[y][x] != golden[y][x]) {
                if (verbose && errors < 10)
                    std::cerr << "    [" << y << "," << x << "]"
                              << " hw=" << (int)hw[y][x]
                              << " expected=" << (int)golden[y][x] << "\n";
                ok = false;
                errors++;
            }
    if (!ok && verbose)
        std::cerr << "    Total errores: " << errors << "\n";
    return ok;
}

// ============================================================
// TEST 1 — Conteo exacto de paquetes de salida
// ============================================================
static bool test_packet_count() {
    const int W = 8, H = 6;
    const int EXP = (W/2) * H;

    Image img(H, std::vector<uint8_t>(W, 100));
    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run(in, out, W, H, MASK_FULL, true);

    int count = 0;
    while (!out.empty()) { out.read(); count++; }
    bool ok = (count == EXP);
    if (!ok)
        std::cerr << "    count=" << count << " esperado=" << EXP << "\n";
    return ok;
}

// ============================================================
// TEST 2 — Bordes en cero para cualquier operación
// ============================================================
static bool test_border_zeros() {
    const int W = 8, H = 6;
    Image img(H, std::vector<uint8_t>(W, 200));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run(in, out, W, H, MASK_FULL, true);

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
// TEST 3 — Dilatación imagen uniforme → misma imagen (interior)
// ============================================================
static bool test_dilation_uniform() {
    const int W = 8, H = 6;
    const uint8_t VAL = 150;
    Image img(H, std::vector<uint8_t>(W, VAL));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run(in, out, W, H, MASK_FULL, true);

    auto hw = read_output(out, W, H);
    bool ok = true;
    for (int y = 1; y < H-1 && ok; y++)
        for (int x = 1; x < W-1 && ok; x++)
            if (hw[y][x] != VAL) {
                std::cerr << "    Dilatación uniforme [" << y << "," << x << "]="
                          << (int)hw[y][x] << " esperado=" << (int)VAL << "\n";
                ok = false;
            }
    return ok;
}

// ============================================================
// TEST 4 — Erosión imagen uniforme → misma imagen (interior)
// ============================================================
static bool test_erosion_uniform() {
    const int W = 8, H = 6;
    const uint8_t VAL = 100;
    Image img(H, std::vector<uint8_t>(W, VAL));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run(in, out, W, H, MASK_FULL, false);

    auto hw = read_output(out, W, H);
    bool ok = true;
    for (int y = 1; y < H-1 && ok; y++)
        for (int x = 1; x < W-1 && ok; x++)
            if (hw[y][x] != VAL) {
                std::cerr << "    Erosión uniforme [" << y << "," << x << "]="
                          << (int)hw[y][x] << " esperado=" << (int)VAL << "\n";
                ok = false;
            }
    return ok;
}

// ============================================================
// TEST 5 — Máscara solo centro = identidad
// ============================================================
static bool test_mask_center_identity() {
    const int W = 8, H = 6;
    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(y*10 + x);

    // Dilatación con solo centro = identidad
    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run(in, out, W, H, MASK_CENTER, true);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H, MASK_CENTER, true);
    bool ok = compare(hw, golden, W, H, true);

    for (int y = 1; y < H-1 && ok; y++)
        for (int x = 1; x < W-1 && ok; x++)
            if (hw[y][x] != img[y][x]) {
                std::cerr << "    Identidad [" << y << "," << x << "]="
                          << (int)hw[y][x] << " esperado=" << (int)img[y][x] << "\n";
                ok = false;
            }
    return ok;
}

// ============================================================
// TEST 6 — Máscara vacía → cero en todo el interior
// ============================================================
static bool test_mask_empty() {
    const int W = 8, H = 6;
    Image img(H, std::vector<uint8_t>(W, 200));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run(in, out, W, H, MASK_EMPTY, true);

    auto hw = read_output(out, W, H);
    bool ok = true;
    for (int y = 0; y < H && ok; y++)
        for (int x = 0; x < W && ok; x++)
            if (hw[y][x] != 0) {
                std::cerr << "    Máscara vacía [" << y << "," << x << "]="
                          << (int)hw[y][x] << " esperado=0\n";
                ok = false;
            }
    return ok;
}

// ============================================================
// TEST 7 — Dilatación impulso → forma de máscara
// Un impulso en el centro con dilatación reproduce la máscara
// ============================================================
static bool test_dilation_impulse() {
    const int W = 8, H = 8;
    Image img(H, std::vector<uint8_t>(W, 0));
    img[4][4] = 255;  // impulso

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run(in, out, W, H, MASK_CROSS, true);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H, MASK_CROSS, true);
    return compare(hw, golden, W, H, true);
}

// ============================================================
// TEST 8 — Erosión impulso → cero en todo el interior
// Un impulso con erosión máscara completa → todo cero interior
// ============================================================
static bool test_erosion_impulse() {
    const int W = 8, H = 8;
    Image img(H, std::vector<uint8_t>(W, 0));
    img[4][4] = 255;

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run(in, out, W, H, MASK_FULL, false);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H, MASK_FULL, false);
    return compare(hw, golden, W, H, true);
}

// ============================================================
// TEST 9 — Dilatación expande regiones brillantes
// Imagen con mitad izquierda=0, mitad derecha=255
// Después de dilatar el borde se expande 1 píxel a la izquierda
// ============================================================
static bool test_dilation_expands() {
    const int W = 10, H = 8;
    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = W/2; x < W; x++)
            img[y][x] = 255;

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run(in, out, W, H, MASK_FULL, true);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H, MASK_FULL, true);
    bool ok = compare(hw, golden, W, H, true);

    // Verificar semánticamente: columna W/2-1 debe ser 255 tras dilatar
    for (int y = 1; y < H-1 && ok; y++)
        if (hw[y][W/2-1] != 255) {
            std::cerr << "    Dilatación no expandió en [" << y << "," << W/2-1 << "]\n";
            ok = false;
        }
    return ok;
}

// ============================================================
// TEST 10 — Erosión contrae regiones brillantes
// ============================================================
static bool test_erosion_contracts() {
    const int W = 10, H = 8;
    Image img(H, std::vector<uint8_t>(W, 255));
    for (int y = 0; y < H; y++)
        img[y][W/2] = 0;  // línea oscura en el centro

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run(in, out, W, H, MASK_FULL, false);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H, MASK_FULL, false);
    bool ok = compare(hw, golden, W, H, true);

    // Verificar semánticamente: vecinos de la línea oscura deben ser 0
    for (int y = 1; y < H-1 && ok; y++)
        if (hw[y][W/2-1] != 0 || hw[y][W/2+1] != 0) {
            std::cerr << "    Erosión no contrajo en y=" << y << "\n";
            ok = false;
        }
    return ok;
}

// ============================================================
// TEST 11 — Golden model pseudo-aleatorio: dilatación
// ============================================================
static bool test_golden_dilation() {
    const int W = 16, H = 12;
    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)((y*37 + x*53 + y*x*7) % 256);

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run(in, out, W, H, MASK_CROSS, true);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H, MASK_CROSS, true);
    return compare(hw, golden, W, H, true);
}

// ============================================================
// TEST 12 — Golden model pseudo-aleatorio: erosión
// ============================================================
static bool test_golden_erosion() {
    const int W = 16, H = 12;
    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)((y*37 + x*53 + y*x*7) % 256);

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run(in, out, W, H, MASK_CROSS, false);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H, MASK_CROSS, false);
    return compare(hw, golden, W, H, true);
}

// ============================================================
// TEST 13 — Croma propagado correctamente
// ============================================================
static bool test_chroma_propagation() {
    const int W = 8, H = 6;
    const uint8_t U_VAL = 0xAB, V_VAL = 0xCD;
    Image img(H, std::vector<uint8_t>(W, 128));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H, U_VAL, V_VAL);
    run(in, out, W, H, MASK_CENTER, true);

    bool ok = true;
    int total = (W/2) * H;
    int errors = 0;

    for (int i = 0; i < total; i++) {
        axis_t pkt = out.read();
        int py = i / (W/2);
        if (py == 0 || py == H-1) continue;
        uint8_t u = (uint8_t)pkt.data.range(15, 8);
        uint8_t v = (uint8_t)pkt.data.range(31, 24);
        if (u != U_VAL || v != V_VAL) {
            if (errors < 5)
                std::cerr << "    Pkt[" << i << "] U=0x" << std::hex
                          << (int)u << " V=0x" << (int)v
                          << " esperado U=0x" << (int)U_VAL
                          << " V=0x" << (int)V_VAL << std::dec << "\n";
            ok = false;
            errors++;
        }
    }
    return ok;
}

// ============================================================
// TEST 14 — Señal LAST en el paquete correcto
// ============================================================
static bool test_last_signal() {
    const int W = 8, H = 6;
    const int TOTAL = (W/2) * H;
    Image img(H, std::vector<uint8_t>(W, 100));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run(in, out, W, H, MASK_FULL, true);

    bool ok = true;
    axis_t last_pkt;
    for (int i = 0; i < TOTAL; i++) {
        last_pkt = out.read();
        bool expected = (i == TOTAL-1);
        bool got      = (last_pkt.last == 1);
        if (got != expected) {
            std::cerr << "    Pkt[" << i << "] last=" << got
                      << " esperado=" << expected << "\n";
            ok = false;
        }
    }
    if (last_pkt.last != 1) {
        std::cerr << "    LAST=0 en el último paquete\n";
        ok = false;
    }
    return ok;
}

// ============================================================
// TEST 15 — Resolución mínima 4x4
// ============================================================
static bool test_minimum_resolution() {
    const int W = 4, H = 4;
    Image img = {
        { 10,  20,  30,  40},
        { 50, 100, 150, 200},
        {200, 150, 100,  50},
        { 40,  30,  20,  10}
    };

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run(in, out, W, H, MASK_FULL, true);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H, MASK_FULL, true);
    return compare(hw, golden, W, H, true);
}

// ============================================================
// TEST 16 — Valores extremos 0x00 y 0xFF
// ============================================================
static bool test_extreme_values() {
    const int W = 8, H = 6;
    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (x % 2 == 0) ? 0x00 : 0xFF;

    // Dilatación
    hls::stream<axis_t> in1, out1;
    load_image(in1, img, W, H);
    run(in1, out1, W, H, MASK_FULL, true);
    auto hw_dil  = read_output(out1, W, H);
    auto ref_dil = compute_golden(img, W, H, MASK_FULL, true);
    bool ok = compare(hw_dil, ref_dil, W, H, true);

    // Erosión
    hls::stream<axis_t> in2, out2;
    load_image(in2, img, W, H);
    run(in2, out2, W, H, MASK_FULL, false);
    auto hw_ero  = read_output(out2, W, H);
    auto ref_ero = compute_golden(img, W, H, MASK_FULL, false);
    ok = ok && compare(hw_ero, ref_ero, W, H, true);

    return ok;
}

// ============================================================
// TEST 17 — Máscara esquina superior izquierda
// Verifica que el desplazamiento de la máscara es correcto
// ============================================================
static bool test_mask_corner() {
    const int W = 8, H = 8;
    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(y*16 + x*2);

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    run(in, out, W, H, MASK_CORNER, true);

    auto hw     = read_output(out, W, H);
    auto golden = compute_golden(img, W, H, MASK_CORNER, true);
    return compare(hw, golden, W, H, true);
}

// ============================================================
// TEST 18 — Idempotencia: dilatar dos veces imagen uniforme
// ============================================================
static bool test_idempotence_uniform() {
    const int W = 8, H = 6;
    const uint8_t VAL = 128;
    Image img(H, std::vector<uint8_t>(W, VAL));

    // Primera pasada
    hls::stream<axis_t> in1, out1;
    load_image(in1, img, W, H);
    run(in1, out1, W, H, MASK_FULL, true);
    auto pass1 = read_output(out1, W, H);

    // Segunda pasada sobre la salida de la primera
    hls::stream<axis_t> in2, out2;
    load_image(in2, pass1, W, H);
    run(in2, out2, W, H, MASK_FULL, true);
    auto pass2 = read_output(out2, W, H);

    // Interior debe ser igual en ambas pasadas
    bool ok = true;
    for (int y = 2; y < H-2 && ok; y++)
        for (int x = 2; x < W-2 && ok; x++)
            if (pass1[y][x] != pass2[y][x]) {
                std::cerr << "    Idempotencia falla [" << y << "," << x << "]"
                          << " pass1=" << (int)pass1[y][x]
                          << " pass2=" << (int)pass2[y][x] << "\n";
                ok = false;
            }
    return ok;
}

// ============================================================
// TEST 19 — Resolución 4K (3840x2160) sin deadlock
// ============================================================
static bool test_4k_resolution() {
    const int W = 3840, H = 2160;
    const int TOTAL = (W/2) * H;

    std::cout << "    [4K] Generando " << W << "x" << H << "...\n";

    const int CHECK_ROWS[] = {1, H/2, H-2};
    const int N_CHECK = 3;
    std::vector<std::vector<uint8_t>> ref_rows(N_CHECK+2,
        std::vector<uint8_t>(W, 0));  // +2 para filas adyacentes

    hls::stream<axis_t> in, out;

    // Guardar filas CHECK_ROWS[i]-1, CHECK_ROWS[i], CHECK_ROWS[i]+1
    // para poder calcular el golden
    std::vector<int> rows_to_save;
    for (int ci = 0; ci < N_CHECK; ci++) {
        rows_to_save.push_back(CHECK_ROWS[ci]-1);
        rows_to_save.push_back(CHECK_ROWS[ci]);
        rows_to_save.push_back(CHECK_ROWS[ci]+1);
    }

    std::vector<std::vector<uint8_t>> saved_rows(H, std::vector<uint8_t>());

    for (int y = 0; y < H; y++) {
        bool save = false;
        for (int r : rows_to_save)
            if (r == y) { save = true; break; }
        if (save) saved_rows[y].resize(W);

        for (int x = 0; x < W; x += 2) {
            uint8_t y0 = (uint8_t)((y*37 + x*53 + y*x % 199) % 256);
            uint8_t y1 = (uint8_t)((y*37 + (x+1)*53 + y*(x+1) % 199) % 256);
            bool last  = (y == H-1) && (x == W-2);
            in.write(make_yuyv(y0, 128, y1, 128, last));
            if (save) {
                saved_rows[y][x]   = y0;
                saved_rows[y][x+1] = y1;
            }
        }
    }

    std::cout << "    [4K] Ejecutando transformation_3x3...\n";
    run(in, out, W, H, MASK_CROSS, true);
    std::cout << "    [4K] Completado, verificando...\n";

    if ((int)out.size() != TOTAL) {
        std::cerr << "    [4K] Paquetes=" << out.size()
                  << " esperado=" << TOTAL << "\n";
        while (!out.empty()) out.read();
        return false;
    }

    bool ok = true;
    axis_t last_pkt;

    for (int y = 0; y < H; y++) {
        int ci = -1;
        for (int i = 0; i < N_CHECK; i++)
            if (CHECK_ROWS[i] == y) { ci = i; break; }

        for (int x = 0; x < W; x += 2) {
            last_pkt = out.read();
            if (ci < 0) continue;

            uint8_t hw_y0 = (uint8_t)last_pkt.data.range(7,  0);
            uint8_t hw_y1 = (uint8_t)last_pkt.data.range(23, 16);
            
            bool is_border = (y == 0 || y == H-1 || x == 0 || x == W-1);
            bool is_border_y1 = (y == 0 || y == H-1 || (x+1) == 0 || (x+1) == W-1);
            // Golden para y0
            uint8_t golden_y0 = 0;
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    if (!MASK_CROSS[ky][kx]) continue;
                    int sy = y + ky - 1;
                    int sx = x + kx - 1;
                    if (sx < 0 || sx >= W) continue;
                    
                    golden_y0 = is_border ? 0: std::max(golden_y0, saved_rows[sy][sx]);
                }

            uint8_t golden_y1 = 0;
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    if (!MASK_CROSS[ky][kx]) continue;
                    int sy = y + ky - 1;
                    int sx = (x+1) + kx - 1;
                    if (sx < 0 || sx >= W) continue;
                    golden_y1 = is_border_y1 ? 0: std::max(golden_y1, saved_rows[sy][sx]);
                }

            if (hw_y0 != golden_y0) {
                std::cerr << "    [4K] Y[" << y << "," << x << "]"
                          << " hw=" << (int)hw_y0
                          << " expected=" << (int)golden_y0 << "\n";
                ok = false;
            }
            if (hw_y1 != golden_y1) {
                std::cerr << "    [4K] Y[" << y << "," << x+1 << "]"
                          << " hw=" << (int)hw_y1
                          << " expected=" << (int)golden_y1 << "\n";
                ok = false;
            }
        }
    }

    if (last_pkt.last != 1) {
        std::cerr << "    [4K] LAST=0 en el último paquete\n";
        ok = false;
    }
    if (!out.empty()) {
        std::cerr << "    [4K] Stream no vacío tras leer " << TOTAL << " paquetes\n";
        ok = false;
    }
    if (ok)
        std::cout << "    [4K] " << TOTAL << " paquetes verificados\n";
    return ok;
}

// ============================================================
// MAIN
// ============================================================
int main_transformation() {
    std::cout << "==========================================\n";
    std::cout << "  Testbench: transformation_3x3\n";
    std::cout << "  Operaciones: dilatación y erosión\n";
    std::cout << "  PPP=" << PPP << "\n";
    std::cout << "==========================================\n\n";

    report("T01 - Conteo exacto de paquetes",          test_packet_count());
    report("T02 - Bordes en cero",                     test_border_zeros());
    report("T03 - Dilatación imagen uniforme",         test_dilation_uniform());
    report("T04 - Erosión imagen uniforme",            test_erosion_uniform());
    report("T05 - Máscara centro = identidad",         test_mask_center_identity());
    report("T06 - Máscara vacía → todo cero",          test_mask_empty());
    report("T07 - Dilatación impulso → forma máscara", test_dilation_impulse());
    report("T08 - Erosión impulso → cero",             test_erosion_impulse());
    report("T09 - Dilatación expande regiones",        test_dilation_expands());
    report("T10 - Erosión contrae regiones",           test_erosion_contracts());
    report("T11 - Golden model dilatación",            test_golden_dilation());
    report("T12 - Golden model erosión",               test_golden_erosion());
    report("T13 - Croma propagado correctamente",      test_chroma_propagation());
    report("T14 - Señal LAST en paquete correcto",     test_last_signal());
    report("T15 - Resolución mínima 4x4",              test_minimum_resolution());
    report("T16 - Valores extremos 0x00 y 0xFF",       test_extreme_values());
    report("T17 - Máscara esquina superior izquierda", test_mask_corner());
    report("T18 - Idempotencia dilatación uniforme",   test_idempotence_uniform());
    report("T19 - Resolución 4K sin deadlock",         test_4k_resolution());

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

int main() {
    return main_transformation();
}