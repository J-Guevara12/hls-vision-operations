// ============================================================
// Testbench: test_conv_engine.cpp
// Componente bajo prueba: stage_conv_engine
//
// Invariantes:
//   - Kernel cero → conv=0 para todas las clases
//   - Ventana uniforme clase C con kernel uniforme → solo conv[C] > 0
//   - Ventana uniforme clase C → conv[C] = suma kernel, conv[resto]=0
//   - Kernel identidad (solo centro) → conv[C] = gauss_centro si píxel central es C
//   - Saturación del acumulador: máximo posible = 169*63 = 10647 < 2^14
//   - Conteo exacto de vectores emitidos
//   - Ventana mixta: golden model
// ============================================================

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include "perceptual_array.hpp"

static int tests_run_c = 0, tests_passed_c = 0, tests_failed_c = 0;
static void report_c(const std::string& name, bool ok) {
    tests_run_c++;
    if (ok) { tests_passed_c++; std::cout << "  [PASS] " << name << "\n"; }
    else     { tests_failed_c++; std::cerr << "  [FAIL] " << name << "\n"; }
}

// Construye una ventana donde todos los píxeles son de clase c
static window_t make_uniform_window(uint8_t c) {
    window_t win = 0;
    for (int i = 0; i < WIN_SIZE*WIN_SIZE; i++) {
        win.range(i*BITS_CLASS+BITS_CLASS-1, i*BITS_CLASS) = c & 0xF;
    }
    return win;
}

// Construye una ventana con patrón pseudo-aleatorio
static window_t make_random_window(int seed) {
    window_t win = 0;
    for (int i = 0; i < WIN_SIZE*WIN_SIZE; i++) {
        uint8_t c = (uint8_t)((seed*37 + i*53) % NUM_CLASSES);
        win.range(i*BITS_CLASS+BITS_CLASS-1, i*BITS_CLASS) = c;
    }
    return win;
}

// Extrae un valor de conv_vec_t para la clase c
static uint16_t get_conv(conv_vec_t v, int c) {
    return (uint16_t)v.range(c*12+11, c*12);
}

// Golden model del conv_engine
static void golden_conv(window_t win, uint8_t gauss[WIN_SIZE][WIN_SIZE],
                         uint16_t result[NUM_CLASSES]) {
    for (int c = 0; c < NUM_CLASSES; c++) result[c] = 0;
    for (int i = 0; i < WIN_SIZE*WIN_SIZE; i++) {
        uint8_t px  = (uint8_t)win.range(i*BITS_CLASS+BITS_CLASS-1, i*BITS_CLASS);
        uint8_t w   = gauss[i/WIN_SIZE][i%WIN_SIZE] & 0x3F;
        result[px] += w;
    }
}

static void run_conv(hls::stream<window_t>& in, hls::stream<conv_vec_t>& out,
                     uint8_t gauss[WIN_SIZE][WIN_SIZE], int n) {
    stage_conv_engine(in, out, gauss, n, 1);  // width=n, height=1 → n ventanas
}

// ============================================================
// TEST 1 — Kernel cero → conv=0 para todas las clases
// ============================================================
static bool test_c_zero_kernel() {
    uint8_t gauss[WIN_SIZE][WIN_SIZE] = {};
    hls::stream<window_t>   in;
    hls::stream<conv_vec_t> out;

    in.write(make_uniform_window(5));
    stage_conv_engine(in, out, gauss, 1, 1);

    conv_vec_t v = out.read();
    bool ok = true;
    for (int c = 0; c < NUM_CLASSES; c++)
        if (get_conv(v,c) != 0) {
            std::cerr << "    conv[" << c << "]=" << get_conv(v,c)
                      << " esperado=0\n";
            ok = false;
        }
    return ok;
}

// ============================================================
// TEST 2 — Ventana uniforme clase C → solo conv[C] > 0
// ============================================================
static bool test_c_uniform_window() {
    uint8_t gauss[WIN_SIZE][WIN_SIZE];
    // Kernel gaussiano simple: todos los pesos = 1
    for (int i = 0; i < WIN_SIZE; i++)
        for (int j = 0; j < WIN_SIZE; j++)
            gauss[i][j] = 1;

    bool ok = true;
    for (int target = 0; target < NUM_CLASSES; target++) {
        hls::stream<window_t>   in;
        hls::stream<conv_vec_t> out;
        in.write(make_uniform_window(target));
        stage_conv_engine(in, out, gauss, 1, 1);

        conv_vec_t v = out.read();
        for (int c = 0; c < NUM_CLASSES; c++) {
            uint16_t got = get_conv(v, c);
            if (c == target) {
                if (got != WIN_SIZE*WIN_SIZE) {
                    std::cerr << "    clase=" << target << " conv[" << c
                              << "]=" << got << " esperado=" << WIN_SIZE*WIN_SIZE << "\n";
                    ok = false;
                }
            } else {
                if (got != 0) {
                    std::cerr << "    clase=" << target << " conv[" << c
                              << "]=" << got << " esperado=0\n";
                    ok = false;
                }
            }
        }
    }
    return ok;
}

// ============================================================
// TEST 3 — Kernel identidad (solo centro) → conv[C] = gauss_centro
// ============================================================
static bool test_c_identity_kernel() {
    uint8_t gauss[WIN_SIZE][WIN_SIZE] = {};
    gauss[HALF_WIN][HALF_WIN] = 42;  // solo el centro activo

    bool ok = true;
    for (int target = 0; target < NUM_CLASSES; target++) {
        hls::stream<window_t>   in;
        hls::stream<conv_vec_t> out;
        in.write(make_uniform_window(target));
        stage_conv_engine(in, out, gauss, 1, 1);

        conv_vec_t v = out.read();
        if (get_conv(v, target) != 42) {
            std::cerr << "    clase=" << target << " conv=" << get_conv(v,target)
                      << " esperado=42\n";
            ok = false;
        }
        for (int c = 0; c < NUM_CLASSES; c++)
            if (c != target && get_conv(v,c) != 0) {
                std::cerr << "    clase=" << target << " conv[" << c
                          << "]=" << get_conv(v,c) << " esperado=0\n";
                ok = false;
            }
    }
    return ok;
}

// ============================================================
// TEST 4 — Valor máximo acumulador sin overflow
// 49 píxeles todos clase 0, kernel todos = 63 → conv[0] = 3087
// ============================================================
static bool test_c_max_accumulator() {
    uint8_t gauss[WIN_SIZE][WIN_SIZE];
    for (int i = 0; i < WIN_SIZE; i++)
        for (int j = 0; j < WIN_SIZE; j++)
            gauss[i][j] = 63;  // máximo valor de 6 bits

    hls::stream<window_t>   in;
    hls::stream<conv_vec_t> out;
    in.write(make_uniform_window(0));
    stage_conv_engine(in, out, gauss, 1, 1);

    conv_vec_t v = out.read();
    uint16_t expected = WIN_SIZE * WIN_SIZE * 63;  // 49 * 63 = 3087
    bool ok = (get_conv(v, 0) == expected);
    if (!ok)
        std::cerr << "    conv[0]=" << get_conv(v,0)
                  << " esperado=" << expected << "\n";
    return ok;
}

// ============================================================
// TEST 5 — Conteo exacto de vectores emitidos
// ============================================================
static bool test_c_output_count() {
    const int N = 16;  // 16 ventanas = width*height
    uint8_t gauss[WIN_SIZE][WIN_SIZE] = {};
    gauss[0][0] = 1;

    hls::stream<window_t>   in;
    hls::stream<conv_vec_t> out;
    for (int i = 0; i < N; i++) in.write(make_uniform_window(i % NUM_CLASSES));
    stage_conv_engine(in, out, gauss, N, 1);

    bool ok = ((int)out.size() == N);
    if (!ok)
        std::cerr << "    emitidos=" << out.size() << " esperado=" << N << "\n";
    while (!out.empty()) out.read();
    return ok;
}

// ============================================================
// TEST 6 — Golden model con ventana mixta pseudo-aleatoria
// ============================================================
static bool test_c_golden_model() {
    uint8_t gauss[WIN_SIZE][WIN_SIZE];
    for (int i = 0; i < WIN_SIZE; i++)
        for (int j = 0; j < WIN_SIZE; j++)
            gauss[i][j] = (uint8_t)((i*WIN_SIZE+j+1) % 64);

    const int N = 8;
    hls::stream<window_t>   in;
    hls::stream<conv_vec_t> out;

    std::vector<window_t> wins(N);
    for (int i = 0; i < N; i++) {
        wins[i] = make_random_window(i*31+7);
        in.write(wins[i]);
    }
    stage_conv_engine(in, out, gauss, N, 1);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        conv_vec_t v = out.read();
        uint16_t ref[NUM_CLASSES];
        golden_conv(wins[i], gauss, ref);
        for (int c = 0; c < NUM_CLASSES; c++) {
            if (get_conv(v,c) != ref[c]) {
                std::cerr << "    win=" << i << " clase=" << c
                          << " hw=" << get_conv(v,c)
                          << " expected=" << ref[c] << "\n";
                ok = false;
            }
        }
    }
    return ok;
}

// ============================================================
// TEST 7 — Bits de gauss se truncan a BITS_GAUSS (6 bits)
// Pasar gauss[i][j]=0xFF → efectivo=63
// ============================================================
static bool test_c_gauss_truncation() {
    uint8_t gauss[WIN_SIZE][WIN_SIZE];
    for (int i = 0; i < WIN_SIZE; i++)
        for (int j = 0; j < WIN_SIZE; j++)
            gauss[i][j] = 0xFF;  // 8 bits pero solo 6 efectivos → 63

    hls::stream<window_t>   in;
    hls::stream<conv_vec_t> out;
    in.write(make_uniform_window(0));
    stage_conv_engine(in, out, gauss, 1, 1);

    conv_vec_t v = out.read();
    uint16_t expected = WIN_SIZE * WIN_SIZE * 63;
    bool ok = (get_conv(v,0) == expected);
    if (!ok)
        std::cerr << "    conv[0]=" << get_conv(v,0)
                  << " esperado=" << expected << " (truncado a 6 bits)\n";
    return ok;
}

// ============================================================
// MAIN
// ============================================================
int main_conv_engine() {
    std::cout << "==========================================\n";
    std::cout << "  Testbench: stage_conv_engine\n";
    std::cout << "==========================================\n\n";

    report_c("T01 - Kernel cero → conv=0",              test_c_zero_kernel());
    report_c("T02 - Ventana uniforme → solo clase C",   test_c_uniform_window());
    report_c("T03 - Kernel identidad (solo centro)",    test_c_identity_kernel());
    report_c("T04 - Valor máximo sin overflow",         test_c_max_accumulator());
    report_c("T05 - Conteo exacto de vectores",         test_c_output_count());
    report_c("T06 - Golden model ventana mixta",        test_c_golden_model());
    report_c("T07 - Truncación gauss a 6 bits",         test_c_gauss_truncation());

    std::cout << "\n==========================================\n";
    std::cout << "  Resultado: " << tests_passed_c << "/"
              << tests_run_c << " tests pasaron\n";
    if (tests_failed_c > 0)
        std::cout << "  FALLARON: " << tests_failed_c << " tests\n";
    else
        std::cout << "  Todos los tests PASARON\n";
    std::cout << "==========================================\n";
    return tests_failed_c > 0 ? 1 : 0;
}