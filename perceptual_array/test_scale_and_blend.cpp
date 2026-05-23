// ============================================================
// Testbench: test_scale_and_blend.cpp
// Componente bajo prueba: stage_scale_and_blend
//
// Invariantes:
//   - alpha=max, inv_k_1malpha=0 → resultado dominado por P0
//   - alpha=0 → resultado = conv*C >> SHIFT (sin término P0)
//   - Clase correcta recibe el término alpha (mux P0)
//   - Saturación a 255 si suma > 255
//   - Conteo exacto de vectores emitidos
//   - Golden model con valores conocidos
//   - P0 se consume al ritmo correcto (1 grupo cada PPP ventanas)
// ============================================================

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>
#include "perceptual_array.hpp"

static int tests_run_s = 0, tests_passed_s = 0, tests_failed_s = 0;
static void report_s(const std::string& name, bool ok) {
    tests_run_s++;
    if (ok) { tests_passed_s++; std::cout << "  [PASS] " << name << "\n"; }
    else     { tests_failed_s++; std::cerr << "  [FAIL] " << name << "\n"; }
}

static uint8_t get_blend(blend_vec_t v, int c) {
    return (uint8_t)v.range(c*8+7, c*8);
}

// Construye un conv_vec_t con todos los valores iguales a val para clase c
static conv_vec_t make_conv_vec(uint16_t vals[NUM_CLASSES]) {
    conv_vec_t v = 0;
    for (int c = 0; c < NUM_CLASSES; c++)
        v.range(c*14+13, c*14) = vals[c] & 0x3FFF;
    return v;
}

// Construye un pgroup_t con PPP píxeles todos de clase c
static pgroup_t make_p0_group(uint8_t classes[PPP]) {
    pgroup_t g = 0;
    for (int p = 0; p < PPP; p++)
        g.range(p*BITS_CLASS+BITS_CLASS-1, p*BITS_CLASS) = classes[p] & 0xF;
    return g;
}

// Golden model del scale_and_blend para 1 píxel
static uint8_t golden_blend(uint16_t conv_val, uint8_t p0_class,
                              int target_class, uint8_t alpha,
                              uint16_t inv_k_1malpha) {
    uint32_t scaled_raw = ((uint32_t)conv_val * inv_k_1malpha) >> SHIFT;
    uint8_t  scaled = (uint8_t)std::min(scaled_raw, (uint32_t)255);
    uint8_t  p0t    = (p0_class == target_class) ? alpha : 0;
    uint32_t sum    = scaled + p0t;
    return (uint8_t)std::min(sum, (uint32_t)255);
}

// ============================================================
// TEST 1 — alpha=63, inv_k_1malpha=0 → resultado = alpha si clase correcta
// ============================================================
static bool test_s_alpha_dominates() {
    const uint8_t ALPHA = 63, C = 0;
    hls::stream<conv_vec_t>  conv_in;
    hls::stream<pgroup_t>    p0_in;
    hls::stream<blend_vec_t> blend_out;

    uint16_t convs[NUM_CLASSES] = {};
    conv_in.write(make_conv_vec(convs));

    uint8_t p0_cls[PPP] = {C, C, C, C};
    p0_in.write(make_p0_group(p0_cls));

    stage_scale_and_blend(conv_in, p0_in, blend_out, ALPHA, 0, 1, 1);

    blend_vec_t v = blend_out.read();
    bool ok = true;
    if (get_blend(v, C) != ALPHA) {
        std::cerr << "    blend[" << C << "]=" << (int)get_blend(v,C)
                  << " esperado=" << (int)ALPHA << "\n";
        ok = false;
    }
    for (int c = 1; c < NUM_CLASSES; c++)
        if (get_blend(v,c) != 0) {
            std::cerr << "    blend[" << c << "]=" << (int)get_blend(v,c)
                      << " esperado=0\n";
            ok = false;
        }
    return ok;
}

// ============================================================
// TEST 2 — alpha=0 → resultado = conv*C >> SHIFT
// ============================================================
static bool test_s_no_alpha() {
    const uint8_t INV_K = 128;  // C = 128
    hls::stream<conv_vec_t>  conv_in;
    hls::stream<pgroup_t>    p0_in;
    hls::stream<blend_vec_t> blend_out;

    uint16_t convs[NUM_CLASSES] = {};
    convs[3] = 1000;
    conv_in.write(make_conv_vec(convs));

    uint8_t p0_cls[PPP] = {3,3,3,3};
    p0_in.write(make_p0_group(p0_cls));

    stage_scale_and_blend(conv_in, p0_in, blend_out, 0, INV_K, 1, 1);

    blend_vec_t v = blend_out.read();
    uint8_t expected = (uint8_t)std::min((uint32_t)((1000u * INV_K) >> SHIFT), 255u);
    bool ok = (get_blend(v,3) == expected);
    if (!ok)
        std::cerr << "    blend[3]=" << (int)get_blend(v,3)
                  << " esperado=" << (int)expected << "\n";
    return ok;
}

// ============================================================
// TEST 3 — Solo la clase correcta recibe el término alpha
// ============================================================
static bool test_s_p0_mux_correct() {
    const uint8_t ALPHA = 50, TARGET = 7;
    hls::stream<conv_vec_t>  conv_in;
    hls::stream<pgroup_t>    p0_in;
    hls::stream<blend_vec_t> blend_out;

    uint16_t convs[NUM_CLASSES] = {};
    conv_in.write(make_conv_vec(convs));

    uint8_t p0_cls[PPP] = {TARGET,TARGET,TARGET,TARGET};
    p0_in.write(make_p0_group(p0_cls));

    stage_scale_and_blend(conv_in, p0_in, blend_out, ALPHA, 0, 1, 1);

    blend_vec_t v = blend_out.read();
    bool ok = true;
    for (int c = 0; c < NUM_CLASSES; c++) {
        uint8_t expected = (c == TARGET) ? ALPHA : 0;
        if (get_blend(v,c) != expected) {
            std::cerr << "    blend[" << c << "]=" << (int)get_blend(v,c)
                      << " esperado=" << (int)expected << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 4 — Saturación a 255
// ============================================================
static bool test_s_saturation() {
    // Con INV_K=1400: scaled = (10647*1400)>>16 = 227; 227+63 = 290 > 255 → satura
    const uint8_t  ALPHA = 63;
    const uint16_t INV_K = 1400;
    hls::stream<conv_vec_t>  conv_in;
    hls::stream<pgroup_t>    p0_in;
    hls::stream<blend_vec_t> blend_out;

    uint16_t convs[NUM_CLASSES] = {};
    convs[0] = 10647;  // máximo posible
    conv_in.write(make_conv_vec(convs));

    uint8_t p0_cls[PPP] = {0,0,0,0};
    p0_in.write(make_p0_group(p0_cls));

    stage_scale_and_blend(conv_in, p0_in, blend_out, ALPHA, INV_K, 1, 1);

    blend_vec_t v = blend_out.read();
    bool ok = (get_blend(v,0) == 255);
    if (!ok)
        std::cerr << "    blend[0]=" << (int)get_blend(v,0) << " esperado=255\n";
    return ok;
}

// ============================================================
// TEST 5 — Conteo exacto de vectores emitidos
// ============================================================
static bool test_s_output_count() {
    const int N = 12;  // 12 píxeles = 3 grupos de PPP
    hls::stream<conv_vec_t>  conv_in;
    hls::stream<pgroup_t>    p0_in;
    hls::stream<blend_vec_t> blend_out;

    uint16_t convs[NUM_CLASSES] = {};
    uint8_t  p0_cls[PPP] = {0,0,0,0};

    for (int i = 0; i < N; i++) conv_in.write(make_conv_vec(convs));
    for (int i = 0; i < N/PPP; i++) p0_in.write(make_p0_group(p0_cls));

    stage_scale_and_blend(conv_in, p0_in, blend_out, 10, 10, N, 1);

    bool ok = ((int)blend_out.size() == N);
    if (!ok)
        std::cerr << "    emitidos=" << blend_out.size() << " esperado=" << N << "\n";
    while (!blend_out.empty()) blend_out.read();
    return ok;
}

// ============================================================
// TEST 6 — Golden model con valores conocidos
// ============================================================
static bool test_s_golden_model() {
    const uint8_t ALPHA = 30, INV_K = 50;
    const int N = 8;
    hls::stream<conv_vec_t>  conv_in;
    hls::stream<pgroup_t>    p0_in;
    hls::stream<blend_vec_t> blend_out;

    std::vector<conv_vec_t> convs_vec(N);
    std::vector<uint8_t>    p0_vals(N);

    for (int i = 0; i < N; i++) {
        uint16_t convs[NUM_CLASSES];
        for (int c = 0; c < NUM_CLASSES; c++)
            convs[c] = (uint16_t)((i*37 + c*53) % 10648);
        convs_vec[i] = make_conv_vec(convs);
        conv_in.write(convs_vec[i]);
        p0_vals[i] = (uint8_t)(i % NUM_CLASSES);
    }

    for (int i = 0; i < N/PPP; i++) {
        uint8_t cls[PPP];
        for (int p = 0; p < PPP; p++) cls[p] = p0_vals[i*PPP + p];
        p0_in.write(make_p0_group(cls));
    }

    stage_scale_and_blend(conv_in, p0_in, blend_out, ALPHA, INV_K, N, 1);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        blend_vec_t v = blend_out.read();
        for (int c = 0; c < NUM_CLASSES; c++) {
            uint16_t cv = (uint16_t)convs_vec[i].range(c*14+13, c*14);
            uint8_t  expected = golden_blend(cv, p0_vals[i], c, ALPHA, INV_K);
            if (get_blend(v,c) != expected) {
                std::cerr << "    i=" << i << " c=" << c
                          << " hw=" << (int)get_blend(v,c)
                          << " expected=" << (int)expected << "\n";
                ok = false;
            }
        }
    }
    return ok;
}

// ============================================================
// MAIN
// ============================================================
int main_scale_and_blend() {
    std::cout << "==========================================\n";
    std::cout << "  Testbench: stage_scale_and_blend\n";
    std::cout << "==========================================\n\n";

    report_s("T01 - Alpha domina cuando inv_k=0",       test_s_alpha_dominates());
    report_s("T02 - Sin alpha solo conv*C>>SHIFT",       test_s_no_alpha());
    report_s("T03 - Solo clase P0 recibe alpha",         test_s_p0_mux_correct());
    report_s("T04 - Saturación a 255",                   test_s_saturation());
    report_s("T05 - Conteo exacto de vectores",          test_s_output_count());
    report_s("T06 - Golden model valores conocidos",     test_s_golden_model());

    std::cout << "\n==========================================\n";
    std::cout << "  Resultado: " << tests_passed_s << "/"
              << tests_run_s << " tests pasaron\n";
    if (tests_failed_s > 0)
        std::cout << "  FALLARON: " << tests_failed_s << " tests\n";
    else
        std::cout << "  Todos los tests PASARON\n";
    std::cout << "==========================================\n";
    return tests_failed_s > 0 ? 1 : 0;
}