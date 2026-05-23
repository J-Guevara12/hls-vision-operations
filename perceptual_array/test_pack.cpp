// ============================================================
// Testbench: test_pack.cpp
// Componente bajo prueba: stage_pack
//
// Invariantes:
//   - Pk+1 en bits [3:0] de cada byte
//   - P0 en bits [7:4] de cada byte
//   - Conteo exacto de paquetes emitidos
//   - TLAST=1 solo en el último paquete
//   - P0 y Pk+1 no se mezclan entre bytes
//   - Valores extremos 0x0 y 0xF
// ============================================================

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include "perceptual_array.hpp"

static int tests_run_p = 0, tests_passed_p = 0, tests_failed_p = 0;
static void report_p(const std::string& name, bool ok) {
    tests_run_p++;
    if (ok) { tests_passed_p++; std::cout << "  [PASS] " << name << "\n"; }
    else     { tests_failed_p++; std::cerr << "  [FAIL] " << name << "\n"; }
}

static result_group_t make_result(uint8_t vals[PPP]) {
    result_group_t g = 0;
    for (int p = 0; p < PPP; p++)
        g.range(p*BITS_CLASS+BITS_CLASS-1, p*BITS_CLASS) = vals[p] & 0xF;
    return g;
}

static pgroup_t make_p0(uint8_t vals[PPP]) {
    pgroup_t g = 0;
    for (int p = 0; p < PPP; p++)
        g.range(p*BITS_CLASS+BITS_CLASS-1, p*BITS_CLASS) = vals[p] & 0xF;
    return g;
}

// ============================================================
// TEST 1 — Pk+1 en bits [3:0] y P0 en bits [7:4] de cada byte
// ============================================================
static bool test_p_bit_positions() {
    const int W = 4, H = 1;
    hls::stream<result_group_t> r_in;
    hls::stream<pgroup_t>       p0_in;
    hls::stream<axis_t>         out;

    uint8_t pk1[PPP] = {1,2,3,4};
    uint8_t p0v[PPP] = {5,6,7,8};
    r_in.write(make_result(pk1));
    p0_in.write(make_p0(p0v));

    stage_pack(r_in, p0_in, out, W, H);

    axis_t pkt = out.read();
    bool ok = true;
    for (int p = 0; p < PPP; p++) {
        uint8_t byte  = (uint8_t)pkt.data.range(p*8+7, p*8);
        uint8_t got_p0  = (byte >> 4) & 0xF;
        uint8_t got_pk1 = byte & 0xF;
        if (got_p0 != p0v[p]) {
            std::cerr << "    P0[" << p << "]=" << (int)got_p0
                      << " esperado=" << (int)p0v[p] << "\n";
            ok = false;
        }
        if (got_pk1 != pk1[p]) {
            std::cerr << "    Pk1[" << p << "]=" << (int)got_pk1
                      << " esperado=" << (int)pk1[p] << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 2 — Conteo exacto de paquetes
// ============================================================
static bool test_p_output_count() {
    const int W = 8, H = 4;
    const int EXP = (W/PPP) * H;
    hls::stream<result_group_t> r_in;
    hls::stream<pgroup_t>       p0_in;
    hls::stream<axis_t>         out;

    uint8_t pk1[PPP] = {0,0,0,0}, p0v[PPP] = {0,0,0,0};
    for (int i = 0; i < EXP; i++) {
        r_in.write(make_result(pk1));
        p0_in.write(make_p0(p0v));
    }

    stage_pack(r_in, p0_in, out, W, H);

    bool ok = ((int)out.size() == EXP);
    if (!ok)
        std::cerr << "    count=" << out.size() << " esperado=" << EXP << "\n";
    while (!out.empty()) out.read();
    return ok;
}

// ============================================================
// TEST 3 — TLAST=1 solo en el último paquete
// ============================================================
static bool test_p_last_signal() {
    const int W = 8, H = 4;
    const int TOTAL = (W/PPP) * H;
    hls::stream<result_group_t> r_in;
    hls::stream<pgroup_t>       p0_in;
    hls::stream<axis_t>         out;

    uint8_t pk1[PPP] = {0,0,0,0}, p0v[PPP] = {0,0,0,0};
    for (int i = 0; i < TOTAL; i++) {
        r_in.write(make_result(pk1));
        p0_in.write(make_p0(p0v));
    }

    stage_pack(r_in, p0_in, out, W, H);

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
// TEST 4 — P0 y Pk+1 no se mezclan entre bytes del paquete
// ============================================================
static bool test_p_no_cross_byte() {
    const int W = 4, H = 1;
    hls::stream<result_group_t> r_in;
    hls::stream<pgroup_t>       p0_in;
    hls::stream<axis_t>         out;

    // Todos Pk1=0xA, todos P0=0x5 → cada byte debe ser 0x5A
    uint8_t pk1[PPP] = {0xA,0xA,0xA,0xA};
    uint8_t p0v[PPP] = {0x5,0x5,0x5,0x5};
    r_in.write(make_result(pk1));
    p0_in.write(make_p0(p0v));

    stage_pack(r_in, p0_in, out, W, H);

    axis_t pkt = out.read();
    bool ok = true;
    for (int p = 0; p < PPP; p++) {
        uint8_t byte = (uint8_t)pkt.data.range(p*8+7, p*8);
        if (byte != 0x5A) {
            std::cerr << "    byte[" << p << "]=0x" << std::hex << (int)byte
                      << " esperado=0x5A\n" << std::dec;
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 5 — Valores extremos 0x0 y 0xF
// ============================================================
static bool test_p_extreme_values() {
    const int W = 4, H = 1;
    hls::stream<result_group_t> r_in;
    hls::stream<pgroup_t>       p0_in;
    hls::stream<axis_t>         out;

    uint8_t pk1[PPP] = {0x0,0xF,0x0,0xF};
    uint8_t p0v[PPP] = {0xF,0x0,0xF,0x0};
    r_in.write(make_result(pk1));
    p0_in.write(make_p0(p0v));

    stage_pack(r_in, p0_in, out, W, H);

    axis_t pkt = out.read();
    bool ok = true;
    // byte 0: P0=0xF, Pk1=0x0 → 0xF0
    // byte 1: P0=0x0, Pk1=0xF → 0x0F
    uint8_t expected[PPP] = {0xF0, 0x0F, 0xF0, 0x0F};
    for (int p = 0; p < PPP; p++) {
        uint8_t byte = (uint8_t)pkt.data.range(p*8+7, p*8);
        if (byte != expected[p]) {
            std::cerr << "    byte[" << p << "]=0x" << std::hex << (int)byte
                      << " esperado=0x" << (int)expected[p] << "\n" << std::dec;
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 6 — keep y strb = 0xF en todos los paquetes
// ============================================================
static bool test_p_keep_strb() {
    const int W = 8, H = 2;
    const int TOTAL = (W/PPP) * H;
    hls::stream<result_group_t> r_in;
    hls::stream<pgroup_t>       p0_in;
    hls::stream<axis_t>         out;

    uint8_t pk1[PPP] = {1,2,3,4}, p0v[PPP] = {5,6,7,8};
    for (int i = 0; i < TOTAL; i++) {
        r_in.write(make_result(pk1));
        p0_in.write(make_p0(p0v));
    }

    stage_pack(r_in, p0_in, out, W, H);

    bool ok = true;
    for (int i = 0; i < TOTAL; i++) {
        axis_t pkt = out.read();
        if (pkt.keep != 0xF || pkt.strb != 0xF) {
            std::cerr << "    Pkt[" << i << "] keep=0x" << std::hex
                      << (int)pkt.keep << " strb=0x" << (int)pkt.strb
                      << " esperado 0xF/0xF\n" << std::dec;
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// MAIN
// ============================================================
int main_pack() {
    std::cout << "==========================================\n";
    std::cout << "  Testbench: stage_pack\n";
    std::cout << "==========================================\n\n";

    report_p("T01 - Pk+1 en [3:0] y P0 en [7:4]",      test_p_bit_positions());
    report_p("T02 - Conteo exacto de paquetes",          test_p_output_count());
    report_p("T03 - TLAST solo en último paquete",       test_p_last_signal());
    report_p("T04 - Sin mezcla entre bytes",             test_p_no_cross_byte());
    report_p("T05 - Valores extremos 0x0 y 0xF",        test_p_extreme_values());
    report_p("T06 - keep y strb = 0xF",                  test_p_keep_strb());

    std::cout << "\n==========================================\n";
    std::cout << "  Resultado: " << tests_passed_p << "/"
              << tests_run_p << " tests pasaron\n";
    if (tests_failed_p > 0)
        std::cout << "  FALLARON: " << tests_failed_p << " tests\n";
    else
        std::cout << "  Todos los tests PASARON\n";
    std::cout << "==========================================\n";
    return tests_failed_p > 0 ? 1 : 0;
}