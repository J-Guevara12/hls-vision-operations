// ============================================================
// Testbench: test_pack.cpp
// Componente bajo prueba: stage_pack
//
// Invariantes (PPP=1: 1 píxel per paquete axis_t):
//   - Pk+1 en bits [3:0] del byte [7:0]
//   - P0 en bits [7:4] del byte [7:0]
//   - Conteo exacto de paquetes emitidos (W×H paquetes)
//   - TLAST=1 solo en el último paquete
//   - P0 y Pk+1 no se mezclan
//   - Valores extremos 0x0 y 0xF
//   - keep y strb = 0xF en todos los paquetes
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
// TEST 1 — Pk+1 en bits [3:0] y P0 en bits [7:4] por paquete
// ============================================================
static bool test_p_bit_positions() {
    const int W = 4, H = 1;
    hls::stream<result_group_t> r_in;
    hls::stream<pgroup_t>       p0_in;
    hls::stream<axis_t>         out;

    uint8_t pk1_vals[4] = {1, 2, 3, 4};
    uint8_t p0_vals[4]  = {5, 6, 7, 8};
    for (int i = 0; i < W; i++) {
        uint8_t pk1[PPP] = {pk1_vals[i]};
        uint8_t p0v[PPP] = {p0_vals[i]};
        r_in.write(make_result(pk1));
        p0_in.write(make_p0(p0v));
    }

    stage_pack(r_in, p0_in, out, W, H);

    bool ok = true;
    for (int i = 0; i < W; i++) {
        axis_t pkt = out.read();
        uint8_t byte    = (uint8_t)pkt.data.range(7, 0);
        uint8_t got_p0  = (byte >> 4) & 0xF;
        uint8_t got_pk1 = byte & 0xF;
        if (got_p0 != p0_vals[i]) {
            std::cerr << "    pkt[" << i << "] P0=" << (int)got_p0
                      << " esperado=" << (int)p0_vals[i] << "\n";
            ok = false;
        }
        if (got_pk1 != pk1_vals[i]) {
            std::cerr << "    pkt[" << i << "] Pk1=" << (int)got_pk1
                      << " esperado=" << (int)pk1_vals[i] << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 2 — Conteo exacto de paquetes (W×H con PPP=1)
// ============================================================
static bool test_p_output_count() {
    const int W = 8, H = 4;
    const int EXP = W * H;
    hls::stream<result_group_t> r_in;
    hls::stream<pgroup_t>       p0_in;
    hls::stream<axis_t>         out;

    uint8_t pk1[PPP] = {0}, p0v[PPP] = {0};
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
    const int TOTAL = W * H;
    hls::stream<result_group_t> r_in;
    hls::stream<pgroup_t>       p0_in;
    hls::stream<axis_t>         out;

    uint8_t pk1[PPP] = {0}, p0v[PPP] = {0};
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
// TEST 4 — P0 y Pk+1 no se mezclan
// ============================================================
static bool test_p_no_cross_byte() {
    const int W = 1, H = 1;
    hls::stream<result_group_t> r_in;
    hls::stream<pgroup_t>       p0_in;
    hls::stream<axis_t>         out;

    // Pk1=0xA, P0=0x5 → byte debe ser 0x5A
    uint8_t pk1[PPP] = {0xA};
    uint8_t p0v[PPP] = {0x5};
    r_in.write(make_result(pk1));
    p0_in.write(make_p0(p0v));

    stage_pack(r_in, p0_in, out, W, H);

    axis_t pkt = out.read();
    uint8_t byte = (uint8_t)pkt.data.range(7, 0);
    bool ok = (byte == 0x5A);
    if (!ok)
        std::cerr << "    byte=0x" << std::hex << (int)byte
                  << " esperado=0x5A\n" << std::dec;
    return ok;
}

// ============================================================
// TEST 5 — Valores extremos 0x0 y 0xF
// ============================================================
static bool test_p_extreme_values() {
    const int W = 2, H = 1;
    hls::stream<result_group_t> r_in;
    hls::stream<pgroup_t>       p0_in;
    hls::stream<axis_t>         out;

    // pkt 0: P0=0xF, Pk1=0x0 → byte=0xF0
    // pkt 1: P0=0x0, Pk1=0xF → byte=0x0F
    uint8_t pk1_a[PPP] = {0x0}; uint8_t p0a[PPP] = {0xF};
    uint8_t pk1_b[PPP] = {0xF}; uint8_t p0b[PPP] = {0x0};
    r_in.write(make_result(pk1_a)); p0_in.write(make_p0(p0a));
    r_in.write(make_result(pk1_b)); p0_in.write(make_p0(p0b));

    stage_pack(r_in, p0_in, out, W, H);

    bool ok = true;
    uint8_t b0 = (uint8_t)out.read().data.range(7, 0);
    uint8_t b1 = (uint8_t)out.read().data.range(7, 0);
    if (b0 != 0xF0) {
        std::cerr << "    pkt0=0x" << std::hex << (int)b0
                  << " esperado=0xF0\n" << std::dec;
        ok = false;
    }
    if (b1 != 0x0F) {
        std::cerr << "    pkt1=0x" << std::hex << (int)b1
                  << " esperado=0x0F\n" << std::dec;
        ok = false;
    }
    return ok;
}

// ============================================================
// TEST 6 — keep y strb = 0xF en todos los paquetes
// ============================================================
static bool test_p_keep_strb() {
    const int W = 8, H = 2;
    const int TOTAL = W * H;
    hls::stream<result_group_t> r_in;
    hls::stream<pgroup_t>       p0_in;
    hls::stream<axis_t>         out;

    uint8_t pk1[PPP] = {1}, p0v[PPP] = {5};
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
    report_p("T04 - Sin mezcla entre P0 y Pk+1",        test_p_no_cross_byte());
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
