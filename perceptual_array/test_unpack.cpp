// ============================================================
// Testbench: test_unpack.cpp
// Componente bajo prueba: stage_unpack
//
// Invariantes (PPP=1: 1 píxel por paquete axis_t):
//   - Conteo exacto de grupos emitidos en los 4 streams (W×H)
//   - P0 extraído correctamente de bits [7:4] del byte [7:0]
//   - Pk extraído correctamente de bits [3:0] del byte [7:0]
//   - Los 4 streams emiten el mismo número de palabras
//   - p0_blend y p0_pack son idénticos
//   - pk_wm y pk_argmax son idénticos
//   - P0 y Pk no se contaminan entre sí
//   - Valores extremos 0x00 y 0xFF
//   - TLAST no afecta el procesamiento
// ============================================================

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include "perceptual_array.hpp"

static int tests_run_u = 0, tests_passed_u = 0, tests_failed_u = 0;
static void report_u(const std::string& name, bool ok) {
    tests_run_u++;
    if (ok) { tests_passed_u++; std::cout << "  [PASS] " << name << "\n"; }
    else     { tests_failed_u++; std::cerr << "  [FAIL] " << name << "\n"; }
}

// PPP=1: 1 byte per pkt at bits[7:0]; bits[7:4]=P0, bits[3:0]=PK
static axis_t make_pkt(uint8_t p0, uint8_t pk, bool last = false) {
    axis_t pkt;
    pkt.data = 0;
    pkt.data.range(7, 0) = ((ap_uint<8>)(p0 & 0xF) << 4) | (ap_uint<8>)(pk & 0xF);
    pkt.keep = 0xF; pkt.strb = 0xF;
    pkt.last = last ? 1 : 0;
    return pkt;
}

// PPP=1: pgroup_t is 4 bits holding exactly 1 class value
static uint8_t get_px(pgroup_t g) {
    return (uint8_t)g.range(BITS_CLASS-1, 0);
}

static void run_unpack(hls::stream<axis_t>& in,
                       hls::stream<pgroup_t>& p0b, hls::stream<pgroup_t>& p0p,
                       hls::stream<pgroup_t>& pkw, hls::stream<pgroup_t>& pka,
                       int w, int h) {
    stage_unpack(in, p0b, p0p, pkw, pka, w, h);
}

// ============================================================
// TEST 1 — Conteo exacto en los 4 streams
// ============================================================
static bool test_u_count() {
    const int W = 8, H = 4;
    const int EXP = W * H;

    hls::stream<axis_t>   in;
    hls::stream<pgroup_t> p0b, p0p, pkw, pka;

    for (int i = 0; i < EXP; i++) in.write(make_pkt(1, 5));

    run_unpack(in, p0b, p0p, pkw, pka, W, H);

    bool ok = ((int)p0b.size()==EXP && (int)p0p.size()==EXP &&
               (int)pkw.size()==EXP && (int)pka.size()==EXP);
    if (!ok)
        std::cerr << "    p0b=" << p0b.size() << " p0p=" << p0p.size()
                  << " pkw=" << pkw.size() << " pka=" << pka.size()
                  << " esperado=" << EXP << "\n";
    while (!p0b.empty()) p0b.read();
    while (!p0p.empty()) p0p.read();
    while (!pkw.empty()) pkw.read();
    while (!pka.empty()) pka.read();
    return ok;
}

// ============================================================
// TEST 2 — P0 extraído correctamente
// ============================================================
static bool test_u_p0_extraction() {
    const int W = 4, H = 2;
    const int N = W * H;
    hls::stream<axis_t>   in;
    hls::stream<pgroup_t> p0b, p0p, pkw, pka;

    std::vector<uint8_t> p0_ref(N);
    for (int i = 0; i < N; i++) {
        p0_ref[i] = (uint8_t)(i % NUM_CLASSES);
        in.write(make_pkt(p0_ref[i], (15 - p0_ref[i]) & 0xF));
    }

    run_unpack(in, p0b, p0p, pkw, pka, W, H);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        uint8_t got = get_px(p0b.read());
        if (got != p0_ref[i]) {
            std::cerr << "    P0[" << i << "]=" << (int)got
                      << " esperado=" << (int)p0_ref[i] << "\n";
            ok = false;
        }
    }
    while (!p0p.empty()) p0p.read();
    while (!pkw.empty()) pkw.read();
    while (!pka.empty()) pka.read();
    return ok;
}

// ============================================================
// TEST 3 — Pk extraído correctamente
// ============================================================
static bool test_u_pk_extraction() {
    const int W = 4, H = 2;
    const int N = W * H;
    hls::stream<axis_t>   in;
    hls::stream<pgroup_t> p0b, p0p, pkw, pka;

    std::vector<uint8_t> pk_ref(N);
    for (int i = 0; i < N; i++) {
        pk_ref[i] = (uint8_t)(i % NUM_CLASSES);
        in.write(make_pkt((15 - pk_ref[i]) & 0xF, pk_ref[i]));
    }

    run_unpack(in, p0b, p0p, pkw, pka, W, H);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        uint8_t got = get_px(pkw.read());
        if (got != pk_ref[i]) {
            std::cerr << "    Pk[" << i << "]=" << (int)got
                      << " esperado=" << (int)pk_ref[i] << "\n";
            ok = false;
        }
    }
    while (!p0b.empty()) p0b.read();
    while (!p0p.empty()) p0p.read();
    while (!pka.empty()) pka.read();
    return ok;
}

// ============================================================
// TEST 4 — p0_blend y p0_pack son idénticos
// ============================================================
static bool test_u_p0_identical() {
    const int W = 8, H = 4;
    const int N = W * H;
    hls::stream<axis_t>   in;
    hls::stream<pgroup_t> p0b, p0p, pkw, pka;

    for (int i = 0; i < N; i++)
        in.write(make_pkt((uint8_t)(i % NUM_CLASSES), 0));

    run_unpack(in, p0b, p0p, pkw, pka, W, H);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        uint8_t gb = get_px(p0b.read());
        uint8_t gp = get_px(p0p.read());
        if (gb != gp) {
            std::cerr << "    [" << i << "] p0_blend=" << (int)gb
                      << " != p0_pack=" << (int)gp << "\n";
            ok = false;
        }
    }
    while (!pkw.empty()) pkw.read();
    while (!pka.empty()) pka.read();
    return ok;
}

// ============================================================
// TEST 5 — pk_wm y pk_argmax son idénticos
// ============================================================
static bool test_u_pk_identical() {
    const int W = 8, H = 4;
    const int N = W * H;
    hls::stream<axis_t>   in;
    hls::stream<pgroup_t> p0b, p0p, pkw, pka;

    for (int i = 0; i < N; i++)
        in.write(make_pkt(0, (uint8_t)(i % NUM_CLASSES)));

    run_unpack(in, p0b, p0p, pkw, pka, W, H);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        uint8_t gw = get_px(pkw.read());
        uint8_t ga = get_px(pka.read());
        if (gw != ga) {
            std::cerr << "    [" << i << "] pk_wm=" << (int)gw
                      << " != pk_argmax=" << (int)ga << "\n";
            ok = false;
        }
    }
    while (!p0b.empty()) p0b.read();
    while (!p0p.empty()) p0p.read();
    return ok;
}

// ============================================================
// TEST 6 — P0 y Pk no se mezclan entre sí
// ============================================================
static bool test_u_no_crosscontamination() {
    const int W = 4, H = 1;
    const int N = W * H;
    hls::stream<axis_t>   in;
    hls::stream<pgroup_t> p0b, p0p, pkw, pka;

    for (int i = 0; i < N; i++)
        in.write(make_pkt(10, 5));  // P0=0xA, Pk=0x5

    run_unpack(in, p0b, p0p, pkw, pka, W, H);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        uint8_t vp0 = get_px(p0b.read());
        uint8_t vpk = get_px(pkw.read());
        if (vp0 != 10) {
            std::cerr << "    [" << i << "] P0=" << (int)vp0 << " esperado=10\n";
            ok = false;
        }
        if (vpk != 5) {
            std::cerr << "    [" << i << "] Pk=" << (int)vpk << " esperado=5\n";
            ok = false;
        }
    }
    while (!p0p.empty()) p0p.read();
    while (!pka.empty()) pka.read();
    return ok;
}

// ============================================================
// TEST 7 — Valores extremos 0x00 y 0xFF
// ============================================================
static bool test_u_extreme_values() {
    const int W = 2, H = 1;
    hls::stream<axis_t>   in;
    hls::stream<pgroup_t> p0b, p0p, pkw, pka;

    in.write(make_pkt(0, 0));         // byte=0x00: P0=0, Pk=0
    in.write(make_pkt(15, 15, true)); // byte=0xFF: P0=15, Pk=15

    run_unpack(in, p0b, p0p, pkw, pka, W, H);

    bool ok = true;
    uint8_t p0_0 = get_px(p0b.read()), p0_f = get_px(p0b.read());
    uint8_t pk_0 = get_px(pkw.read()), pk_f = get_px(pkw.read());

    if (p0_0 != 0 || pk_0 != 0) {
        std::cerr << "    Pkt 0x00: P0=" << (int)p0_0
                  << " Pk=" << (int)pk_0 << " esperado 0,0\n";
        ok = false;
    }
    if (p0_f != 15 || pk_f != 15) {
        std::cerr << "    Pkt 0xFF: P0=" << (int)p0_f
                  << " Pk=" << (int)pk_f << " esperado 15,15\n";
        ok = false;
    }
    while (!p0p.empty()) p0p.read();
    while (!pka.empty()) pka.read();
    return ok;
}

// ============================================================
// TEST 8 — TLAST no afecta el procesamiento
// ============================================================
static bool test_u_last_ignored() {
    const int W = 8, H = 2;
    const int N = W * H;
    hls::stream<axis_t>   in;
    hls::stream<pgroup_t> p0b, p0p, pkw, pka;

    for (int i = 0; i < N; i++) {
        bool last = (i == N-1);
        in.write(make_pkt(1, 5, last));
    }

    run_unpack(in, p0b, p0p, pkw, pka, W, H);

    bool ok = ((int)p0b.size() == N);
    if (!ok)
        std::cerr << "    recibidos=" << p0b.size() << " esperado=" << N << "\n";
    while (!p0b.empty()) p0b.read();
    while (!p0p.empty()) p0p.read();
    while (!pkw.empty()) pkw.read();
    while (!pka.empty()) pka.read();
    return ok;
}

// ============================================================
// MAIN
// ============================================================
int main_unpack() {
    std::cout << "==========================================\n";
    std::cout << "  Testbench: stage_unpack\n";
    std::cout << "==========================================\n\n";

    report_u("T01 - Conteo exacto 4 streams",          test_u_count());
    report_u("T02 - P0 extraído correctamente",         test_u_p0_extraction());
    report_u("T03 - Pk extraído correctamente",         test_u_pk_extraction());
    report_u("T04 - p0_blend == p0_pack",               test_u_p0_identical());
    report_u("T05 - pk_wm == pk_argmax",                test_u_pk_identical());
    report_u("T06 - P0 y Pk no se contaminan",          test_u_no_crosscontamination());
    report_u("T07 - Valores extremos 0x00 y 0xFF",      test_u_extreme_values());
    report_u("T08 - TLAST no afecta procesamiento",     test_u_last_ignored());

    std::cout << "\n==========================================\n";
    std::cout << "  Resultado: " << tests_passed_u << "/"
              << tests_run_u << " tests pasaron\n";
    if (tests_failed_u > 0)
        std::cout << "  FALLARON: " << tests_failed_u << " tests\n";
    else
        std::cout << "  Todos los tests PASARON\n";
    std::cout << "==========================================\n";
    return tests_failed_u > 0 ? 1 : 0;
}
