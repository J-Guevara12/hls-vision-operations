// ============================================================
// Testbench: test_unpack.cpp
// Componente bajo prueba: stage_unpack
//
// Invariantes:
//   - Conteo exacto de grupos emitidos en los 4 streams
//   - P0 extraído correctamente de bits [7:4] de cada byte
//   - Pk extraído correctamente de bits [3:0] de cada byte
//   - Orden de empaquetado: píxel p en bits [p*4+3:p*4]
//   - Los 4 streams emiten el mismo número de palabras
//   - p0_blend y p0_pack son idénticos
//   - pk_wm y pk_argmax son idénticos
//   - Valores extremos 0x00 y 0xFF
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

// Construye un paquete axis_t con 4 píxeles
static axis_t make_pkt(uint8_t p0[PPP], uint8_t pk[PPP], bool last = false) {
    axis_t pkt;
    pkt.data = 0;
    for (int p = 0; p < PPP; p++) {
        #pragma HLS UNROLL
        ap_uint<8> byte = ((ap_uint<8>)(p0[p] & 0xF) << 4) |
                           (ap_uint<8>)(pk[p] & 0xF);
        pkt.data.range(p*8+7, p*8) = byte;
    }
    pkt.keep = 0xF; pkt.strb = 0xF;
    pkt.last = last ? 1 : 0;
    return pkt;
}

// Extrae el píxel p del pgroup_t
static uint8_t get_px(pgroup_t g, int p) {
    return (uint8_t)g.range(p*BITS_CLASS+BITS_CLASS-1, p*BITS_CLASS);
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
    const int EXP = (W/PPP) * H;

    hls::stream<axis_t>   in;
    hls::stream<pgroup_t> p0b, p0p, pkw, pka;

    uint8_t p0[PPP] = {1,2,3,4}, pk[PPP] = {5,6,7,8};
    for (int i = 0; i < EXP; i++) in.write(make_pkt(p0, pk));

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
    hls::stream<axis_t>   in;
    hls::stream<pgroup_t> p0b, p0p, pkw, pka;

    // P0 = {0,1,2,3}, Pk = {4,5,6,7}
    uint8_t p0v[PPP] = {0,1,2,3}, pkv[PPP] = {4,5,6,7};
    for (int i = 0; i < (W/PPP)*H; i++) in.write(make_pkt(p0v, pkv));

    run_unpack(in, p0b, p0p, pkw, pka, W, H);

    bool ok = true;
    while (!p0b.empty()) {
        pgroup_t g = p0b.read();
        for (int p = 0; p < PPP; p++) {
            uint8_t got = get_px(g, p);
            if (got != p0v[p]) {
                std::cerr << "    P0[" << p << "]=" << (int)got
                          << " esperado=" << (int)p0v[p] << "\n";
                ok = false;
            }
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
    hls::stream<axis_t>   in;
    hls::stream<pgroup_t> p0b, p0p, pkw, pka;

    uint8_t p0v[PPP] = {15,14,13,12}, pkv[PPP] = {0,1,2,3};
    for (int i = 0; i < (W/PPP)*H; i++) in.write(make_pkt(p0v, pkv));

    run_unpack(in, p0b, p0p, pkw, pka, W, H);

    bool ok = true;
    while (!pkw.empty()) {
        pgroup_t g = pkw.read();
        for (int p = 0; p < PPP; p++) {
            uint8_t got = get_px(g, p);
            if (got != pkv[p]) {
                std::cerr << "    Pk[" << p << "]=" << (int)got
                          << " esperado=" << (int)pkv[p] << "\n";
                ok = false;
            }
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
    hls::stream<axis_t>   in;
    hls::stream<pgroup_t> p0b, p0p, pkw, pka;

    for (int i = 0; i < (W/PPP)*H; i++) {
        uint8_t p0v[PPP] = {(uint8_t)(i%16),(uint8_t)((i+1)%16),
                             (uint8_t)((i+2)%16),(uint8_t)((i+3)%16)};
        uint8_t pkv[PPP] = {0,0,0,0};
        in.write(make_pkt(p0v, pkv));
    }

    run_unpack(in, p0b, p0p, pkw, pka, W, H);

    bool ok = true;
    while (!p0b.empty() && !p0p.empty()) {
        pgroup_t gb = p0b.read();
        pgroup_t gp = p0p.read();
        if (gb != gp) {
            std::cerr << "    p0_blend != p0_pack\n";
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
    hls::stream<axis_t>   in;
    hls::stream<pgroup_t> p0b, p0p, pkw, pka;

    for (int i = 0; i < (W/PPP)*H; i++) {
        uint8_t p0v[PPP] = {0,0,0,0};
        uint8_t pkv[PPP] = {(uint8_t)(i%16),(uint8_t)((i+1)%16),
                             (uint8_t)((i+2)%16),(uint8_t)((i+3)%16)};
        in.write(make_pkt(p0v, pkv));
    }

    run_unpack(in, p0b, p0p, pkw, pka, W, H);

    bool ok = true;
    while (!pkw.empty() && !pka.empty()) {
        pgroup_t gw = pkw.read();
        pgroup_t ga = pka.read();
        if (gw != ga) {
            std::cerr << "    pk_wm != pk_argmax\n";
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
    hls::stream<axis_t>   in;
    hls::stream<pgroup_t> p0b, p0p, pkw, pka;

    // P0 = 0xA (10), Pk = 0x5 (5) — valores distintos fácil de distinguir
    uint8_t p0v[PPP] = {10,10,10,10}, pkv[PPP] = {5,5,5,5};
    in.write(make_pkt(p0v, pkv));

    run_unpack(in, p0b, p0p, pkw, pka, W, H);

    bool ok = true;
    pgroup_t gp0 = p0b.read();
    pgroup_t gpk = pkw.read();

    for (int p = 0; p < PPP; p++) {
        if (get_px(gp0, p) != 10) {
            std::cerr << "    P0[" << p << "]=" << (int)get_px(gp0,p)
                      << " esperado=10\n";
            ok = false;
        }
        if (get_px(gpk, p) != 5) {
            std::cerr << "    Pk[" << p << "]=" << (int)get_px(gpk,p)
                      << " esperado=5\n";
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
    const int W = 4, H = 1;
    hls::stream<axis_t>   in;
    hls::stream<pgroup_t> p0b, p0p, pkw, pka;

    // Byte 0x00: P0=0, Pk=0. Byte 0xFF: P0=15, Pk=15
    axis_t pkt;
    pkt.data.range(7,  0)  = 0x00;
    pkt.data.range(15, 8)  = 0xFF;
    pkt.data.range(23, 16) = 0x00;
    pkt.data.range(31, 24) = 0xFF;
    pkt.keep = 0xF; pkt.strb = 0xF; pkt.last = 1;
    in.write(pkt);

    run_unpack(in, p0b, p0p, pkw, pka, W, H);

    bool ok = true;
    pgroup_t gp0 = p0b.read();
    pgroup_t gpk = pkw.read();

    // Píxel 0: P0=0, Pk=0
    if (get_px(gp0,0) != 0 || get_px(gpk,0) != 0) {
        std::cerr << "    Píxel 0: P0=" << (int)get_px(gp0,0)
                  << " Pk=" << (int)get_px(gpk,0) << " esperado 0,0\n";
        ok = false;
    }
    // Píxel 1: P0=15, Pk=15
    if (get_px(gp0,1) != 15 || get_px(gpk,1) != 15) {
        std::cerr << "    Píxel 1: P0=" << (int)get_px(gp0,1)
                  << " Pk=" << (int)get_px(gpk,1) << " esperado 15,15\n";
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
    const int GROUPS = (W/PPP)*H;
    hls::stream<axis_t>   in;
    hls::stream<pgroup_t> p0b, p0p, pkw, pka;

    uint8_t p0v[PPP] = {1,2,3,4}, pkv[PPP] = {5,6,7,8};
    for (int i = 0; i < GROUPS; i++) {
        bool last = (i == GROUPS-1);
        in.write(make_pkt(p0v, pkv, last));
    }

    run_unpack(in, p0b, p0p, pkw, pka, W, H);

    // Todos los grupos deben haberse procesado independientemente del LAST
    bool ok = ((int)p0b.size() == GROUPS);
    if (!ok)
        std::cerr << "    Grupos recibidos=" << p0b.size()
                  << " esperado=" << GROUPS << "\n";
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