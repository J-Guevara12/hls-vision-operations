//============================================================
// Testbench: test_hls_lib::stage_write.cpp
// Componente bajo prueba: hls_lib::stage_write
//
// Responsabilidad del stage:
//   - Leer palabras de PPP píxeles Y empaquetados del y_stream
//   - Leer 1 par UV del uv_stream por cada palabra Y
//   - Empaquetar en paquetes YUYV de salida
//   - Generar señal LAST correctamente en el último paquete
//
// Invariantes a verificar:
//   - Conteo exacto de paquetes = (width/PPP) * height
//   - Y0 en bits [7:0], Y1 en bits [23:16]
//   - U  en bits [15:8], V en bits [31:24]
//   - LAST=1 solo en el último paquete
//   - UV del stream se asigna al paquete correcto
//============================================================

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdint>

#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "hls_vision_utils.hpp"

#define PPP 2

// ============================================================
// Contadores
// ============================================================
static int tests_run_w = 0, tests_passed_w = 0, tests_failed_w = 0;
static void report_w(const std::string& name, bool ok) {
    tests_run_w++;
    if (ok) { tests_passed_w++; std::cout << "  [PASS] " << name << "\n"; }
    else     { tests_failed_w++; std::cerr << "  [FAIL] " << name << "\n"; }
}

// ============================================================
// Utilidades
// ============================================================

// Empaqueta PPP píxeles individuales en una palabra ap_uint<PPP*8>
// píxel p queda en bits [p*8+7 : p*8]
static void load_y(hls::stream<ap_uint<PPP*8>>& s,
                   const std::vector<uint8_t>& y_vals) {
    for (size_t i = 0; i < y_vals.size(); i += PPP) {
        ap_uint<PPP*8> word = 0;
        for (int p = 0; p < PPP; p++)
            word.range(p*8+7, p*8) = y_vals[i + p];
        s.write(word);
    }
}

// Carga uv_stream con pares (V<<8)|U
static void load_uv(hls::stream<uint16_t>& s,
                    const std::vector<std::pair<uint8_t,uint8_t>>& uvs) {
    for (auto [u, v] : uvs)
        s.write(((uint16_t)v << 8) | u);
}

// ============================================================
// TEST 1 — Conteo exacto de paquetes de salida
// ============================================================
static bool test_packet_count() {
    const int W = 8, H = 4;
    const int PKTS = (W / PPP) * H;

    hls::stream<ap_uint<PPP*8>> y_in;
    hls::stream<uint16_t>       uv_in;
    hls::stream<axis_t>         out;

    std::vector<uint8_t> y_vals(W * H, 100);
    load_y(y_in, y_vals);
    for (int i = 0; i < PKTS; i++) uv_in.write(0x8080);

    hls_lib::stage_write(y_in, uv_in, out, W, H);

    int count = 0;
    while (!out.empty()) { out.read(); count++; }
    bool ok = (count == PKTS);
    if (!ok) std::cerr << "    count=" << count << " esperado=" << PKTS << "\n";
    return ok;
}

// ============================================================
// TEST 2 — Y0 e Y1 en los bits correctos del paquete
// ============================================================
static bool test_y_bit_positions() {
    const int W = 4, H = 2;
    const int PKTS = (W / PPP) * H;

    hls::stream<ap_uint<PPP*8>> y_in;
    hls::stream<uint16_t>       uv_in;
    hls::stream<axis_t>         out;

    std::vector<uint8_t> y_vals = {0x11,0x22,0x33,0x44, 0x55,0x66,0x77,0x88};
    load_y(y_in, y_vals);
    for (int i = 0; i < PKTS; i++) uv_in.write(0);

    hls_lib::stage_write(y_in, uv_in, out, W, H);

    // Con PPP=2: paquete i tiene y0=y_vals[i*2], y1=y_vals[i*2+1]
    bool ok = true;
    for (int i = 0; i < PKTS && ok; i++) {
        axis_t pkt = out.read();
        uint8_t y0 = pkt.data.range(7,  0);
        uint8_t y1 = pkt.data.range(23, 16);
        uint8_t exp_y0 = y_vals[i * PPP];
        uint8_t exp_y1 = y_vals[i * PPP + 1];
        if (y0 != exp_y0) {
            std::cerr << "    Pkt[" << i << "] Y0=0x" << std::hex << (int)y0
                      << " esperado=0x" << (int)exp_y0 << std::dec << "\n";
            ok = false;
        }
        if (y1 != exp_y1) {
            std::cerr << "    Pkt[" << i << "] Y1=0x" << std::hex << (int)y1
                      << " esperado=0x" << (int)exp_y1 << std::dec << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 3 — U y V en los bits correctos del paquete
// ============================================================
static bool test_uv_bit_positions() {
    const int W = 4, H = 2;
    const int PKTS = (W / PPP) * H;

    hls::stream<ap_uint<PPP*8>> y_in;
    hls::stream<uint16_t>       uv_in;
    hls::stream<axis_t>         out;

    std::vector<uint8_t> y_vals(W * H, 0);
    load_y(y_in, y_vals);

    uint8_t us[] = {0xAA, 0xBB, 0xCC, 0xDD};
    uint8_t vs[] = {0x11, 0x22, 0x33, 0x44};
    for (int i = 0; i < PKTS; i++)
        uv_in.write(((uint16_t)vs[i] << 8) | us[i]);

    hls_lib::stage_write(y_in, uv_in, out, W, H);

    bool ok = true;
    for (int i = 0; i < PKTS && ok; i++) {
        axis_t pkt = out.read();
        uint8_t u = pkt.data.range(15, 8);
        uint8_t v = pkt.data.range(31, 24);
        if (u != us[i]) {
            std::cerr << "    Pkt[" << i << "] U=0x" << std::hex << (int)u
                      << " esperado=0x" << (int)us[i] << std::dec << "\n";
            ok = false;
        }
        if (v != vs[i]) {
            std::cerr << "    Pkt[" << i << "] V=0x" << std::hex << (int)v
                      << " esperado=0x" << (int)vs[i] << std::dec << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 4 — LAST=1 solo en el último paquete
// ============================================================
static bool test_last_signal() {
    const int W = 8, H = 4;
    const int PKTS = (W / PPP) * H;

    hls::stream<ap_uint<PPP*8>> y_in;
    hls::stream<uint16_t>       uv_in;
    hls::stream<axis_t>         out;

    std::vector<uint8_t> y_vals(W * H, 100);
    load_y(y_in, y_vals);
    for (int i = 0; i < PKTS; i++) uv_in.write(0x8080);

    hls_lib::stage_write(y_in, uv_in, out, W, H);

    bool ok = true;
    for (int i = 0; i < PKTS && ok; i++) {
        axis_t pkt = out.read();
        bool expected = (i == PKTS - 1);
        bool got      = (pkt.last == 1);
        if (got != expected) {
            std::cerr << "    Pkt[" << i << "] last=" << got
                      << " esperado=" << expected << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 5 — Alineamiento UV por grupo
// ============================================================
static bool test_uv_alignment() {
    const int W = 6, H = 3;
    const int PKTS = (W / PPP) * H;

    hls::stream<ap_uint<PPP*8>> y_in;
    hls::stream<uint16_t>       uv_in;
    hls::stream<axis_t>         out;

    std::vector<uint8_t> y_vals(W * H, 0);
    load_y(y_in, y_vals);

    for (int y = 0; y < H; y++)
        for (int g = 0; g < W / PPP; g++) {
            uint8_t u = (uint8_t)(y * 10 + g);
            uint8_t v = (uint8_t)(y * 10 + g + 100);
            uv_in.write(((uint16_t)v << 8) | u);
        }

    hls_lib::stage_write(y_in, uv_in, out, W, H);

    bool ok = true;
    for (int y = 0; y < H && ok; y++) {
        for (int g = 0; g < W / PPP && ok; g++) {
            axis_t pkt = out.read();
            uint8_t u_exp = (uint8_t)(y * 10 + g);
            uint8_t v_exp = (uint8_t)(y * 10 + g + 100);
            uint8_t u_got = pkt.data.range(15, 8);
            uint8_t v_got = pkt.data.range(31, 24);
            if (u_got != u_exp || v_got != v_exp) {
                std::cerr << "    Pkt[y=" << y << ",g=" << g << "]"
                          << " U=" << (int)u_got << " V=" << (int)v_got
                          << " esperado U=" << (int)u_exp
                          << " V=" << (int)v_exp << "\n";
                ok = false;
            }
        }
    }
    return ok;
}

// ============================================================
// TEST 6 — Valores extremos 0x00 y 0xFF
// ============================================================
static bool test_extreme_values() {
    const int W = 4, H = 1;
    const int PKTS = W / PPP;

    hls::stream<ap_uint<PPP*8>> y_in;
    hls::stream<uint16_t>       uv_in;
    hls::stream<axis_t>         out;

    // Paquete 0: Y0=0x00, Y1=0xFF — Paquete 1: Y0=0xFF, Y1=0x00
    std::vector<uint8_t> y_vals = {0x00, 0xFF, 0xFF, 0x00};
    load_y(y_in, y_vals);
    uv_in.write(0xFF00); // U=0x00, V=0xFF
    uv_in.write(0x00FF); // U=0xFF, V=0x00

    hls_lib::stage_write(y_in, uv_in, out, W, H);

    bool ok = true;
    axis_t p0 = out.read();
    if ((uint8_t)p0.data.range(7,  0) != 0x00 ||
        (uint8_t)p0.data.range(15, 8) != 0x00 ||
        (uint8_t)p0.data.range(23,16) != 0xFF ||
        (uint8_t)p0.data.range(31,24) != 0xFF) {
        std::cerr << "    Pkt0: 0x" << std::hex << (uint32_t)p0.data
                  << " esperado 0xFF_FF_00_00\n" << std::dec;
        ok = false;
    }
    axis_t p1 = out.read();
    if ((uint8_t)p1.data.range(7,  0) != 0xFF ||
        (uint8_t)p1.data.range(15, 8) != 0xFF ||
        (uint8_t)p1.data.range(23,16) != 0x00 ||
        (uint8_t)p1.data.range(31,24) != 0x00) {
        std::cerr << "    Pkt1: 0x" << std::hex << (uint32_t)p1.data
                  << " esperado 0x00_00_FF_FF\n" << std::dec;
        ok = false;
    }
    return ok;
}

// ============================================================
// TEST 7 — Roundtrip: hls_lib::stage_read → hls_lib::stage_write sin filtrado
// ============================================================
static bool test_roundtrip() {
    const int W = 8, H = 4;
    const uint8_t U_VAL = 0xAB, V_VAL = 0xCD;

    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(y * 10 + x);

    hls::stream<axis_t> in_stream;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x += 2) {
            axis_t p;
            p.data.range(7,  0)  = img[y][x];
            p.data.range(15, 8)  = U_VAL;
            p.data.range(23, 16) = img[y][x+1];
            p.data.range(31, 24) = V_VAL;
            p.keep = 0xF; p.strb = 0xF;
            p.last = (y == H-1 && x == W-2) ? 1 : 0;
            in_stream.write(p);
        }

    hls::stream<ap_uint<PPP*8>> y_mid;
    hls::stream<uint16_t>       uv_mid;
    hls::stream<axis_t>         out_stream;

    hls_lib::stage_read(in_stream, y_mid, uv_mid, W, H);
    hls_lib::stage_write(y_mid, uv_mid, out_stream, W, H);

    bool ok = true;
    for (int y = 0; y < H && ok; y++) {
        for (int x = 0; x < W; x += PPP) {
            axis_t pkt = out_stream.read();
            uint8_t y0 = pkt.data.range(7,  0);
            uint8_t u  = pkt.data.range(15, 8);
            uint8_t y1 = pkt.data.range(23, 16);
            uint8_t v  = pkt.data.range(31, 24);
            if (y0 != img[y][x] || y1 != img[y][x+1]) {
                std::cerr << "    Y roundtrip [" << y << "," << x << "]"
                          << " y0=" << (int)y0 << "/" << (int)img[y][x]
                          << " y1=" << (int)y1 << "/" << (int)img[y][x+1] << "\n";
                ok = false;
            }
            if (u != U_VAL || v != V_VAL) {
                std::cerr << "    UV roundtrip [" << y << "," << x << "]"
                          << " U=" << (int)u << " V=" << (int)v << "\n";
                ok = false;
            }
        }
    }
    return ok;
}

// ============================================================
// MAIN
// ============================================================
int main_write() {
    std::cout << "======================================\n";
    std::cout << "  Testbench: hls_lib::stage_write\n";
    std::cout << "  Reempaqueta Y filtrado + UV → YUYV\n";
    std::cout << "======================================\n\n";

    report_w("T01 - Conteo exacto de paquetes",         test_packet_count());
    report_w("T02 - Y0/Y1 en bits correctos",           test_y_bit_positions());
    report_w("T03 - U/V en bits correctos",             test_uv_bit_positions());
    report_w("T04 - LAST=1 solo en último paquete",     test_last_signal());
    report_w("T05 - Alineamiento UV por grupo",         test_uv_alignment());
    report_w("T06 - Valores extremos 0x00 y 0xFF",      test_extreme_values());
    report_w("T07 - Roundtrip read→write sin filtrado", test_roundtrip());

    std::cout << "\n======================================\n";
    std::cout << "  Resultado: " << tests_passed_w << "/"
              << tests_run_w << " tests pasaron\n";
    if (tests_failed_w > 0)
        std::cout << "  FALLARON: " << tests_failed_w << " tests\n";
    else
        std::cout << "  Todos los tests PASARON\n";
    std::cout << "======================================\n";
    return tests_failed_w > 0 ? 1 : 0;
}
