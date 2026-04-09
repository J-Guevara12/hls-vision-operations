// ============================================================
// Testbench: test_resize_half_2x.cpp
// Componente bajo prueba: resize_half_2x (top level)
//
// Valida el pipeline completo:
//   stage_read → resize_half_y + resize_half_uv → stage_write
//
// Complementa test_resizer_half.cpp que prueba los stages
// internos de forma aislada.
// ============================================================

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdint>

#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "resizer.hpp"

// ============================================================
// Contadores
// ============================================================
static int tests_run_tl = 0, tests_passed_tl = 0, tests_failed_tl = 0;
static void report_tl(const std::string& name, bool ok) {
    tests_run_tl++;
    if (ok) { tests_passed_tl++; std::cout << "  [PASS] " << name << "\n"; }
    else     { tests_failed_tl++; std::cerr << "  [FAIL] " << name << "\n"; }
}

// ============================================================
// Tipos
// ============================================================
typedef std::vector<std::vector<uint8_t>> Image;

// ============================================================
// Utilidades
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

static void load_stream(hls::stream<axis_t>& s, const Image& img,
                         int width, int height,
                         uint8_t u_val = 128, uint8_t v_val = 128) {
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x += 2) {
            bool last = (y == height-1) && (x == width-2);
            s.write(make_yuyv(img[y][x], u_val, img[y][x+1], v_val, last));
        }
}

// Golden: toma píxeles pares de filas pares
static Image golden_y(const Image& img, int width, int height) {
    int out_h = height / 2, out_w = width / 2;
    Image out(out_h, std::vector<uint8_t>(out_w, 0));
    for (int y = 0; y < out_h; y++)
        for (int x = 0; x < out_w; x++)
            out[y][x] = img[y*2][x*2];
    return out;
}

// ============================================================
// TEST 1 — Conteo exacto de paquetes de salida
// ============================================================
static bool test_tl_packet_count() {
    const int W = 8, H = 4;
    const int EXP = (W/2/2) * (H/2);  // (out_w/2) * out_h paquetes YUYV

    Image img(H, std::vector<uint8_t>(W, 100));
    hls::stream<axis_t> in, out;
    load_stream(in, img, W, H);

    resize_half_2x(in, out, W, H);

    int count = 0;
    while (!out.empty()) { out.read(); count++; }
    bool ok = (count == EXP);
    if (!ok)
        std::cerr << "    count=" << count << " esperado=" << EXP << "\n";
    return ok;
}

// ============================================================
// TEST 2 — Valores Y correctos end-to-end
// ============================================================
static bool test_tl_y_values() {
    const int W = 8, H = 4;
    const int OUT_W = W/2, OUT_H = H/2;

    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(y*10 + x);

    hls::stream<axis_t> in, out;
    load_stream(in, img, W, H);
    resize_half_2x(in, out, W, H);

    auto ref = golden_y(img, W, H);
    bool ok = true;
    int errors = 0;

    for (int y = 0; y < OUT_H; y++) {
        for (int x = 0; x < OUT_W; x += 2) {
            axis_t pkt = out.read();
            uint8_t hw_y0 = (uint8_t)pkt.data.range(7,  0);
            uint8_t hw_y1 = (uint8_t)pkt.data.range(23, 16);
            if (hw_y0 != ref[y][x]) {
                if (errors < 10)
                    std::cerr << "    Y[" << y << "," << x << "]"
                              << " hw=" << (int)hw_y0
                              << " expected=" << (int)ref[y][x] << "\n";
                ok = false; errors++;
            }
            if (hw_y1 != ref[y][x+1]) {
                if (errors < 10)
                    std::cerr << "    Y[" << y << "," << x+1 << "]"
                              << " hw=" << (int)hw_y1
                              << " expected=" << (int)ref[y][x+1] << "\n";
                ok = false; errors++;
            }
        }
    }
    if (!ok) std::cerr << "    Total errores: " << errors << "\n";
    return ok;
}

// ============================================================
// TEST 3 — UV propagado correctamente end-to-end
// ============================================================
static bool test_tl_uv_propagation() {
    const int W = 8, H = 4;
    const int OUT_W = W/2, OUT_H = H/2;
    const uint8_t U_VAL = 0xAB, V_VAL = 0xCD;

    Image img(H, std::vector<uint8_t>(W, 128));
    hls::stream<axis_t> in, out;
    load_stream(in, img, W, H, U_VAL, V_VAL);
    resize_half_2x(in, out, W, H);

    bool ok = true;
    int errors = 0;
    int total = (OUT_W/2) * OUT_H;

    for (int i = 0; i < total; i++) {
        axis_t pkt = out.read();
        uint8_t u = (uint8_t)pkt.data.range(15, 8);
        uint8_t v = (uint8_t)pkt.data.range(31, 24);
        if (u != U_VAL || v != V_VAL) {
            if (errors < 10)
                std::cerr << "    Pkt[" << i << "] U=0x" << std::hex
                          << (int)u << " V=0x" << (int)v
                          << " esperado U=0x" << (int)U_VAL
                          << " V=0x" << (int)V_VAL << std::dec << "\n";
            ok = false; errors++;
        }
    }
    if (!ok) std::cerr << "    Total errores UV: " << errors << "\n";
    return ok;
}

// ============================================================
// TEST 4 — LAST=1 solo en el último paquete
// ============================================================
static bool test_tl_last_signal() {
    const int W = 8, H = 4;
    const int OUT_W = W/2, OUT_H = H/2;
    const int TOTAL = (OUT_W/2) * OUT_H;

    Image img(H, std::vector<uint8_t>(W, 100));
    hls::stream<axis_t> in, out;
    load_stream(in, img, W, H);
    resize_half_2x(in, out, W, H);

    bool ok = true;
    axis_t last_pkt;
    for (int i = 0; i < TOTAL; i++) {
        axis_t pkt = out.read();
        bool expected = (i == TOTAL-1);
        bool got      = (pkt.last == 1);
        if (got != expected) {
            std::cerr << "    Pkt[" << i << "] last=" << got
                      << " esperado=" << expected << "\n";
            ok = false;
        }
        last_pkt = pkt;
    }
    if (last_pkt.last != 1) {
        std::cerr << "    LAST=0 en el último paquete\n";
        ok = false;
    }
    return ok;
}

// ============================================================
// TEST 5 — Resolución mínima (4x4 → 2x2)
// ============================================================
static bool test_tl_minimum_resolution() {
    const int W = 4, H = 4;
    const int OUT_W = W/2, OUT_H = H/2;

    Image img = {
        { 10,  20,  30,  40},
        { 50,  60,  70,  80},
        { 90, 100, 110, 120},
        {130, 140, 150, 160}
    };

    hls::stream<axis_t> in, out;
    load_stream(in, img, W, H);
    resize_half_2x(in, out, W, H);

    auto ref = golden_y(img, W, H);
    bool ok = true;

    // 4x4 → 2x2: 1 paquete YUYV de salida
    
    axis_t pkt;
     for (int y = 0; y < OUT_H; y++) {
        pkt = out.read();
        uint8_t hw_y0 = (uint8_t)pkt.data.range(7,  0);
        uint8_t hw_y1 = (uint8_t)pkt.data.range(23, 16);

        if (hw_y0 != ref[y][0]) {
            std::cerr << "    Y[0,0] hw=" << (int)hw_y0
                    << " expected=" << (int)ref[0][0] << "\n";
            ok = false;
        }
        if (hw_y1 != ref[y][1]) {
            std::cerr << "    Y[0,1] hw=" << (int)hw_y1
                    << " expected=" << (int)ref[0][1] << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 6 — 4K (3840x2160 → 1920x1080): sin deadlock
// ============================================================
static bool test_tl_4k() {
    const int W = 3840, H = 2160;
    const int OUT_W = W/2, OUT_H = H/2;
    const int EXP_PKTS = (OUT_W/2) * OUT_H;

    std::cout << "    [4K] Generando " << W << "x" << H << "...\n";

    // Guardar filas de muestra para verificación
    const int CHECK_IN_ROWS[] = {0, H/2, H-2};
    const int N_CHECK = 3;
    std::vector<std::vector<uint8_t>> ref_rows(N_CHECK,
        std::vector<uint8_t>(W, 0));

    hls::stream<axis_t> in, out;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x += 2) {
            uint8_t y0 = (uint8_t)((y*37 + x*53 + y*x % 199) % 256);
            uint8_t y1 = (uint8_t)((y*37 + (x+1)*53 + y*(x+1) % 199) % 256);
            bool last  = (y == H-1) && (x == W-2);
            in.write(make_yuyv(y0, 128, y1, 128, last));
            for (int ci = 0; ci < N_CHECK; ci++)
                if (y == CHECK_IN_ROWS[ci]) {
                    ref_rows[ci][x]   = y0;
                    ref_rows[ci][x+1] = y1;
                }
        }
    }

    std::cout << "    [4K] Ejecutando resize_half_2x...\n";
    resize_half_2x(in, out, W, H);
    std::cout << "    [4K] Completado, verificando...\n";

    if ((int)out.size() != EXP_PKTS) {
        std::cerr << "    [4K] Paquetes=" << out.size()
                  << " esperado=" << EXP_PKTS << "\n";
        while (!out.empty()) out.read();
        return false;
    }

    bool ok = true;
    axis_t last_pkt;

    for (int y = 0; y < OUT_H; y++) {
        int src_row = y * 2;
        int ci = -1;
        for (int i = 0; i < N_CHECK; i++)
            if (CHECK_IN_ROWS[i] == src_row) { ci = i; break; }

        for (int x = 0; x < OUT_W; x += 2) {
            last_pkt = out.read();
            if (ci < 0) continue;

            uint8_t hw_y0 = (uint8_t)last_pkt.data.range(7,  0);
            uint8_t hw_y1 = (uint8_t)last_pkt.data.range(23, 16);
            uint8_t ex_y0 = ref_rows[ci][x*2];
            uint8_t ex_y1 = ref_rows[ci][(x+1)*2];

            if (hw_y0 != ex_y0) {
                std::cerr << "    [4K] Y[" << y << "," << x << "]"
                          << " hw=" << (int)hw_y0
                          << " expected=" << (int)ex_y0 << "\n";
                ok = false;
            }
            if (hw_y1 != ex_y1) {
                std::cerr << "    [4K] Y[" << y << "," << x+1 << "]"
                          << " hw=" << (int)hw_y1
                          << " expected=" << (int)ex_y1 << "\n";
                ok = false;
            }
        }
    }

    if (last_pkt.last != 1) {
        std::cerr << "    [4K] LAST=0 en el último paquete\n";
        ok = false;
    }

    if (!out.empty()) {
        std::cerr << "    [4K] Stream no vacío tras leer " << EXP_PKTS
                  << " paquetes\n";
        ok = false;
    }

    if (ok)
        std::cout << "    [4K] " << EXP_PKTS << " paquetes verificados\n";
    return ok;
}

// ============================================================
// TEST 7 — Altura impar (floor(H/2) filas de salida)
// Con H=5 se esperan 2 filas de salida (filas 0 y 2 de entrada)
// La fila 4 de entrada se consume pero no produce salida
// ============================================================
static bool test_tl_odd_height() {
    const int W = 240, H = 135;
    const int OUT_W = W/2, OUT_H = H/2;  // 5/2 = 2
    const int EXP_PKTS = (OUT_W/2) * OUT_H;
 
    Image img(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(y*10 + x);
 
    hls::stream<axis_t> in, out;
    load_stream(in, img, W, H);
    resize_half_2x(in, out, W, H);
 
    // Verificar conteo
    int count = (int)out.size();
    if (count != EXP_PKTS) {
        std::cerr << "    Paquetes=" << count << " esperado=" << EXP_PKTS << "\n";
        while (!out.empty()) out.read();
        return false;
    }
 
    // Verificar que las 2 filas de salida corresponden a filas 0 y 2 de entrada
    bool ok = true;
    axis_t last_pkt;
    for (int y = 0; y < OUT_H; y++) {
        int src_row = y * 2;
        for (int x = 0; x < OUT_W; x += 2) {
            last_pkt = out.read();
            uint8_t hw_y0 = (uint8_t)last_pkt.data.range(7,  0);
            uint8_t hw_y1 = (uint8_t)last_pkt.data.range(23, 16);
            uint8_t ex_y0 = img[src_row][x*2];
            uint8_t ex_y1 = img[src_row][(x+1)*2];
            if (hw_y0 != ex_y0) {
                std::cerr << "    Y[out_y=" << y << ",x=" << x << "]"
                          << " hw=" << (int)hw_y0
                          << " expected=" << (int)ex_y0
                          << " (src_row=" << src_row << ")\n";
                ok = false;
            }
            if (hw_y1 != ex_y1) {
                std::cerr << "    Y[out_y=" << y << ",x=" << x+1 << "]"
                          << " hw=" << (int)hw_y1
                          << " expected=" << (int)ex_y1
                          << " (src_row=" << src_row << ")\n";
                ok = false;
            }
        }
    }
 
    // Verificar LAST
    if (last_pkt.last != 1) {
        std::cerr << "    LAST=0 en el último paquete\n";
        ok = false;
    }
 
    // Verificar que la fila impar sobrante (fila 4) no apareció en la salida
    if (!out.empty()) {
        std::cerr << "    Stream no vacío — fila sobrante contaminó la salida\n";
        ok = false;
    }
 
    return ok;
}

// ============================================================
// MAIN
// ============================================================
int main_resize_half_stages();

int main_resize_half() {
    if (main_resize_half_stages() != 0) return 1;

    std::cout << "==========================================\n";
    std::cout << "  Testbench: resize_half_2x (top level)\n";
    std::cout << "  Pipeline: stage_read → resize → stage_write\n";
    std::cout << "  PPP=" << PPP << "\n";
    std::cout << "==========================================\n\n";

    report_tl("T01 - Conteo exacto de paquetes",     test_tl_packet_count());
    report_tl("T02 - Valores Y correctos",           test_tl_y_values());
    report_tl("T03 - UV propagado correctamente",    test_tl_uv_propagation());
    report_tl("T04 - LAST=1 en último paquete",      test_tl_last_signal());
    report_tl("T05 - Resolución mínima 4x4→2x2",    test_tl_minimum_resolution());
    report_tl("T06 - 4K sin deadlock",               test_tl_4k());
    report_tl("T07 - Altura impar H=5 → 2 filas",   test_tl_odd_height());

    std::cout << "\n==========================================\n";
    std::cout << "  Resultado: " << tests_passed_tl << "/"
              << tests_run_tl << " tests pasaron\n";
    if (tests_failed_tl > 0)
        std::cout << "  FALLARON: " << tests_failed_tl << " tests\n";
    else
        std::cout << "  Todos los tests PASARON\n";
    std::cout << "==========================================\n";

    return tests_failed_tl > 0 ? 1 : 0;
}
