// ============================================================
// Testbench: tb_hls_lib::stage_read.cpp
// Componente bajo prueba: hls_lib::stage_read
//
// Responsabilidad del stage:
//   - Leer paquetes YUYV del stream de entrada
//   - Escribir PPP píxeles Y individuales por paquete a y_out
//   - Escribir 1 par UV por paquete a uv_out
//   - No modificar los valores, solo desempaquetar
//
// Invariantes a verificar:
//   - y_out recibe exactamente width*height píxeles
//   - uv_out recibe exactamente (width/PPP)*height pares
//   - Los valores Y extraídos corresponden a los bits correctos
//   - Los valores UV extraídos corresponden a los bits correctos
//   - El orden de los píxeles es raster (fila a fila, izq a der)
// ============================================================

#include <iostream>
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
static int tests_run = 0, tests_passed = 0, tests_failed = 0;
static void report(const std::string& name, bool ok) {
    tests_run++;
    if (ok) { tests_passed++; std::cout << "  [PASS] " << name << "\n"; }
    else     { tests_failed++; std::cerr << "  [FAIL] " << name << "\n"; }
}

// ============================================================
// Utilidades
// ============================================================
static axis_t make_yuyv(uint8_t y0, uint8_t u, uint8_t y1, uint8_t v,
                         bool last = false) {
    axis_t p;
    p.data.range(7,  0)  = y0;
    p.data.range(15, 8)  = u;
    p.data.range(23, 16) = y1;
    p.data.range(31, 24) = v;
    p.keep = 0xF; p.strb = 0xF;
    p.last = last ? 1 : 0;
    return p;
}

// Carga una imagen con U y V uniformes
static void load_stream(hls::stream<axis_t>& s,
                         const std::vector<std::vector<uint8_t>>& img,
                         int W, int H,
                         uint8_t u_val = 128, uint8_t v_val = 200) {
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x += 2) {
            bool last = (y == H-1) && (x == W-2);
            s.write(make_yuyv(img[y][x], u_val, img[y][x+1], v_val, last));
        }
}

// ============================================================
// TEST 1 — Conteo exacto de muestras en y_out y uv_out
// ============================================================
static bool test_sample_count() {
    const int W = 8, H = 4;
    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W, 100));

    hls::stream<axis_t>   in;
    hls::stream<ap_uint<PPP*8>>  y_out;

    hls::stream<uint16_t> uv_out;

    load_stream(in, img, W, H);
    hls_lib::stage_read(in, y_out, uv_out, W, H);

    int y_count = 0, uv_count = 0;
    while (!y_out.empty())  { y_out.read();  y_count++;  }
    while (!uv_out.empty()) { uv_out.read(); uv_count++; }

    bool ok = true;
    if (y_count != W/PPP * H) {
        std::cerr << "    y_count=" << y_count << " esperado=" << W*H << "\n";
        ok = false;
    }
    if (uv_count != (W/PPP) * H) {
        std::cerr << "    uv_count=" << uv_count
                  << " esperado=" << (W/2)*H << "\n";
        ok = false;
    }
    return ok;
}

// ============================================================
// TEST 2 — Valores Y extraídos correctamente
// Verifica que y_out[i*2] = Y0 y y_out[i*2+1] = Y1 del paquete i
// ============================================================
static bool test_y_values() {
    const int W = 6, H = 3;
    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(y * 10 + x);

    hls::stream<axis_t>   in;
    hls::stream<ap_uint<PPP*8>>  y_out;
    hls::stream<uint16_t> uv_out;

    load_stream(in, img, W, H);
    hls_lib::stage_read(in, y_out, uv_out, W, H);

    bool ok = true;
    for (int y = 0; y < H && ok; y++) {
        for (int g = 0; g < W/PPP && ok; g++) {
            ap_uint<PPP*8> packet      = y_out.read();
            uv_out.read();

            for (int p = 0; p < PPP && ok; p++) {
                uint8_t got = (packet >> (p*8)) & 0xFF;
                int x = g*PPP + p;
                uint8_t expected = img[y][x];
                if (got != expected) {
                    std::cerr << "    Y[" << y << "," << x << "]"
                        << " got=" << (int)got
                        << " expected=" << (int)expected << "\n";
                    ok = false;
                }
            }
        }
    }
    return ok;
}

// ============================================================
// TEST 3 — Valores UV extraídos correctamente
// Verifica que uv_out contiene (V<<8)|U de cada paquete
// ============================================================
static bool test_uv_values() {
    const int W = 6, H = 3;
    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W, 128));

    // U y V distintos por grupo para verificar posición de bits
    const uint8_t U_VAL = 0xAB, V_VAL = 0xCD;

    hls::stream<axis_t>   in;
    hls::stream<ap_uint<PPP*8>>  y_out;
    hls::stream<uint16_t> uv_out;

    load_stream(in, img, W, H, U_VAL, V_VAL);
    hls_lib::stage_read(in, y_out, uv_out, W, H);

    bool ok = true;
    int total_uv = (W/2) * H;
    for (int i = 0; i < total_uv && ok; i++) {
        uint16_t got = uv_out.read();
        uint8_t u_got = got & 0xFF;
        uint8_t v_got = (got >> 8) & 0xFF;
        y_out.read();
        if (u_got != U_VAL || v_got != V_VAL) {
            std::cerr << "    UV[" << i << "]"
                      << " U=" << std::hex << (int)u_got
                      << " V=" << (int)v_got
                      << " esperado U=" << (int)U_VAL
                      << " V=" << (int)V_VAL << std::dec << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 4 — Orden raster: los píxeles salen fila a fila
// ============================================================
static bool test_raster_order() {
    const int W = 6, H = 4;
    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W));
    // Cada píxel tiene un valor único = y*W + x
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(y * W + x);

    hls::stream<axis_t>   in;
    hls::stream<ap_uint<PPP*8>> y_out;
    hls::stream<uint16_t> uv_out;

    load_stream(in, img, W, H);
    hls_lib::stage_read(in, y_out, uv_out, W, H);

    bool ok = true;
    for (int z = 0; z < W/PPP * H && ok; z++) {
        ap_uint<PPP*8> packet      = y_out.read();
        uv_out.read();
        for (int p = 0; p < PPP && ok; p++) {
            int i =z*PPP + p;

            uint8_t got = (packet >> (p*8)) & 0xFF;
            uint8_t expected = (uint8_t)i; // y*W + x = i en orden raster
                                           //
            if (got != expected) {
                std::cerr << "    Raster[" << i << "]"
                    << " got=" << (int)got
                    << " expected=" << (int)expected << "\n";
                ok = false;
            }
        }
    }
    return ok;
}

// ============================================================
// TEST 5 — UV por grupo: 1 par UV por cada 2 píxeles Y
// Verifica la proporción y el alineamiento temporal
// ============================================================
static bool test_uv_interleaving() {
    const int W = 8, H = 2;
    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W, 0));

    // UV distinto por grupo para verificar alineamiento
    hls::stream<axis_t> in;
    for (int y = 0; y < H; y++) {
        for (int g = 0; g < W/2; g++) {
            bool last = (y == H-1) && (g == W/2-1);
            // U = grupo*10, V = grupo*10 + 5
            in.write(make_yuyv(0, g*10, 0, g*10+5, last));
        }
    }

    hls::stream<ap_uint<PPP*8>>  y_out;
    hls::stream<uint16_t> uv_out;
    hls_lib::stage_read(in, y_out, uv_out, W, H);

    bool ok = true;
    // Drenar y_out
    while (!y_out.empty()) y_out.read();

    // Verificar UV por grupo
    for (int y = 0; y < H && ok; y++) {
        for (int g = 0; g < W/2 && ok; g++) {
            uint16_t uv = uv_out.read();
            uint8_t u_got = uv & 0xFF;
            uint8_t v_got = (uv >> 8) & 0xFF;
            if (u_got != g*10 || v_got != g*10+5) {
                std::cerr << "    Grupo [y=" << y << ",g=" << g << "]"
                          << " U=" << (int)u_got << " V=" << (int)v_got
                          << " esperado U=" << g*10
                          << " V=" << g*10+5 << "\n";
                ok = false;
            }
        }
    }
    return ok;
}

// ============================================================
// TEST 6 — Valores extremos: 0x00 y 0xFF
// ============================================================
static bool test_extreme_values() {
    const int W = 4, H = 2;
    hls::stream<axis_t> in;

    // Paquete 1: Y0=0, U=0, Y1=0, V=0
    in.write(make_yuyv(0x00, 0x00, 0x00, 0x00, false));
    // Paquete 2: Y0=255, U=255, Y1=255, V=255
    in.write(make_yuyv(0xFF, 0xFF, 0xFF, 0xFF, false));
    // Paquete 3: Y0=0, U=255, Y1=255, V=0
    in.write(make_yuyv(0x00, 0xFF, 0xFF, 0x00, false));
    // Paquete 4: Y0=128, U=64, Y1=192, V=32
    in.write(make_yuyv(0x80, 0x40, 0xC0, 0x20, true));

    hls::stream<ap_uint<PPP*8>>  y_out;
    hls::stream<uint16_t> uv_out;
    hls_lib::stage_read(in, y_out, uv_out, W, H);

    uint8_t y_expected[] = {0,0, 255,255, 0,255, 128,192};
    uint16_t uv_expected[] = {
        (uint16_t)(0x00<<8|0x00),
        (uint16_t)(0xFF<<8|0xFF),
        (uint16_t)(0x00<<8|0xFF),
        (uint16_t)(0x20<<8|0x40)
    };

    bool ok = true;

    for (int z = 0; z < W/PPP*H && ok; z++) {
        ap_uint<PPP*8> packet      = y_out.read();
        for (int p = 0; p < PPP && ok; p++) {
            int i =z*PPP + p;

            uint8_t got = (packet >> (p*8)) & 0xFF;
                                           //
            if (got != y_expected[i]) {
                std::cerr << "    Y[" << i << "] got=" << (int)got
                    << " expected=" << (int)y_expected[i] << "\n";
                ok = false;
            }
        }
    }
    for (int i = 0; i < (W/2)*H && ok; i++) {
        uint16_t got = uv_out.read();
        if (got != uv_expected[i]) {
            std::cerr << "    UV[" << i << "] got=0x" << std::hex << got
                      << " expected=0x" << uv_expected[i] << std::dec << "\n";
            ok = false;
        }
    }
    return ok;
}

int main_read() {
    std::cout << "======================================\n";
    std::cout << "  Testbench: hls_lib::stage_read\n";
    std::cout << "  Desempaqueta YUYV → y_stream + uv_stream\n";
    std::cout << "======================================\n\n";

    report("T01 - Conteo exacto de muestras",         test_sample_count());
    report("T02 - Valores Y extraídos correctamente", test_y_values());
    report("T03 - Valores UV extraídos correctamente",test_uv_values());
    report("T04 - Orden raster preservado",           test_raster_order());
    report("T05 - 1 UV por cada 2 píxeles Y",         test_uv_interleaving());
    report("T06 - Valores extremos 0x00 y 0xFF",      test_extreme_values());

    std::cout << "\n======================================\n";
    std::cout << "  Resultado: " << tests_passed << "/"
              << tests_run << " tests pasaron\n";
    if (tests_failed > 0)
        std::cout << "  FALLARON: " << tests_failed << " tests\n";
    else
        std::cout << "  Todos los tests PASARON\n";
    std::cout << "======================================\n";
    return tests_failed > 0 ? 1 : 0;
}
