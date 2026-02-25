// ============================================================
// Testbench: tb_image_invert.cpp
// Función bajo prueba: image_invert
//
// Formato asumido: BGRA (32 bits/pixel)
//   bits  7:0  = B
//   bits 15:8  = G
//   bits 23:16 = R
//   bits 31:24 = A (preservado por máscara 0x00FFFFFF)
//
// Comportamiento esperado:
//   B_out = ~B_in
//   G_out = ~G_in
//   R_out = ~R_in
//   A_out =  A_in  (sin cambio)
// ============================================================

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>

#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"

typedef ap_axiu<32,0,0,0> pixel_stream;

// ============================================================
// Prototipo de la función bajo prueba
// ============================================================
void image_invert(hls::stream<pixel_stream>& stream_in,
                  hls::stream<pixel_stream>& stream_out,
                  int width, int height);

// ============================================================
// Contadores globales
// ============================================================
static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

static void report(const std::string& name, bool ok) {
    tests_run++;
    if (ok) { tests_passed++; std::cout << "  [PASS] " << name << "\n"; }
    else     { tests_failed++; std::cerr << "  [FAIL] " << name << "\n"; }
}

// ============================================================
// Utilidades
// ============================================================

struct BGRA { uint8_t b, g, r, a; };

static pixel_stream make_pixel(uint8_t b, uint8_t g, uint8_t r, uint8_t a,
                                bool last = false) {
    pixel_stream p;
    p.data.range(7,  0)  = b;
    p.data.range(15, 8)  = g;
    p.data.range(23, 16) = r;
    p.data.range(31, 24) = a;
    p.keep = 0xF;
    p.strb = 0xF;
    p.last = last ? 1 : 0;
    return p;
}

static BGRA unpack(const pixel_stream& p) {
    return {
        (uint8_t)p.data.range(7,  0),
        (uint8_t)p.data.range(15, 8),
        (uint8_t)p.data.range(23, 16),
        (uint8_t)p.data.range(31, 24)
    };
}

// Carga una imagen en el stream
static void load_image(hls::stream<pixel_stream>& s,
                        const std::vector<uint32_t>& pixels,
                        int width, int height) {
    int total = width * height;
    for (int i = 0; i < total; i++) {
        bool last = (i == total - 1);
        pixel_stream p;
        p.data  = pixels[i];
        p.keep  = 0xF;
        p.strb  = 0xF;
        p.last  = last ? 1 : 0;
        s.write(p);
    }
}

// Modelo golden en software
static uint32_t golden_invert(uint32_t pixel) {
    return pixel ^ 0x00FFFFFF;
}

// ============================================================
// TEST 1 — Inversión básica: pixel conocido
// Verifica que B, G, R se invierten y A se preserva
// ============================================================
static bool test_basic_inversion() {
    const int W = 2, H = 1;
    hls::stream<pixel_stream> in, out;

    // B=0x10, G=0x20, R=0x30, A=0xFF
    in.write(make_pixel(0x10, 0x20, 0x30, 0xFF, false));
    // B=0xAA, G=0xBB, R=0xCC, A=0x00
    in.write(make_pixel(0xAA, 0xBB, 0xCC, 0x00, true));

    image_invert(in, out, W, H);

    bool ok = true;

    auto p0 = unpack(out.read());
    if (p0.b != 0xEF || p0.g != 0xDF || p0.r != 0xCF || p0.a != 0xFF) {
        std::cerr << "    Pixel 0: got BGRA=("
                  << std::hex
                  << (int)p0.b << "," << (int)p0.g << ","
                  << (int)p0.r << "," << (int)p0.a
                  << ") expected (EF,DF,CF,FF)\n" << std::dec;
        ok = false;
    }

    auto p1 = unpack(out.read());
    if (p1.b != 0x55 || p1.g != 0x44 || p1.r != 0x33 || p1.a != 0x00) {
        std::cerr << "    Pixel 1: got BGRA=("
                  << std::hex
                  << (int)p1.b << "," << (int)p1.g << ","
                  << (int)p1.r << "," << (int)p1.a
                  << ") expected (55,44,33,00)\n" << std::dec;
        ok = false;
    }
    return ok;
}

// ============================================================
// TEST 2 — Imagen en cero: 0x00000000 → 0x00FFFFFF
// ============================================================
static bool test_black_image() {
    const int W = 4, H = 4;
    hls::stream<pixel_stream> in, out;

    std::vector<uint32_t> pixels(W * H, 0x00000000);
    load_image(in, pixels, W, H);
    image_invert(in, out, W, H);

    bool ok = true;
    for (int i = 0; i < W * H && ok; i++) {
        auto p = unpack(out.read());
        if (p.b != 0xFF || p.g != 0xFF || p.r != 0xFF || p.a != 0x00) {
            std::cerr << "    Pixel " << i << ": BGRA=("
                      << std::hex
                      << (int)p.b << "," << (int)p.g << ","
                      << (int)p.r << "," << (int)p.a
                      << ") esperado (FF,FF,FF,00)\n" << std::dec;
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 3 — Imagen blanca: 0x00FFFFFF → 0x00000000
// ============================================================
static bool test_white_image() {
    const int W = 4, H = 4;
    hls::stream<pixel_stream> in, out;

    std::vector<uint32_t> pixels(W * H, 0x00FFFFFF);
    load_image(in, pixels, W, H);
    image_invert(in, out, W, H);

    bool ok = true;
    for (int i = 0; i < W * H && ok; i++) {
        auto p = unpack(out.read());
        if (p.b != 0x00 || p.g != 0x00 || p.r != 0x00 || p.a != 0x00) {
            std::cerr << "    Pixel " << i << ": BGRA=("
                      << std::hex
                      << (int)p.b << "," << (int)p.g << ","
                      << (int)p.r << "," << (int)p.a
                      << ") esperado (00,00,00,00)\n" << std::dec;
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 4 — Doble inversión: f(f(x)) == x
// Propiedad fundamental: invertir dos veces debe dar la imagen original
// ============================================================
static bool test_double_inversion() {
    const int W = 8, H = 8;
    hls::stream<pixel_stream> in, mid, out;

    // Imagen con patrón variado
    std::vector<uint32_t> original(W * H);
    for (int i = 0; i < W * H; i++)
        original[i] = (uint32_t)(i * 0x010203 + 0x050A0F);

    load_image(in, original, W, H);
    image_invert(in, mid, W, H);
    image_invert(mid, out, W, H);

    bool ok = true;
    for (int i = 0; i < W * H && ok; i++) {
        pixel_stream p = out.read();
        uint32_t got      = (uint32_t)p.data;
        uint32_t expected = original[i];
        if (got != expected) {
            std::cerr << "    Pixel " << i
                      << ": got=0x" << std::hex << got
                      << " expected=0x" << expected << std::dec << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 5 — Canal Alpha preservado para distintos valores
// Verifica que alpha nunca cambia independientemente de su valor
// ============================================================
static bool test_alpha_preserved() {
    const int W = 4, H = 1;
    hls::stream<pixel_stream> in, out;

    uint8_t alphas[] = {0x00, 0x7F, 0x80, 0xFF};
    for (int i = 0; i < 4; i++) {
        bool last = (i == 3);
        in.write(make_pixel(0xAA, 0xBB, 0xCC, alphas[i], last));
    }

    image_invert(in, out, W, H);

    bool ok = true;
    for (int i = 0; i < 4 && ok; i++) {
        auto p = unpack(out.read());
        if (p.a != alphas[i]) {
            std::cerr << "    Pixel " << i
                      << ": alpha got=" << std::hex << (int)p.a
                      << " expected=" << (int)alphas[i] << std::dec << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 6 — Golden model: barrido completo píxel a píxel
// Compara HW vs SW para un patrón que cubre todo el rango
// ============================================================
static bool test_golden_model() {
    const int W = 16, H = 16;
    hls::stream<pixel_stream> in, out;

    std::vector<uint32_t> pixels(W * H);
    for (int i = 0; i < W * H; i++)
        pixels[i] = (uint32_t)(i * 0x01010101);

    load_image(in, pixels, W, H);
    image_invert(in, out, W, H);

    bool ok = true;
    for (int i = 0; i < W * H && ok; i++) {
        uint32_t got      = (uint32_t)out.read().data;
        uint32_t expected = golden_invert(pixels[i]);
        if (got != expected) {
            std::cerr << "    Pixel " << i
                      << ": got=0x"      << std::hex << got
                      << " expected=0x"  << expected << std::dec << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 7 — Señal LAST: solo el último pixel del frame tiene last=1
// ============================================================
static bool test_last_signal() {
    const int W = 6, H = 4;
    hls::stream<pixel_stream> in, out;

    std::vector<uint32_t> pixels(W * H, 0x00808080);
    load_image(in, pixels, W, H);
    image_invert(in, out, W, H);

    bool ok = true;
    int total = W * H;
    for (int i = 0; i < total; i++) {
        pixel_stream p = out.read();
        bool expected_last = (i == total - 1);
        bool got_last      = (p.last == 1);
        if (got_last != expected_last) {
            std::cerr << "    Pixel " << i << "/" << total
                      << ": last=" << got_last
                      << " esperado=" << expected_last << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 8 — keep y strb se propagan intactos
// ============================================================
static bool test_sideband_propagation() {
    const int W = 4, H = 1;
    hls::stream<pixel_stream> in, out;

    uint8_t keep_vals[] = {0xF, 0xA, 0x5, 0x3};
    uint8_t strb_vals[] = {0xF, 0xC, 0x6, 0x1};

    for (int i = 0; i < 4; i++) {
        pixel_stream p;
        p.data = 0x00123456;
        p.keep = keep_vals[i];
        p.strb = strb_vals[i];
        p.last = (i == 3) ? 1 : 0;
        in.write(p);
    }

    image_invert(in, out, W, H);

    bool ok = true;
    for (int i = 0; i < 4 && ok; i++) {
        pixel_stream p = out.read();
        if ((uint8_t)p.keep != keep_vals[i]) {
            std::cerr << "    Pixel " << i
                      << ": keep got=" << std::hex << (int)(uint8_t)p.keep
                      << " expected=" << (int)keep_vals[i] << std::dec << "\n";
            ok = false;
        }
        if ((uint8_t)p.strb != strb_vals[i]) {
            std::cerr << "    Pixel " << i
                      << ": strb got=" << std::hex << (int)(uint8_t)p.strb
                      << " expected=" << (int)strb_vals[i] << std::dec << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 9 — Imagen 1x1 (caso mínimo)
// Verifica que el IP maneja frames de un solo pixel
// ============================================================
static bool test_single_pixel() {
    const int W = 1, H = 1;
    hls::stream<pixel_stream> in, out;

    in.write(make_pixel(0x12, 0x34, 0x56, 0x78, true));
    image_invert(in, out, W, H);

    auto p = unpack(out.read());
    bool ok = (p.b == 0xED && p.g == 0xCB && p.r == 0xA9 && p.a == 0x78);
    if (!ok)
        std::cerr << "    got BGRA=("
                  << std::hex
                  << (int)p.b << "," << (int)p.g << ","
                  << (int)p.r << "," << (int)p.a
                  << ") expected (ED,CB,A9,78)\n" << std::dec;
    return ok;
}

// ============================================================
// TEST 10 — Resolución no potencia de 2 (ej: 10x7)
// ============================================================
static bool test_odd_resolution() {
    const int W = 10, H = 7;
    hls::stream<pixel_stream> in, out;

    std::vector<uint32_t> pixels(W * H);
    for (int i = 0; i < W * H; i++)
        pixels[i] = (uint32_t)(i * 0x030507 + 0x112233);

    load_image(in, pixels, W, H);
    image_invert(in, out, W, H);

    bool ok = true;
    for (int i = 0; i < W * H && ok; i++) {
        uint32_t got      = (uint32_t)out.read().data;
        uint32_t expected = golden_invert(pixels[i]);
        if (got != expected) {
            std::cerr << "    Pixel " << i
                      << ": got=0x"     << std::hex << got
                      << " expected=0x" << expected << std::dec << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 11 — Conteo exacto de paquetes de salida
// El IP debe producir exactamente width*height paquetes
// ============================================================
static bool test_packet_count() {
    const int W = 6, H = 5;
    hls::stream<pixel_stream> in, out;

    std::vector<uint32_t> pixels(W * H, 0x00AABBCC);
    load_image(in, pixels, W, H);
    image_invert(in, out, W, H);

    int count = 0;
    while (!out.empty()) { out.read(); count++; }

    bool ok = (count == W * H);
    if (!ok)
        std::cerr << "    Paquetes recibidos: " << count
                  << " esperados: " << W * H << "\n";
    return ok;
}

// ============================================================
// TEST 12 — Compatibilidad con pipeline YUYV2BGRA
// Simula el formato real que produciría tu IP de conversión:
// A=0xFF para todos los píxeles, valores BGR variados
// ============================================================
static bool test_bgra_pipeline_format() {
    const int W = 8, H = 4;
    hls::stream<pixel_stream> in, out;

    // Simula salida típica de YUYV2BGRA: alpha siempre 0xFF
    std::vector<uint32_t> pixels(W * H);
    for (int i = 0; i < W * H; i++) {
        uint8_t b = (uint8_t)(i * 3);
        uint8_t g = (uint8_t)(i * 5);
        uint8_t r = (uint8_t)(i * 7);
        uint8_t a = 0xFF;
        pixels[i] = ((uint32_t)a << 24) | ((uint32_t)r << 16) |
                    ((uint32_t)g << 8)  |  (uint32_t)b;
    }

    load_image(in, pixels, W, H);
    image_invert(in, out, W, H);

    bool ok = true;
    for (int i = 0; i < W * H && ok; i++) {
        uint32_t got      = (uint32_t)out.read().data;
        uint32_t expected = golden_invert(pixels[i]);
        // Alpha debe seguir siendo 0xFF
        uint8_t a_out = (uint8_t)(got >> 24);
        if (a_out != 0xFF) {
            std::cerr << "    Pixel " << i
                      << ": alpha=" << std::hex << (int)a_out
                      << " esperado FF\n" << std::dec;
            ok = false;
        }
        if (got != expected) {
            std::cerr << "    Pixel " << i
                      << ": got=0x"     << std::hex << got
                      << " expected=0x" << expected << std::dec << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// MAIN
// ============================================================
int main() {
    std::cout << "======================================\n";
    std::cout << "  Testbench: image_invert\n";
    std::cout << "  Formato: BGRA 32bpp\n";
    std::cout << "  Máscara: 0x00FFFFFF (preserva alpha)\n";
    std::cout << "======================================\n\n";

    report("T01 - Inversión básica pixel conocido",      test_basic_inversion());
    report("T02 - Imagen negra → blanca",                test_black_image());
    report("T03 - Imagen blanca → negra",                test_white_image());
    report("T04 - Doble inversión = identidad",          test_double_inversion());
    report("T05 - Canal alpha preservado",               test_alpha_preserved());
    report("T06 - Golden model barrido completo",        test_golden_model());
    report("T07 - Señal LAST en pixel correcto",         test_last_signal());
    report("T08 - keep y strb propagados intactos",      test_sideband_propagation());
    report("T09 - Frame mínimo 1x1",                     test_single_pixel());
    report("T10 - Resolución no potencia de 2 (10x7)",  test_odd_resolution());
    report("T11 - Conteo exacto de paquetes",            test_packet_count());
    report("T12 - Formato BGRA con alpha=0xFF",          test_bgra_pipeline_format());

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