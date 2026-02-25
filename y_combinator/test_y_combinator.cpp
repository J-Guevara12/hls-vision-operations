// ============================================================
// Testbench: tb_y_combinator.cpp
// Función bajo prueba: y_combinator
// Descripción: Combina dos streams Gx, Gy aplicando |Gx|+|Gy|
//              saturado sobre canal Y en formato YUYV (32 bits)
// ============================================================

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <string>

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

typedef ap_axiu<32,0,0,0> axis_t;

// ============================================================
// Prototipo de la función bajo prueba
// ============================================================
void y_combinator(hls::stream<axis_t>& gx_stream,
                  hls::stream<axis_t>& gy_stream,
                  hls::stream<axis_t>& out_stream,
                  int width, int height);

// ============================================================
// Utilidades
// ============================================================

// Saturación en software (referencia golden)
static uint8_t sat_add_sw(uint8_t a, uint8_t b) {
    uint16_t s = (uint16_t)a + (uint16_t)b;
    return s > 255 ? 255 : (uint8_t)s;
}

// Empaqueta un par de píxeles YUYV en un paquete AXI Stream
static axis_t make_packet(uint8_t y0, uint8_t u,
                           uint8_t y1, uint8_t v,
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

// Extrae campos de un paquete
struct Pixel32 { uint8_t y0, u, y1, v; };
static Pixel32 unpack(const axis_t& pkt) {
    return {
        (uint8_t)pkt.data.range(7,  0),
        (uint8_t)pkt.data.range(15, 8),
        (uint8_t)pkt.data.range(23, 16),
        (uint8_t)pkt.data.range(31, 24)
    };
}

// Rellena un stream con un patrón uniforme
static void fill_stream_uniform(hls::stream<axis_t>& s,
                                 int width, int height,
                                 uint8_t y0_val, uint8_t u_val,
                                 uint8_t y1_val, uint8_t v_val) {
    int packets = (width * height) / 2;
    for (int i = 0; i < packets; i++) {
        bool last = (i == packets - 1);
        s.write(make_packet(y0_val, u_val, y1_val, v_val, last));
    }
}

// ============================================================
// Contadores globales de test
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
// TEST 1 — Caso cero: ambas entradas en 0 → salida Y = 0
// ============================================================
static bool test_zeros() {
    const int W = 8, H = 4;
    hls::stream<axis_t> gx, gy, out;

    fill_stream_uniform(gx, W, H, 0, 128, 0, 128);
    fill_stream_uniform(gy, W, H, 0, 128, 0, 128);

    y_combinator(gx, gy, out, W, H);

    bool ok = true;
    int packets = (W * H) / 2;
    for (int i = 0; i < packets && ok; i++) {
        auto p = unpack(out.read());
        if (p.y0 != 0 || p.y1 != 0) {
            std::cerr << "    Paquete " << i
                      << ": y0=" << (int)p.y0
                      << " y1=" << (int)p.y1 << " (esperado 0)\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 2 — Suma simple sin saturación
// ============================================================
static bool test_simple_sum() {
    const int W = 8, H = 4;
    hls::stream<axis_t> gx, gy, out;

    // Gx: Y0=100, Y1=50   Gy: Y0=50, Y1=100
    // Esperado: Y0=150, Y1=150
    fill_stream_uniform(gx, W, H, 100, 200, 50,  100);
    fill_stream_uniform(gy, W, H,  50, 200, 100, 100);

    y_combinator(gx, gy, out, W, H);

    bool ok = true;
    int packets = (W * H) / 2;
    for (int i = 0; i < packets && ok; i++) {
        auto p = unpack(out.read());
        if (p.y0 != 150 || p.y1 != 150) {
            std::cerr << "    Paquete " << i
                      << ": y0=" << (int)p.y0
                      << " y1=" << (int)p.y1
                      << " (esperado 150, 150)\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 3 — Saturación: suma supera 255 → debe quedar en 255
// ============================================================
static bool test_saturation() {
    const int W = 8, H = 4;
    hls::stream<axis_t> gx, gy, out;

    // 200 + 200 = 400 → saturado a 255
    fill_stream_uniform(gx, W, H, 200, 128, 200, 128);
    fill_stream_uniform(gy, W, H, 200, 128, 200, 128);

    y_combinator(gx, gy, out, W, H);

    bool ok = true;
    int packets = (W * H) / 2;
    for (int i = 0; i < packets && ok; i++) {
        auto p = unpack(out.read());
        if (p.y0 != 255 || p.y1 != 255) {
            std::cerr << "    Paquete " << i
                      << ": y0=" << (int)p.y0
                      << " y1=" << (int)p.y1
                      << " (esperado 255, 255)\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 4 — Exactamente en el límite: 128+127=255, 128+128=256→255
// ============================================================
static bool test_boundary() {
    const int W = 4, H = 2;
    hls::stream<axis_t> gx, gy, out;

    // Par 0: gx.y0=128 + gy.y0=127 = 255 (sin saturar)
    // Par 0: gx.y1=128 + gy.y1=128 = 256 → 255 (saturado)
    gx.write(make_packet(128, 0, 128, 0, false));
    gx.write(make_packet(128, 0, 128, 0, false));
    gx.write(make_packet(128, 0, 128, 0, false));
    gx.write(make_packet(128, 0, 128, 0, true));

    gy.write(make_packet(127, 0, 128, 0, false));
    gy.write(make_packet(127, 0, 128, 0, false));
    gy.write(make_packet(127, 0, 128, 0, false));
    gy.write(make_packet(127, 0, 128, 0, true));

    y_combinator(gx, gy, out, W, H);

    bool ok = true;
    int packets = (W * H) / 2;
    for (int i = 0; i < packets && ok; i++) {
        auto p = unpack(out.read());
        if (p.y0 != 255 || p.y1 != 255) {
            std::cerr << "    Paquete " << i
                      << ": y0=" << (int)p.y0
                      << " y1=" << (int)p.y1
                      << " (esperado 255, 255)\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 5 — Croma se propaga desde Gx, no desde Gy
// ============================================================
static bool test_chroma_propagation() {
    const int W = 4, H = 2;
    hls::stream<axis_t> gx, gy, out;

    // Gx tiene U=0xAA, V=0xBB
    // Gy tiene U=0x11, V=0x22
    // Salida debe usar U/V de Gx
    int packets = (W * H) / 2;
    for (int i = 0; i < packets; i++) {
        bool last = (i == packets - 1);
        gx.write(make_packet(50, 0xAA, 50, 0xBB, last));
        gy.write(make_packet(50, 0x11, 50, 0x22, last));
    }

    y_combinator(gx, gy, out, W, H);

    bool ok = true;
    for (int i = 0; i < packets && ok; i++) {
        auto p = unpack(out.read());
        if (p.u != 0xAA || p.v != 0xBB) {
            std::cerr << "    Paquete " << i
                      << ": U=0x" << std::hex << (int)p.u
                      << " V=0x" << (int)p.v << std::dec
                      << " (esperado U=0xAA V=0xBB)\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 6 — Señal LAST se propaga correctamente
// ============================================================
static bool test_last_signal() {
    const int W = 4, H = 2;
    hls::stream<axis_t> gx, gy, out;

    int packets = (W * H) / 2;
    for (int i = 0; i < packets; i++) {
        bool last = (i == packets - 1);
        gx.write(make_packet(10, 128, 10, 128, last));
        gy.write(make_packet(10, 128, 10, 128, last));
    }

    y_combinator(gx, gy, out, W, H);

    bool ok = true;
    for (int i = 0; i < packets; i++) {
        axis_t p = out.read();
        bool expected_last = (i == packets - 1);
        bool got_last = (p.last == 1);
        if (got_last != expected_last) {
            std::cerr << "    Paquete " << i
                      << ": last=" << got_last
                      << " (esperado " << expected_last << ")\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 7 — Golden model: barrido completo con valores aleatorios
//           Compara HW vs SW píxel a píxel
// ============================================================
static bool test_golden_model() {
    const int W = 16, H = 8;
    hls::stream<axis_t> gx, gy, out;

    // Valores de prueba con patrones que cubran rangos variados
    const uint8_t gx_y0[] = {  0, 10, 50, 100, 128, 200, 254, 255,
                                 1, 30, 75, 110, 130, 180, 240, 250,
                                 5, 20, 60, 120, 140, 160, 220, 245,
                                15, 25, 80, 115, 145, 170, 230, 248 };
    const uint8_t gy_y0[] = {  0,  5, 50, 100, 128, 100, 100, 100,
                               10, 40, 80, 120, 100, 100, 100, 100,
                               20, 50, 90, 130, 100, 100, 100, 100,
                               30, 60, 95, 140, 100, 100, 100, 100 };

    int packets = (W * H) / 2;

    // Cargar streams
    for (int i = 0; i < packets; i++) {
        bool last = (i == packets - 1);
        uint8_t gx0 = gx_y0[i % 32];
        uint8_t gy0 = gy_y0[i % 32];
        uint8_t gx1 = gx_y0[(i + 1) % 32];
        uint8_t gy1 = gy_y0[(i + 1) % 32];
        gx.write(make_packet(gx0, 128, gx1, 128, last));
        gy.write(make_packet(gy0, 128, gy1, 128, last));
    }

    y_combinator(gx, gy, out, W, H);

    bool ok = true;
    for (int i = 0; i < packets && ok; i++) {
        uint8_t gx0 = gx_y0[i % 32];
        uint8_t gy0 = gy_y0[i % 32];
        uint8_t gx1 = gx_y0[(i + 1) % 32];
        uint8_t gy1 = gy_y0[(i + 1) % 32];

        uint8_t exp_y0 = sat_add_sw(gx0, gy0);
        uint8_t exp_y1 = sat_add_sw(gx1, gy1);

        auto p = unpack(out.read());
        if (p.y0 != exp_y0 || p.y1 != exp_y1) {
            std::cerr << "    Paquete " << i
                      << ": got=(" << (int)p.y0 << "," << (int)p.y1 << ")"
                      << " expected=(" << (int)exp_y0 << "," << (int)exp_y1 << ")"
                      << " [gx=(" << (int)gx0 << "," << (int)gx1 << ")"
                      << " gy=(" << (int)gy0 << "," << (int)gy1 << ")]\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 8 — Resolución no potencia de 2 (ej: 10x6)
// ============================================================
static bool test_odd_resolution() {
    const int W = 10, H = 6;
    hls::stream<axis_t> gx, gy, out;

    int packets = (W * H) / 2;
    for (int i = 0; i < packets; i++) {
        bool last = (i == packets - 1);
        uint8_t val = (uint8_t)(i % 128);
        gx.write(make_packet(val, 128, val, 128, last));
        gy.write(make_packet(val, 128, val, 128, last));
    }

    y_combinator(gx, gy, out, W, H);

    bool ok = true;
    for (int i = 0; i < packets && ok; i++) {
        uint8_t val = (uint8_t)(i % 128);
        uint8_t expected = sat_add_sw(val, val);
        auto p = unpack(out.read());
        if (p.y0 != expected || p.y1 != expected) {
            std::cerr << "    Paquete " << i
                      << ": y0=" << (int)p.y0
                      << " y1=" << (int)p.y1
                      << " (esperado " << (int)expected << ")\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 9 — Stream vacío no produce salida (width=0 o height=0)
// ============================================================
static bool test_empty_stream() {
    const int W = 0, H = 4;
    hls::stream<axis_t> gx, gy, out;

    y_combinator(gx, gy, out, W, H);

    bool ok = out.empty();
    if (!ok)
        std::cerr << "    Stream de salida no debería tener datos\n";
    return ok;
}

// ============================================================
// TEST 10 — Simetría: Gx y Gy intercambiados → mismo resultado
// ============================================================
static bool test_symmetry() {
    const int W = 8, H = 4;
    hls::stream<axis_t> gx1, gy1, out1;
    hls::stream<axis_t> gx2, gy2, out2;

    int packets = (W * H) / 2;
    for (int i = 0; i < packets; i++) {
        bool last = (i == packets - 1);
        gx1.write(make_packet(100, 200, 50, 100, last));
        gy1.write(make_packet( 50, 200, 100, 100, last));
        // intercambiado
        gx2.write(make_packet( 50, 200, 100, 100, last));
        gy2.write(make_packet(100, 200,  50, 100, last));
    }

    y_combinator(gx1, gy1, out1, W, H);
    y_combinator(gx2, gy2, out2, W, H);

    bool ok = true;
    for (int i = 0; i < packets && ok; i++) {
        auto p1 = unpack(out1.read());
        auto p2 = unpack(out2.read());
        if (p1.y0 != p2.y0 || p1.y1 != p2.y1) {
            std::cerr << "    Paquete " << i
                      << ": out1=(" << (int)p1.y0 << "," << (int)p1.y1 << ")"
                      << " out2=(" << (int)p2.y0 << "," << (int)p2.y1 << ")\n";
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
    std::cout << "  Testbench: y_combinator\n";
    std::cout << "======================================\n\n";

    report("T01 - Ambas entradas en cero",           test_zeros());
    report("T02 - Suma simple sin saturación",        test_simple_sum());
    report("T03 - Saturación a 255",                  test_saturation());
    report("T04 - Casos límite 255/256",              test_boundary());
    report("T05 - Croma propagado desde Gx",          test_chroma_propagation());
    report("T06 - Señal LAST propagada",              test_last_signal());
    report("T07 - Golden model barrido completo",     test_golden_model());
    report("T08 - Resolución no potencia de 2",       test_odd_resolution());
    report("T09 - Stream vacío (W=0)",                test_empty_stream());
    report("T10 - Simetría Gx/Gy intercambiados",    test_symmetry());

    std::cout << "\n======================================\n";
    std::cout << "  Resultado: " << tests_passed << "/" << tests_run << " tests pasaron\n";
    if (tests_failed > 0)
        std::cout << "  FALLARON: " << tests_failed << " tests\n";
    else
        std::cout << "  Todos los tests PASARON\n";
    std::cout << "======================================\n";

    return tests_failed > 0 ? 1 : 0;
}