// ============================================================
// Testbench: tb_histogram_yuyv.cpp
// Función bajo prueba: histogram_yuyv
//
// Protocolo de salida esperado por frame:
//   [0..height-1]       : acumulado de Y por fila (y_col)
//   [height..height+width-1] : acumulado de Y por columna (col_register)
//   [height+width..height+width+255] : histograma Y (y_counter)
//
// Nota futura: cuando se agreguen canales U/V se extenderá
// el protocolo de salida con 512 paquetes adicionales (u_counter + v_counter)
// ============================================================

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

typedef ap_axiu<32,0,0,0> axis_t;

// ============================================================
// Prototipo de la función bajo prueba
// ============================================================
void histogram_yuyv(hls::stream<axis_t>& in_stream,
                    hls::stream<axis_t>& out_stream,
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
// Estructuras de resultado
// ============================================================
struct HistogramResult {
    std::vector<uint32_t> row_sums;    // height valores
    std::vector<uint32_t> col_sums;    // width valores
    uint32_t              y_hist[256]; // histograma Y
};

// ============================================================
// Utilidades
// ============================================================

// Empaqueta dos píxeles YUYV en un paquete AXI Stream
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

// Lee la salida completa del IP y la estructura en HistogramResult
static HistogramResult read_output(hls::stream<axis_t>& out,
                                    int width, int height) {
    HistogramResult r;
    r.row_sums.resize(height);
    r.col_sums.resize(width);
    memset(r.y_hist, 0, sizeof(r.y_hist));

    // 1. Acumulados por fila
    for (int y = 0; y < height; y++)
        r.row_sums[y] = out.read().data;

    // 2. Acumulados por columna
    for (int x = 0; x < width; x++)
        r.col_sums[x] = out.read().data;

    // 3. Histograma Y
    for (int b = 0; b < 256; b++)
        r.y_hist[b] = out.read().data;

    return r;
}

// Modelo golden en software
struct GoldenResult {
    std::vector<uint32_t> row_sums;
    std::vector<uint32_t> col_sums;
    uint32_t              y_hist[256];
};

// image[y][x] contiene el valor Y de cada píxel
static GoldenResult compute_golden(const std::vector<std::vector<uint8_t>>& image,
                                    int width, int height) {
    GoldenResult g;
    g.row_sums.assign(height, 0);
    g.col_sums.assign(width,  0);
    memset(g.y_hist, 0, sizeof(g.y_hist));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint8_t val = image[y][x];
            g.row_sums[y] += val;
            g.col_sums[x] += val;
            g.y_hist[val]++;
        }
    }
    return g;
}

// Carga una imagen en el stream YUYV (U=128, V=128 por defecto)
static void load_image(hls::stream<axis_t>& s,
                        const std::vector<std::vector<uint8_t>>& image,
                        int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 2) {
            bool last = (y == height-1) && (x == width-2);
            s.write(make_yuyv(image[y][x], 128, image[y][x+1], 128, last));
        }
    }
}

// Compara HistogramResult con GoldenResult y reporta diferencias
static bool compare_results(const HistogramResult& hw,
                             const GoldenResult& sw,
                             int width, int height,
                             bool verbose = false) {
    bool ok = true;

    // Comparar filas
    for (int y = 0; y < height; y++) {
        if (hw.row_sums[y] != sw.row_sums[y]) {
            if (verbose)
                std::cerr << "    row_sum[" << y << "]: hw="
                          << hw.row_sums[y] << " sw=" << sw.row_sums[y] << "\n";
            ok = false;
        }
    }

    // Comparar columnas
    for (int x = 0; x < width; x++) {
        if (hw.col_sums[x] != sw.col_sums[x]) {
            if (verbose)
                std::cerr << "    col_sum[" << x << "]: hw="
                          << hw.col_sums[x] << " sw=" << sw.col_sums[x] << "\n";
            ok = false;
        }
    }

    // Comparar histograma
    for (int b = 0; b < 256; b++) {
        if (hw.y_hist[b] != sw.y_hist[b]) {
            if (verbose)
                std::cerr << "    y_hist[" << b << "]: hw="
                          << hw.y_hist[b] << " sw=" << sw.y_hist[b] << "\n";
            ok = false;
        }
    }

    return ok;
}

// ============================================================
// TEST 1 — Imagen uniforme (todos los píxeles el mismo valor)
// Verifica la lógica básica de acumulación
// ============================================================
static bool test_uniform() {
    const int W = 8, H = 4;
    const uint8_t VAL = 100;

    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W, VAL));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    histogram_yuyv(in, out, W, H);

    auto hw = read_output(out, W, H);
    auto sw = compute_golden(img, W, H);

    bool ok = compare_results(hw, sw, W, H, true);

    // Verificaciones adicionales para imagen uniforme
    // row_sum debe ser W * VAL para cada fila
    if (hw.row_sums[0] != (uint32_t)(W * VAL)) {
        std::cerr << "    row_sum[0]=" << hw.row_sums[0]
                  << " esperado=" << (W * VAL) << "\n";
        ok = false;
    }
    // hist[VAL] debe ser W*H, el resto 0
    if (hw.y_hist[VAL] != (uint32_t)(W * H)) {
        std::cerr << "    y_hist[" << (int)VAL << "]=" << hw.y_hist[VAL]
                  << " esperado=" << (W * H) << "\n";
        ok = false;
    }
    for (int b = 0; b < 256; b++) {
        if (b != VAL && hw.y_hist[b] != 0) {
            std::cerr << "    y_hist[" << b << "]=" << hw.y_hist[b]
                      << " esperado=0\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 2 — Imagen en cero
// Verifica que todos los contadores queden en cero
// ============================================================
static bool test_all_zeros() {
    const int W = 8, H = 4;
    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W, 0));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    histogram_yuyv(in, out, W, H);

    auto hw = read_output(out, W, H);

    bool ok = true;
    for (int y = 0; y < H; y++)
        if (hw.row_sums[y] != 0) { ok = false; break; }
    for (int x = 0; x < W; x++)
        if (hw.col_sums[x] != 0) { ok = false; break; }
    for (int b = 1; b < 256; b++)
        if (hw.y_hist[b] != 0) { ok = false; break; }

    if (!ok)
        std::cerr << "    Imagen en cero debería dar todos los contadores en 0\n";
    return ok;
}

// ============================================================
// TEST 3 — Imagen con valor máximo (255)
// Verifica saturación del histograma en el bin 255
// ============================================================
static bool test_all_max() {
    const int W = 8, H = 4;
    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W, 255));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    histogram_yuyv(in, out, W, H);

    auto hw = read_output(out, W, H);
    auto sw = compute_golden(img, W, H);

    return compare_results(hw, sw, W, H, true);
}

// ============================================================
// TEST 4 — Gradiente horizontal
// Cada columna x tiene valor x (0..W-1)
// Verifica que col_sums refleja el gradiente correctamente
// ============================================================
static bool test_horizontal_gradient() {
    const int W = 16, H = 8;
    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W));

    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)x;

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    histogram_yuyv(in, out, W, H);

    auto hw = read_output(out, W, H);
    auto sw = compute_golden(img, W, H);

    bool ok = compare_results(hw, sw, W, H, true);

    // col_sum[x] debe ser x * H para todo x
    for (int x = 0; x < W && ok; x++) {
        uint32_t expected = (uint32_t)(x * H);
        if (hw.col_sums[x] != expected) {
            std::cerr << "    col_sum[" << x << "]=" << hw.col_sums[x]
                      << " esperado=" << expected << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 5 — Gradiente vertical
// Cada fila y tiene valor y (0..H-1)
// Verifica que row_sums refleja el gradiente correctamente
// ============================================================
static bool test_vertical_gradient() {
    const int W = 16, H = 8;
    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W));

    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)y;

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    histogram_yuyv(in, out, W, H);

    auto hw = read_output(out, W, H);
    auto sw = compute_golden(img, W, H);

    bool ok = compare_results(hw, sw, W, H, true);

    // row_sum[y] debe ser y * W para todo y
    for (int y = 0; y < H && ok; y++) {
        uint32_t expected = (uint32_t)(y * W);
        if (hw.row_sums[y] != expected) {
            std::cerr << "    row_sum[" << y << "]=" << hw.row_sums[y]
                      << " esperado=" << expected << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 6 — Un solo píxel activo
// Solo (y=1, x=2) tiene valor 200, el resto 0
// Verifica que los acumuladores son selectivos
// ============================================================
static bool test_single_pixel() {
    const int W = 8, H = 4;
    const int PX = 2, PY = 1;
    const uint8_t VAL = 200;

    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W, 0));
    img[PY][PX] = VAL;

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    histogram_yuyv(in, out, W, H);

    auto hw = read_output(out, W, H);
    auto sw = compute_golden(img, W, H);

    bool ok = compare_results(hw, sw, W, H, true);

    // Verificaciones explícitas
    if (hw.row_sums[PY] != VAL) {
        std::cerr << "    row_sum[" << PY << "]=" << hw.row_sums[PY]
                  << " esperado=" << (int)VAL << "\n";
        ok = false;
    }
    if (hw.col_sums[PX] != VAL) {
        std::cerr << "    col_sum[" << PX << "]=" << hw.col_sums[PX]
                  << " esperado=" << (int)VAL << "\n";
        ok = false;
    }
    if (hw.y_hist[VAL] != 1) {
        std::cerr << "    y_hist[" << (int)VAL << "]=" << hw.y_hist[VAL]
                  << " esperado=1\n";
        ok = false;
    }
    if (hw.y_hist[0] != (uint32_t)(W * H - 1)) {
        std::cerr << "    y_hist[0]=" << hw.y_hist[0]
                  << " esperado=" << (W * H - 1) << "\n";
        ok = false;
    }
    return ok;
}

// ============================================================
// TEST 7 — Dependencia RAW: y0 == y1 en el mismo paquete
// Si el DEPENDENCE pragma oculta este caso, y_counter dará
// resultados incorrectos. Este test lo expone.
// ============================================================
static bool test_raw_dependency() {
    const int W = 8, H = 2;
    const uint8_t VAL = 42; // y0 == y1 en todos los paquetes

    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W, VAL));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    histogram_yuyv(in, out, W, H);

    auto hw = read_output(out, W, H);

    uint32_t expected_count = (uint32_t)(W * H);
    bool ok = true;

    if (hw.y_hist[VAL] != expected_count) {
        std::cerr << "    y_hist[" << (int)VAL << "]=" << hw.y_hist[VAL]
                  << " esperado=" << expected_count
                  << " (posible fallo RAW en y_counter++)\n";
        ok = false;
    }
    return ok;
}

// ============================================================
// TEST 8 — Histograma completo: un píxel de cada valor 0..255
// Verifica que todos los bins tienen exactamente count=1
// Requiere imagen de 16x16 = 256 píxeles con valores 0..255
// ============================================================
static bool test_full_histogram() {
    const int W = 16, H = 16;
    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W));

    int val = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(val++);

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    histogram_yuyv(in, out, W, H);

    auto hw = read_output(out, W, H);

    bool ok = true;
    for (int b = 0; b < 256; b++) {
        if (hw.y_hist[b] != 1) {
            std::cerr << "    y_hist[" << b << "]=" << hw.y_hist[b]
                      << " esperado=1\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 9 — Señal LAST en el paquete correcto
// El último paquete de salida (bin 255) debe tener last=1
// Todos los anteriores deben tener last=0
// ============================================================
static bool test_last_signal() {
    const int W = 8, H = 4;
    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W, 50));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    histogram_yuyv(in, out, W, H);

    int total_packets = H + W + 256;
    bool ok = true;

    for (int i = 0; i < total_packets; i++) {
        axis_t pkt = out.read();
        bool is_last_packet = (i == total_packets - 1);
        bool got_last = (pkt.last == 1);

        if (got_last != is_last_packet) {
            std::cerr << "    Paquete " << i << "/" << total_packets
                      << ": last=" << got_last
                      << " esperado=" << is_last_packet << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 10 — Golden model completo con imagen realista
// Patrón que simula una imagen con región oscura central
// (similar a los tests del morfológico)
// ============================================================
static bool test_golden_realistic() {
    const int W = 16, H = 12;
    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W, 200));

    // Región oscura central 4x4 con valor 50
    for (int y = 4; y < 8; y++)
        for (int x = 4; x < 8; x++)
            img[y][x] = 50;

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    histogram_yuyv(in, out, W, H);

    auto hw = read_output(out, W, H);
    auto sw = compute_golden(img, W, H);

    bool ok = compare_results(hw, sw, W, H, true);

    // Verificaciones semánticas adicionales
    int dark_pixels  = 4 * 4;
    int light_pixels = W * H - dark_pixels;

    if (hw.y_hist[50]  != (uint32_t)dark_pixels) {
        std::cerr << "    y_hist[50]=" << hw.y_hist[50]
                  << " esperado=" << dark_pixels << "\n";
        ok = false;
    }
    if (hw.y_hist[200] != (uint32_t)light_pixels) {
        std::cerr << "    y_hist[200]=" << hw.y_hist[200]
                  << " esperado=" << light_pixels << "\n";
        ok = false;
    }
    return ok;
}

// ============================================================
// TEST 11 — Preparación para croma: estructura de salida extensible
// Verifica que el stream de salida tiene exactamente
// H + W + 256 paquetes (y en el futuro H + W + 256 + 256 + 256)
// ============================================================
static bool test_output_packet_count() {
    const int W = 8, H = 4;
    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W, 100));

    hls::stream<axis_t> in, out;
    load_image(in, img, W, H);
    histogram_yuyv(in, out, W, H);

    int expected = H + W + 256;
    int count = 0;
    while (!out.empty()) { out.read(); count++; }

    if (count != expected) {
        std::cerr << "    Paquetes de salida: " << count
                  << " esperado: " << expected
                  << "\n    (futuro con U+V: " << H + W + 256*3 << ")\n";
        return false;
    }
    return true;
}

// ============================================================
// MAIN
// ============================================================
int main() {
    std::cout << "======================================\n";
    std::cout << "  Testbench: histogram_yuyv\n";
    std::cout << "  Protocolo salida: [row_sums | col_sums | y_hist]\n";
    std::cout << "======================================\n\n";

    report("T01 - Imagen uniforme",                   test_uniform());
    report("T02 - Imagen en cero",                    test_all_zeros());
    report("T03 - Imagen en máximo (255)",             test_all_max());
    report("T04 - Gradiente horizontal",              test_horizontal_gradient());
    report("T05 - Gradiente vertical",                test_vertical_gradient());
    report("T06 - Un solo píxel activo",              test_single_pixel());
    report("T07 - Dependencia RAW (y0==y1)",          test_raw_dependency());
    report("T08 - Histograma completo 0..255",        test_full_histogram());
    report("T09 - Señal LAST en paquete correcto",    test_last_signal());
    report("T10 - Golden model imagen realista",      test_golden_realistic());
    report("T11 - Conteo de paquetes de salida",      test_output_packet_count());

    std::cout << "\n======================================\n";
    std::cout << "  Resultado: " << tests_passed << "/"
              << tests_run << " tests pasaron\n";
    if (tests_failed > 0)
        std::cout << "  FALLARON: " << tests_failed << " tests\n";
    else
        std::cout << "  Todos los tests PASARON\n";
    std::cout << "======================================\n";

    // Nota para extensión futura con canales U y V:
    // Cuando se agreguen u_counter y v_counter al IP, extender:
    //   read_output()       → leer 256 paquetes adicionales por canal
    //   compute_golden()    → acumular u_hist y v_hist
    //   compare_results()   → comparar los tres histogramas
    //   test_output_packet_count() → esperado = H + W + 256*3

    return tests_failed > 0 ? 1 : 0;
}