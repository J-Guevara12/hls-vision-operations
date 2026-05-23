    // ============================================================
    // Testbench: tb_stage_filter.cpp
    // Componente bajo prueba: stage_filter
    //
    // Responsabilidad del stage:
    //   - Recibir stream de píxeles Y individuales
    //   - Aplicar convolución 3x3 con kernel configurable
    //   - Emitir stream de píxeles Y filtrados
    //   - Zerear bordes (primera/última fila y columna)
    //   - Procesar PPP píxeles por ciclo
    //
    // Invariantes a verificar:
    //   - Conteo exacto de píxeles de salida = width * height
    //   - Bordes en cero
    //   - Convolución correcta en interior vs golden model
    //   - Desfase de salida: (width+PPP)*(height+1) iteraciones internas
    //     pero width*height píxeles de salida
    // ============================================================

    #include <iostream>
    #include <iomanip>
    #include <string>
    #include <vector>
    #include <cstdint>
    #include <cstring>
    #include <algorithm>

    #include "ap_int.h"
    #include "hls_stream.h"
    #include "kernels.hpp"


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
    // Tipos
    // ============================================================
    typedef std::vector<std::vector<uint8_t>> Image;

    struct Kernel { short k[3][3]; uint8_t shift; };

    static const Kernel K_IDENTITY  = {{{0,0,0},{0,1,0},{0,0,0}}, 0};
    static const Kernel K_GAUSSIAN  = {{{1,2,1},{2,4,2},{1,2,1}}, 4};
    static const Kernel K_LAPLACIAN = {{{0,1,0},{1,-4,1},{0,1,0}}, 0};
    static const Kernel K_SOBEL_GX  = {{{-1,0,1},{-2,0,2},{-1,0,1}}, 0};
    static const Kernel K_ZEROS     = {{{0,0,0},{0,0,0},{0,0,0}}, 0};

    // ============================================================
    // Utilidades
    // ============================================================
    static uint8_t saturate(int v) {
        return v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v;
    }

    // Carga imagen en y_stream (píxel a píxel en orden raster)
    static void load_y_stream(hls::stream<ap_uint<PPP*8>>& s,
                            const Image& img, int W, int H) {
        for (int y = 0; y < H; y++)
            for (int g = 0; g < W/PPP; g++){
                ap_uint<PPP*8> packet = 0;
                for (int p = 0; p < PPP; p++)
                    packet |= (ap_uint<PPP*8>) img[y][g*PPP+p]<<(p<<3);
                //std::cout << "Writing " << (g*PPP) << ", " << y << std::endl;
                s.write(packet);
            }
    }

    // Lee width*height píxeles del stream de salida
    static Image read_y_stream(hls::stream<ap_uint<PPP*8>>& s, int W, int H) {
        Image out(H, std::vector<uint8_t>(W, 0));
        for (int y = 0; y < H; y++)
            for (int g = 0; g < W/PPP; g++){
                auto packet = s.read();
                for (int p = 0; p < PPP; p++)
                    out[y][g*PPP+p] = (packet >> (p<<3)) & 0xFF ;
            }
        return out;
    }

    // Golden model: convolución 3x3 con bordes en cero
    static Image golden(const Image& img, int W, int H, const Kernel& k) {
        Image out(H, std::vector<uint8_t>(W, 0));
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                if (y == 0 || y == H-1 || x == 0 || x == W-1) continue;
                int acc = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                        acc += img[y+dy][x+dx] * k.k[dy+1][dx+1];
                out[y][x] = saturate(acc >> k.shift);
            }
        return out;
    }

    static void print_image(const Image& img, int W, int H, const std::string& label = "") {
        if (!label.empty()) std::cout << "  --- " << label << " ---\n";
        for (int y = 0; y < H; y++) {
            std::cout << "  ";
            for (int x = 0; x < W; x++)
                std::cout << std::setw(4) << (int)img[y][x];
            std::cout << "\n";
        }
    }

    static bool compare(const Image& hw, const Image& ref,
                        int W, int H, bool verbose = true) {
        bool ok = true; int errs = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                if (hw[y][x] != ref[y][x]) {
                    if (verbose && errs < 8)
                        std::cerr << "    [" << y << "," << x << "]"
                                << " hw=" << (int)hw[y][x]
                                << " ref=" << (int)ref[y][x] << "\n";
                    ok = false; errs++;
                }
        if (!ok && verbose) std::cerr << "    Total: " << errs << " errores\n";
        return ok;
    }

    // Wrapper que construye el array k[] para stage_filter
    static void run_filter(hls::stream<ap_uint<PPP*8>>& in,
                            hls::stream<ap_uint<PPP*8>>& out,
                            int W, int H, const Kernel& k) {
        short ka[3][3];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                ka[r][c] = k.k[r][c];
        stage_filter(in, out, W, H, ka, k.shift);
    }

    // ============================================================
    // TEST 1 — Conteo exacto: stage_filter emite width*height píxeles
    // ============================================================
    static bool test_output_count() {
        const int W = 8, H = 6;
        Image img(H, std::vector<uint8_t>(W, 128));

        hls::stream<ap_uint<PPP*8>> in, out;
        load_y_stream(in, img, W, H);
        run_filter(in, out, W, H, K_GAUSSIAN);

        int count = 0;
        while (!out.empty()) { out.read(); count+=PPP; }
        bool ok = (count == W * H);
        if (!ok)
            std::cerr << "    Píxeles emitidos=" << count
                    << " esperados=" << W*H << "\n";
        return ok;
    }

    // ============================================================
    // TEST 2 — Bordes en cero para cualquier kernel
    // ============================================================
    static bool test_border_zeros() {
        const int W = 8, H = 6;
        Image img(H, std::vector<uint8_t>(W, 200));
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                img[y][x] = (uint8_t)(y * 10 + x);

        hls::stream<ap_uint<PPP*8>> in, out;
        load_y_stream(in, img, W, H);
        run_filter(in, out, W, H, K_GAUSSIAN);

        auto hw = read_y_stream(out, W, H);
        bool ok = true;

        //print_image(img, W, H);
        //print_image(hw, W, H);


        for (int x = 0; x < W; x++) {
            if (hw[0][x]   != 0) { std::cerr << "    hw[0]["   << x << "]=" << (int)hw[0][x]   << "\n"; ok=false; }
            if (hw[H-1][x] != 0) { std::cerr << "    hw[H-1][" << x << "]=" << (int)hw[H-1][x] << "\n"; ok=false; }
        }
        for (int y = 0; y < H; y++) {
            if (hw[y][0]   != 0) { std::cerr << "    hw[" << y << "][0]="   << (int)hw[y][0]   << "\n"; ok=false; }
            if (hw[y][W-1] != 0) { std::cerr << "    hw[" << y << "][W-1]=" << (int)hw[y][W-1] << "\n"; ok=false; }
        }
        return ok;
    }

    // ============================================================
    // TEST 3 — Kernel identidad: interior igual a entrada
    // ============================================================
    static bool test_identity() {
        const int W = 8, H = 6;
        Image img(H, std::vector<uint8_t>(W));
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                img[y][x] = (uint8_t)(y * 10 + x);
        
        hls::stream<ap_uint<PPP*8>> in, out;
        load_y_stream(in, img, W, H);
        run_filter(in, out, W, H, K_IDENTITY);


        auto hw  = read_y_stream(out, W, H);
        auto ref = golden(img, W, H, K_IDENTITY);
        bool ok  = compare(hw, ref, W, H);

        //print_image(img, W, H);
        //print_image(hw, W, H);

        // Verificación semántica adicional
        for (int y = 1; y < H-1 && ok; y++)
            for (int x = 1; x < W-1 && ok; x++)
                if (hw[y][x] != img[y][x]) {
                    std::cerr << "    Identidad [" << y << "," << x << "]"
                            << " hw=" << (int)hw[y][x]
                            << " original=" << (int)img[y][x] << "\n";
                    ok = false;
                }
        return ok;
    }

    // ============================================================
    // TEST 4 — Gaussiano imagen uniforme: interior = mismo valor
    // ============================================================
    static bool test_gaussian_uniform() {
        const int W = 8, H = 6;
        const uint8_t VAL = 100;
        Image img(H, std::vector<uint8_t>(W, VAL));

        hls::stream<ap_uint<PPP*8>> in, out;
        load_y_stream(in, img, W, H);
        run_filter(in, out, W, H, K_GAUSSIAN);



        auto hw = read_y_stream(out, W, H);
        bool ok = true;

        //print_image(img, W, H);
        //print_image(hw, W, H);

        for (int y = 1; y < H-1 && ok; y++)
            for (int x = 1; x < W-1 && ok; x++)
                if (hw[y][x] != VAL) {
                    std::cerr << "    Gaussiano uniforme [" << y << "," << x << "]"
                            << " hw=" << (int)hw[y][x] << " esperado=" << (int)VAL << "\n";
                    ok = false;
                }
        return ok;
    }

    // ============================================================
    // TEST 5 — Laplaciano imagen uniforme: interior = 0
    // ============================================================
    static bool test_laplacian_uniform() {
        const int W = 8, H = 6;
        Image img(H, std::vector<uint8_t>(W, 128));

        hls::stream<ap_uint<PPP*8>> in, out;
        load_y_stream(in, img, W, H);
        run_filter(in, out, W, H, K_LAPLACIAN);

        auto hw = read_y_stream(out, W, H);
        bool ok = true;
        for (int y = 1; y < H-1 && ok; y++)
            for (int x = 1; x < W-1 && ok; x++)
                if (hw[y][x] != 0) {
                    std::cerr << "    Laplaciano uniforme [" << y << "," << x << "]"
                            << " hw=" << (int)hw[y][x] << " esperado=0\n";
                    ok = false;
                }
        return ok;
    }

    // ============================================================
    // TEST 6 — Golden model completo: patrón pseudo-aleatorio
    // ============================================================
    static bool test_golden(const Kernel& k, const std::string& name) {
        const int W = 16, H = 12;
        Image img(H, std::vector<uint8_t>(W));
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                img[y][x] = (uint8_t)((y*37 + x*53 + y*x*7) % 256);

        hls::stream<ap_uint<PPP*8>> in, out;
        load_y_stream(in, img, W, H);
        run_filter(in, out, W, H, k);

        auto hw  = read_y_stream(out, W, H);
        auto ref = golden(img, W, H, k);
        bool ok  = compare(hw, ref, W, H);
        if (!ok) std::cerr << "    Kernel: " << name << "\n";
        return ok;
    }

    // ============================================================
    // TEST 7 — Kernel ceros: toda la salida debe ser cero
    // ============================================================
    static bool test_zero_kernel() {
        const int W = 8, H = 6;
        Image img(H, std::vector<uint8_t>(W, 200));

        hls::stream<ap_uint<PPP*8>> in, out;
        load_y_stream(in, img, W, H);
        run_filter(in, out, W, H, K_ZEROS);

        auto hw = read_y_stream(out, W, H);
        bool ok = true;
        for (int y = 0; y < H && ok; y++)
            for (int x = 0; x < W && ok; x++)
                if (hw[y][x] != 0) {
                    std::cerr << "    Kernel cero [" << y << "," << x << "]="
                            << (int)hw[y][x] << "\n";
                    ok = false;
                }
        return ok;
    }

    // ============================================================
    // TEST 8 — Saturación: valores negativos → 0, >255 → 255
    // ============================================================
    static bool test_saturation() {
        const int W = 8, H = 8;
        Image img(H, std::vector<uint8_t>(W, 128));
        img[4][4] = 255;

        hls::stream<ap_uint<PPP*8>> in, out;
        load_y_stream(in, img, W, H);
        run_filter(in, out, W, H, K_LAPLACIAN);

        auto hw  = read_y_stream(out, W, H);
        auto ref = golden(img, W, H, K_LAPLACIAN);
        return compare(hw, ref, W, H);
    }

    // ============================================================
    // TEST 9 — Shift_val: verificar normalización correcta
    // ============================================================
    static bool test_shift_val() {
        const int W = 8, H = 6;
        Image img(H, std::vector<uint8_t>(W, 100));

        // Con shift=4: (16*100)>>4 = 100
        hls::stream<ap_uint<PPP*8>> in1, out1;
        load_y_stream(in1, img, W, H);
        run_filter(in1, out1, W, H, K_GAUSSIAN);
        auto hw4 = read_y_stream(out1, W, H);

        // Con shift=0: 16*100 = 1600 → saturado a 255
        Kernel k0 = K_GAUSSIAN; k0.shift = 0;
        hls::stream<ap_uint<PPP*8>> in2, out2;
        load_y_stream(in2, img, W, H);
        run_filter(in2, out2, W, H, k0);
        auto hw0 = read_y_stream(out2, W, H);

        bool ok = true;
        if (hw4[2][2] != 100) {
            std::cerr << "    Shift=4: hw[2][2]=" << (int)hw4[2][2] << " esperado=100\n";
            ok = false;
        }
        if (hw0[2][2] != 255) {
            std::cerr << "    Shift=0: hw[2][2]=" << (int)hw0[2][2] << " esperado=255\n";
            ok = false;
        }
        return ok;
    }

    // ============================================================
    // TEST 10 — Resolución mínima 4x4
    // ============================================================
    static bool test_min_resolution() {
        const int W = 4, H = 4;
        Image img = {{10,20,30,40},{50,60,70,80},{90,100,110,120},{130,140,150,160}};

        hls::stream<ap_uint<PPP*8>> in1, in, out;
        load_y_stream(in, img, W, H);
        run_filter(in, out, W, H, K_GAUSSIAN);

        auto hw  = read_y_stream(out, W, H);
        auto ref = golden(img, W, H, K_GAUSSIAN);
        return compare(hw, ref, W, H);
    }

    // ============================================================
    // TEST 11 — Resolución no potencia de 2 (10x7)
    // ============================================================
    static bool test_odd_resolution() {
        const int W = 10, H = 7;
        Image img(H, std::vector<uint8_t>(W));
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                img[y][x] = (uint8_t)((y*31 + x*47) % 256);

        hls::stream<ap_uint<PPP*8>> in, out;
        load_y_stream(in, img, W, H);
        run_filter(in, out, W, H, K_GAUSSIAN);

        auto hw  = read_y_stream(out, W, H);
        auto ref = golden(img, W, H, K_GAUSSIAN);
        return compare(hw, ref, W, H);
    }

    // ============================================================
    // MAIN
    // ============================================================
    int main_filter() {
        std::cout << "======================================\n";
        std::cout << "  Testbench: stage_filter\n";
        std::cout << "  Convolución 3x3 sobre stream Y\n";
        std::cout << "======================================\n\n";

        report("T01 - Conteo exacto de píxeles de salida", test_output_count());
        report("T02 - Bordes en cero",                     test_border_zeros());
        report("T03 - Kernel identidad",                   test_identity());
        report("T04 - Gaussiano imagen uniforme",          test_gaussian_uniform());
        report("T05 - Laplaciano imagen uniforme → cero",  test_laplacian_uniform());
        report("T06a- Golden model Gaussiano",             test_golden(K_GAUSSIAN,  "Gaussiano"));
        report("T06b- Golden model Laplaciano",            test_golden(K_LAPLACIAN, "Laplaciano"));
        report("T06c- Golden model Sobel Gx",              test_golden(K_SOBEL_GX,  "Sobel Gx"));
        report("T07 - Kernel de todos ceros",              test_zero_kernel());
        report("T08 - Saturación correcta",                test_saturation());
        report("T09 - Shift_val correcto",                 test_shift_val());
        report("T10 - Resolución mínima 4x4",              test_min_resolution());
        report("T11 - Resolución no potencia de 2 (10x7)", test_odd_resolution());

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
