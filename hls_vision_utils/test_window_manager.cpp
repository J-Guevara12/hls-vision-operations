// ============================================================
// Testbench: tb_window_manager.cpp
// Componente bajo prueba: hls_lib::WindowManager<T, WIN_SIZE, MAX_W, PPP>
//
// Estrategia:
//   - Verificar que la ventana contiene los valores correctos
//     después de cada shift, para PPP=1 y PPP=2
//   - Verificar el orden de filas dentro de la ventana
//   - Verificar el comportamiento en bordes de fila
//   - Verificar que PPP=1 y PPP=2 producen los mismos resultados
//     cuando se alimentan con los mismos píxeles
// ============================================================

#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "hls_vision_utils.hpp"

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

// Imprime una ventana 3x3
static void print_win(const uint8_t win[3][3], const std::string& label) {
    std::cout << "  --- " << label << " ---\n";
    for (int r = 0; r < 3; r++) {
        std::cout << "  ";
        for (int c = 0; c < 3; c++)
            std::cout << std::setw(4) << (int)win[r][c];
        std::cout << "\n";
    }
}

// Verifica que dos ventanas son iguales
static bool win_eq(const uint8_t a[3][3], const uint8_t b[3][3],
                    const std::string& label = "") {
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            if (a[r][c] != b[r][c]) {
                if (!label.empty())
                    std::cerr << "    " << label
                              << " mismatch en [" << r << "," << c << "]"
                              << " got=" << (int)a[r][c]
                              << " expected=" << (int)b[r][c] << "\n";
                return false;
            }
    return true;
}

// Simula el comportamiento esperado del WindowManager en software
// Devuelve la ventana centrada en (row, col) de la imagen
static void expected_window(const std::vector<std::vector<uint8_t>>& img,
                              int row, int col,
                              uint8_t win[3][3],
                              int width, int height) {
    for (int dr = -1; dr <= 1; dr++)
        for (int dc = -1; dc <= 1; dc++) {
            int r = row + dr;
            int c = col + dc;
            bool valid = (r >= 0 && r < height && c >= 0 && c < width);
            win[dr+1][dc+1] = valid ? img[r][c] : 0;
        }
}

// ============================================================
// TEST 1 — PPP=1: secuencia simple de 3 filas
// Verifica que después de llenar la ventana con datos conocidos
// el contenido es el correcto
// ============================================================
static bool test_ppp1_basic_window() {
    const int W = 6, H = 3;

    // Imagen de prueba con valores únicos por posición
    // img[y][x] = y*10 + x
    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(y * 10 + x);

    hls_lib::WindowManager<uint8_t, 3, 8, 1> wm;

    bool ok = true;


    // Alimentar píxel a píxel y verificar la ventana cuando está completa
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            uint8_t pixels[1] = {img[y][x]};
            uint8_t wins[1][3][3];
            wm.shiftN(pixels, x, wins);

            // La ventana es válida cuando tenemos al menos 2 filas
            // y al menos 2 columnas (desfase de 1 fila + 1 col)
            if (y >= 1 && x >= 1) {
                // El píxel central de la ventana debe ser img[y-1][x-1]
                // porque la ventana está centrada en (y-1, x-1)
                uint8_t expected_center = img[y-1][x-1];
                if (wins[0][1][1] != expected_center) {
                    std::cerr << "    PPP=1 [y=" << y << ",x=" << x << "]"
                              << " centro=" << (int)wins[0][1][1]
                              << " esperado=" << (int)expected_center << "\n";
                    ok = false;
                }

                // Verificar ventana completa si estamos en interior
                if (y >= 2 && x >= 2 && y < H && x < W) {
                    uint8_t exp_win[3][3];
                    expected_window(img, y-1, x-1, exp_win, W, H);
                    if (!win_eq(wins[0], exp_win)) {
                        std::cerr << "    PPP=1 ventana incorrecta en [y="
                                  << y-1 << ",x=" << x-1 << "]\n";
                        print_win(wins[0], "obtenida");
                        print_win(exp_win, "esperada");
                        ok = false;
                    }
                }
            }
        }
    }
    return ok;
}

// ============================================================
// TEST 2 — PPP=1: verificar orden de filas en la ventana
// La fila 0 debe ser la más antigua, fila 2 la más reciente
// ============================================================
static bool test_ppp1_row_order() {
    // Imagen de 3 filas con valores distintos por fila
    // Fila 0: todos 10, Fila 1: todos 20, Fila 2: todos 30
    const int W = 4, H = 3;
    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            img[y][x] = (uint8_t)((y+1) * 10);
    }
            

    hls_lib::WindowManager<uint8_t, 3, 8, 1> wm;

    bool ok = true;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            uint8_t pixels[1] = {img[y][x]};
            uint8_t wins[1][3][3];
            wm.shiftN(pixels, x, wins);

            // Cuando tenemos las 3 filas y estamos en posición interior
            if (y == 2 && x == 3) {
                // win[0] = fila más antigua = img[0] = 10
                // win[1] = fila media       = img[1] = 20
                // win[2] = fila más reciente = img[2] = 30
                for (int c = 0; c < 2 && ok; c++) {
                    if (wins[0][0][c] != 10) {
                        std::cerr << "    Fila 0 de ventana: got="
                                  << (int)wins[0][0][c] << " esperado=10\n";
                        ok = false;
                    }
                    if (wins[0][1][c] != 20) {
                        std::cerr << "    Fila 1 de ventana: got="
                                  << (int)wins[0][1][c] << " esperado=20\n";
                        ok = false;
                    }
                    if (wins[0][2][c] != 30) {
                        std::cerr << "    Fila 2 de ventana: got="
                                  << (int)wins[0][2][c] << " esperado=30\n";
                        ok = false;
                    }
                }
            }
        }
    }
    return ok;
}

// ============================================================
// TEST 3 — PPP=1: verificar orden de columnas en la ventana
// La columna 0 debe ser la más antigua, columna 2 la más reciente
// ============================================================
static bool test_ppp1_col_order() {
    // Imagen donde cada columna tiene un valor único
    const int W = 5, H = 3;
    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)((x+1) * 10); // col 0=10, 1=20, 2=30...

    hls_lib::WindowManager<uint8_t, 3, 8, 1> wm;

    bool ok = true;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            uint8_t pixels[1] = {img[y][x]};
            uint8_t wins[1][3][3];
            wm.shiftN(pixels, x, wins);

            if (y == 1 && x == 3) {
                // Ventana centrada en (1, 2): cols 1,2,3 = 20,30,40
                if (wins[0][1][0] != 20 || wins[0][1][1] != 30 || wins[0][1][2] != 40) {
                    std::cerr << "    Orden columnas: got=("
                              << (int)wins[0][1][0] << ","
                              << (int)wins[0][1][1] << ","
                              << (int)wins[0][1][2]
                              << ") esperado=(20,30,40)\n";
                    ok = false;
                }
            }
        }
    }
    return ok;
}

// ============================================================
// TEST 4 — PPP=1: píxel impulso
// Un solo píxel de valor 255 rodeado de ceros
// Verifica que el impulso aparece en la posición correcta
// de la ventana y se desplaza correctamente
// ============================================================
static bool test_ppp1_impulse() {
    const int W = 6, H = 4;
    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W, 0));
    img[1][2] = 255; // impulso en (1,2)

    hls_lib::WindowManager<uint8_t, 3, 8, 1> wm;

    bool ok = true;
    int impulse_found = 0;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            uint8_t pixels[1] = {img[y][x]};
            uint8_t wins[1][3][3];
            wm.shiftN(pixels, x, wins);

            // Contar cuántas veces aparece 255 en la ventana
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    if (wins[0][r][c] == 255)
                        impulse_found++;
        }
    }

    // El impulso debe aparecer exactamente una vez en cada una de las
    // 9 posiciones de la ventana mientras el kernel lo recorre → 9 veces
    // en ventanas distintas, pero en cada ventana máximo 1 vez
    if (impulse_found != 9) {
        std::cerr << "    Impulso encontrado " << impulse_found
                  << " veces en ventanas, esperado 9\n";
        ok = false;
    }
    return ok;
}

// ============================================================
// TEST 5 — PPP=2: verificar que wins[0] y wins[1] son correctas
// y corresponden a píxeles consecutivos
// ============================================================
static bool test_ppp2_basic_windows() {
    const int W = 6, H = 3;

    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(y * 10 + x);

    hls_lib::WindowManager<uint8_t, 3, 8, 2> wm;

    bool ok = true;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W/2; x++) {
            int x0 = x * 2;
            int x1 = x * 2 + 1;

            uint8_t pixels[2] = {img[y][x0], img[y][x1]};
            uint8_t wins[2][3][3];
            wm.shiftN(pixels, x, wins);

            // Verificar cuando la ventana está completa
            if (y >= 2 && x >= 2) {
                // wins[0] centrada en (y-1, x0-1) = (y-1, g*2-1) → impar anterior
                // wins[1] centrada en (y-1, x0)   = (y-1, g*2)   → par actual
                uint8_t exp0[3][3], exp1[3][3];
                expected_window(img, y-1, (x-1)*2, exp0, W, H);
                expected_window(img, y-1, (x-1)*2 + 1,   exp1, W, H);

                if (!win_eq(wins[0], exp0)) {
                    std::cerr << "    PPP=2 wins[0] incorrecta en grupo x="
                              << x << " y=" << y << "\n";
                    print_win(wins[0], "obtenida wins[0]");
                    print_win(exp0,    "esperada wins[0]");
                    ok = false;
                }
                if (!win_eq(wins[1], exp1)) {
                    std::cerr << "    PPP=2 wins[1] incorrecta en grupo x="
                              << x << " y=" << y << "\n";
                    print_win(wins[1], "obtenida wins[1]");
                    print_win(exp1,    "esperada wins[1]");
                    ok = false;
                }
            }
        }
    }
    return ok;
}

// ============================================================
// TEST 6 — Equivalencia PPP=1 vs PPP=2
// Alimentando los mismos píxeles, los centros de ventana
// deben ser idénticos entre PPP=1 y PPP=2
// ============================================================
static bool test_ppp1_vs_ppp2_equivalence() {
    const int W = 8, H = 4;

    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(y * 13 + x * 17 + 5);

    hls_lib::WindowManager<uint8_t, 3, 16, 1> wm1;
    hls_lib::WindowManager<uint8_t, 3, 16, 2> wm2;

    bool ok = true;

    // Resultado PPP=1: centros de ventana para cada posición
    std::vector<std::vector<uint8_t>> centers1(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            uint8_t pixels[1] = {img[y][x]};
            uint8_t wins[1][3][3];
            wm1.shiftN(pixels, x, wins);
            if (y >= 1 && x >= 1)
                centers1[y-1][x-1] = wins[0][1][1];
        }
    }

    // Resultado PPP=2: centros de ventana para cada posición
    // Desfase real del WM con PPP=2: en ciclo (y, g)
    //   wins[0] centrada en img[y-1][(g-1)*2 + 0]
    //   wins[1] centrada en img[y-1][(g-1)*2 + 1]
    std::vector<std::vector<uint8_t>> centers2(H, std::vector<uint8_t>(W, 0));
    for (int y = 0; y < H; y++) {
        for (int g = 0; g < W/2; g++) {
            uint8_t pixels[2] = {img[y][g*2], img[y][g*2+1]};
            uint8_t wins[2][3][3];
            wm2.shiftN(pixels, g, wins);
            if (y >= 1 && g >= 1) {
                centers2[y-1][(g-1)*2]     = wins[0][1][1];
                centers2[y-1][(g-1)*2 + 1] = wins[1][1][1];
            }
        }
    }

    // Comparar centros en zona interior:
    // PPP=1 tiene borde en x=0, PPP=2 tiene artefacto en g=1 (x=0,1)
    // → comparar desde x=2 para que ambos estén en zona limpia
    for (int y = 1; y < H-1 && ok; y++) {
        for (int x = 2; x < W-2 && ok; x++) {
            if (centers1[y][x] != centers2[y][x]) {
                std::cerr << "    PPP1 vs PPP2 difieren en [" << y << "," << x << "]"
                          << " ppp1=" << (int)centers1[y][x]
                          << " ppp2=" << (int)centers2[y][x] << "\n";
                ok = false;
            }
        }
    }
    return ok;
}

// ============================================================
// TEST 7 — PPP=2: impulso aislado
// Verifica que el impulso recorre correctamente ambas ventanas
// ============================================================
static bool test_ppp2_impulse() {
    const int W = 6, H = 4;

    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W, 0));
    img[1][2] = 255; // impulso en posición par del grupo 1

    hls_lib::WindowManager<uint8_t, 3, 8, 2> wm;

    bool ok = true;
    int impulse_count = 0;

    for (int y = 0; y < H; y++) {
        for (int g = 0; g < W/2; g++) {
            uint8_t pixels[2] = {img[y][g*2], img[y][g*2+1]};
            uint8_t wins[2][3][3];
            wm.shiftN(pixels, g, wins);

            for (int p = 0; p < 2; p++)
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        if (wins[p][r][c] == 255)
                            impulse_count++;
        }
    }

    // El impulso debe aparecer 9 veces en total
    if (impulse_count != 9) {
        std::cerr << "    PPP=2 impulso encontrado " << impulse_count
                  << " veces, esperado 9\n";
        ok = false;
    }
    return ok;
}

// ============================================================
// TEST 8 — PPP=2: ventanas de píxeles pares e impares
// correctamente separadas con gradiente horizontal
// ============================================================
static bool test_ppp2_gradient() {
    const int W = 8, H = 4;

    // Gradiente horizontal puro: img[y][x] = x * 10
    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(x * 10);

    hls_lib::WindowManager<uint8_t, 3, 16, 2> wm;

    bool ok = true;

    for (int y = 0; y < H; y++) {
        for (int g = 0; g < W/2; g++) {
            uint8_t pixels[2] = {img[y][g*2], img[y][g*2+1]};
            uint8_t wins[2][3][3];
            wm.shiftN(pixels, g, wins);

            if (y >= 2 && g >= 2 && g < W/2) {
                // wins[0] centrada en col (g*2-1)
                // fila central debe ser: col-1, col, col+1
                int center_col0 = (g-1)*2;
                if (center_col0 >= 1 && center_col0 < W-1) {
                    uint8_t exp_left   = (uint8_t)((center_col0-1)*10);
                    uint8_t exp_center = (uint8_t)( center_col0   *10);
                    uint8_t exp_right  = (uint8_t)((center_col0+1)*10);

                    if (wins[0][1][0] != exp_left ||
                        wins[0][1][1] != exp_center ||
                        wins[0][1][2] != exp_right) {
                        std::cerr << "    Gradiente wins[0] fila central g=" << g
                                  << " got=(" << (int)wins[0][1][0] << ","
                                  << (int)wins[0][1][1] << ","
                                  << (int)wins[0][1][2] << ")"
                                  << " esperado=(" << (int)exp_left << ","
                                  << (int)exp_center << ","
                                  << (int)exp_right << ")\n";
                        ok = false;
                    }
                }
            }
        }
    }
    return ok;
}

// ============================================================
// TEST 9 — PPP=1: convolución identidad completa
// Aplica kernel identidad usando los datos del WindowManager
// y verifica que el resultado es igual a la imagen original
// ============================================================
static bool test_ppp1_identity_convolution() {
    const int W = 8, H = 6;

    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(y * 15 + x * 11 + 3);

    hls_lib::WindowManager<uint8_t, 3, 16, 1> wm;

    // Kernel identidad
    short k[3][3] = {{0,0,0},{0,1,0},{0,0,0}};

    bool ok = true;

    for (int y = 0; y < H+1; y++) {
        for (int x = 0; x < W+1; x++) {
            uint8_t new_px = (y < H && x < W) ? img[y][x] : 0;
            uint8_t pixels[1] = {new_px};
            uint8_t wins[1][3][3];
            wm.shiftN(pixels, x < W+1 ? x : W, wins);

            if (y >= 1 && x >= 1) {
                bool is_border = (y == 1 || y == H || x == 1 || x == W);
                if (!is_border) {
                    int acc = 0;
                    for (int r = 0; r < 3; r++)
                        for (int c = 0; c < 3; c++)
                            acc += wins[0][r][c] * k[r][c];

                    uint8_t result = (uint8_t)acc;
                    uint8_t expected = img[y-1][x-1];

                    if (result != expected) {
                        std::cerr << "    Identidad conv [" << y-1 << "," << x-1 << "]"
                                  << " got=" << (int)result
                                  << " expected=" << (int)expected << "\n";
                        ok = false;
                    }
                }
            }
        }
    }
    return ok;
}

// ============================================================
// TEST 10 — PPP=2: convolución identidad completa
// Mismo test que T10 pero con PPP=2 para verificar equivalencia
// ============================================================
static bool test_ppp2_identity_convolution() {
    const int W = 8, H = 6;

    std::vector<std::vector<uint8_t>> img(H, std::vector<uint8_t>(W));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] = (uint8_t)(y * 15 + x * 11 + 3);

    hls_lib::WindowManager<uint8_t, 3, 16, 2> wm;

    short k[3][3] = {{0,0,0},{0,1,0},{0,0,0}};

    bool ok = true;
    const int GROUPS = W / 2;

    for (int y = 0; y < H+1; y++) {
        for (int g = 0; g < GROUPS+1; g++) {
            int x0 = g * 2;
            int x1 = g * 2 + 1;

            uint8_t p0 = (y < H && x0 < W) ? img[y][x0] : 0;
            uint8_t p1 = (y < H && x1 < W) ? img[y][x1] : 0;
            uint8_t pixels[2] = {p0, p1};
            uint8_t wins[2][3][3];
            wm.shiftN(pixels, g, wins);

            if (y >= 1 && g >= 1) {
                for (int p = 0; p < 2 && ok; p++) {
                    int out_x = (g-1)*2 + p;
                    int out_y = y - 1;

                    bool is_border = (out_y == 0 || out_y == H-1 ||
                                      out_x == 0 || out_x == W-1);
                    if (!is_border && out_x < W && out_y < H) {
                        int acc = 0;
                        for (int r = 0; r < 3; r++)
                            for (int c = 0; c < 3; c++)
                                acc += wins[p][r][c] * k[r][c];

                        uint8_t result   = (uint8_t)acc;
                        uint8_t expected = img[out_y][out_x];

                        if (result != expected) {
                            std::cerr << "    PPP=2 identidad conv ["
                                      << out_y << "," << out_x << "]"
                                      << " got=" << (int)result
                                      << " expected=" << (int)expected << "\n";
                            ok = false;
                        }
                    }
                }
            }
        }
    }
    return ok;
}

// ============================================================
// MAIN
// ============================================================
int main_window_manager() {
    std::cout << "======================================\n";
    std::cout << "  Testbench: WindowManager\n";
    std::cout << "  WIN_SIZE=3, PPP=1 y PPP=2\n";
    std::cout << "======================================\n\n";

    report("T01 - PPP=1 ventana básica correcta",          test_ppp1_basic_window());
    report("T02 - PPP=1 orden de filas (antigua→reciente)",test_ppp1_row_order());
    report("T03 - PPP=1 orden de columnas (izq→der)",      test_ppp1_col_order());
    report("T04 - PPP=1 impulso recorre 9 posiciones",     test_ppp1_impulse());
    report("T05 - PPP=2 wins[0] y wins[1] correctas",      test_ppp2_basic_windows());
    report("T06 - PPP=1 vs PPP=2 centros equivalentes",    test_ppp1_vs_ppp2_equivalence());
    report("T07 - PPP=2 impulso recorre 9 posiciones",     test_ppp2_impulse());
    report("T08 - PPP=2 gradiente horizontal correcto",    test_ppp2_gradient());
    report("T09 - PPP=1 convolución identidad completa",   test_ppp1_identity_convolution());
    report("T10 - PPP=2 convolución identidad completa",   test_ppp2_identity_convolution());

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
