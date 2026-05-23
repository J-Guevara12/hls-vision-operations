// ============================================================
// Testbench: test_argmax_and_count.cpp
// Componente bajo prueba: stage_argmax_and_count
//
// Invariantes:
//   - Ganador correcto cuando una clase domina claramente
//   - Empate: gana la clase de menor índice (0 >= 1 → 0 gana)
//   - changed_count=0 cuando Pk+1 == Pk en todos los píxeles
//   - changed_count=N cuando todos los píxeles cambian
//   - changed_count correcto con mezcla de cambios y no cambios
//   - Conteo exacto de grupos emitidos
//   - Empaquetado correcto: PPP índices en result_group_t
// ============================================================

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include "perceptual_array.hpp"

static int tests_run_a = 0, tests_passed_a = 0, tests_failed_a = 0;
static void report_a(const std::string& name, bool ok) {
    tests_run_a++;
    if (ok) { tests_passed_a++; std::cout << "  [PASS] " << name << "\n"; }
    else     { tests_failed_a++; std::cerr << "  [FAIL] " << name << "\n"; }
}

// Construye un blend_vec_t con una sola clase dominante
static blend_vec_t make_blend_winner(int winner, uint8_t win_val = 200,
                                      uint8_t rest_val = 10) {
    blend_vec_t v = 0;
    for (int c = 0; c < NUM_CLASSES; c++)
        v.range(c*8+7, c*8) = (c == winner) ? win_val : rest_val;
    return v;
}

// Construye un blend_vec_t con todos los valores iguales (empate)
static blend_vec_t make_blend_tie(uint8_t val = 100) {
    blend_vec_t v = 0;
    for (int c = 0; c < NUM_CLASSES; c++)
        v.range(c*8+7, c*8) = val;
    return v;
}

static uint8_t get_result(result_group_t g, int p) {
    return (uint8_t)g.range(p*BITS_CLASS+BITS_CLASS-1, p*BITS_CLASS);
}

static pgroup_t make_pk_group(uint8_t classes[PPP]) {
    pgroup_t g = 0;
    for (int p = 0; p < PPP; p++)
        g.range(p*BITS_CLASS+BITS_CLASS-1, p*BITS_CLASS) = classes[p] & 0xF;
    return g;
}

// ============================================================
// TEST 1 — Ganador correcto cuando una clase domina
// ============================================================
static bool test_a_correct_winner() {
    hls::stream<blend_vec_t>    blend_in;
    hls::stream<pgroup_t>       pk_in;
    hls::stream<result_group_t> result_out;
    int changed = 0;

    bool ok = true;
    for (int target = 0; target < NUM_CLASSES; target++) {
        hls::stream<blend_vec_t>    b_in;
        hls::stream<pgroup_t>       p_in;
        hls::stream<result_group_t> r_out;
        int ch = 0;

        // PPP ventanas para completar 1 grupo
        for (int p = 0; p < PPP; p++) b_in.write(make_blend_winner(target));
        uint8_t pk_cls[PPP] = {0,0,0,0};
        p_in.write(make_pk_group(pk_cls));

        stage_argmax_and_count(b_in, p_in, r_out, ch, PPP, 1);

        result_group_t g = r_out.read();
        for (int p = 0; p < PPP; p++) {
            if (get_result(g,p) != target) {
                std::cerr << "    target=" << target << " p=" << p
                          << " got=" << (int)get_result(g,p) << "\n";
                ok = false;
            }
        }
    }
    return ok;
}

// ============================================================
// TEST 2 — Empate: gana clase de menor índice
// ============================================================
static bool test_a_tie_lower_wins() {
    hls::stream<blend_vec_t>    b_in;
    hls::stream<pgroup_t>       p_in;
    hls::stream<result_group_t> r_out;
    int ch = 0;

    for (int p = 0; p < PPP; p++) b_in.write(make_blend_tie(100));
    uint8_t pk_cls[PPP] = {15,15,15,15};
    p_in.write(make_pk_group(pk_cls));

    stage_argmax_and_count(b_in, p_in, r_out, ch, PPP, 1);

    result_group_t g = r_out.read();
    bool ok = true;
    for (int p = 0; p < PPP; p++) {
        uint8_t winner = get_result(g,p);
        // Con empate total, el árbol de comparadores con >= favorece
        // al de menor índice en cada nivel
        if (winner != 0) {
            std::cerr << "    Empate: p=" << p << " winner=" << (int)winner
                      << " esperado=0\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// TEST 3 — changed_count=0 cuando Pk+1 == Pk
// ============================================================
static bool test_a_no_changes() {
    const int N = 8;
    hls::stream<blend_vec_t>    b_in;
    hls::stream<pgroup_t>       p_in;
    hls::stream<result_group_t> r_out;
    int ch = 0;

    // Ganador siempre es clase 5, Pk también es clase 5
    for (int i = 0; i < N; i++) b_in.write(make_blend_winner(5));
    for (int i = 0; i < N/PPP; i++) {
        uint8_t pk_cls[PPP] = {5,5,5,5};
        p_in.write(make_pk_group(pk_cls));
    }

    stage_argmax_and_count(b_in, p_in, r_out, ch, N, 1);

    while (!r_out.empty()) r_out.read();
    bool ok = (ch == 0);
    if (!ok) std::cerr << "    changed=" << ch << " esperado=0\n";
    return ok;
}

// ============================================================
// TEST 4 — changed_count=N cuando todos los píxeles cambian
// ============================================================
static bool test_a_all_changes() {
    const int N = 8;
    hls::stream<blend_vec_t>    b_in;
    hls::stream<pgroup_t>       p_in;
    hls::stream<result_group_t> r_out;
    int ch = 0;

    // Ganador siempre es clase 3, Pk siempre es clase 7
    for (int i = 0; i < N; i++) b_in.write(make_blend_winner(3));
    for (int i = 0; i < N/PPP; i++) {
        uint8_t pk_cls[PPP] = {7,7,7,7};
        p_in.write(make_pk_group(pk_cls));
    }

    stage_argmax_and_count(b_in, p_in, r_out, ch, N, 1);

    while (!r_out.empty()) r_out.read();
    bool ok = (ch == N);
    if (!ok) std::cerr << "    changed=" << ch << " esperado=" << N << "\n";
    return ok;
}

// ============================================================
// TEST 5 — changed_count correcto con mezcla
// ============================================================
static bool test_a_partial_changes() {
    const int N = 8;
    hls::stream<blend_vec_t>    b_in;
    hls::stream<pgroup_t>       p_in;
    hls::stream<result_group_t> r_out;
    int ch = 0;

    // Píxeles pares: ganador=2, Pk=2 (no cambia)
    // Píxeles impares: ganador=3, Pk=7 (cambia)
    for (int i = 0; i < N; i++)
        b_in.write(make_blend_winner(i%2==0 ? 2 : 3));

    for (int i = 0; i < N/PPP; i++) {
        // PPP=4: p0=2,p1=7,p2=2,p3=7
        uint8_t pk_cls[PPP] = {2,7,2,7};
        p_in.write(make_pk_group(pk_cls));
    }

    stage_argmax_and_count(b_in, p_in, r_out, ch, N, 1);

    while (!r_out.empty()) r_out.read();
    int expected = N/2;  // la mitad cambian
    bool ok = (ch == expected);
    if (!ok) std::cerr << "    changed=" << ch << " esperado=" << expected << "\n";
    return ok;
}

// ============================================================
// TEST 6 — Conteo exacto de grupos emitidos
// ============================================================
static bool test_a_output_count() {
    const int N = 16;
    hls::stream<blend_vec_t>    b_in;
    hls::stream<pgroup_t>       p_in;
    hls::stream<result_group_t> r_out;
    int ch = 0;

    for (int i = 0; i < N; i++) b_in.write(make_blend_winner(0));
    for (int i = 0; i < N/PPP; i++) {
        uint8_t pk_cls[PPP] = {0,0,0,0};
        p_in.write(make_pk_group(pk_cls));
    }

    stage_argmax_and_count(b_in, p_in, r_out, ch, N, 1);

    int expected_groups = N/PPP;
    bool ok = ((int)r_out.size() == expected_groups);
    if (!ok)
        std::cerr << "    grupos=" << r_out.size()
                  << " esperado=" << expected_groups << "\n";
    while (!r_out.empty()) r_out.read();
    return ok;
}

// ============================================================
// TEST 7 — Empaquetado correcto: PPP índices distintos en grupo
// ============================================================
static bool test_a_packing() {
    hls::stream<blend_vec_t>    b_in;
    hls::stream<pgroup_t>       p_in;
    hls::stream<result_group_t> r_out;
    int ch = 0;

    // Cada píxel del grupo tiene un ganador distinto: 0,1,2,3
    b_in.write(make_blend_winner(0));
    b_in.write(make_blend_winner(1));
    b_in.write(make_blend_winner(2));
    b_in.write(make_blend_winner(3));

    uint8_t pk_cls[PPP] = {15,15,15,15};
    p_in.write(make_pk_group(pk_cls));

    stage_argmax_and_count(b_in, p_in, r_out, ch, PPP, 1);

    result_group_t g = r_out.read();
    bool ok = true;
    for (int p = 0; p < PPP; p++) {
        if (get_result(g,p) != p) {
            std::cerr << "    p=" << p << " got=" << (int)get_result(g,p)
                      << " esperado=" << p << "\n";
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// MAIN
// ============================================================
int main_argmax_and_count() {
    std::cout << "==========================================\n";
    std::cout << "  Testbench: stage_argmax_and_count\n";
    std::cout << "==========================================\n\n";

    report_a("T01 - Ganador correcto (domina claramente)", test_a_correct_winner());
    report_a("T02 - Empate: gana menor índice",            test_a_tie_lower_wins());
    report_a("T03 - changed_count=0 sin cambios",          test_a_no_changes());
    report_a("T04 - changed_count=N todos cambian",        test_a_all_changes());
    report_a("T05 - changed_count con mezcla",             test_a_partial_changes());
    report_a("T06 - Conteo exacto de grupos",              test_a_output_count());
    report_a("T07 - Empaquetado correcto PPP índices",     test_a_packing());

    std::cout << "\n==========================================\n";
    std::cout << "  Resultado: " << tests_passed_a << "/"
              << tests_run_a << " tests pasaron\n";
    if (tests_failed_a > 0)
        std::cout << "  FALLARON: " << tests_failed_a << " tests\n";
    else
        std::cout << "  Todos los tests PASARON\n";
    std::cout << "==========================================\n";
    return tests_failed_a > 0 ? 1 : 0;
}