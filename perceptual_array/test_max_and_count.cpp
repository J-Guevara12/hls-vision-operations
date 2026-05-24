// ============================================================
// Testbench: test_argmax_and_count.cpp
// Componente bajo prueba: stage_argmax_and_count
//
// Invariantes (PPP=1: 1 pixel por iteración):
//   - Ganador correcto cuando una clase domina claramente
//   - Empate: gana la clase de menor índice (0 >= 1 → 0 gana)
//   - changed_count=0 cuando Pk+1 == Pk en todos los píxeles
//   - changed_count=N cuando todos los píxeles cambian
//   - changed_count correcto con mezcla de cambios y no cambios
//   - Conteo exacto de resultados emitidos (N por N píxeles)
//   - Ganadores distintos en píxeles consecutivos
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

static blend_vec_t make_blend_winner(int winner, uint8_t win_val = 200,
                                      uint8_t rest_val = 10) {
    blend_vec_t v = 0;
    for (int c = 0; c < NUM_CLASSES; c++)
        v.range(c*8+7, c*8) = (c == winner) ? win_val : rest_val;
    return v;
}

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
    bool ok = true;
    for (int target = 0; target < NUM_CLASSES; target++) {
        hls::stream<blend_vec_t>    b_in;
        hls::stream<pgroup_t>       p_in;
        hls::stream<result_group_t> r_out;
        int ch = 0;

        b_in.write(make_blend_winner(target));
        uint8_t pk_cls[PPP] = {0};
        p_in.write(make_pk_group(pk_cls));

        stage_argmax_and_count(b_in, p_in, r_out, ch, 1, 1);

        result_group_t g = r_out.read();
        if (get_result(g, 0) != (uint8_t)target) {
            std::cerr << "    target=" << target
                      << " got=" << (int)get_result(g, 0) << "\n";
            ok = false;
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

    b_in.write(make_blend_tie(100));
    uint8_t pk_cls[PPP] = {15};
    p_in.write(make_pk_group(pk_cls));

    stage_argmax_and_count(b_in, p_in, r_out, ch, 1, 1);

    result_group_t g = r_out.read();
    uint8_t winner = get_result(g, 0);
    bool ok = (winner == 0);
    if (!ok)
        std::cerr << "    Empate: winner=" << (int)winner << " esperado=0\n";
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

    for (int i = 0; i < N; i++) {
        b_in.write(make_blend_winner(5));
        uint8_t pk_cls[PPP] = {5};
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

    for (int i = 0; i < N; i++) {
        b_in.write(make_blend_winner(3));
        uint8_t pk_cls[PPP] = {7};
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

    // Píxeles pares: winner=2, pk=2 (no cambia)
    // Píxeles impares: winner=3, pk=7 (cambia)
    for (int i = 0; i < N; i++) {
        int  winner = (i % 2 == 0) ? 2 : 3;
        uint8_t pk  = (i % 2 == 0) ? 2 : 7;
        b_in.write(make_blend_winner(winner));
        uint8_t pk_cls[PPP] = {pk};
        p_in.write(make_pk_group(pk_cls));
    }

    stage_argmax_and_count(b_in, p_in, r_out, ch, N, 1);

    while (!r_out.empty()) r_out.read();
    int expected = N / 2;
    bool ok = (ch == expected);
    if (!ok) std::cerr << "    changed=" << ch << " esperado=" << expected << "\n";
    return ok;
}

// ============================================================
// TEST 6 — Conteo exacto de resultados emitidos
// ============================================================
static bool test_a_output_count() {
    const int N = 16;
    hls::stream<blend_vec_t>    b_in;
    hls::stream<pgroup_t>       p_in;
    hls::stream<result_group_t> r_out;
    int ch = 0;

    for (int i = 0; i < N; i++) {
        b_in.write(make_blend_winner(0));
        uint8_t pk_cls[PPP] = {0};
        p_in.write(make_pk_group(pk_cls));
    }

    stage_argmax_and_count(b_in, p_in, r_out, ch, N, 1);

    bool ok = ((int)r_out.size() == N);
    if (!ok)
        std::cerr << "    resultados=" << r_out.size() << " esperado=" << N << "\n";
    while (!r_out.empty()) r_out.read();
    return ok;
}

// ============================================================
// TEST 7 — Ganadores distintos en 4 píxeles consecutivos
// ============================================================
static bool test_a_consecutive_winners() {
    hls::stream<blend_vec_t>    b_in;
    hls::stream<pgroup_t>       p_in;
    hls::stream<result_group_t> r_out;
    int ch = 0;

    // 4 píxeles con ganadores 0,1,2,3; todos Pk=15 (cambio esperado)
    for (int w = 0; w < 4; w++) {
        b_in.write(make_blend_winner(w));
        uint8_t pk_cls[PPP] = {15};
        p_in.write(make_pk_group(pk_cls));
    }

    stage_argmax_and_count(b_in, p_in, r_out, ch, 4, 1);

    bool ok = true;
    for (int w = 0; w < 4; w++) {
        result_group_t g = r_out.read();
        if (get_result(g, 0) != (uint8_t)w) {
            std::cerr << "    pixel=" << w << " got=" << (int)get_result(g,0)
                      << " esperado=" << w << "\n";
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
    report_a("T06 - Conteo exacto de resultados",          test_a_output_count());
    report_a("T07 - Ganadores consecutivos correctos",     test_a_consecutive_winners());

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
