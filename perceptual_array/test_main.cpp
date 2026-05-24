// ============================================================
// test_main.cpp — Testbench principal de perceptual_array
//
// Ejecuta todos los testbenches en orden, de más básico a
// integración. Si cualquier suite falla, aborta inmediatamente
// sin ejecutar las siguientes.
//
// Orden de ejecución:
//   1. stage_unpack          (desempaquetado de entrada)
//   2. stage_window_manager  (ventana deslizante 7x7)
//   3. stage_conv_engine     (convolución gaussiana)
//   4. stage_scale_and_blend (mezcla α)
//   5. stage_argmax_and_count(argmax + conteo de cambios)
//   6. stage_pack            (empaquetado de salida)
//   7. perceptual_array      (integración end-to-end)
// ============================================================

#include "perceptual_array.hpp"
#include "test_unpack.cpp"
#include "test_window_manager.cpp"
#include "test_conv_engine.cpp"
#include "test_scale_and_blend.cpp"
#include "test_max_and_count.cpp"
#include "test_pack.cpp"
#include "test_perceptual_array.cpp"

// Declaraciones (definidas en cada test_*.cpp)
int main_unpack();
int main_window_manager();
int main_conv_engine();
int main_scale_and_blend();
int main_argmax_and_count();
int main_pack();
int main_perceptual_array();

struct Suite {
    const char* name;
    int (*fn)();
};

int main() {
    static const Suite suites[] = {
        { "stage_unpack",           main_unpack           },
        { "stage_window_manager",   main_window_manager   },
        { "stage_conv_engine",      main_conv_engine      },
        { "stage_scale_and_blend",  main_scale_and_blend  },
        { "stage_argmax_and_count", main_argmax_and_count },
        { "stage_pack",             main_pack             },
        { "perceptual_array",       main_perceptual_array },
    };
    const int N = sizeof(suites) / sizeof(suites[0]);

    std::cout << "\n";
    std::cout << "##############################################\n";
    std::cout << "#   PERCEPTUAL ARRAY — TEST SUITE COMPLETO  #\n";
    std::cout << "##############################################\n\n";

    for (int i = 0; i < N; i++) {
        std::cout << ">>> [" << (i+1) << "/" << N << "] "
                  << suites[i].name << "\n\n";

        int rc = suites[i].fn();

        if (rc != 0) {
            std::cout << "\n";
            std::cout << "##############################################\n";
            std::cerr << "  ABORTANDO: fallos en suite ["
                      << suites[i].name << "]\n";
            std::cerr << "  Los stages dependientes no se ejecutaron.\n";
            std::cout << "##############################################\n";
            return rc;
        }
        std::cout << "\n";
    }

    std::cout << "##############################################\n";
    std::cout << "#   TODOS LOS TESTS PASARON                 #\n";
    std::cout << "##############################################\n";
    return 0;
}
