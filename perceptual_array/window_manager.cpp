#include "perceptual_array.hpp"

// G_DELAY: grupos de retraso antes de emitir (ceil(HALF_WIN/PPP) = ceil(3/1) = 3)
// SLIDE_W: columnas en el buffer deslizante = (G_DELAY+1)*PPP + HALF_WIN = 7
// Con PPP=1, SLIDE_W = WIN_SIZE: el slide ES exactamente la ventana.
// Invariante: tras insertar grupo g, slide[r][c] contiene el pixel en
//   abs_col = g*1 + c - (SLIDE_W-1) = g + c - 6
//   abs_row = y - (WIN_SIZE-1-r)
// La ventana para pixel p del grupo g-G_DELAY usa slide[wy][p+wx].
#define G_DELAY  3
#define SLIDE_W  ((G_DELAY + 1) * PPP + HALF_WIN)   // = 7

void stage_window_manager(
    hls::stream<pgroup_t>& pk_in,
    hls::stream<window_t>& win_out,
    int width, int height) {

    // Sintesis: static → BRAM persistente entre frames.
    // Csim:     local  → se reinicializa en cada llamada (tests independientes).
#ifdef __SYNTHESIS__
    static ap_uint<BITS_CLASS> line_buf[WIN_SIZE-1][MAX_WIDTH/PPP][PPP];
#else
    ap_uint<BITS_CLASS> line_buf[WIN_SIZE-1][MAX_WIDTH/PPP][PPP] = {};
#endif
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    // dim=3 tiene tamaño PPP=1 — partición innecesaria

    // Buffer deslizante: [WIN_SIZE][SLIDE_W] completamente en registros.
    ap_uint<BITS_CLASS> slide[WIN_SIZE][SLIDE_W];
    #pragma HLS ARRAY_PARTITION variable=slide complete

    // Columna post-shift guardada en registros para eliminar la dependencia BRAM
    // en la transición borde-horizontal (g=GROUPS-1 escribe line_buf, g=GROUPS lee).
    // saved_col[r][p] = line_buf[r][g][p] después del shift (actualizado cada g válido).
    ap_uint<BITS_CLASS> saved_col[WIN_SIZE-1][PPP];
    #pragma HLS ARRAY_PARTITION variable=saved_col complete

    for (int r = 0; r < WIN_SIZE-1; r++) {
        #pragma HLS UNROLL
        for (int p = 0; p < PPP; p++) {
            #pragma HLS UNROLL
            saved_col[r][p] = 0;
        }
    }

    for (int r = 0; r < WIN_SIZE; r++) {
        #pragma HLS UNROLL
        for (int c = 0; c < SLIDE_W; c++) {
            #pragma HLS UNROLL
            slide[r][c] = 0;
        }
    }

    const int GROUPS = width / PPP;

    Row_Loop: for (int y = 0; y < height + HALF_WIN; y++) {
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT max=2160+3

        Col_Loop: for (int g = 0; g < GROUPS + G_DELAY; g++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT max=MAX_WIDTH/PPP+G_DELAY

            const bool vert_border  = (y >= height);
            const bool horiz_border = (g >= GROUPS);

            // ── Paso 1: leer pixeles o replicar borde ──────────────────
            ap_uint<BITS_CLASS> new_pixels[PPP];
            #pragma HLS ARRAY_PARTITION variable=new_pixels complete

            if (!vert_border && !horiz_border) {
                pgroup_t grp = pk_in.read();
                for (int p = 0; p < PPP; p++) {
                    #pragma HLS UNROLL
                    new_pixels[p] = grp.range(p*BITS_CLASS + BITS_CLASS-1, p*BITS_CLASS);
                }
            } else if (horiz_border) {
                // Borde horizontal: usar registro saved_col (evita dependencia BRAM).
                // saved_col[WIN_SIZE-2][PPP-1] = new_pixels del ultimo grupo valido.
                for (int p = 0; p < PPP; p++) {
                    #pragma HLS UNROLL
                    new_pixels[p] = saved_col[WIN_SIZE-2][PPP-1];
                }
            } else {
                // Borde vertical (y >= height), g valido: leer line_buf normalmente.
                // sg=g → no hay conflicto inter-iteracion (direcciones distintas).
                for (int p = 0; p < PPP; p++) {
                    #pragma HLS UNROLL
                    new_pixels[p] = line_buf[WIN_SIZE-2][g][p];
                }
            }

            // ── Paso 2: construir columna completa (WIN_SIZE filas) ────
            ap_uint<BITS_CLASS> new_col[PPP][WIN_SIZE];
            #pragma HLS ARRAY_PARTITION variable=new_col complete

            {
                for (int p = 0; p < PPP; p++) {
                    #pragma HLS UNROLL
                    if (horiz_border) {
                        // Usar registros saved_col (sin acceso a BRAM) para evitar
                        // la dependencia llevada en la transicion GROUPS-1 → GROUPS.
                        for (int r = 0; r < WIN_SIZE-1; r++) {
                            #pragma HLS UNROLL
                            new_col[p][r] = saved_col[r][PPP-1];
                        }
                    } else {
                        for (int r = 0; r < WIN_SIZE-1; r++) {
                            #pragma HLS UNROLL
                            new_col[p][r] = line_buf[r][g][p];
                        }
                    }
                    new_col[p][WIN_SIZE-1] = new_pixels[p];
                }
            }

            // ── Actualizar saved_col con la columna post-shift actual ──
            // Necesario para que las iteraciones de borde horizontal no lean BRAM.
            // Post-shift row r = pre-shift row r+1 = new_col[p][r+1] (ya leido arriba).
            if (!horiz_border) {
                for (int r = 0; r < WIN_SIZE-2; r++) {
                    #pragma HLS UNROLL
                    for (int p = 0; p < PPP; p++) {
                        #pragma HLS UNROLL
                        saved_col[r][p] = new_col[p][r+1];
                    }
                }
                for (int p = 0; p < PPP; p++) {
                    #pragma HLS UNROLL
                    saved_col[WIN_SIZE-2][p] = new_pixels[p];
                }
            }

            // ── Paso 3: desplazar slide y insertar nueva columna ───────
            for (int r = 0; r < WIN_SIZE; r++) {
                #pragma HLS UNROLL
                for (int c = 0; c < SLIDE_W - PPP; c++) {
                    #pragma HLS UNROLL
                    slide[r][c] = slide[r][c + PPP];
                }
                for (int p = 0; p < PPP; p++) {
                    #pragma HLS UNROLL
                    slide[r][SLIDE_W - PPP + p] = new_col[p][r];
                }
            }

            // ── Paso 4: emitir PPP ventanas ────────────────────────────
            if (y >= HALF_WIN && g >= G_DELAY) {
                for (int p = 0; p < PPP; p++) {
                    #pragma HLS UNROLL
                    window_t win = 0;
                    for (int wy = 0; wy < WIN_SIZE; wy++) {
                        #pragma HLS UNROLL
                        for (int wx = 0; wx < WIN_SIZE; wx++) {
                            #pragma HLS UNROLL
                            int bit_idx = (wy * WIN_SIZE + wx) * BITS_CLASS;
                            win.range(bit_idx + BITS_CLASS-1, bit_idx) = slide[wy][p + wx];
                        }
                    }
                    win_out.write(win);
                }
            }

            // ── Paso 5: actualizar line_buf ────────────────────────────
            // Se actualiza para todos los grupos validos, incluyendo y >= height
            // (con new_pixels ya cargado con replicacion de borde inferior).
            if (!horiz_border) {
                for (int r = 0; r < WIN_SIZE-2; r++) {
                    #pragma HLS UNROLL
                    for (int p = 0; p < PPP; p++) {
                        #pragma HLS UNROLL
                        line_buf[r][g][p] = line_buf[r+1][g][p];
                    }
                }
                for (int p = 0; p < PPP; p++) {
                    #pragma HLS UNROLL
                    line_buf[WIN_SIZE-2][g][p] = new_pixels[p];
                }
            }
        }
    }
}
