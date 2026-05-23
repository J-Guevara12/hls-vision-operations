#include "perceptual_array.hpp"

// ============================================================
// stage_window_manager
//
// Window Manager custom 13x13 para píxeles de BITS_CLASS bits.
// PPP=4 píxeles por grupo. Emite PPP ventanas por grupo —
// 1 ventana de WIN_SIZE×WIN_SIZE por píxel, en PPP ciclos consecutivos.
//
// Replicación de borde: los píxeles fuera del dominio de la
// imagen replican el píxel del borde más cercano.
//
// Almacenamiento:
//   line_buf: (WIN_SIZE-1) filas × (MAX_WIDTH/PPP) grupos × PPP píxeles
//   Cada píxel ocupa BITS_CLASS bits.
//
// Desfase de salida: igual que el WindowManager genérico,
// la ventana completa para el píxel (y,x) está disponible
// en el ciclo (y+HALF_WIN, x/PPP + HALF_WIN/PPP + 1).
//
// Cadena de cachés de columna (en iter g, construyendo ventanas de grupo g-1):
//   lb_curr      → datos del grupo g     (leído de line_buf al inicio)
//   lb_cache_curr → grupo g-1  (fila hist: lb_curr anterior; fila actual: pixels del iter g-1)
//   lb_cache2    → grupo g-2  (igual para el iter g-2)
//   lb_cache3    → grupo g-3  (igual para el iter g-3)
//
// Limitación conocida: eff_group = g+1 (borde derecho de ventana) usa datos de
// la fila anterior (mejor aproximación disponible sin aumentar el look-ahead).
// ============================================================

// Tipo interno para un píxel individual de la imagen
typedef ap_uint<BITS_CLASS> px_t;

void stage_window_manager(
    hls::stream<pgroup_t>& pk_in,
    hls::stream<window_t>& win_out,
    int width, int height) {

    static px_t line_buf[WIN_SIZE-1][MAX_WIDTH/PPP][PPP];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=3

    static px_t lb_cache_curr[WIN_SIZE-1][PPP];
    #pragma HLS ARRAY_PARTITION variable=lb_cache_curr complete

    static px_t lb_cache2[WIN_SIZE-1][PPP];
    #pragma HLS ARRAY_PARTITION variable=lb_cache2 complete

    static px_t lb_cache3[WIN_SIZE-1][PPP];
    #pragma HLS ARRAY_PARTITION variable=lb_cache3 complete

    const int GROUPS = width / PPP;

    // Inicializar cachés (line_buf persiste entre frames intencionalmente)
    for (int r = 0; r < WIN_SIZE-1; r++) {
        #pragma HLS UNROLL
        for (int p = 0; p < PPP; p++) {
            #pragma HLS UNROLL
            lb_cache_curr[r][p] = 0;
            lb_cache2[r][p]     = 0;
            lb_cache3[r][p]     = 0;
        }
    }

    // Pipeline principal: height+HALF_WIN filas para drenar la ventana
    Row_Loop: for (int y = 0; y < height + HALF_WIN; y++) {
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS LOOP_TRIPCOUNT max=2160+6

        Col_Loop: for (int g = 0; g <= GROUPS + HALF_WIN/PPP; g++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT max=MAX_WIDTH/PPP+2

            // Leer píxeles del stream — con replicación vertical
            px_t pixels[PPP];
            #pragma HLS ARRAY_PARTITION variable=pixels complete

            if (y < height && g < GROUPS) {
                pgroup_t grp = pk_in.read();
                for (int p = 0; p < PPP; p++) {
                    #pragma HLS UNROLL
                    pixels[p] = grp.range(p*BITS_CLASS+BITS_CLASS-1, p*BITS_CLASS);
                }
            } else if (y >= height) {
                // Replicación vertical inferior: repetir última fila
                for (int p = 0; p < PPP; p++) {
                    #pragma HLS UNROLL
                    pixels[p] = line_buf[WIN_SIZE-2][g < GROUPS ? g : GROUPS-1][p];
                }
            } else {
                // g >= GROUPS: replicación horizontal derecha
                for (int p = 0; p < PPP; p++) {
                    #pragma HLS UNROLL
                    pixels[p] = line_buf[0][GROUPS-1][PPP-1];
                }
            }

            // Leer lb_curr del grupo actual (datos históricos de columna g)
            px_t lb_curr[WIN_SIZE-1][PPP];
            #pragma HLS ARRAY_PARTITION variable=lb_curr complete
            for (int r = 0; r < WIN_SIZE-1; r++) {
                #pragma HLS UNROLL
                for (int p = 0; p < PPP; p++) {
                    #pragma HLS UNROLL
                    lb_curr[r][p] = line_buf[r][g < GROUPS ? g : GROUPS-1][p];
                }
            }

            // Construir y emitir PPP ventanas (una por píxel del grupo)
            // Ventana del grupo anterior (desfase de 1 grupo)
            if (y >= HALF_WIN && g >= 1 && g <= GROUPS) {
                int abs_col_base = (g-1) * PPP;
                int abs_row = y - HALF_WIN;  // fila de imagen correspondiente

                for (int p = 0; p < PPP; p++) {
                    #pragma HLS UNROLL
                    window_t win = 0;
                    int abs_col = abs_col_base + p;

                    for (int wy = 0; wy < WIN_SIZE; wy++) {
                        #pragma HLS UNROLL
                        for (int wx = 0; wx < WIN_SIZE; wx++) {
                            #pragma HLS UNROLL
                            px_t val;
                            int src_col = abs_col + wx - HALF_WIN;
                            int src_row = abs_row + wy - HALF_WIN;

                            // Replicación vertical
                            int eff_row = src_row;
                            if (eff_row < 0)       eff_row = 0;
                            if (eff_row >= height)  eff_row = height-1;

                            // Replicación horizontal
                            int eff_col = src_col;
                            if (eff_col < 0)      eff_col = 0;
                            if (eff_col >= width)  eff_col = width-1;

                            int rel_row = eff_row - abs_row + HALF_WIN;
                            // rel_row: 0=top de ventana, WIN_SIZE-1=bottom (fila actual)

                            int rel_col_in_group = eff_col % PPP;
                            int eff_group = eff_col / PPP;
                            // group_diff: desplazamiento respecto al grupo central (g-1)
                            int group_diff = eff_group - (g - 1);

                            if (rel_row == WIN_SIZE-1) {
                                // Fila actual (row y):
                                //   lb_cache_curr[WIN_SIZE-2] = pixels del iter g-1 ✓
                                //   lb_cache2[WIN_SIZE-2]     = pixels del iter g-2 ✓
                                //   lb_cache3[WIN_SIZE-2]     = pixels del iter g-3 ✓
                                //   pixels[]                  = grupo g actual       ✓
                                //   lb_curr[WIN_SIZE-2]       = fila anterior (aprox. para g+1)
                                if (group_diff <= -2) {
                                    val = lb_cache3[WIN_SIZE-2][rel_col_in_group];
                                } else if (group_diff == -1) {
                                    val = lb_cache2[WIN_SIZE-2][rel_col_in_group];
                                } else if (group_diff == 0) {
                                    val = lb_cache_curr[WIN_SIZE-2][rel_col_in_group];
                                } else {
                                    // group_diff == 1 (grupo g): correcto
                                    // group_diff >= 2 (grupo g+1): approx. con pixels[g]
                                    val = pixels[rel_col_in_group];
                                }
                            } else {
                                // Filas históricas: cachés contienen los datos correctos
                                int buf_row = rel_row;
                                if (group_diff <= -2) {
                                    val = lb_cache3[buf_row][rel_col_in_group];
                                } else if (group_diff == -1) {
                                    val = lb_cache2[buf_row][rel_col_in_group];
                                } else if (group_diff == 0) {
                                    val = lb_cache_curr[buf_row][rel_col_in_group];
                                } else {
                                    // group_diff == 1 (grupo g): correcto
                                    // group_diff >= 2 (grupo g+1): approx. con lb_curr[g]
                                    val = lb_curr[buf_row][rel_col_in_group];
                                }
                            }

                            int bit_idx = (wy * WIN_SIZE + wx) * BITS_CLASS;
                            win.range(bit_idx + BITS_CLASS-1, bit_idx) = val;
                        }
                    }
                    win_out.write(win);
                }
            }

            // Actualizar cadena de cachés DESPUÉS de emitir las ventanas.
            // Para la fila actual (r == WIN_SIZE-2) se captura pixels[] en lugar
            // de lb_curr[WIN_SIZE-2] (que aún tiene la fila anterior), de modo que
            // lb_cache_curr[WIN_SIZE-2] en el siguiente ciclo = fila actual del grupo g.
            for (int r = 0; r < WIN_SIZE-1; r++) {
                #pragma HLS UNROLL
                for (int p = 0; p < PPP; p++) {
                    #pragma HLS UNROLL
                    lb_cache3[r][p] = lb_cache2[r][p];
                    lb_cache2[r][p] = lb_cache_curr[r][p];
                    lb_cache_curr[r][p] = (r == WIN_SIZE-2) ? pixels[p] : lb_curr[r][p];
                }
            }

            // Actualizar line buffer
            for (int r = 0; r < WIN_SIZE-2; r++) {
                #pragma HLS UNROLL
                for (int p = 0; p < PPP; p++) {
                    #pragma HLS UNROLL
                    line_buf[r][g < GROUPS ? g : GROUPS-1][p] =
                        line_buf[r+1][g < GROUPS ? g : GROUPS-1][p];
                }
            }
            for (int p = 0; p < PPP; p++) {
                #pragma HLS UNROLL
                line_buf[WIN_SIZE-2][g < GROUPS ? g : GROUPS-1][p] = pixels[p];
            }
        }
    }
}
