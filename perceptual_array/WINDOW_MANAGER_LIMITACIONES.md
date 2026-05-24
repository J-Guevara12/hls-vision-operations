# Limitaciones conocidas del stage_window_manager

**Módulo:** `perceptual_array/window_manager.cpp`  
**Implementación:** sliding-column buffer (`slide[WIN_SIZE][SLIDE_W]`)  
**Parámetros de referencia:** WIN_SIZE=7, HALF_WIN=3, PPP=1, MAX_WIDTH=4096

---

## Resumen

El `stage_window_manager` produce ventanas de WIN_SIZE×WIN_SIZE **exactas para todos
los píxeles interiores** de la imagen. Existen cuatro zonas de borde donde el
comportamiento se desvía de la replicación ideal (clamp-to-edge). Todas son
heredadas del diseño original o consecuencia directa del enfoque de slide-column.

---

## Limitación 1 — Borde superior frío (primer frame)

| Campo | Detalle |
|---|---|
| **Zona afectada** | Filas 0 .. HALF_WIN-1 del primer frame (3 filas) |
| **Síntoma** | `win[wy][wx]` con `row + wy - HALF_WIN < 0` retorna 0 en lugar de `img[0][col]` |
| **Impacto en hardware** | Solo en el primer frame desde reset. Frames posteriores usan datos reales de la fila final del frame anterior como warm-up |
| **Impacto en algoritmo** | Franja de 3 píxeles en la parte superior del primer frame con valores de ventana incorrectos. Para video continuo es prácticamente invisible |

### Causa

`slide` es un array local que se inicializa a 0 en cada llamada a
`stage_window_manager`. Las posiciones que representan filas con `abs_row < 0`
(borde superior) nunca son escritas con datos reales; permanecen en 0.

`line_buf` es `static` en síntesis (persistente entre frames) pero en la primera
llamada también parte de 0 por el comportamiento de inicialización de BRAM en
Xilinx (valor por defecto = 0).

### Posibles mitigaciones

- **Pre-cargar el primer frame con la primera fila duplicada** (software): enviar
  HALF_WIN filas extra al inicio del stream con el contenido de `img[0]` antes
  del frame real. Costo: HALF_WIN ciclos de latencia adicional al arrancar.
- **Aceptarlo** (recomendado): para video de 30 fps a 200 MHz, el primer frame
  dura ~33 ms y el warm-up representa <0.1% del tiempo total de operación.

---

## Limitación 2 — Borde izquierdo (todas las filas excepto la primera del frame)

| Campo | Detalle |
|---|---|
| **Zona afectada** | `col < HALF_WIN` → posiciones `wx` donde `col + wx - HALF_WIN < 0` |
| **Síntoma** | En lugar de replicar `img[row][0]`, las posiciones de ventana con `abs_col < 0` contienen datos del **borde derecho de la fila anterior** |
| **Impacto en hardware** | Franja de 3 píxeles en el borde izquierdo con valores cruzados entre filas |
| **Impacto en algoritmo** | El `perceptual_array` clasifica píxeles; los 3 píxeles del borde izquierdo recibirán una clasificación incorrecta derivada de la última columna de la fila anterior |

### Causa

El slide-buffer es una ventana deslizante de columnas. Al terminar cada fila, se
procesan G_DELAY=3 grupos de borde derecho (g = GROUPS..GROUPS+G_DELAY-1) que
insertan datos del **último píxel** de la imagen en las posiciones altas del slide
(`[SLIDE_W-1]`). Debido al tamaño del slide y al número de
desplazamientos hasta la siguiente fila, esos datos llegan a las posiciones bajas
(`[0..2]`) exactamente cuando se emite el primer grupo de salida de la fila siguiente
— que corresponde a `abs_col = -3..-1` (borde izquierdo).

El diseño original tenía el mismo problema de forma encubierta (los cachés
`lb_cache_left`, `lb_cache2`, `lb_cache3` también partían de 0 y no recibían
actualización especial para el borde izquierdo).

### Posibles mitigaciones

- **Registro `left_col_save[WIN_SIZE]`**: guardar la columna 0 al procesarla
  (g=0) y usarla explícitamente en el paso de salida cuando `p + wx < HALF_WIN -
  (g - G_DELAY)*PPP`. Coste: WIN_SIZE registros extra + MUX en el paso de salida.
- **Aumentar SLIDE_W a `G_DELAY*PPP + HALF_WIN + G_DELAY*PPP` = 9**:
  matemáticamente los datos del borde derecho no alcanzan la posición [0] antes
  del primer output. Coste: mayor latencia horizontal (+2 píxeles) y slide más
  grande (+2×WIN_SIZE=14 bits de FFs).
- **Zeroing explícito al inicio de cada fila**: resetear `slide[r][0..HALF_WIN-1]`
  a 0 en la primera iteración de cada fila. Coste: rompe el II=1 de la primera
  iteración si no se maneja con doble-buffer.
- **Aceptarlo**: para el caso de uso actual (tracking perceptual de clases en video)
  los 6 píxeles del borde izquierdo son estadísticamente irrelevantes.

---

## Limitación 3 — Borde derecho: off-by-one de fila

| Campo | Detalle |
|---|---|
| **Zona afectada** | `col + HALF_WIN >= width` → posiciones `wx` donde `col + wx - HALF_WIN >= width` |
| **Síntoma** | Las posiciones con `abs_src_col >= width` contienen `img[row+1][width-1]` en lugar de `img[row][width-1]` (1 fila desplazada) |
| **Impacto en hardware** | Franja de HALF_WIN=3 píxeles en el borde derecho |
| **Impacto en algoritmo** | Igual que borde izquierdo: clasificación incorrecta en los 3 píxeles del borde derecho |

### Causa

Para grupos de borde derecho (`g >= GROUPS`), el paso 2 lee
`line_buf[r][GROUPS-1][PPP-1]` para construir `new_col`. Sin embargo, el paso 5
ya actualizó `line_buf[*][GROUPS-1]` en la iteración `g = GROUPS-1` de la misma
fila, insertando los datos de la fila **actual** (`img[y]`) y desplazando todo
hacia arriba. Al leer en `g = GROUPS`, `line_buf[6][GROUPS-1][PPP-1]` contiene
`img[y - (WIN_SIZE-2-6)] = img[y-5]` en lugar del esperado `img[y-6] = img[row]`.

El paso 5 actualiza la columna `g` con los datos del pixel `g` de la fila actual.
Para `g < GROUPS` esto es correcto porque step-2 lee su propia columna antes de
que step-5 la actualice. Pero para borde derecho (`g >= GROUPS`) se lee
`GROUPS-1`, una columna que ya fue actualizada en `g = GROUPS-1`.

### Posibles mitigaciones

- **Registro `right_col_save[WIN_SIZE-1]`**: en step-2 cuando `g == GROUPS-1`,
  guardar `new_col[PPP-1][r]` (que es `line_buf[r][GROUPS-1][PPP-1]` antes de
  step-5). Usarlo en lugar de `line_buf[r][GROUPS-1][PPP-1]` para iteraciones
  `g >= GROUPS`. Coste: WIN_SIZE-1 registros extra, lógica condicional mínima.
  Esta es la **mitigación más limpia y de menor coste**.

---

## Limitación 4 — Borde inferior (sin impacto)

El borde inferior (`row + HALF_WIN >= height`) **funciona correctamente**. El
paso 1 replica explícitamente la última fila cuando `y >= height` y el paso 5
(sin restricción `y < height` desde la corrección) propaga esa replicación por el
`line_buf`, llenándolo con los datos de la última fila antes de que las ventanas
de drenaje sean emitidas.

---

## Tabla resumen

| Borde | Zona | Comportamiento real | Comportamiento ideal | Impacto |
|---|---|---|---|---|
| Superior | Filas 0..2 del primer frame | 0 | `img[0][col]` | Cosmético (solo primer frame) |
| Izquierdo | Columnas 0..2, todas las filas | Último píxel fila anterior | `img[row][0]` | 3 px borde izq. incorrectos |
| Derecho | Columnas W-3..W-1, todas las filas | `img[row+1][W-1]` | `img[row][W-1]` | 3 px borde der. incorrectos |
| Inferior | Filas H-6..H-1 | `img[H-1][col]` ✓ | `img[H-1][col]` | Ninguno |

---

## Contexto: comparación con el diseño original

El `stage_window_manager` original (basado en caches de columna `lb_cache_curr`,
`lb_cache2`, `lb_cache3`) presentaba las mismas limitaciones de borde izquierdo
y superior (inicializados a 0), más una adicional: el borde derecho de cada
ventana (`group_diff >= 2`) usaba una **aproximación** que retornaba datos de la
fila anterior en lugar de la fila correcta. El testbench original excluía
explícitamente esas posiciones con `if (group_diff >= 2) continue`.

La nueva implementación **elimina la aproximación de `group_diff >= 2`** para
todos los píxeles interiores, al coste de introducir el off-by-one de borde
derecho (limitación 3), que es más acotado y mitigable.

---

## Tests afectados

| Test | Cobertura | Limitaciones excluidas |
|---|---|---|
| T01 — Conteo de ventanas | Completo (W×H) | Ninguna |
| T02 — Imagen uniforme | Interior + borde derecho/inferior | abs_row < 0, abs_col < 0 |
| T03 — Centro de ventana | Todos los píxeles | — |
| T04 — Zona interior | col ∈ [HALF_WIN, W-HALF_WIN-1], row ídem | Todos los bordes |
| T05 — Golden model interior | Mismo recorte que T04, ventana completa | Todos los bordes |
