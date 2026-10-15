# Juego de la Vida paralelo — Parcial I (750023C)


Infraestructuras Paralelas y Distribuidas · Universidad del Valle · 2026-2

Juego de la Vida de Conway en cuatro versiones que comparten **el mismo kernel de cálculo**:

| Versión | Modelo | Archivo |
|---|---|---|
| Secuencial (línea base) | 1 hilo | `vida/motor_secuencial.py` |
| Hilos | memoria compartida: `threading`, barreras, doble buffer, planificación estática/dinámica | `vida/motor_hilos.py` |
| Hilos nativos (comparación) | fork-join nativo con `numba.prange` (estilo OpenMP) | `vida/motor_nativo.py` |
| MPI | memoria distribuida: `mpi4py`, anillo periódico, filas fantasma con `Sendrecv` | `vida/motor_mpi.py` |

Además incluye una interfaz gráfica (pygame) para todas las versiones, pruebas de correctitud, scripts de
medición que generan CSV y gráficas, y el informe en LaTeX, que toma las cifras directamente de los CSV.

## Instalación

**Opción recomendada (conda/mamba, incluye MPI):**

```bash
mamba env create -f environment.yml
mamba activate vida-paralela
```

**Con pip:** primero instale una implementación de MPI del sistema (Arch: `sudo pacman -S openmpi`;
Debian/Ubuntu: `sudo apt install libopenmpi-dev openmpi-bin`) y luego:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
python -m vida.gui                                  # GUI: cambie de modo con M, hilos con +/-
python -m vida.gui --modo hilos --hilos 8 --tamano 1000 --iniciar
mpiexec -n 4 python -m vida.gui_mpi --tamano 1000   # GUI de la versión MPI

python -m pytest -q                                 # 45 pruebas: paralelo == secuencial
python bench/demo_condicion_carrera.py              # qué pasa sin barrera / sin doble buffer
```

**Controles de la GUI:** `Espacio` iniciar/pausar · `N` un paso · `M` modo · `D` planificación ·
`+`/`-` hilos · `T` tamaño · `P` patrón · `R` reiniciar · `↑`/`↓` generaciones por cuadro ·
clic izquierdo/derecho en pausa para dibujar o borrar células. Para ver el speedup en vivo, deje correr
unos segundos en modo **Secuencial** y luego cambie a **Hilos**: el panel muestra la razón entre ambos.

## Mediciones e informe

```bash
python bench/ejecutar_todo.py          # 5 tamaños × {1,2,4,6,8,...} hilos/procesos × 5 reps (~5-10 min)
python bench/capturas.py               # capturas de la GUI para el informe
python informe/generar_datos.py        # CSV -> tablas y cifras LaTeX
cd informe && latexmk -pdf informe.tex # -> informe/informe.pdf
```

(o `make todo`). Antes de medir: portátil conectado a la corriente, perfil de energía en "rendimiento" y
otros programas cerrados.

Los resultados quedan en `resultados/` (`hilos.csv`, `mpi.csv`, `gil.csv`, `sincronizacion.csv`,
`carrera.csv`, `hardware.json`) y las gráficas en `resultados/figuras/`. El informe usa la macro
`\dato{...}` para todas las cifras, así que se actualiza solo. **Solo hay que releer los párrafos de la
sección "Análisis"** (marcados con `REVISAR` en `informe.tex`), porque interpretan las cifras.

Para editar el informe en equipo con **Overleaf**: suba la carpeta `informe/` junto con
`resultados/figuras/`, conservando la estructura de carpetas.

## Estructura

```
vida/            código del juego (kernels, motores, patrones, GUI)
bench/           mediciones, experimentos y gráficas
tests/           pruebas de correctitud (pytest) y verificación MPI
resultados/      CSV y figuras generadas
informe/         informe LaTeX (+ generar_datos.py)
SUSTENTACION.md  guion y preguntas probables para la sustentación oral
```
Fin del comunicado...
