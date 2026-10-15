"""
Corre TODAS las mediciones y genera las gráficas.

    python bench/ejecutar_todo.py                 # configuración completa
    python bench/ejecutar_todo.py --rapido        # prueba corta (~1 min)

Pasos:
  1. benchmark_hilos.py   -> resultados/hilos.csv, gens.json, hardware.json
  2. benchmark_mpi.py     -> resultados/mpi.csv (una corrida de mpiexec por P)
  3. experimento_gil.py   -> resultados/gil.csv
  4. sobrecosto_sincronizacion.py -> resultados/sincronizacion.csv
  5. demo_condicion_carrera.py -> resultados/carrera.csv
  6. graficas.py          -> resultados/figuras/*.pdf y *.png

Consejo: cierre otros programas (navegador, etc.) y conecte el portátil a la
corriente antes de medir; el modo de ahorro de energía distorsiona los tiempos.
"""
import argparse
import os
import shutil
import subprocess
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def correr(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tamanos", type=int, nargs="+", default=[100, 500, 1000, 2000, 4000])
    ap.add_argument("--trabajadores", type=int, nargs="+", default=None,
                    help="hilos/procesos a probar (por defecto 1, 2, 4, 6, 8, 12... hasta los núcleos lógicos)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--sin-mpi", action="store_true")
    a = ap.parse_args()

    n_cpu = os.cpu_count() or 4
    trab = a.trabajadores or sorted({k for k in (1, 2, 4, 6, 8, 12, 16) if k <= n_cpu} | {n_cpu})
    reps, tamanos, objetivo = a.reps, a.tamanos, "0.6"
    if a.rapido:
        reps, tamanos, objetivo = 2, [100, 500, 1000], "0.2"

    t = [str(x) for x in trab]
    correr([PY, os.path.join(AQUI, "benchmark_hilos.py"), "--tamanos", *map(str, tamanos),
            "--hilos", *t, "--reps", str(reps), "--objetivo", objetivo])

    if not a.sin_mpi:
        mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
        if mpiexec is None:
            print("\n[aviso] No se encontró mpiexec: se omite la medición MPI.")
        else:
            version = subprocess.run([mpiexec, "--version"], capture_output=True, text=True).stdout
            extra = ["--oversubscribe"] if "Open MPI" in version or "OpenRTE" in version else []
            for k, P in enumerate(trab):
                cmd = [mpiexec, *extra, "-n", str(P), PY, os.path.join(AQUI, "benchmark_mpi.py"),
                       "--reps", str(reps)]
                correr(cmd + (["--nuevo"] if k == 0 else []))

    correr([PY, os.path.join(AQUI, "experimento_gil.py"), "--hilos", *[x for x in t if int(x) <= 8]])
    correr([PY, os.path.join(AQUI, "sobrecosto_sincronizacion.py"), "--hilos", *t])
    correr([PY, os.path.join(AQUI, "demo_condicion_carrera.py"), "--hilos", str(min(4, n_cpu))])
    correr([PY, os.path.join(AQUI, "graficas.py")])
    print("\nListo. Resultados en resultados/ y figuras en resultados/figuras/")


if __name__ == "__main__":
    main()
