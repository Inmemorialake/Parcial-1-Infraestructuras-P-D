"""
Experimento: ¿por qué los hilos de Python NO siempre aceleran? (el GIL)

Ejecuta la misma versión con hilos (planificación estática) con tres kernels:
  - python: bucle en Python puro  -> el GIL serializa los hilos.
  - numpy:  operaciones vectorizadas; NumPy suelta el GIL dentro de cada
            operación, pero lo retoma entre operaciones.
  - numba:  bucle compilado con nogil=True -> paralelismo real.

    python bench/experimento_gil.py --tamano 1000 --hilos 1 2 4
"""
import argparse
import os
import statistics
import time

import comun
from vida import MotorHilos, MotorSecuencial, patrones

GENS = {"python": 3, "numpy": 60, "numba": 60}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tamano", type=int, default=1000)
    ap.add_argument("--hilos", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    tablero = patrones.aleatorio(a.tamano, semilla=1)
    filas = []
    for kernel, gens in GENS.items():
        def tiempo(fabrica):
            ts = []
            for _ in range(a.reps):
                with fabrica() as m:
                    m.paso(1)
                    t0 = time.perf_counter()
                    m.paso(gens)
                    ts.append(time.perf_counter() - t0)
            return statistics.median(ts) / gens * 1000
        base = tiempo(lambda: MotorSecuencial(tablero, kernel=kernel))
        filas.append(dict(kernel=kernel, hilos=0, ms_gen=base))
        print(f"{kernel:6s} secuencial: {base:9.3f} ms/gen")
        for T in a.hilos:
            ms = tiempo(lambda: MotorHilos(tablero, T, kernel=kernel))
            filas.append(dict(kernel=kernel, hilos=T, ms_gen=ms))
            print(f"{kernel:6s} hilos={T}:    {ms:9.3f} ms/gen   speedup {base / ms:4.2f}")
    ruta = os.path.join(comun.RESULTADOS, "gil.csv")
    os.makedirs(comun.RESULTADOS, exist_ok=True)
    with open(ruta, "w") as f:
        f.write("tamano,kernel,hilos,ms_gen\n")
        for r in filas:
            f.write(f"{a.tamano},{r['kernel']},{r['hilos']},{r['ms_gen']}\n")


if __name__ == "__main__":
    main()
