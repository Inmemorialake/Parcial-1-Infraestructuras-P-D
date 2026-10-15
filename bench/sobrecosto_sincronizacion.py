"""
Mide el costo PURO de sincronizar una generación (kernel vacío, sin cálculo):

  * threading.Barrier en Python (MotorHilos con un kernel que no hace nada)
  * fork-join nativo de Numba (prange con cuerpo vacío)

Este costo se paga en CADA generación sin importar el tamaño del tablero, por
eso domina en tableros pequeños (ley de Amdahl: es una parte que no se reduce
al agregar hilos; de hecho crece).

    python bench/sobrecosto_sincronizacion.py --hilos 1 2 4 8
"""
import argparse
import os
import time

import numba
import numpy as np
from numba import njit, prange

import comun
from vida import kernels
from vida.motor_hilos import MotorHilos

kernels.KERNELS["vacio"] = lambda src, dst, r0, r1: None


@njit(parallel=True)
def _forkjoin_vacio(x, n_gens, n_bandas):
    for g in range(n_gens):
        for t in prange(n_bandas):
            x[t] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hilos", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--gens", type=int, default=3000)
    a = ap.parse_args()
    t = np.zeros((64, 64), np.uint8)
    filas = []
    for T in a.hilos:
        with MotorHilos(t, T, kernel="vacio") as m:
            m.paso(50)
            t0 = time.perf_counter()
            m.paso(a.gens)
            us_py = (time.perf_counter() - t0) / a.gens * 1e6
        Tn = min(T, numba.config.NUMBA_NUM_THREADS)
        numba.set_num_threads(Tn)
        x = np.zeros(max(a.hilos), np.int64)
        _forkjoin_vacio(x, 50, Tn)
        t0 = time.perf_counter()
        _forkjoin_vacio(x, a.gens, Tn)
        us_nat = (time.perf_counter() - t0) / a.gens * 1e6
        filas.append((T, us_py, us_nat))
        print(f"T={T:2d}: threading.Barrier {us_py:8.2f} µs/gen   ·   prange nativo {us_nat:6.2f} µs/gen")
    os.makedirs(comun.RESULTADOS, exist_ok=True)
    with open(os.path.join(comun.RESULTADOS, "sincronizacion.csv"), "w") as f:
        f.write("hilos,us_threading,us_nativo\n")
        for T, p, n in filas:
            f.write(f"{T},{p},{n}\n")


if __name__ == "__main__":
    main()
