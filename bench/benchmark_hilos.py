"""
Mide la versión secuencial y la versión con hilos.

    python bench/benchmark_hilos.py --tamanos 100 500 1000 2000 4000 --hilos 1 2 4 8 --reps 5

Para cada tamaño:
  1. Calibra el número de generaciones para que la corrida secuencial dure
     ~`--objetivo` segundos (así los tableros pequeños no se miden con ruido).
     Ese mismo número de generaciones se usa para TODAS las configuraciones del
     tamaño (y se guarda en resultados/gens.json para la versión MPI).
  2. Mide la secuencial, los hilos (planificación estática y dinámica) y la
     variante de hilos nativos (Numba prange).
  3. Verifica que cada resultado paralelo sea idéntico al secuencial.

La creación del pool de hilos NO se incluye en el tiempo (se hace una sola vez
por corrida); sí se incluye toda la sincronización por generación.
"""
import argparse
import os
import statistics
import time

import numpy as np

import comun
from vida import MotorHilos, MotorNativo, MotorSecuencial, patrones

CALENTAMIENTO = 3          # generaciones previas sin medir


def calibrar(tablero, objetivo):
    m = MotorSecuencial(tablero)
    m.paso(CALENTAMIENTO)
    t0 = time.perf_counter()
    k = 0
    while time.perf_counter() - t0 < 0.15:
        m.paso(1)
        k += 1
    por_gen = (time.perf_counter() - t0) / k
    return int(min(20000, max(20, objetivo / por_gen)))


def medir(motor, gens):
    motor.paso(CALENTAMIENTO)
    t0 = time.perf_counter()
    motor.paso(gens)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tamanos", type=int, nargs="+", default=[100, 500, 1000, 2000, 4000])
    ap.add_argument("--hilos", type=int, nargs="+", default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--objetivo", type=float, default=0.6,
                    help="segundos aproximados por corrida secuencial")
    ap.add_argument("--planificaciones", nargs="+", default=["estatica", "dinamica"])
    ap.add_argument("--sin-nativo", action="store_true", help="no medir la variante prange")
    ap.add_argument("--salida", default=os.path.join(comun.RESULTADOS, "hilos.csv"))
    a = ap.parse_args()

    n_cpu = os.cpu_count() or 4
    hilos = a.hilos or sorted({1, 2, 4, 6, 8, 12, 16, n_cpu} & set(range(1, n_cpu + 1)) | {n_cpu})

    comun.guardar_json(os.path.join(comun.RESULTADOS, "hardware.json"), comun.info_hardware())
    ruta_gens = os.path.join(comun.RESULTADOS, "gens.json")
    gens_por_tamano = {}
    primera = True

    for n in a.tamanos:
        tablero = patrones.aleatorio(n, semilla=n)
        gens = calibrar(tablero, a.objetivo)
        gens_por_tamano[str(n)] = gens
        ref = MotorSecuencial(tablero)
        ref.paso(CALENTAMIENTO + gens)
        referencia = ref.tablero.copy()

        filas = []
        tiempos = []
        for r in range(a.reps):
            m = MotorSecuencial(tablero)
            t = medir(m, gens)
            tiempos.append(t)
            filas.append(dict(implementacion="secuencial", tamano=n, trabajadores=1,
                              gens=gens, rep=r, tiempo_s=t, ms_gen=1000 * t / gens,
                              correcto=np.array_equal(m.tablero, referencia)))
        t_seq = statistics.median(tiempos)
        print(f"\n[{n}x{n}] {gens} generaciones · secuencial {1000 * t_seq / gens:.3f} ms/gen")

        for plan in a.planificaciones:
            for T in hilos:
                tiempos = []
                for r in range(a.reps):
                    with MotorHilos(tablero, T, plan) as m:
                        t = medir(m, gens)
                        comp = m.t_computo
                        ok = np.array_equal(m.tablero, referencia)
                    media = sum(comp) / len(comp)
                    tiempos.append(t)
                    filas.append(dict(implementacion=f"hilos_{plan}", tamano=n, trabajadores=T,
                                      gens=gens, rep=r, tiempo_s=t, ms_gen=1000 * t / gens,
                                      t_computo_s=media,
                                      desbalance=(max(comp) / media) if media > 0 else 1.0,
                                      correcto=ok))
                    if not ok:
                        print("  ¡ERROR! resultado distinto al secuencial")
                tm = statistics.median(tiempos)
                print(f"  hilos {plan:9s} T={T:2d}: {1000 * tm / gens:8.3f} ms/gen  speedup {t_seq / tm:5.2f}")

        if not a.sin_nativo:
            for T in hilos:
                tiempos = []
                for r in range(a.reps):
                    with MotorNativo(tablero, T) as m:
                        t = medir(m, gens)
                        ok = np.array_equal(m.tablero, referencia)
                    tiempos.append(t)
                    filas.append(dict(implementacion="nativo", tamano=n, trabajadores=T,
                                      gens=gens, rep=r, tiempo_s=t, ms_gen=1000 * t / gens,
                                      correcto=ok))
                tm = statistics.median(tiempos)
                print(f"  nativo (prange)  T={T:2d}: {1000 * tm / gens:8.3f} ms/gen  speedup {t_seq / tm:5.2f}")

        comun.escribir_filas(a.salida, filas, nuevo=primera)
        primera = False
        comun.guardar_json(ruta_gens, gens_por_tamano)


if __name__ == "__main__":
    main()
