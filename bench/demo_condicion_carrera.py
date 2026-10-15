"""
Demostración para la sustentación: ¿qué pasa si se quita la sincronización?

  1. Hilos SIN barrera entre generaciones: un hilo rápido empieza la
     generación g+1 leyendo filas del vecino que todavía están en la g.
     -> condición de carrera: el resultado cambia de una ejecución a otra.
  2. Actualización EN EL MISMO arreglo (sin doble buffer), incluso con un solo
     hilo: las células ya actualizadas contaminan el conteo de sus vecinas.
  3. La versión correcta (barrera + doble buffer) coincide siempre.

    python bench/demo_condicion_carrera.py --hilos 4 --tamano 300 --gens 200
"""
import argparse
import os
import threading

import numpy as np

import comun  # ajusta sys.path
from vida import MotorHilos, MotorSecuencial, patrones
from vida.kernels import paso_numba
from vida.motor_hilos import particion_filas


def hilos_sin_barrera(tablero, T, gens):
    buf = [tablero.copy(), np.empty_like(tablero)]
    bandas = particion_filas(tablero.shape[0], T)

    def trabajador(k):
        r0, r1 = bandas[k]
        for g in range(gens):              # ¡sin barrera!
            paso_numba(buf[g % 2], buf[(g + 1) % 2], r0, r1)

    hs = [threading.Thread(target=trabajador, args=(k,)) for k in range(T)]
    for h in hs:
        h.start()
    for h in hs:
        h.join()
    return buf[gens % 2]


def en_el_mismo_arreglo(tablero, gens):
    a = tablero.copy()
    for _ in range(gens):
        paso_numba(a, a, 0, a.shape[0])    # src y dst son el mismo arreglo
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--tamano", type=int, default=300)
    ap.add_argument("--gens", type=int, default=200)
    ap.add_argument("--intentos", type=int, default=5)
    a = ap.parse_args()

    t = patrones.aleatorio(a.tamano, semilla=3)
    s = MotorSecuencial(t)
    s.paso(a.gens)
    ref = s.tablero
    print(f"Tablero {a.tamano}x{a.tamano}, {a.gens} generaciones, {a.hilos} hilos\n")

    filas = []
    print("1) Hilos SIN barrera (células distintas al resultado secuencial):")
    for i in range(a.intentos):
        d = int(np.count_nonzero(hilos_sin_barrera(t, a.hilos, a.gens) != ref))
        filas.append(("sin_barrera", i + 1, d))
        print(f"   intento {i + 1}: {d:6d} células distintas")

    d = int(np.count_nonzero(en_el_mismo_arreglo(t, a.gens) != ref))
    filas.append(("mismo_arreglo", 1, d))
    print(f"\n2) Secuencial actualizando en el mismo arreglo: {d} células distintas")

    print("\n3) Versión correcta (barrera + doble buffer):")
    for i in range(a.intentos):
        with MotorHilos(t, a.hilos) as m:
            m.paso(a.gens)
            d = int(np.count_nonzero(m.tablero != ref))
        filas.append(("correcta", i + 1, d))
        print(f"   intento {i + 1}: {d:6d} células distintas")

    os.makedirs(comun.RESULTADOS, exist_ok=True)
    with open(os.path.join(comun.RESULTADOS, "carrera.csv"), "w") as f:
        f.write("variante,intento,celdas_distintas,tamano,gens,hilos\n")
        for v, i, d in filas:
            f.write(f"{v},{i},{d},{a.tamano},{a.gens},{a.hilos}\n")


if __name__ == "__main__":
    main()
