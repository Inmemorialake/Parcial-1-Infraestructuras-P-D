"""
Mide la versión MPI. Debe lanzarse con mpiexec, una vez por número de procesos:

    mpiexec -n 4 python bench/benchmark_mpi.py --reps 5

Usa el mismo número de generaciones por tamaño que la medición de hilos
(resultados/gens.json), así el speedup se calcula contra la misma línea base
secuencial. El reparto inicial (Scatterv) y la recolección final (Gatherv) NO
se incluyen en el tiempo; el intercambio de filas fantasma de cada generación
SÍ (es parte del costo real del algoritmo).
"""
import argparse
import os

import numpy as np
from mpi4py import MPI

import comun
from vida import MotorSecuencial, patrones
from vida.motor_mpi import MotorMPI

CALENTAMIENTO = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--salida", default=os.path.join(comun.RESULTADOS, "mpi.csv"))
    ap.add_argument("--nuevo", action="store_true", help="sobrescribir el CSV")
    a = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, P = comm.rank, comm.size
    gens_por_tamano = comun.leer_json(os.path.join(comun.RESULTADOS, "gens.json")) if rank == 0 else None
    gens_por_tamano = comm.bcast(gens_por_tamano, root=0)

    filas = []
    for clave, gens in gens_por_tamano.items():
        n = int(clave)
        tablero = referencia = None
        if rank == 0:
            tablero = patrones.aleatorio(n, semilla=n)
            ref = MotorSecuencial(tablero)
            ref.paso(CALENTAMIENTO + gens)
            referencia = ref.tablero
        for r in range(a.reps):
            m = MotorMPI(comm, tablero, (n, n))
            m.paso(CALENTAMIENTO)
            m.t_computo = m.t_comunicacion = 0.0
            comm.Barrier()
            t0 = MPI.Wtime()
            m.paso(gens)
            comm.Barrier()                      # el tiempo es el del proceso más lento
            t = MPI.Wtime() - t0
            comp = comm.gather(m.t_computo, root=0)
            comu = comm.gather(m.t_comunicacion, root=0)
            final = m.recolectar()
            m.cerrar()
            if rank == 0:
                media = sum(comp) / P
                filas.append(dict(implementacion="mpi", tamano=n, trabajadores=P, gens=gens,
                                  rep=r, tiempo_s=t, ms_gen=1000 * t / gens,
                                  t_computo_s=media, t_comunicacion_s=sum(comu) / P,
                                  desbalance=(max(comp) / media) if media > 0 else 1.0,
                                  correcto=np.array_equal(final, referencia)))
        if rank == 0:
            ult = [f for f in filas if f["tamano"] == n]
            ms = sorted(f["ms_gen"] for f in ult)[len(ult) // 2]
            ok = all(f["correcto"] for f in ult)
            print(f"  mpi P={P:2d} [{n}x{n}]: {ms:8.3f} ms/gen  {'OK' if ok else '¡ERROR!'}", flush=True)

    if rank == 0:
        comun.escribir_filas(a.salida, filas, nuevo=a.nuevo)


if __name__ == "__main__":
    main()
