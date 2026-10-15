"""Verifica que la versión MPI produce exactamente lo mismo que la secuencial.

Uso:  mpiexec -n P python tests/verificar_mpi.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from mpi4py import MPI

from vida import MotorSecuencial, patrones
from vida.motor_mpi import MotorMPI

comm = MPI.COMM_WORLD
ok = True
for n, gens in ((50, 31), (101, 40), (257, 25)):
    t = patrones.aleatorio(n, semilla=n) if comm.rank == 0 else None
    m = MotorMPI(comm, t, (n, n))
    m.paso(gens)
    final = m.recolectar()
    m.cerrar()
    if comm.rank == 0:
        s = MotorSecuencial(t)
        s.paso(gens)
        igual = np.array_equal(s.tablero, final)
        ok &= igual
        print(f"P={comm.size} n={n} gens={gens}: {'OK' if igual else 'DIFERENTE'}")
if comm.rank == 0:
    sys.exit(0 if ok else 1)
