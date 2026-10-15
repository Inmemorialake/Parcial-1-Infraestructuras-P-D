"""
Variante de comparación: hilos NATIVOS con Numba `prange` (estilo OpenMP).

Usa exactamente la misma descomposición (T bandas de filas) y el mismo kernel
que MotorHilos, pero el fork-join de cada generación lo hace el runtime de
Numba (capa OpenMP/TBB/workqueue) en código nativo, sin pasar por el
intérprete de Python. Sirve para separar dos costos en el informe:

  * el costo del algoritmo paralelo (igual en ambas versiones), y
  * el costo de sincronizar con threading.Barrier en Python (solo en MotorHilos).

Al final de cada `prange` hay una barrera implícita (join): ningún hilo empieza
la generación siguiente hasta que todos terminan la actual.
"""
import numba
import numpy as np
from numba import njit, prange

from .kernels import paso_numba


@njit(parallel=True, nogil=True, cache=True)
def _generaciones(a, b, n_gens, n_bandas):
    H = a.shape[0]
    for g in range(n_gens):
        src = a if g % 2 == 0 else b
        dst = b if g % 2 == 0 else a
        for t in prange(n_bandas):                 # fork: una banda por hilo
            paso_numba(src, dst, H * t // n_bandas, H * (t + 1) // n_bandas)
        # join implícito aquí


class MotorNativo:
    nombre = "Hilos nativos"

    def __init__(self, tablero, n_hilos=4):
        self.n_hilos = max(1, min(n_hilos, numba.config.NUMBA_NUM_THREADS))
        self._a = np.array(tablero, dtype=np.uint8, order="C")
        self._b = np.empty_like(self._a)
        self.generacion = 0

    @property
    def tablero(self):
        return self._a

    def paso(self, n=1):
        if n <= 0:
            return
        numba.set_num_threads(self.n_hilos)
        _generaciones(self._a, self._b, n, self.n_hilos)
        if n % 2 == 1:                              # el resultado quedó en _b
            self._a, self._b = self._b, self._a
        self.generacion += n

    def cerrar(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cerrar()
