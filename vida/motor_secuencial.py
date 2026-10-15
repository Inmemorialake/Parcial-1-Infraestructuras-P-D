"""Versión secuencial (línea base para medir el speedup)."""
import time

import numpy as np

from .kernels import obtener_kernel


class MotorSecuencial:
    """Calcula las generaciones con un solo hilo, recorriendo todas las filas."""

    nombre = "Secuencial"

    def __init__(self, tablero, kernel="numba"):
        self._buf = [np.array(tablero, dtype=np.uint8, order="C"),
                     np.empty(tablero.shape, dtype=np.uint8)]
        self._actual = 0                     # índice del buffer con la generación vigente
        self._kernel = obtener_kernel(kernel)
        self.generacion = 0
        self.t_computo = 0.0                 # segundos acumulados dentro del kernel

    @property
    def tablero(self):
        return self._buf[self._actual]

    def paso(self, n=1):
        H = self.tablero.shape[0]
        for _ in range(n):
            src = self._buf[self._actual]
            dst = self._buf[1 - self._actual]
            t0 = time.perf_counter()
            self._kernel(src, dst, 0, H)
            self.t_computo += time.perf_counter() - t0
            self._actual = 1 - self._actual  # intercambio de buffers (doble buffer)
        self.generacion += n

    def cerrar(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cerrar()
