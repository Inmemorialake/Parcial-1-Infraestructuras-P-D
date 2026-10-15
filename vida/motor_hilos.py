"""
Versión paralela con hilos (memoria compartida, modelo SMP).

Estrategia
----------
* Descomposición por dominio: el tablero se divide en BANDAS DE FILAS.
* Doble buffer: todos los hilos leen la generación g de `src` y escriben la
  generación g+1 en `dst`. Como cada hilo escribe solo sus filas, no hay
  escrituras en conflicto y no se necesita un lock para el tablero.
* Barrera por generación: ningún hilo puede empezar la generación g+1 hasta que
  TODOS hayan terminado la g (si no, leería filas vecinas a medio actualizar:
  condición de carrera). La acción de la barrera (la ejecuta un único hilo
  cuando todos llegaron) intercambia los buffers.
* Pool persistente: los hilos se crean una sola vez y esperan órdenes; crear y
  destruir hilos en cada generación costaría más que el cálculo en tableros
  pequeños.

Planificación
-------------
* "estatica": cada hilo recibe una banda fija de H/T filas.
* "dinamica": el tablero se parte en muchos bloques pequeños y los hilos los
  toman de una cola compartida (un contador protegido con un Lock). Así un
  núcleo rápido procesa más bloques que uno lento (balanceo de carga, útil en
  CPUs híbridas con núcleos P y E).

¿Y el GIL?
----------
El kernel "numba" se compila con nogil=True: mientras calcula una banda, el
hilo suelta el GIL y los demás hilos corren en paralelo en otros núcleos. Con
el kernel "python" el GIL serializa los hilos y no hay aceleración.
"""
import threading
import time

import numpy as np

from .kernels import obtener_kernel


def particion_filas(H, partes):
    """Divide H filas en `partes` bandas contiguas lo más parejas posible."""
    return [(H * k // partes, H * (k + 1) // partes) for k in range(partes)]


class MotorHilos:
    nombre = "Hilos"

    def __init__(self, tablero, n_hilos=4, planificacion="estatica",
                 kernel="numba", bloques_por_hilo=8):
        if planificacion not in ("estatica", "dinamica"):
            raise ValueError("planificacion debe ser 'estatica' o 'dinamica'")
        self._buf = [np.array(tablero, dtype=np.uint8, order="C"),
                     np.empty(tablero.shape, dtype=np.uint8)]
        self._actual = 0
        self._kernel = obtener_kernel(kernel)
        self.n_hilos = n_hilos
        self.planificacion = planificacion
        self.generacion = 0

        H = self._buf[0].shape[0]
        self._bandas = particion_filas(H, n_hilos)
        n_bloques = min(H, n_hilos * bloques_por_hilo)
        self._bloques = particion_filas(H, n_bloques)

        # Cola de bloques para la planificación dinámica
        self._lock_cola = threading.Lock()
        self._siguiente_bloque = 0

        # Sincronización:
        #  - _inicio / _fin: el hilo principal (+1) y los trabajadores se citan
        #    para empezar y terminar un lote de generaciones.
        #  - _barrera_gen: separa una generación de la siguiente.
        self._inicio = threading.Barrier(n_hilos + 1)
        self._fin = threading.Barrier(n_hilos + 1)
        self._barrera_gen = threading.Barrier(n_hilos, action=self._fin_de_generacion)

        self._gens_pedidas = 0
        self._detener = False
        self._error = None
        self.t_computo = [0.0] * n_hilos     # tiempo dentro del kernel, por hilo

        self._hilos = [threading.Thread(target=self._trabajador, args=(k,),
                                        name=f"vida-{k}", daemon=True)
                       for k in range(n_hilos)]
        for h in self._hilos:
            h.start()

    # ------------------------------------------------------------------ API
    @property
    def tablero(self):
        return self._buf[self._actual]

    def paso(self, n=1):
        """Avanza n generaciones usando todos los hilos. Bloquea hasta terminar."""
        if n <= 0:
            return
        self._gens_pedidas = n
        self._inicio.wait()      # libera a los trabajadores
        self._fin.wait()         # espera a que terminen las n generaciones
        if self._error is not None:
            raise RuntimeError("Falló un hilo trabajador") from self._error
        self.generacion += n

    def cerrar(self):
        if self._detener:
            return
        self._detener = True
        try:
            self._inicio.wait(timeout=5)
        except threading.BrokenBarrierError:
            pass
        for h in self._hilos:
            h.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cerrar()

    # ------------------------------------------------------------ internos
    def _fin_de_generacion(self):
        """Acción de la barrera: la ejecuta UN solo hilo cuando todos llegaron."""
        self._actual = 1 - self._actual      # intercambio de buffers
        self._siguiente_bloque = 0           # reinicia la cola dinámica

    def _tomar_bloque(self):
        # Sección crítica: sin el lock, dos hilos podrían leer el mismo valor
        # del contador y procesar el mismo bloque (o saltarse uno).
        with self._lock_cola:
            k = self._siguiente_bloque
            self._siguiente_bloque += 1
        return k

    def _trabajador(self, idx):
        kernel = self._kernel
        r0, r1 = self._bandas[idx]
        try:
            while True:
                self._inicio.wait()
                if self._detener:
                    return
                for _ in range(self._gens_pedidas):
                    src = self._buf[self._actual]
                    dst = self._buf[1 - self._actual]
                    t0 = time.perf_counter()
                    if self.planificacion == "estatica":
                        kernel(src, dst, r0, r1)
                    else:
                        while True:
                            k = self._tomar_bloque()
                            if k >= len(self._bloques):
                                break
                            b0, b1 = self._bloques[k]
                            kernel(src, dst, b0, b1)
                    self.t_computo[idx] += time.perf_counter() - t0
                    self._barrera_gen.wait()
                self._fin.wait()
        except threading.BrokenBarrierError:
            return
        except BaseException as e:          # no dejar a los demás bloqueados
            self._error = e
            for b in (self._barrera_gen, self._fin, self._inicio):
                b.abort()
