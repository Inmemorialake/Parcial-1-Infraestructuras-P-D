"""
Kernels del Juego de la Vida.

Todos los kernels tienen la MISMA firma y semántica:

    kernel(src, dst, r0, r1)

Calculan la generación siguiente de las filas [r0, r1) de `src` y la escriben
en las mismas filas de `dst`. Nunca escriben en `src` (doble buffer) y solo
escriben las filas [r0, r1) de `dst`, por eso varios hilos pueden ejecutar el
kernel a la vez sobre bandas disjuntas sin necesidad de locks.

Fronteras toroidales: la fila de arriba de la fila 0 es la última fila y la
columna a la izquierda de la columna 0 es la última columna (y viceversa).

Kernels disponibles:
  - "numba":  bucle compilado con Numba (JIT) y nogil=True. Libera el GIL
              mientras calcula, así que los hilos se ejecutan en paralelo real.
              Es el kernel por defecto en todo el proyecto.
  - "numpy":  versión vectorizada (operaciones sobre arreglos completos, SIMD).
  - "python": bucle en Python puro. Solo se usa para demostrar el efecto del
              GIL (los hilos NO aceleran este kernel).
"""
import numpy as np
from numba import njit

# Firma explícita: compilación "eager" al importar (no en la primera llamada),
# así la compilación JIT no contamina las mediciones de tiempo.
_FIRMA = "void(uint8[:, ::1], uint8[:, ::1], int64, int64)"


@njit(inline="always")
def _regla(vecinos, viva):
    """Reglas de Conway: nace con 3 vecinos; sobrevive con 2 o 3."""
    if vecinos == 3 or (vecinos == 2 and viva == 1):
        return 1
    return 0


@njit(_FIRMA, nogil=True, cache=True)
def paso_numba(src, dst, r0, r1):
    H, W = src.shape
    for i in range(r0, r1):
        arriba = i - 1 if i > 0 else H - 1          # frontera toroidal
        abajo = i + 1 if i < H - 1 else 0
        a = src[arriba]
        b = src[i]
        c = src[abajo]
        d = dst[i]
        # Columna 0 y W-1: necesitan "envolver" (toroide)
        for j in (0, W - 1):
            iz = (j - 1) % W
            de = (j + 1) % W
            n = a[iz] + a[j] + a[de] + b[iz] + b[de] + c[iz] + c[j] + c[de]
            d[j] = _regla(n, b[j])
        # Columnas interiores: sin módulo (bucle más rápido)
        for j in range(1, W - 1):
            n = (a[j - 1] + a[j] + a[j + 1]
                 + b[j - 1] + b[j + 1]
                 + c[j - 1] + c[j] + c[j + 1])
            d[j] = _regla(n, b[j])


def paso_numpy(src, dst, r0, r1):
    if r1 <= r0:
        return
    # Bloque de filas [r0-1, r1] con envoltura toroidal en filas...
    filas = src.take(np.arange(r0 - 1, r1 + 1), axis=0, mode="wrap")
    # ...y una columna extra a cada lado (envoltura en columnas)
    ext = np.empty((filas.shape[0], filas.shape[1] + 2), dtype=np.uint8)
    ext[:, 1:-1] = filas
    ext[:, 0] = filas[:, -1]
    ext[:, -1] = filas[:, 0]
    n = (ext[:-2, :-2] + ext[:-2, 1:-1] + ext[:-2, 2:]
         + ext[1:-1, :-2] + ext[1:-1, 2:]
         + ext[2:, :-2] + ext[2:, 1:-1] + ext[2:, 2:])
    viva = ext[1:-1, 1:-1]
    dst[r0:r1] = (n == 3) | ((n == 2) & (viva == 1))


def paso_python(src, dst, r0, r1):
    H, W = src.shape
    for i in range(r0, r1):
        a = src[(i - 1) % H].tolist()
        b = src[i].tolist()
        c = src[(i + 1) % H].tolist()
        fila = [0] * W
        for j in range(W):
            iz, de = (j - 1) % W, (j + 1) % W
            n = a[iz] + a[j] + a[de] + b[iz] + b[de] + c[iz] + c[j] + c[de]
            fila[j] = 1 if (n == 3 or (n == 2 and b[j] == 1)) else 0
        dst[i] = fila


KERNELS = {
    "numba": paso_numba,
    "numpy": paso_numpy,
    "python": paso_python,
}


def obtener_kernel(nombre):
    try:
        return KERNELS[nombre]
    except KeyError:
        raise ValueError(f"Kernel desconocido: {nombre!r}. Opciones: {list(KERNELS)}")
