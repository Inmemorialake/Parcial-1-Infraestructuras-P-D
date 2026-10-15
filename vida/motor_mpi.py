"""
Versión paralela con paso de mensajes (MPI, memoria distribuida).

Se ejecuta con:  mpiexec -n P python -m <script que use MotorMPI>

Estrategia
----------
* Descomposición por dominio en BANDAS DE FILAS: el proceso k guarda solo sus
  h_k filas, más DOS FILAS FANTASMA (ghost/halo): una copia de la última fila
  del proceso de arriba y otra de la primera fila del proceso de abajo.

      fila 0        <- fantasma (viene del vecino de arriba)
      filas 1..h    <- filas propias
      fila h+1      <- fantasma (viene del vecino de abajo)

* Topología cartesiana 1-D PERIÓDICA (un anillo): el vecino de arriba del
  proceso 0 es el proceso P-1. Así se respeta la frontera toroidal del
  tablero sin casos especiales.
* En cada generación:
    1. Intercambio de fronteras con MPI_Sendrecv (envía y recibe en una sola
       llamada: evita el interbloqueo que tendría un Send bloqueante seguido de
       un Recv si todos los procesos envían a la vez).
    2. Cálculo local con el mismo kernel de la versión secuencial.
    3. Intercambio de buffers (doble buffer).
  No se necesita una barrera explícita: el Sendrecv ya sincroniza a cada
  proceso con sus dos vecinos (sincronización punto a punto).
* Reparto inicial con Scatterv y recolección con Gatherv (filas de tamaño
  desigual cuando H no es múltiplo de P).
"""
import numpy as np
from mpi4py import MPI

from .kernels import obtener_kernel
from .motor_hilos import particion_filas


class MotorMPI:
    nombre = "MPI"

    def __init__(self, comm, tablero_raiz, forma, kernel="numba", raiz=0):
        """
        comm:          comunicador (p. ej. MPI.COMM_WORLD)
        tablero_raiz:  tablero completo (solo se usa en el proceso raíz)
        forma:         (H, W), debe conocerse en todos los procesos
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.n_procesos = comm.Get_size()
        self.raiz = raiz
        H, W = forma
        if H < self.n_procesos:
            raise ValueError(f"Hay más procesos ({self.n_procesos}) que filas ({H})")
        self.forma = (H, W)
        self._kernel = obtener_kernel(kernel)
        self.generacion = 0
        self.t_computo = 0.0
        self.t_comunicacion = 0.0

        # Anillo periódico de procesos
        self.cart = comm.Create_cart(dims=[self.n_procesos], periods=[True], reorder=False)
        self.arriba, self.abajo = self.cart.Shift(0, 1)   # (origen, destino) con desplazamiento +1

        # Reparto de filas
        bandas = particion_filas(H, self.n_procesos)
        self._filas = [b1 - b0 for b0, b1 in bandas]
        self._cuentas = [f * W for f in self._filas]
        self._despl = [b0 * W for b0, _ in bandas]
        self.h = self._filas[self.rank]
        self.fila_inicial = bandas[self.rank][0]

        self._buf = [np.zeros((self.h + 2, W), dtype=np.uint8),
                     np.zeros((self.h + 2, W), dtype=np.uint8)]
        self._actual = 0

        envio = None
        if self.rank == raiz:
            envio = [np.ascontiguousarray(tablero_raiz, dtype=np.uint8),
                     self._cuentas, self._despl, MPI.UNSIGNED_CHAR]
        self.cart.Scatterv(envio, self._buf[0][1:self.h + 1], root=raiz)

    # ----------------------------------------------------------------------
    def _intercambiar_fronteras(self, a):
        h = self.h
        # Mi primera fila sube; recibo en mi fantasma inferior la del vecino de abajo
        self.cart.Sendrecv(sendbuf=a[1], dest=self.arriba, sendtag=0,
                           recvbuf=a[h + 1], source=self.abajo, recvtag=0)
        # Mi última fila baja; recibo en mi fantasma superior la del vecino de arriba
        self.cart.Sendrecv(sendbuf=a[h], dest=self.abajo, sendtag=1,
                           recvbuf=a[0], source=self.arriba, recvtag=1)

    def paso(self, n=1):
        for _ in range(n):
            src = self._buf[self._actual]
            dst = self._buf[1 - self._actual]
            t0 = MPI.Wtime()
            self._intercambiar_fronteras(src)
            t1 = MPI.Wtime()
            # Filas 1..h; las filas 0 y h+1 de src son las fantasma.
            self._kernel(src, dst, 1, self.h + 1)
            t2 = MPI.Wtime()
            self.t_comunicacion += t1 - t0
            self.t_computo += t2 - t1
            self._actual = 1 - self._actual
        self.generacion += n

    def recolectar(self):
        """Reúne el tablero completo en la raíz (None en los demás procesos)."""
        local = self._buf[self._actual][1:self.h + 1]
        destino = None
        completo = None
        if self.rank == self.raiz:
            completo = np.empty(self.forma, dtype=np.uint8)
            destino = [completo, self._cuentas, self._despl, MPI.UNSIGNED_CHAR]
        self.cart.Gatherv(local, destino, root=self.raiz)
        return completo

    def cerrar(self):
        if self.cart != MPI.COMM_NULL:
            self.cart.Free()
            self.cart = MPI.COMM_NULL
