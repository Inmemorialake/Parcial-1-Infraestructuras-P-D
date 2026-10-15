"""
Interfaz gráfica para la versión MPI.

    mpiexec -n 4 python -m vida.gui_mpi --tamano 1000

El proceso 0 abre la ventana y coordina; los demás procesos solo calculan.
Cada orden de la interfaz (avanzar, reiniciar, salir...) se difunde con
MPI_Bcast; después de cada lote de generaciones el proceso 0 reúne el tablero
con MPI_Gatherv para dibujarlo.
"""
import numpy as np
from mpi4py import MPI

from .gui import App, argumentos, estado_desde
from .motor_mpi import MotorMPI

comm = MPI.COMM_WORLD


class ProxyMPI:
    """Lo usa la GUI en el proceso 0 como si fuera un motor local."""

    def __init__(self, tablero):
        comm.bcast(("nuevo", tablero.shape), root=0)
        self._m = MotorMPI(comm, tablero, tablero.shape)
        self._tablero = tablero.copy()
        self.generacion = 0

    @property
    def tablero(self):
        return self._tablero

    def paso(self, n=1):
        comm.bcast(("paso", n), root=0)
        self._m.paso(n)
        self._tablero = self._m.recolectar()
        self.generacion += n

    def cerrar(self):
        comm.bcast(("cerrar", None), root=0)
        self._m.cerrar()


class FabricaMPI:
    modos = ["mpi"]

    def crear(self, tablero, estado):
        return ProxyMPI(tablero)

    def editable(self):
        # El tablero de la GUI es una copia: editarlo no cambiaría los datos
        # repartidos entre procesos.
        return False


def trabajador():
    """Bucle de los procesos != 0: obedecen las órdenes del proceso 0."""
    motor = None
    while True:
        orden, dato = comm.bcast(None, root=0)
        if orden == "nuevo":
            motor = MotorMPI(comm, None, dato)
        elif orden == "paso":
            motor.paso(dato)
            motor.recolectar()
        elif orden == "cerrar":
            motor.cerrar()
            motor = None
        elif orden == "salir":
            return


def main():
    a = argumentos(modos=("mpi",))
    if comm.rank == 0:
        try:
            estado = estado_desde(a, modo="mpi", procesos=comm.size)
            App(FabricaMPI(), estado, a.captura, a.frames).ejecutar()
        finally:
            comm.bcast(("salir", None), root=0)
    else:
        trabajador()


if __name__ == "__main__":
    main()
