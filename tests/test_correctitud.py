"""
Pruebas de correctitud:   python -m pytest -q

La propiedad central: TODAS las versiones paralelas producen exactamente el
mismo tablero que la secuencial, para cualquier número de hilos/procesos.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vida import MotorHilos, MotorNativo, MotorSecuencial, patrones  # noqa: E402
from vida.kernels import KERNELS  # noqa: E402


def secuencial(t, gens):
    m = MotorSecuencial(t)
    m.paso(gens)
    return m.tablero


# ---------------------------------------------------------------- reglas
def test_blinker_oscila_con_periodo_2():
    t = np.zeros((5, 5), np.uint8)
    t[2, 1:4] = 1                                   # horizontal
    g1 = secuencial(t, 1)
    assert g1[1:4, 2].all() and g1.sum() == 3       # vertical
    assert np.array_equal(secuencial(t, 2), t)


def test_bloque_es_estable():
    t = np.zeros((6, 6), np.uint8)
    t[2:4, 2:4] = 1
    assert np.array_equal(secuencial(t, 10), t)


def test_planeador_se_desplaza_y_cruza_el_borde_toroidal():
    t = np.zeros((10, 10), np.uint8)
    patrones.colocar(t, patrones.parsear_rle(patrones.RLE["glider"]), 7, 7)
    # cada 4 generaciones se mueve (1, 1); tras 40 da la vuelta completa al toroide
    assert np.array_equal(secuencial(t, 4), np.roll(t, (1, 1), (0, 1)))
    assert np.array_equal(secuencial(t, 40), t)


def test_canon_de_gosper_dispara_un_planeador_cada_30_generaciones():
    t = np.zeros((200, 200), np.uint8)
    patrones.colocar(t, patrones.parsear_rle(patrones.RLE["canon_gosper"]), 10, 10)
    assert t.sum() == 36
    assert secuencial(t, 120).sum() == 36 + 4 * 5     # 4 planeadores de 5 células


@pytest.mark.parametrize("nombre", list(KERNELS))
def test_todos_los_kernels_coinciden(nombre):
    t = patrones.aleatorio(64, semilla=9)
    esperado = np.zeros_like(t)
    KERNELS["numba"](t, esperado, 0, 64)
    d = np.zeros_like(t)
    KERNELS[nombre](t, d, 0, 64)
    assert np.array_equal(d, esperado)


# ------------------------------------------------------ paralelo == secuencial
@pytest.mark.parametrize("n", [1, 7, 100, 257])
@pytest.mark.parametrize("hilos", [1, 2, 3, 8])
@pytest.mark.parametrize("plan", ["estatica", "dinamica"])
def test_hilos_igual_a_secuencial(n, hilos, plan):
    t = patrones.aleatorio(n, semilla=n)
    with MotorHilos(t, hilos, plan) as m:
        m.paso(11)
        m.paso(14)                    # varios lotes seguidos reutilizan el pool
        assert np.array_equal(m.tablero, secuencial(t, 25))


@pytest.mark.parametrize("hilos", [1, 2, 3])
def test_nativo_igual_a_secuencial(hilos):
    t = patrones.aleatorio(150, semilla=4)
    with MotorNativo(t, hilos) as m:
        m.paso(9)
        m.paso(8)
        assert np.array_equal(m.tablero, secuencial(t, 17))


@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="MPI no instalado")
@pytest.mark.parametrize("procesos", [1, 2, 3])
def test_mpi_igual_a_secuencial(procesos):
    script = os.path.join(os.path.dirname(__file__), "verificar_mpi.py")
    r = subprocess.run(["mpiexec", "-n", str(procesos), sys.executable, script],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
