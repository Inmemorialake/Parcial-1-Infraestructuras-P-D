"""
Genera capturas de la interfaz para el informe (sin abrir ventana).

    python bench/capturas.py
"""
import os
import subprocess
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import comun  # noqa: E402
import pygame  # noqa: E402

from vida.gui import App, FabricaLocal  # noqa: E402

SALIDA = os.path.join(comun.RAIZ, "informe", "figuras")


def capturar(nombre, estado, pasos_seq, pasos_par, modo_final="hilos"):
    app = App(FabricaLocal(), dict(estado, iniciar=False))
    k = app.e["gens_cuadro"]
    for _ in range(pasos_seq):                 # mide la línea base secuencial
        app._avanzar(k)
    if modo_final != "secuencial":
        while app.e["modo"] != modo_final:
            app.cambiar_modo()
        for _ in range(pasos_par):
            app._avanzar(k)
    app.corriendo = True
    app._gens_ventana, app.gens_s = 0, min(60 * k, 1000 / max(app.ms_gen, 1e-3))
    app.dibujar()
    pygame.image.save(app.pantalla, os.path.join(SALIDA, nombre))
    app.motor.cerrar()
    pygame.quit()
    print("  captura:", nombre)


def main():
    os.makedirs(SALIDA, exist_ok=True)
    hilos = min(8, os.cpu_count() or 2)
    base = dict(modo="secuencial", hilos=hilos, planificacion="estatica", semilla=42, gens_cuadro=1)
    capturar("gui_hilos_1000.png", dict(base, tamano=1000, patron="aleatorio", gens_cuadro=8), 30, 30)
    capturar("gui_gosper_200.png", dict(base, tamano=200, patron="canon_gosper"), 150, 150, "secuencial")
    # Captura de la versión MPI (2 procesos)
    mpiexec = "mpiexec"
    subprocess.run([mpiexec, "-n", "2", sys.executable, "-m", "vida.gui_mpi", "--tamano", "500",
                    "--captura", os.path.join(SALIDA, "gui_mpi_500.png"), "--frames", "90"],
                   cwd=comun.RAIZ, check=False)
    print("  captura: gui_mpi_500.png")


if __name__ == "__main__":
    main()
