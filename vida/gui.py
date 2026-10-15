"""
Interfaz gráfica (pygame) para simular el Juego de la Vida.

    python -m vida.gui                       # secuencial / hilos / nativo (se cambia en vivo)
    python -m vida.gui --modo hilos --hilos 8 --tamano 1000
    mpiexec -n 4 python -m vida.gui_mpi      # versión MPI (ver gui_mpi.py)

Controles (teclado o botones del panel):
    Espacio  iniciar / pausar          N / ->  avanzar una generación
    M        cambiar modo              D       planificación estática / dinámica
    + / -    más / menos hilos         T       cambiar tamaño del tablero
    P        cambiar patrón            R       reiniciar el patrón
    Arriba / Abajo  más / menos generaciones por cuadro
    Clic izq. / der. sobre el tablero (en pausa): dar vida / matar células
    Q / Esc  salir

El tiempo de cálculo (ms/gen) se mide SOLO alrededor de motor.paso(); el
dibujado no se incluye. El panel guarda la última medición de cada modo para
cada tamaño, así puede verse el speedup en vivo frente al secuencial.
"""
import argparse
import os
import time

import numpy as np
import pygame

from . import patrones
from .motor_hilos import MotorHilos
from .motor_nativo import MotorNativo
from .motor_secuencial import MotorSecuencial

# ----------------------------------------------------------------- estilo
FONDO = (22, 22, 30)
PANEL = (31, 33, 46)
MUERTA = (26, 27, 38)
VIVA = (122, 162, 247)
TEXTO = (192, 202, 245)
TENUE = (115, 122, 162)
ACENTO = (158, 206, 106)
BOTON = (52, 56, 80)
BOTON_HOVER = (68, 74, 106)
BOTON_ACTIVO = (122, 162, 247)

ANCHO, ALTO = 1200, 800
LIENZO = 780                    # el tablero se dibuja en un cuadrado de LIENZO px
TAMANOS = [100, 200, 500, 1000, 2000]
VELOCIDADES = [1, 2, 4, 8, 16, 32]


class Boton:
    def __init__(self, rect, texto, accion, resaltado=None):
        self.rect = pygame.Rect(rect)
        self.texto = texto            # str o función que devuelve str
        self.accion = accion
        self.resaltado = resaltado    # función -> bool

    def dibujar(self, sup, fuente, raton):
        activo = self.resaltado() if self.resaltado else False
        color = BOTON_ACTIVO if activo else (BOTON_HOVER if self.rect.collidepoint(raton) else BOTON)
        pygame.draw.rect(sup, color, self.rect, border_radius=6)
        txt = self.texto() if callable(self.texto) else self.texto
        img = fuente.render(txt, True, FONDO if activo else TEXTO)
        sup.blit(img, img.get_rect(center=self.rect.center))


class FabricaLocal:
    """Crea motores secuencial / hilos en este mismo proceso."""
    modos = ["secuencial", "hilos", "nativo"]

    def crear(self, tablero, estado):
        if estado["modo"] == "secuencial":
            return MotorSecuencial(tablero)
        if estado["modo"] == "nativo":
            return MotorNativo(tablero, estado["hilos"])
        return MotorHilos(tablero, estado["hilos"], estado["planificacion"])

    def editable(self):
        return True


class App:
    def __init__(self, fabrica, estado, captura=None, frames=0):
        self.fabrica = fabrica
        self.e = estado
        self.captura = captura
        self.frames_captura = frames
        self.corriendo = estado.pop("iniciar", False) or bool(captura)
        self.mediciones = {}           # (n, clave_modo) -> ms/gen (media móvil)
        self.ms_gen = None
        self.gens_s = 0.0
        self._t_ultimo = time.perf_counter()
        self._gens_ventana = 0

        pygame.init()
        pygame.display.set_caption("Juego de la Vida — Infraestructuras Paralelas y Distribuidas")
        self.pantalla = pygame.display.set_mode((ANCHO, ALTO))
        self.reloj = pygame.time.Clock()
        self.f_titulo = pygame.font.Font(None, 30)
        self.f_texto = pygame.font.Font(None, 23)
        self.f_peque = pygame.font.Font(None, 19)

        self.motor = None
        self.reiniciar()
        self._crear_botones()

    # ------------------------------------------------------- estado/motor
    def clave_modo(self):
        if self.e["modo"] == "secuencial":
            return "Secuencial"
        if self.e["modo"] == "mpi":
            return f"MPI ({self.e['procesos']} procesos)"
        if self.e["modo"] == "nativo":
            return f"Hilos nativos ({self.e['hilos']}, prange)"
        plan = "estática" if self.e["planificacion"] == "estatica" else "dinámica"
        return f"Hilos ({self.e['hilos']}, {plan})"

    def _nuevo_motor(self, tablero, generacion=0):
        if self.motor is not None:
            self.motor.cerrar()
        self.motor = self.fabrica.crear(tablero, self.e)
        self.motor.generacion = generacion
        self.ms_gen = None

    def reiniciar(self):
        t = patrones.crear(self.e["patron"], self.e["tamano"], self.e["semilla"])
        self._nuevo_motor(t, 0)

    def reconfigurar(self):
        """Cambia de motor conservando el tablero actual y la generación."""
        self._nuevo_motor(self.motor.tablero.copy(), self.motor.generacion)

    # ---------------------------------------------------------- acciones
    def alternar(self):
        self.corriendo = not self.corriendo

    def paso_manual(self):
        self.corriendo = False
        self._avanzar(1)

    def cambiar_modo(self):
        modos = self.fabrica.modos
        if len(modos) > 1:
            self.e["modo"] = modos[(modos.index(self.e["modo"]) + 1) % len(modos)]
            self.reconfigurar()

    def cambiar_planificacion(self):
        if self.e["modo"] == "hilos":
            self.e["planificacion"] = "dinamica" if self.e["planificacion"] == "estatica" else "estatica"
            self.reconfigurar()

    def hilos(self, delta):
        if self.e["modo"] in ("hilos", "nativo"):
            nuevo = min(64, max(1, self.e["hilos"] + delta))
            if nuevo != self.e["hilos"]:
                self.e["hilos"] = nuevo
                self.reconfigurar()

    def cambiar_tamano(self):
        i = TAMANOS.index(self.e["tamano"]) if self.e["tamano"] in TAMANOS else 0
        self.e["tamano"] = TAMANOS[(i + 1) % len(TAMANOS)]
        self.reiniciar()

    def cambiar_patron(self):
        orden = patrones.ORDEN
        self.e["patron"] = orden[(orden.index(self.e["patron"]) + 1) % len(orden)]
        self.reiniciar()

    def velocidad(self, delta):
        i = VELOCIDADES.index(self.e["gens_cuadro"])
        self.e["gens_cuadro"] = VELOCIDADES[min(len(VELOCIDADES) - 1, max(0, i + delta))]

    # ------------------------------------------------------------- lógica
    def _avanzar(self, k):
        t0 = time.perf_counter()
        self.motor.paso(k)
        ms = (time.perf_counter() - t0) * 1000 / k
        self.ms_gen = ms if self.ms_gen is None else 0.8 * self.ms_gen + 0.2 * ms
        self.mediciones[(self.e["tamano"], self.clave_modo())] = self.ms_gen
        self._gens_ventana += k

    def _actualizar_gens_s(self):
        ahora = time.perf_counter()
        if ahora - self._t_ultimo >= 0.5:
            self.gens_s = self._gens_ventana / (ahora - self._t_ultimo)
            self._t_ultimo, self._gens_ventana = ahora, 0

    def _celda_bajo_raton(self, pos):
        n = self.e["tamano"]
        x, y = pos[0] - 10, pos[1] - 10
        if 0 <= x < LIENZO and 0 <= y < LIENZO:
            return int(y * n / LIENZO), int(x * n / LIENZO)
        return None

    def _dibujar_celulas(self):
        botones = pygame.mouse.get_pressed()
        if self.corriendo or not (botones[0] or botones[2]) or not self.fabrica.editable():
            return
        c = self._celda_bajo_raton(pygame.mouse.get_pos())
        if c:
            self.motor.tablero[c] = 1 if botones[0] else 0

    # ------------------------------------------------------------ dibujo
    def _crear_botones(self):
        x0, y0, w, h, sep = 815, 330, 180, 38, 8
        col2 = x0 + w + 10
        filas = [
            ((lambda: "Pausar" if self.corriendo else "Iniciar", self.alternar, lambda: self.corriendo),
             ("Paso", self.paso_manual, None)),
            ((lambda: "Modo: " + {"secuencial": "Secuencial", "hilos": "Hilos", "nativo": "Nativo", "mpi": "MPI"}[self.e["modo"]],
              self.cambiar_modo, None),
             (lambda: "Planif.: " + ("estática" if self.e["planificacion"] == "estatica" else "dinámica"),
              self.cambiar_planificacion, None)),
            (("Hilos  -", lambda: self.hilos(-1), None),
             ("Hilos  +", lambda: self.hilos(+1), None)),
            ((lambda: f"Tamaño: {self.e['tamano']}", self.cambiar_tamano, None),
             ("Patrón", self.cambiar_patron, None)),
            (("Más lento", lambda: self.velocidad(-1), None),
             ("Más rápido", lambda: self.velocidad(+1), None)),
            (("Reiniciar", self.reiniciar, None),
             ("Salir", self.salir, None)),
        ]
        self.botones = []
        for k, (a, b) in enumerate(filas):
            y = y0 + k * (h + sep)
            self.botones.append(Boton((x0, y, w, h), a[0], a[1], a[2]))
            self.botones.append(Boton((col2, y, w, h), b[0], b[1], b[2]))

    def _superficie_tablero(self):
        t = self.motor.tablero
        sup = pygame.surfarray.make_surface(t.T)          # 8 bits con paleta
        sup.set_palette_at(0, MUERTA)
        sup.set_palette_at(1, VIVA)
        n = t.shape[0]
        if n <= LIENZO:
            return pygame.transform.scale(sup, (LIENZO, LIENZO))
        return pygame.transform.smoothscale(sup.convert(24), (LIENZO, LIENZO))

    def _texto(self, s, pos, fuente=None, color=TEXTO):
        self.pantalla.blit((fuente or self.f_texto).render(s, True, color), pos)

    def dibujar(self):
        p = self.pantalla
        p.fill(FONDO)
        p.blit(self._superficie_tablero(), (10, 10))
        pygame.draw.rect(p, PANEL, (800, 0, ANCHO - 800, ALTO))

        n = self.e["tamano"]
        pob = int(self.motor.tablero.sum(dtype=np.int64))
        ms = f"{self.ms_gen:.3f} ms/gen" if self.ms_gen is not None else "—"
        self._texto("Juego de la Vida paralelo", (815, 18), self.f_titulo)
        self._texto("750023C · Infraestructuras Paralelas y Distribuidas", (815, 46), self.f_peque, TENUE)
        info = [
            ("Modo", self.clave_modo()),
            ("Tablero", f"{n} x {n}  ·  {patrones.NOMBRES[self.e['patron']]}"),
            ("Generación", f"{self.motor.generacion:,}".replace(",", ".")),
            ("Población", f"{pob:,}".replace(",", ".")),
            ("Cálculo", ms),
            ("Ritmo", f"{self.gens_s:.0f} gen/s  ({self.e['gens_cuadro']} gen/cuadro)"),
            ("Estado", "corriendo" if self.corriendo else "en pausa"),
        ]
        for k, (et, val) in enumerate(info):
            self._texto(et, (815, 80 + k * 26), color=TENUE)
            self._texto(val, (930, 80 + k * 26))

        # Speedup en vivo frente al secuencial del mismo tamaño
        base = self.mediciones.get((n, "Secuencial"))
        actual = self.mediciones.get((n, self.clave_modo()))
        y = 270
        if self.e["modo"] == "secuencial":
            self._texto("Línea base secuencial medida.", (815, y), color=ACENTO)
        elif base and actual:
            self._texto(f"Speedup vs secuencial: {base / actual:.2f}x", (815, y), color=ACENTO)
        elif self.e["modo"] != "mpi":
            self._texto("Mida primero en modo Secuencial para ver el speedup.", (815, y), self.f_peque, TENUE)
        if self.e["modo"] == "mpi":
            self._texto("Procesos fijados por mpiexec -n P", (815, y + 24), self.f_peque, TENUE)

        raton = pygame.mouse.get_pos()
        for b in self.botones:
            b.dibujar(p, self.f_texto, raton)

        ayuda = ["Espacio: iniciar/pausar   N: paso   R: reiniciar",
                 "M: modo   D: planificación   +/-: hilos",
                 "T: tamaño   P: patrón   Arriba/Abajo: velocidad",
                 "Clic izq./der. en pausa: dibujar / borrar células"]
        for k, s in enumerate(ayuda):
            self._texto(s, (815, 650 + k * 22), self.f_peque, TENUE)
        pygame.display.flip()

    # ------------------------------------------------------------ bucle
    def salir(self):
        self._vivo = False

    def _teclado(self, ev):
        teclas = {
            pygame.K_SPACE: self.alternar, pygame.K_n: self.paso_manual,
            pygame.K_RIGHT: self.paso_manual, pygame.K_m: self.cambiar_modo,
            pygame.K_d: self.cambiar_planificacion, pygame.K_t: self.cambiar_tamano,
            pygame.K_p: self.cambiar_patron, pygame.K_r: self.reiniciar,
            pygame.K_UP: lambda: self.velocidad(+1), pygame.K_DOWN: lambda: self.velocidad(-1),
            pygame.K_PLUS: lambda: self.hilos(+1), pygame.K_KP_PLUS: lambda: self.hilos(+1),
            pygame.K_EQUALS: lambda: self.hilos(+1),
            pygame.K_MINUS: lambda: self.hilos(-1), pygame.K_KP_MINUS: lambda: self.hilos(-1),
            pygame.K_q: self.salir, pygame.K_ESCAPE: self.salir,
        }
        if ev.key in teclas:
            teclas[ev.key]()

    def ejecutar(self):
        self._vivo = True
        frame = 0
        try:
            while self._vivo:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        self.salir()
                    elif ev.type == pygame.KEYDOWN:
                        self._teclado(ev)
                    elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                        for b in self.botones:
                            if b.rect.collidepoint(ev.pos):
                                b.accion()
                self._dibujar_celulas()
                if self.corriendo:
                    self._avanzar(self.e["gens_cuadro"])
                self._actualizar_gens_s()
                self.dibujar()
                frame += 1
                if self.captura and frame >= self.frames_captura:
                    pygame.image.save(self.pantalla, self.captura)
                    self.salir()
                self.reloj.tick(60)
        finally:
            self.motor.cerrar()
            pygame.quit()


def argumentos(modos=("secuencial", "hilos", "nativo")):
    ap = argparse.ArgumentParser(description="Juego de la Vida con interfaz gráfica")
    ap.add_argument("--modo", choices=modos, default=modos[0])
    ap.add_argument("--hilos", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--planificacion", choices=["estatica", "dinamica"], default="estatica")
    ap.add_argument("--tamano", type=int, default=200)
    ap.add_argument("--patron", choices=patrones.ORDEN, default="aleatorio")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--velocidad", type=int, choices=VELOCIDADES, default=1,
                    help="generaciones por cuadro")
    ap.add_argument("--iniciar", action="store_true", help="arrancar corriendo")
    ap.add_argument("--captura", help="guardar una captura PNG y salir (uso interno)")
    ap.add_argument("--frames", type=int, default=60)
    return ap.parse_args()


def estado_desde(a, **extra):
    e = dict(modo=a.modo, hilos=a.hilos, planificacion=a.planificacion, tamano=a.tamano,
             patron=a.patron, semilla=a.semilla, gens_cuadro=a.velocidad, iniciar=a.iniciar)
    e.update(extra)
    return e


def main():
    a = argumentos()
    App(FabricaLocal(), estado_desde(a), a.captura, a.frames).ejecutar()


if __name__ == "__main__":
    main()
