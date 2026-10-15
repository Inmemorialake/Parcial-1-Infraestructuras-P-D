"""Juego de la Vida de Conway: versiones secuencial, con hilos y con MPI."""
from .motor_secuencial import MotorSecuencial
from .motor_hilos import MotorHilos
from .motor_nativo import MotorNativo
from . import patrones

__all__ = ["MotorSecuencial", "MotorHilos", "MotorNativo", "patrones"]
