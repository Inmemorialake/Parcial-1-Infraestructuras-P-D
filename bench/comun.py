"""Utilidades compartidas por los scripts de medición."""
import csv
import json
import os
import platform
import subprocess
import sys

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RAIZ)
RESULTADOS = os.path.join(RAIZ, "resultados")

CAMPOS = ["implementacion", "tamano", "trabajadores", "gens", "rep", "tiempo_s",
          "ms_gen", "t_computo_s", "t_comunicacion_s", "desbalance", "correcto"]


def escribir_filas(ruta, filas, nuevo=False):
    existe = os.path.exists(ruta) and not nuevo
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "a" if existe else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        if not existe:
            w.writeheader()
        for fila in filas:
            w.writerow({k: fila.get(k, "") for k in CAMPOS})


def leer_csv(ruta):
    with open(ruta, newline="") as f:
        return list(csv.DictReader(f))


def _lscpu():
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5).stdout
        return dict(l.split(":", 1) for l in out.splitlines() if ":" in l)
    except Exception:
        return {}


def info_hardware():
    info = {
        "sistema": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "nucleos_logicos": os.cpu_count(),
        "nucleos_fisicos": None,
    }
    l = {k.strip(): v.strip() for k, v in _lscpu().items()}
    if "Model name" in l:
        info["cpu"] = l["Model name"]
    try:
        info["nucleos_fisicos"] = int(l["Core(s) per socket"]) * int(l.get("Socket(s)", 1))
    except (KeyError, ValueError):
        pass
    if platform.system() == "Darwin":
        try:
            info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                         capture_output=True, text=True).stdout.strip()
        except Exception:
            pass
    try:
        import numpy, numba
        info["numpy"], info["numba"] = numpy.__version__, numba.__version__
    except Exception:
        pass
    try:
        from mpi4py import MPI
        info["mpi"] = " ".join(MPI.Get_library_version().splitlines()[0].split())
    except Exception:
        info["mpi"] = "no disponible"
    return info


def guardar_json(ruta, datos):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)


def leer_json(ruta):
    with open(ruta) as f:
        return json.load(f)
