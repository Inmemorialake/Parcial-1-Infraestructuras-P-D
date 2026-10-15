"""
Patrones iniciales para el tablero.

Los patrones clásicos se describen en formato RLE (Run Length Encoded), el
formato estándar de la comunidad de Life: 'b' = muerta, 'o' = viva,
'$' = fin de fila, '!' = fin del patrón; un número antes repite el símbolo.
"""
import numpy as np

RLE = {
    "glider": "bo$2bo$3o!",
    "nave_ligera": "bo2bo$o4b$o3bo$4o!",
    "r_pentomino": "b2o$2ob$bo!",
    "bellota": "bo5b$3bo3b$2o2b3o!",
    "diehard": "6bob$2o6b$bo3b3o!",
    "canon_gosper": ("24bo$22bobo$12b2o6b2o12b2o$11bo3bo4b2o12b2o$"
                     "2o8bo5bo3b2o$2o8bo3bob2o4bobo$10bo5bo7bo$11bo3bo$12b2o!"),
}

NOMBRES = {
    "aleatorio": "Aleatorio (30 %)",
    "canon_gosper": "Cañón de Gosper",
    "r_pentomino": "R-pentominó",
    "bellota": "Bellota (acorn)",
    "diehard": "Diehard",
    "glider": "Planeador (glider)",
    "nave_ligera": "Nave ligera (LWSS)",
    "vacio": "Vacío (dibujar)",
}

ORDEN = ["aleatorio", "canon_gosper", "r_pentomino", "bellota", "diehard",
         "glider", "nave_ligera", "vacio"]


def parsear_rle(rle):
    """Convierte un RLE en una matriz uint8."""
    filas, fila, num = [], [], ""
    for ch in rle.strip():
        if ch.isdigit():
            num += ch
            continue
        k = int(num) if num else 1
        num = ""
        if ch == "b":
            fila += [0] * k
        elif ch == "o":
            fila += [1] * k
        elif ch == "$":
            filas.append(fila)
            filas += [[] for _ in range(k - 1)]
            fila = []
        elif ch == "!":
            break
    filas.append(fila)
    ancho = max(len(f) for f in filas)
    m = np.zeros((len(filas), ancho), dtype=np.uint8)
    for i, f in enumerate(filas):
        m[i, :len(f)] = f
    return m


def colocar(tablero, patron, fila=None, col=None):
    """Coloca `patron` en `tablero` (centrado por defecto)."""
    H, W = tablero.shape
    h, w = patron.shape
    if h > H or w > W:
        raise ValueError(f"El patrón ({h}x{w}) no cabe en el tablero ({H}x{W})")
    fila = (H - h) // 2 if fila is None else fila
    col = (W - w) // 2 if col is None else col
    tablero[fila:fila + h, col:col + w] = patron
    return tablero


def aleatorio(n, densidad=0.3, semilla=42):
    rng = np.random.default_rng(semilla)
    return (rng.random((n, n)) < densidad).astype(np.uint8)


def crear(nombre, n, semilla=42):
    """Crea un tablero n x n con el patrón indicado."""
    if nombre == "aleatorio":
        return aleatorio(n, semilla=semilla)
    t = np.zeros((n, n), dtype=np.uint8)
    if nombre == "vacio":
        return t
    patron = parsear_rle(RLE[nombre])
    if nombre == "canon_gosper":
        # En tableros grandes se ponen varios cañones para que haya más acción
        h, w = patron.shape
        paso_f, paso_c = max(h + 30, n // 4), max(w + 30, n // 4)
        for f in range(5, n - h, paso_f):
            for c in range(5, n - w, paso_c):
                colocar(t, patron, f, c)
        return t
    return colocar(t, patron)
