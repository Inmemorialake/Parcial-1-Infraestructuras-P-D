"""
Genera las gráficas de rendimiento a partir de resultados/*.csv.

    python bench/graficas.py

Salida: resultados/figuras/*.pdf (vectorial, para el informe) y *.png.
Todas las cifras usan la MEDIANA de las repeticiones.
"""
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import comun  # noqa: E402

FIG = os.path.join(comun.RESULTADOS, "figuras")

# Paleta (categórica en orden fijo; rampa azul para tamaños, que son ordinales)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#008300", "#eda100", "#e34948"]
RAMPA = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
TINTA, TINTA2, TENUE, REJILLA, EJE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

IMPL = {
    "hilos_estatica": "Hilos (threading), estática",
    "hilos_dinamica": "Hilos (threading), dinámica",
    "nativo": "Hilos nativos (Numba prange)",
    "mpi": "MPI (mpi4py)",
}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": EJE, "axes.labelcolor": TINTA2, "axes.titlecolor": TINTA,
    "xtick.color": TENUE, "ytick.color": TENUE, "text.color": TINTA,
    "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "lines.linewidth": 2, "lines.markersize": 5,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})


def cargar():
    filas = []
    for nombre in ("hilos.csv", "mpi.csv"):
        ruta = os.path.join(comun.RESULTADOS, nombre)
        if os.path.exists(ruta):
            filas += comun.leer_csv(ruta)
    med = defaultdict(list)       # (impl, n, p) -> [ms/gen...]
    extra = defaultdict(lambda: defaultdict(list))
    for f in filas:
        k = (f["implementacion"], int(f["tamano"]), int(f["trabajadores"]))
        med[k].append(float(f["ms_gen"]))
        for c in ("t_computo_s", "t_comunicacion_s", "desbalance", "tiempo_s"):
            if f.get(c):
                extra[k][c].append(float(f[c]))
    ms = {k: statistics.median(v) for k, v in med.items()}
    ext = {k: {c: statistics.median(v) for c, v in d.items()} for k, d in extra.items()}
    return ms, ext


def guardar(fig, nombre):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, nombre + ".pdf"))
    fig.savefig(os.path.join(FIG, nombre + ".png"))
    plt.close(fig)
    print("  figura:", nombre)


def paneles(k, sharey=False, alto=2.9):
    """Rejilla de k paneles en 2 o 3 columnas (legible al ancho de una página)."""
    ncol = 2 if k <= 4 else 3
    nfil = (k + ncol - 1) // ncol
    fig, axs = plt.subplots(nfil, ncol, figsize=(3.4 * ncol, alto * nfil), sharey=sharey, squeeze=False)
    axs = list(axs.flat)
    for ax in axs[k:]:
        ax.set_visible(False)
    fig.subplots_adjust(hspace=0.55, wspace=0.25)
    return fig, axs[:k]


def ejes_trabajadores(ax, ps):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ps)
    ax.set_xticklabels([str(p) for p in ps])
    ax.minorticks_off()


def main():
    ms, ext = cargar()
    tamanos = sorted({n for (_, n, _) in ms})
    impls = [i for i in IMPL if any(k[0] == i for k in ms)]
    ps = sorted({p for (i, _, p) in ms if i != "secuencial"})
    seq = {n: ms[("secuencial", n, 1)] for n in tamanos if ("secuencial", n, 1) in ms}
    col_t = {n: RAMPA[min(len(RAMPA) - 1, k + max(0, len(RAMPA) - len(tamanos)))]
             for k, n in enumerate(tamanos)}
    hw = comun.leer_json(os.path.join(comun.RESULTADOS, "hardware.json"))
    pie = f"{hw.get('cpu')} · {hw.get('nucleos_logicos')} núcleos lógicos"

    # 1) Speedup vs trabajadores: un panel por implementación, una línea por tamaño
    fig, axs = paneles(len(impls), sharey=True)
    for ax, imp in zip(axs, impls):
        ax.plot(ps, ps, ls="--", lw=1, color=TENUE, label="Ideal (lineal)")
        for n in tamanos:
            xs = [p for p in ps if (imp, n, p) in ms]
            ax.plot(xs, [seq[n] / ms[(imp, n, p)] for p in xs], marker="o",
                    color=col_t[n], label=f"{n}×{n}")
        ax.axhline(1, color=EJE, lw=0.8)
        ax.set_title(IMPL[imp])
        ejes_trabajadores(ax, ps)
        ax.set_xlabel("Hilos / procesos")
    axs[0].set_ylabel("Speedup (T_secuencial / T_paralelo)")
    axs[0].set_ylim(bottom=0)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=min(len(l), 3), bbox_to_anchor=(0.5, 1.03))
    fig.text(0.99, -0.10, pie, ha="right", color=TENUE, fontsize=7)
    guardar(fig, "speedup_por_implementacion")

    # 2) Comparación de implementaciones: un panel por tamaño
    fig, axs = paneles(len(tamanos))
    for ax, n in zip(axs, tamanos):
        ax.plot(ps, ps, ls="--", lw=1, color=TENUE, label="Ideal")
        for c, imp in enumerate(impls):
            xs = [p for p in ps if (imp, n, p) in ms]
            ax.plot(xs, [seq[n] / ms[(imp, n, p)] for p in xs], marker="o",
                    color=CAT[c], label=IMPL[imp])
        ax.axhline(1, color=EJE, lw=0.8)
        ax.set_title(f"{n}×{n}\nsecuencial: {seq[n]:.3g} ms/gen", fontsize=9)
        ejes_trabajadores(ax, ps)
        ax.set_xlabel("Hilos / procesos")
        ax.set_ylim(bottom=0)
    axs[0].set_ylabel("Speedup")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=min(len(l), 3), bbox_to_anchor=(0.5, 1.03))
    fig.text(0.99, -0.10, pie, ha="right", color=TENUE, fontsize=7)
    guardar(fig, "speedup_por_tamano")

    # 3) Tiempo por generación vs tamaño (log-log), un panel por implementación
    fig, axs = paneles(len(impls), sharey=True)
    col_p = {p: CAT[k % len(CAT)] for k, p in enumerate(ps)}
    for ax, imp in zip(axs, impls):
        ax.plot(tamanos, [seq[n] for n in tamanos], color=TINTA, lw=1.5, ls=":",
                marker="s", label="Secuencial")
        for p in ps:
            xs = [n for n in tamanos if (imp, n, p) in ms]
            ax.plot(xs, [ms[(imp, n, p)] for n in xs], marker="o", color=col_p[p], label=f"P = {p}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(tamanos)
        ax.set_xticklabels([str(n) for n in tamanos])
        ax.minorticks_off()
        ax.set_title(IMPL[imp])
        ax.set_xlabel("Lado del tablero (N×N)")
    axs[0].set_ylabel("Tiempo por generación (ms, escala log)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=min(len(l), 3), bbox_to_anchor=(0.5, 1.03))
    guardar(fig, "tiempo_vs_tamano")

    # 4) Eficiencia = speedup / P
    fig, axs = paneles(len(impls), sharey=True)
    for ax, imp in zip(axs, impls):
        ax.axhline(1, ls="--", lw=1, color=TENUE, label="Ideal (100 %)")
        for n in tamanos:
            xs = [p for p in ps if (imp, n, p) in ms]
            ax.plot(xs, [seq[n] / ms[(imp, n, p)] / p for p in xs], marker="o",
                    color=col_t[n], label=f"{n}×{n}")
        ax.set_title(IMPL[imp])
        ejes_trabajadores(ax, ps)
        ax.set_xlabel("Hilos / procesos")
        ax.set_ylim(0, 1.15)
    axs[0].set_ylabel("Eficiencia (speedup / P)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=min(len(l), 3), bbox_to_anchor=(0.5, 1.03))
    guardar(fig, "eficiencia")

    # 5) MPI: fracción del tiempo en comunicación
    if "mpi" in impls:
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        for n in tamanos:
            xs = [p for p in ps if ("mpi", n, p) in ext and "t_comunicacion_s" in ext[("mpi", n, p)]]
            ys = [100 * ext[("mpi", n, p)]["t_comunicacion_s"] / ext[("mpi", n, p)]["tiempo_s"] for p in xs]
            ax.plot(xs, ys, marker="o", color=col_t[n], label=f"{n}×{n}")
        ejes_trabajadores(ax, ps)
        ax.set_xlabel("Procesos MPI")
        ax.set_ylabel("% del tiempo en intercambio de bordes")
        ax.set_ylim(0, 100)
        ax.legend(ncol=2)
        ax.set_title("MPI: costo de comunicación")
        guardar(fig, "mpi_comunicacion")

    # 6) Desbalance de carga (hilos estática vs dinámica) en el tablero más grande
    n = tamanos[-1]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for c, imp in enumerate(("hilos_estatica", "hilos_dinamica")):
        xs = [p for p in ps if (imp, n, p) in ext and "desbalance" in ext[(imp, n, p)]]
        ax.plot(xs, [ext[(imp, n, p)]["desbalance"] for p in xs], marker="o", color=CAT[c], label=IMPL[imp])
    ax.axhline(1, ls="--", lw=1, color=TENUE, label="Balance perfecto")
    ejes_trabajadores(ax, ps)
    ax.set_xlabel("Hilos")
    ax.set_ylabel("Cómputo máx. / cómputo medio")
    ax.set_title(f"Desbalance de carga ({n}×{n})")
    ax.legend()
    guardar(fig, "desbalance")

    # 7) Experimento del GIL
    ruta = os.path.join(comun.RESULTADOS, "gil.csv")
    if os.path.exists(ruta):
        g = comun.leer_csv(ruta)
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        hs = sorted({int(f["hilos"]) for f in g if int(f["hilos"]) > 0})
        ax.plot(hs, hs, ls="--", lw=1, color=TENUE, label="Ideal")
        etiquetas = {"python": "Python puro", "numpy": "NumPy (vectorizado)", "numba": "Numba (nogil)"}
        for c, k in enumerate(("python", "numpy", "numba")):
            base = [float(f["ms_gen"]) for f in g if f["kernel"] == k and int(f["hilos"]) == 0]
            if not base:
                continue
            ys = [base[0] / float(f["ms_gen"]) for h in hs for f in g if f["kernel"] == k and int(f["hilos"]) == h]
            ax.plot(hs, ys, marker="o", color=CAT[c], label=etiquetas[k])
        ax.axhline(1, color=EJE, lw=0.8)
        ejes_trabajadores(ax, hs)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Hilos (threading)")
        ax.set_ylabel("Speedup vs. mismo kernel secuencial")
        ax.set_title(f"Efecto del GIL ({g[0]['tamano']}×{g[0]['tamano']})")
        ax.legend()
        guardar(fig, "gil")


def grafica_sincronizacion():
    ruta = os.path.join(comun.RESULTADOS, "sincronizacion.csv")
    if not os.path.exists(ruta):
        return
    g = comun.leer_csv(ruta)
    hs = [int(f["hilos"]) for f in g]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(hs, [float(f["us_threading"]) for f in g], marker="o", color=CAT[0],
            label="threading.Barrier (Python)")
    ax.plot(hs, [float(f["us_nativo"]) for f in g], marker="o", color=CAT[2],
            label="fork-join nativo (prange)")
    ejes_trabajadores(ax, hs)
    ax.set_yscale("log")
    ax.set_xlabel("Hilos")
    ax.set_ylabel("µs por generación (escala log)")
    ax.set_title("Costo de sincronizar una generación vacía")
    ax.legend()
    guardar(fig, "sincronizacion")


if __name__ == "__main__":
    main()
    grafica_sincronizacion()
