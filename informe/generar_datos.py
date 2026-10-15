"""
Convierte resultados/*.csv en archivos LaTeX que el informe incluye:

    informe/generado/datos.tex          macros con cifras (\\dato{clave})
    informe/generado/tabla_speedup.tex  speedup por implementación, tamaño y P
    informe/generado/tabla_tiempos.tex  ms/gen (anexo)
    informe/generado/tabla_karp.tex     fracción serial de Karp-Flatt

    python informe/generar_datos.py [--borrador]

Así, si se repiten las mediciones en otro equipo, el informe se actualiza solo
(salvo los párrafos de análisis marcados con "REVISAR" en informe.tex).
"""
import argparse
import os
import statistics
import sys
from collections import defaultdict

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "bench"))
import comun  # noqa: E402

GEN = os.path.join(AQUI, "generado")
IMPL = ["hilos_estatica", "hilos_dinamica", "nativo", "mpi"]
NOMBRE = {"hilos_estatica": "Hilos, estática", "hilos_dinamica": "Hilos, dinámica",
          "nativo": "Hilos nativos (prange)", "mpi": "MPI"}


def num(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def tex(s):
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    return s


def mediana_por(filas, *claves, valor="ms_gen"):
    d = defaultdict(list)
    for f in filas:
        if f.get(valor) not in (None, ""):
            d[tuple(f[c] for c in claves)].append(float(f[valor]))
    return {k: statistics.median(v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--borrador", action="store_true")
    a = ap.parse_args()
    os.makedirs(GEN, exist_ok=True)
    R = comun.RESULTADOS

    filas = comun.leer_csv(os.path.join(R, "hilos.csv"))
    if os.path.exists(os.path.join(R, "mpi.csv")):
        filas += comun.leer_csv(os.path.join(R, "mpi.csv"))
    for f in filas:
        f["tamano"], f["trabajadores"] = int(f["tamano"]), int(f["trabajadores"])
    ms = mediana_por(filas, "implementacion", "tamano", "trabajadores")
    tcom = mediana_por(filas, "implementacion", "tamano", "trabajadores", valor="t_comunicacion_s")
    ttot = mediana_por(filas, "implementacion", "tamano", "trabajadores", valor="tiempo_s")
    tamanos = sorted({n for (_, n, _) in ms})
    impls = [i for i in IMPL if any(k[0] == i for k in ms)]
    ps = sorted({p for (i, _, p) in ms if i != "secuencial"})
    seq = {n: ms[("secuencial", n, 1)] for n in tamanos}
    gens = {f["tamano"]: int(f["gens"]) for f in filas}
    reps = max(int(f["rep"]) for f in filas) + 1
    correctas = sum(1 for f in filas if f["correcto"] == "True")
    hw = comun.leer_json(os.path.join(R, "hardware.json"))

    D = {}
    D["cpu"] = tex(str(hw.get("cpu")))
    D["nucleos-log"] = str(hw.get("nucleos_logicos"))
    D["nucleos-fis"] = str(hw.get("nucleos_fisicos") or "?")
    D["sistema"] = tex(hw.get("sistema", ""))
    D["python"] = hw.get("python", "")
    D["numpy"] = hw.get("numpy", "")
    D["numba"] = hw.get("numba", "")
    D["mpi"] = tex(hw.get("mpi", ""))
    D["memoria"] = tex(hw.get("memoria", "no registrada"))
    D["reps"] = str(reps)
    D["corridas"] = str(len(filas))
    D["corridas-ok"] = str(correctas)
    D["pmax"] = str(ps[-1])
    D["ps"] = ", ".join(map(str, ps))
    D["tamanos"] = ", ".join(f"{n}$\\times${n}" for n in tamanos)
    D["nmax"] = str(tamanos[-1])
    D["nmin"] = str(tamanos[0])

    sp = lambda i, n, p: seq[n] / ms[(i, n, p)]  # noqa: E731
    for n in tamanos:
        D[f"seq-{n}"] = num(seq[n], 3)
        D[f"gens-{n}"] = str(gens[n])
        for i in impls:
            disp = [p for p in ps if (i, n, p) in ms]
            if not disp:
                continue
            mejor = max(disp, key=lambda p: sp(i, n, p))
            D[f"best-{i}-{n}"] = num(sp(i, n, mejor))
            D[f"bestp-{i}-{n}"] = str(mejor)
            if (i, n, ps[-1]) in ms:
                D[f"spmax-{i}-{n}"] = num(sp(i, n, ps[-1]))
                D[f"efmax-{i}-{n}"] = num(100 * sp(i, n, ps[-1]) / ps[-1], 0)
            for p in disp:
                D[f"sp-{i}-{n}-{p}"] = num(sp(i, n, p))
                D[f"ef-{i}-{n}-{p}"] = num(100 * sp(i, n, p) / p, 0)
    for n in tamanos:
        cand = [(i, p) for (i, m_, p) in ms if m_ == n and i != "secuencial"]
        i_b, p_b = max(cand, key=lambda k: sp(k[0], n, k[1]))
        D[f"bestany-{n}"] = num(sp(i_b, n, p_b))
        D[f"bestany-impl-{n}"] = NOMBRE[i_b]
        D[f"bestany-p-{n}"] = str(p_b)
    glob = max(((i, n, p) for (i, n, p) in ms if i != "secuencial"), key=lambda k: sp(*k))
    D["mejor-sp"] = num(sp(*glob))
    D["mejor-impl"] = NOMBRE[glob[0]]
    D["mejor-n"] = str(glob[1])
    D["mejor-p"] = str(glob[2])

    # MPI: porcentaje de tiempo en comunicación con el P máximo
    if "mpi" in impls:
        for n in tamanos:
            k = ("mpi", n, ps[-1])
            if k in tcom and k in ttot:
                D[f"com-{n}"] = num(100 * tcom[k] / ttot[k], 0)
            for p in ps:
                k = ("mpi", n, p)
                if k in tcom and k in ttot:
                    D[f"com-{n}-{p}"] = num(100 * tcom[k] / ttot[k], 0)
    desb = mediana_por(filas, "implementacion", "tamano", "trabajadores", valor="desbalance")
    for (i, n, p), v in desb.items():
        D[f"desb-{i}-{n}-{p}"] = num(v)

    # Costo de sincronización y predicción simple T_P = T_1/P + s(P)
    ruta = os.path.join(R, "sincronizacion.csv")
    if os.path.exists(ruta):
        sinc = {int(f["hilos"]): (float(f["us_threading"]), float(f["us_nativo"]))
                for f in comun.leer_csv(ruta)}
        for T, (py, nat) in sinc.items():
            D[f"sync-py-{T}"] = num(py, 1)
            D[f"sync-nat-{T}"] = num(nat, 1)
        tmax = max(sinc)
        D["sync-tmax"] = str(tmax)
        n_ej = 1000 if 1000 in tamanos else tamanos[len(tamanos) // 2]
        D["pred-n"] = str(n_ej)
        D["sync-razon"] = f"{sinc[tmax][0] / sinc[tmax][1]:.0f}"
        for etiqueta, T in (("4", 4), ("max", tmax)):
            if T in sinc and ("hilos_estatica", n_ej, T) in ms:
                pred = seq[n_ej] / (seq[n_ej] / T + sinc[T][0] / 1000)
                D[f"pred{etiqueta}-sp"] = num(pred)
                D[f"pred{etiqueta}-real"] = num(sp("hilos_estatica", n_ej, T))
                D[f"pred{etiqueta}-ideal-ms"] = num(seq[n_ej] / T, 3)

    # Experimento del GIL
    ruta = os.path.join(R, "gil.csv")
    if os.path.exists(ruta):
        g = comun.leer_csv(ruta)
        base = {f["kernel"]: float(f["ms_gen"]) for f in g if f["hilos"] == "0"}
        hmax = max(int(f["hilos"]) for f in g)
        D["gil-n"] = g[0]["tamano"]
        D["gil-hmax"] = str(hmax)
        for f in g:
            if f["hilos"] != "0":
                D[f"gil-{f['kernel']}-{f['hilos']}"] = num(base[f["kernel"]] / float(f["ms_gen"]))
        for k, v in base.items():
            D[f"gil-ms-{k}"] = num(v, 3 if v < 10 else 1)
        if "python" in base and "numba" in base:
            D["gil-razon"] = f"{base['python'] / base['numba']:.0f}"

    # Demostración de condición de carrera
    ruta = os.path.join(R, "carrera.csv")
    if os.path.exists(ruta):
        c = comun.leer_csv(ruta)
        sb = [int(f["celdas_distintas"]) for f in c if f["variante"] == "sin_barrera"]
        D["carr-min"], D["carr-max"] = f"{min(sb):,}".replace(",", "."), f"{max(sb):,}".replace(",", ".")
        D["carr-mismo"] = f"{int(next(f['celdas_distintas'] for f in c if f['variante'] == 'mismo_arreglo')):,}".replace(",", ".")
        D["carr-ok"] = str(sum(int(f["celdas_distintas"]) for f in c if f["variante"] == "correcta"))
        D["carr-n"], D["carr-gens"], D["carr-hilos"] = c[0]["tamano"], c[0]["gens"], c[0]["hilos"]
        D["carr-intentos"] = str(len(sb))

    with open(os.path.join(GEN, "datos.tex"), "w") as f:
        f.write("% Generado por informe/generar_datos.py — no editar a mano\n")
        f.write("\\newif\\ifborrador\\borrador" + ("true" if a.borrador else "false") + "\n")
        for k, v in D.items():
            f.write(f"\\expandafter\\def\\csname dato@{k}\\endcsname{{{v}}}\n")

    # ------------------------------------------------- tabla de speedup
    with open(os.path.join(GEN, "tabla_speedup.tex"), "w") as f:
        f.write("\\begin{tabular}{ll" + "r" * len(ps) + "}\n\\toprule\n")
        f.write("Tablero & Implementación & " + " & ".join(f"$P={p}$" for p in ps) + " \\\\\n\\midrule\n")
        for n in tamanos:
            for k, i in enumerate(impls):
                celdas = []
                mejor = max((sp(i, n, p) for p in ps if (i, n, p) in ms), default=0)
                for p in ps:
                    if (i, n, p) in ms:
                        v = sp(i, n, p)
                        s = num(v)
                        celdas.append(f"\\textbf{{{s}}}" if v == mejor and v > 1 else s)
                    else:
                        celdas.append("--")
                cab = f"\\multirow{{{len(impls)}}}{{*}}{{\\shortstack[l]{{{n}$\\times${n}\\\\\\footnotesize sec.: {num(seq[n], 3)} ms}}}}" if k == 0 else ""
                f.write(f"{cab} & {NOMBRE[i]} & " + " & ".join(celdas) + " \\\\\n")
            f.write("\\midrule\n" if n != tamanos[-1] else "\\bottomrule\n")
        f.write("\\end{tabular}\n")

    # ------------------------------------------------- tabla de tiempos
    with open(os.path.join(GEN, "tabla_tiempos.tex"), "w") as f:
        f.write("\\begin{tabular}{lrl" + "r" * len(ps) + "}\n\\toprule\n")
        f.write("Tablero & Gens. & Implementación & " + " & ".join(f"$P={p}$" for p in ps) + " \\\\\n\\midrule\n")
        for n in tamanos:
            f.write(f"{n}$\\times${n} & {gens[n]} & Secuencial & {num(seq[n], 4)} & "
                    + " & ".join([""] * (len(ps) - 1)) + " \\\\\n")
            for i in impls:
                f.write(f" & & {NOMBRE[i]} & " + " & ".join(
                    num(ms[(i, n, p)], 4) if (i, n, p) in ms else "--" for p in ps) + " \\\\\n")
            f.write("\\midrule\n" if n != tamanos[-1] else "\\bottomrule\n")
        f.write("\\end{tabular}\n")

    # ------------------------------------------------- Karp-Flatt
    ps2 = [p for p in ps if p > 1]
    ns = [n for n in tamanos if n >= 500] or tamanos
    with open(os.path.join(GEN, "tabla_karp.tex"), "w") as f:
        f.write("\\begin{tabular}{ll" + "r" * len(ps2) + "}\n\\toprule\n")
        f.write("Tablero & Implementación & " + " & ".join(f"$P={p}$" for p in ps2) + " \\\\\n\\midrule\n")
        for n in ns:
            for k, i in enumerate(impls):
                celdas = []
                for p in ps2:
                    if (i, n, p) in ms:
                        S = sp(i, n, p)
                        e = (1 / S - 1 / p) / (1 - 1 / p)
                        celdas.append(num(e, 3))
                    else:
                        celdas.append("--")
                cab = f"\\multirow{{{len(impls)}}}{{*}}{{{n}$\\times${n}}}" if k == 0 else ""
                f.write(f"{cab} & {NOMBRE[i]} & " + " & ".join(celdas) + " \\\\\n")
            f.write("\\midrule\n" if n != ns[-1] else "\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print("Archivos generados en", GEN)


if __name__ == "__main__":
    main()
