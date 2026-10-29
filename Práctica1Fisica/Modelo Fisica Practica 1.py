#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prácticas de resultados de física · FUN-01 Física I · CENFOTEC · C3-2026
Grupo 6 — Detección y frenado antes de un obstáculo

MODELO FÍSICO
  Fase 1 (detección y procesamiento):  a = 0      ->  x(t) = v0·t
  Fase 2 (frenado):                    a = -|a|   ->  x(t) = d_r + v0·τ - ½|a|τ²,  τ = t - t_r
  Después de detenerse:                x = d_total (constante)

USO
  python modelo_frenado.py             -> tabla de validación + gráficas + panel interactivo
  python modelo_frenado.py --guardar   -> guarda las figuras en ./salidas sin abrir ventanas

REQUISITOS
  pip install numpy matplotlib
"""

import os
import sys
import itertools
import numpy as np
import matplotlib

if "--guardar" in sys.argv:          # modo sin ventanas (solo guarda imágenes)
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch

try:                                  # para que se vean ✔ y tildes en la consola de Windows
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# 1. PARÁMETROS DEL PROBLEMA (Sistema Internacional)
# ══════════════════════════════════════════════════════════════════════════════
MASA = 650.0              # kg
V0_KMH = 36.0             # km/h
V0 = V0_KMH / 3.6         # m/s  -> 10 m/s
D = 24.0                  # m, distancia inicial al obstáculo
A = 3.5                   # m/s², desaceleración de frenado (valor absoluto)
MARGEN = 2.0              # m, margen mínimo de seguridad
TR_BASE = 0.60            # s, tiempo de detección y procesamiento (caso base)
TR_MEJORA = 0.30          # s, escenario de mejora

# Paleta de colores
C_BASE = "#E4572E"
C_MEJORA = "#1F78B4"
C_OK = "#2BA84A"
C_NO = "#D62839"
C_PROC = "#F4B942"
C_FRENO = "#EF476F"
C_TEXTO = "#1D2433"
C_GRID = "#D9DEE7"
C_FONDO = "#F6F8FB"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#9AA3B2",
    "axes.labelcolor": C_TEXTO,
    "axes.titleweight": "bold",
    "axes.titlesize": 11.5,
    "axes.facecolor": "white",
    "figure.facecolor": C_FONDO,
    "xtick.color": C_TEXTO,
    "ytick.color": C_TEXTO,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODELO FÍSICO
# ══════════════════════════════════════════════════════════════════════════════
def evaluar(t_r, a=A, v0=V0, masa=MASA):
    """Calcula todas las magnitudes del problema para un tiempo de procesamiento t_r."""
    d_r = v0 * t_r                       # distancia durante el procesamiento
    t_f = v0 / a                         # tiempo de frenado
    d_f = v0 ** 2 / (2 * a)              # distancia de frenado
    d_total = d_r + d_f                  # distancia total de detención
    restante = D - d_total               # espacio que queda hasta el obstáculo
    k_i = 0.5 * masa * v0 ** 2           # energía cinética inicial
    return {
        "v0": v0, "t_r": t_r, "a": a,
        "d_r": d_r, "t_f": t_f, "d_f": d_f,
        "d_total": d_total, "restante": restante,
        "cumple": restante >= MARGEN,
        "F": masa * a,                   # fuerza neta media de frenado
        "K": k_i,
        "W": -k_i,                       # trabajo neto de frenado (teorema trabajo-energía)
        "t_total": t_r + t_f,
    }


def t_r_maximo(a=A, v0=V0):
    """Mayor tiempo de procesamiento que todavía respeta el margen de seguridad."""
    return (D - MARGEN - v0 ** 2 / (2 * a)) / v0


def trayectoria(t_r, a=A, v0=V0, extra=0.8, n=700):
    """Posición y velocidad por tramos en función del tiempo."""
    r = evaluar(t_r, a, v0)
    t = np.linspace(0, r["t_total"] + extra, n)
    tau = t - t_r                                        # tiempo desde que inicia el frenado
    x = np.where(t <= t_r, v0 * t,
                 np.where(tau <= r["t_f"], r["d_r"] + v0 * tau - 0.5 * a * tau ** 2, r["d_total"]))
    v = np.where(t <= t_r, v0,
                 np.where(tau <= r["t_f"], v0 - a * tau, 0.0))
    return t, x, v, r


# ══════════════════════════════════════════════════════════════════════════════
# 3. VALIDACIÓN: CÁLCULOS A MANO vs MODELO
# ══════════════════════════════════════════════════════════════════════════════
def tabla_validacion():
    b, m = evaluar(TR_BASE), evaluar(TR_MEJORA)
    # (concepto, valor a mano, unidad, valor del modelo, tolerancia = media unidad de la última cifra)
    filas = [
        ("Velocidad inicial v0",              10.0,    "m/s", V0,           0.05),
        ("Distancia de procesamiento d_r",     6.0,    "m",   b["d_r"],     0.05),
        ("Tiempo de frenado t_f",              2.86,   "s",   b["t_f"],     0.005),
        ("Distancia de frenado d_f",          14.3,    "m",   b["d_f"],     0.05),
        ("Distancia total de detención",      20.3,    "m",   b["d_total"], 0.05),
        ("Espacio restante (t_r = 0,60 s)",    3.7,    "m",   b["restante"], 0.05),
        ("Fuerza neta media de frenado",    2275.0,    "N",   b["F"],       0.5),
        ("Energía cinética inicial",       32500.0,    "J",   b["K"],       0.5),
        ("Trabajo neto de frenado",       -32500.0,    "J",   b["W"],       0.5),
        ("d_r (t_r = 0,30 s)",                 3.0,    "m",   m["d_r"],     0.05),
        ("Distancia total (t_r = 0,30 s)",    17.3,    "m",   m["d_total"], 0.05),
        ("Espacio restante (t_r = 0,30 s)",    6.7,    "m",   m["restante"], 0.05),
        ("Tiempo máximo de procesamiento",     0.77,   "s",   t_r_maximo(), 0.005),
    ]
    ancho = 74
    print("\n" + "═" * ancho)
    print(" VALIDACIÓN: CÁLCULOS A MANO vs MODELO (Grupo 6 · Detección y frenado)")
    print("═" * ancho)
    print(f" {'Concepto':<36}{'A mano':>11}{'Modelo':>13}  {'Unidad':<6}{'':>3}")
    print("─" * ancho)
    todo_ok = True
    for nombre, mano, unidad, modelo, tol in filas:
        ok = abs(modelo - mano) <= tol
        todo_ok &= ok
        print(f" {nombre:<36}{mano:>11,.2f}{modelo:>13,.2f}  {unidad:<6}{'✔' if ok else '✘':>3}")
    print("─" * ancho)
    print(" Resultado:", "todos los valores coinciden con los cálculos a mano ✔" if todo_ok
          else "hay diferencias, revisar ✘")
    print("═" * ancho + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 4. FIGURA ESTÁTICA PARA EL INFORME
# ══════════════════════════════════════════════════════════════════════════════
def sombrear_fases(ax, t_r, t_f):
    ax.axvspan(0, t_r, color=C_PROC, alpha=0.18, lw=0)
    ax.axvspan(t_r, t_r + t_f, color=C_FRENO, alpha=0.10, lw=0)


def figura_informe(ruta=None):
    fig = plt.figure(figsize=(14.5, 9.2))
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.97, top=0.88, bottom=0.08,
                  hspace=0.38, wspace=0.18)
    fig.suptitle("Detección y frenado antes de un obstáculo", fontsize=19, fontweight="bold",
                 color=C_TEXTO, y=0.975)
    fig.text(0.5, 0.925, "FUN-01 Física I · CENFOTEC · Grupo 6 · v₀ = 10 m/s · |a| = 3,5 m/s² · "
             "obstáculo a 24 m · margen mínimo 2,0 m", ha="center", fontsize=10.5, color="#586174")

    base = trayectoria(TR_BASE)
    mejora = trayectoria(TR_MEJORA)
    rb, rm = base[3], mejora[3]

    # --- A) Posición vs tiempo
    ax = fig.add_subplot(gs[0, 0])
    sombrear_fases(ax, TR_BASE, rb["t_f"])
    ax.plot(base[0], base[1], color=C_BASE, lw=2.8, label=f"Base  (t_r = {TR_BASE:.2f} s)")
    ax.plot(mejora[0], mejora[1], color=C_MEJORA, lw=2.8, label=f"Mejora (t_r = {TR_MEJORA:.2f} s)")
    ax.axhline(D, color=C_TEXTO, ls="--", lw=1.4)
    ax.axhline(D - MARGEN, color="#E67E22", ls="--", lw=1.4)
    ax.text(0.05, D + 0.35, "Obstáculo (24 m)", fontsize=9, color=C_TEXTO)
    ax.text(0.05, D - MARGEN - 1.25, "Límite de seguridad (22 m)", fontsize=9, color="#E67E22")
    for r, c in ((rb, C_BASE), (rm, C_MEJORA)):
        ax.plot(r["t_total"], r["d_total"], "o", color=c, ms=8, mec="white", mew=1.5, zorder=5)
        ax.annotate(f"{r['d_total']:.1f} m", (r["t_total"], r["d_total"]), xytext=(8, -16),
                    textcoords="offset points", color=c, fontsize=10, fontweight="bold")
    ax.text(TR_BASE / 2, 1.2, "procesamiento", ha="center", fontsize=8.5, color="#9A6B00", rotation=90)
    ax.text(TR_BASE + rb["t_f"] / 2, 1.0, "frenado (caso base)", ha="center", fontsize=8.5, color="#B23A5A")
    ax.set_title("Posición en función del tiempo")
    ax.set_xlabel("t (s)"); ax.set_ylabel("x (m)")
    ax.set_xlim(0, 4.3); ax.set_ylim(0, 27)
    ax.grid(color=C_GRID, lw=0.8); ax.legend(loc="lower right", frameon=True, framealpha=0.95)

    # --- B) Velocidad vs tiempo
    ax = fig.add_subplot(gs[0, 1])
    sombrear_fases(ax, TR_BASE, rb["t_f"])
    ax.plot(base[0], base[2], color=C_BASE, lw=2.8, label="Base")
    ax.plot(mejora[0], mejora[2], color=C_MEJORA, lw=2.8, label="Mejora")
    ax.set_title("Velocidad en función del tiempo")
    ax.set_xlabel("t (s)"); ax.set_ylabel("v (m/s)")
    ax.set_xlim(0, 4.3); ax.set_ylim(0, 11.5)
    ax.annotate("v constante\n(a = 0)", (TR_BASE / 2, 10), xytext=(TR_BASE / 2 + 0.55, 6.2),
                fontsize=9, color="#9A6B00", ha="center",
                arrowprops=dict(arrowstyle="->", color="#9A6B00"))
    ax.annotate("pendiente = −3,5 m/s²", (TR_BASE + 1.4, 10 - A * 1.4), xytext=(TR_BASE + 1.95, 7.6),
                fontsize=9, color="#B23A5A", ha="left",
                arrowprops=dict(arrowstyle="->", color="#B23A5A"))
    ax.grid(color=C_GRID, lw=0.8); ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    # --- C) Barrido de t_r
    ax = fig.add_subplot(gs[1, 0])
    t_rs = np.linspace(0.05, 1.1, 300)
    rest = np.array([evaluar(t)["restante"] for t in t_rs])
    ax.plot(t_rs, rest, color=C_TEXTO, lw=2.6)
    ax.fill_between(t_rs, rest, MARGEN, where=rest >= MARGEN, color=C_OK, alpha=0.22, interpolate=True)
    ax.fill_between(t_rs, rest, MARGEN, where=rest < MARGEN, color=C_NO, alpha=0.22, interpolate=True)
    ax.axhline(MARGEN, color="#E67E22", ls="--", lw=1.4)
    ax.text(1.09, MARGEN + 0.25, "margen mínimo 2,0 m", ha="right", color="#E67E22", fontsize=9)
    tmax = t_r_maximo()
    ax.axvline(tmax, color=C_NO, ls=":", lw=1.6)
    ax.annotate(f"t_r máx ≈ {tmax:.2f} s", (tmax, MARGEN), xytext=(tmax + 0.05, 6.4),
                fontsize=10, color=C_NO, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_NO))
    for tr, c, nm in ((TR_BASE, C_BASE, "Base"), (TR_MEJORA, C_MEJORA, "Mejora")):
        y = evaluar(tr)["restante"]
        ax.plot(tr, y, "o", color=c, ms=9, mec="white", mew=1.5, zorder=5)
        ax.annotate(f"{nm}: {y:.1f} m", (tr, y), xytext=(10, 9), textcoords="offset points",
                    color=c, fontsize=10, fontweight="bold", ha="left")
    ax.set_title("Variación de t_r: ¿hasta cuándo se cumple el margen?")
    ax.set_xlabel("Tiempo de procesamiento t_r (s)"); ax.set_ylabel("Espacio restante (m)")
    ax.set_xlim(0.05, 1.1); ax.grid(color=C_GRID, lw=0.8)

    # --- D) Reparto de los 24 m
    ax = fig.add_subplot(gs[1, 1])
    for i, (nombre, r, c) in enumerate((("Mejora\n t_r = 0,30 s", rm, C_MEJORA),
                                        ("Base\n t_r = 0,60 s", rb, C_BASE))):
        izq = 0
        segmentos = [(r["d_r"], C_PROC, f"d_r\n{r['d_r']:.1f} m"),
                     (r["d_f"], C_FRENO, f"d_f\n{r['d_f']:.1f} m"),
                     (r["restante"] - MARGEN, C_OK, f"holgura\n{r['restante'] - MARGEN:.1f} m"),
                     (MARGEN, "#C9CED8", f"margen\n{MARGEN:.1f} m")]
        for ancho, col, txt in segmentos:
            ax.barh(i, ancho, left=izq, color=col, edgecolor="white", height=0.55, lw=2)
            if ancho < 2.6:                       # segmento angosto: solo el valor
                txt = f"{ancho:.1f}"
            ax.text(izq + ancho / 2, i, txt, ha="center", va="center",
                    fontsize=8.5 if ancho < 2.6 else 9,
                    color="white" if col in (C_FRENO, C_OK) else C_TEXTO, fontweight="bold")
            izq += ancho
        ax.text(24.4, i, "CUMPLE ✔" if r["cumple"] else "NO CUMPLE ✘", va="center",
                fontsize=11, fontweight="bold", color=C_OK if r["cumple"] else C_NO)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Mejora\n t_r = 0,30 s", "Base\n t_r = 0,60 s"])
    ax.set_xlim(0, 28); ax.set_ylim(-0.85, 1.6)
    ax.axvline(D, color=C_TEXTO, lw=2)
    ax.text(D, 1.52, "obstáculo", ha="center", fontsize=9, color=C_TEXTO)
    ax.set_xlabel("Distancia desde el punto de detección (m)")
    ax.set_title("Cómo se reparten los 24 m disponibles")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_PROC, label="d_r procesamiento"), Patch(color=C_FRENO, label="d_f frenado"),
                       Patch(color=C_OK, label="holgura"), Patch(color="#C9CED8", label="margen mínimo")],
              loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    ax.grid(axis="x", color=C_GRID, lw=0.8); ax.set_axisbelow(True)

    fig.text(0.5, 0.008,
             "Zonas sombreadas: amarillo = procesamiento (v constante) · rosado = frenado (a constante)",
             ha="center", fontsize=9, color="#586174")
    if ruta:
        fig.savefig(ruta, dpi=170, facecolor=fig.get_facecolor())
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. PANEL INTERACTIVO CON ANIMACIÓN Y DESLIZADORES
# ══════════════════════════════════════════════════════════════════════════════
def panel_interactivo(ruta_snapshot=None):
    fig = plt.figure(figsize=(14.5, 9))
    fig.suptitle("Simulador interactivo · detección y frenado", fontsize=18,
                 fontweight="bold", color=C_TEXTO, y=0.965)

    ax_lane = fig.add_axes([0.04, 0.62, 0.92, 0.27])
    ax_x = fig.add_axes([0.065, 0.27, 0.40, 0.27])
    ax_v = fig.add_axes([0.565, 0.27, 0.40, 0.27])
    ax_info = fig.add_axes([0.58, 0.02, 0.38, 0.17]); ax_info.axis("off")

    # ---- Carril (vista superior) ----
    ax_lane.set_xlim(-3, D + 4); ax_lane.set_ylim(-1.5, 4.4)
    ax_lane.axis("off")
    ax_lane.add_patch(Rectangle((-3, -0.15), D + 7, 0.15, color="#3A4150"))          # asfalto
    ax_lane.add_patch(Rectangle((-3, -0.15), D + 7, 2.9, color="#EEF1F6", zorder=0))  # carril
    ax_lane.plot([-3, D + 4], [2.75, 2.75], color="#C9CED8", lw=2)
    zona_margen = Rectangle((D - MARGEN, 0), MARGEN, 2.6, color="#E67E22", alpha=0.20, lw=0)
    ax_lane.add_patch(zona_margen)
    ax_lane.plot([D - MARGEN] * 2, [0, 2.6], color="#E67E22", ls="--", lw=1.4)
    ax_lane.text(D - MARGEN - 0.2, 2.95, "margen mínimo 2,0 m", ha="right", fontsize=8.5, color="#E67E22")
    ax_lane.add_patch(Rectangle((D, 0), 0.8, 2.6, fc="#D62839", ec="#7A0F1B", hatch="///", lw=1.4, zorder=3))
    ax_lane.text(D + 0.4, 3.1, "OBSTÁCULO", ha="center", fontsize=8.5, color="#7A0F1B", fontweight="bold")
    ax_lane.plot([0, 0], [0, 2.6], color=C_TEXTO, ls=":", lw=1.3)
    ax_lane.text(0, 2.95, "detección\nx = 0", ha="center", fontsize=8.5, color=C_TEXTO)

    # Barra de fases bajo el carril
    seg_proc = Rectangle((0, -1.15), 1, 0.7, fc=C_PROC, ec="white", lw=2)
    seg_freno = Rectangle((0, -1.15), 1, 0.7, fc=C_FRENO, ec="white", lw=2)
    seg_rest = Rectangle((0, -1.15), 1, 0.7, fc=C_OK, ec="white", lw=2)
    for s in (seg_proc, seg_freno, seg_rest):
        ax_lane.add_patch(s)
    txt_proc = ax_lane.text(0, -0.8, "", ha="center", va="center", fontsize=9, fontweight="bold", color=C_TEXTO)
    txt_freno = ax_lane.text(0, -0.8, "", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    txt_rest = ax_lane.text(0, -0.8, "", ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    # Vehículo
    LARGO = 2.6
    cuerpo = FancyBboxPatch((-LARGO, 0.35), LARGO, 1.5, boxstyle="round,pad=0.02,rounding_size=0.35",
                            fc=C_MEJORA, ec="#0F3F63", lw=1.8, zorder=5)
    ventana = Rectangle((-0.9, 0.95), 0.55, 0.65, fc="#BFE3FF", ec="#0F3F63", lw=1, zorder=6)
    rueda1 = Circle((-LARGO + 0.6, 0.35), 0.3, fc="#1D2433", ec="white", lw=1, zorder=7)
    rueda2 = Circle((-0.5, 0.35), 0.3, fc="#1D2433", ec="white", lw=1, zorder=7)
    luz = Rectangle((-LARGO, 0.7), 0.16, 0.7, fc="#5A0F1B", ec="none", zorder=6)
    for p in (cuerpo, ventana, rueda1, rueda2, luz):
        ax_lane.add_patch(p)
    ghost = Rectangle((0, 0.35), LARGO, 1.5, fc="none", ec=C_TEXTO, ls="--", lw=1.2, alpha=0.5, zorder=4)
    ax_lane.add_patch(ghost)
    estado = ax_lane.text(0, 2.15, "", ha="center", fontsize=9.5, fontweight="bold", color=C_TEXTO, zorder=8)
    banner = ax_lane.text(D / 2, 3.85, "", ha="center", va="center", fontsize=14, fontweight="bold",
                          color="white", bbox=dict(boxstyle="round,pad=0.45", fc=C_OK, ec="none"))

    # ---- Gráficas x(t) y v(t) ----
    linea_x, = ax_x.plot([], [], color=C_MEJORA, lw=2.8)
    linea_v, = ax_v.plot([], [], color=C_MEJORA, lw=2.8)
    punto_x, = ax_x.plot([], [], "o", color=C_BASE, ms=9, mec="white", mew=1.5, zorder=6)
    punto_v, = ax_v.plot([], [], "o", color=C_BASE, ms=9, mec="white", mew=1.5, zorder=6)
    ax_x.axhline(D, color=C_TEXTO, ls="--", lw=1.3)
    ax_x.axhline(D - MARGEN, color="#E67E22", ls="--", lw=1.3)
    ax_x.set_title("Posición x(t)"); ax_x.set_xlabel("t (s)"); ax_x.set_ylabel("x (m)")
    ax_v.set_title("Velocidad v(t)"); ax_v.set_xlabel("t (s)"); ax_v.set_ylabel("v (m/s)")
    for a_ in (ax_x, ax_v):
        a_.grid(color=C_GRID, lw=0.8)
    sombras = []

    # ---- Controles ----
    ax_s1 = fig.add_axes([0.10, 0.145, 0.36, 0.022])
    ax_s2 = fig.add_axes([0.10, 0.095, 0.36, 0.022])
    ax_s3 = fig.add_axes([0.10, 0.045, 0.36, 0.022])
    s_tr = Slider(ax_s1, "t_r (s)", 0.05, 1.20, valinit=TR_BASE, valstep=0.01, color=C_PROC)
    s_a = Slider(ax_s2, "|a| (m/s²)", 1.5, 8.0, valinit=A, valstep=0.1, color=C_FRENO)
    s_v = Slider(ax_s3, "v₀ (km/h)", 10, 60, valinit=V0_KMH, valstep=1, color=C_MEJORA)
    ax_btn = fig.add_axes([0.505, 0.088, 0.07, 0.038])
    boton = Button(ax_btn, "↻ Reiniciar", color="#E3E8F1", hovercolor="#CBD5E6")

    info = ax_info.text(0, 1, "", va="top", fontsize=10.5, family="monospace", color=C_TEXTO)
    estado_sim = {"t": 0.0, "t_arr": None, "x_arr": None, "v_arr": None, "r": None}

    def recalcular(_=None):
        t_r, a, v0 = s_tr.val, s_a.val, s_v.val / 3.6
        t, x, v, r = trayectoria(t_r, a, v0)
        estado_sim.update(t_arr=t, x_arr=x, v_arr=v, r=r, t=0.0)
        linea_x.set_data(t, x); linea_v.set_data(t, v)
        ax_x.set_xlim(0, t[-1]); ax_v.set_xlim(0, t[-1])
        ax_x.set_ylim(0, max(D, r["d_total"]) + 3); ax_v.set_ylim(0, v0 * 1.15)
        for s in sombras:
            s.remove()
        sombras.clear()
        for a_ in (ax_x, ax_v):
            sombras.append(a_.axvspan(0, t_r, color=C_PROC, alpha=0.18, lw=0))
            sombras.append(a_.axvspan(t_r, r["t_total"], color=C_FRENO, alpha=0.10, lw=0))

        # barra de fases
        seg_proc.set_x(0); seg_proc.set_width(r["d_r"])
        seg_freno.set_x(r["d_r"]); seg_freno.set_width(r["d_f"])
        if r["restante"] > 0:
            seg_rest.set_x(r["d_total"]); seg_rest.set_width(r["restante"]); seg_rest.set_facecolor(C_OK)
        else:                                   # el vehículo atraviesa el obstáculo
            seg_rest.set_x(D); seg_rest.set_width(-r["restante"]); seg_rest.set_facecolor(C_NO)
        txt_proc.set_position((r["d_r"] / 2, -0.8)); txt_proc.set_text(f"d_r {r['d_r']:.1f} m")
        txt_freno.set_position((r["d_r"] + r["d_f"] / 2, -0.8)); txt_freno.set_text(f"d_f {r['d_f']:.1f} m")
        if r["restante"] > 0:
            txt_rest.set_position((r["d_total"] + r["restante"] / 2, -0.8))
            txt_rest.set_text(f"{r['restante']:.1f} m" if r["restante"] > 1.6 else "")
        else:
            txt_rest.set_position((D - r["restante"] / 2, -0.8)); txt_rest.set_text("")
        ghost.set_xy((r["d_total"] - LARGO, 0.35))

        # veredicto
        if r["cumple"]:
            banner.set_text(f"CUMPLE ✔   quedan {r['restante']:.2f} m  (mínimo {MARGEN:.1f} m)")
            banner.get_bbox_patch().set_facecolor(C_OK)
        elif r["restante"] > 0:
            banner.set_text(f"NO CUMPLE ✘   quedan {r['restante']:.2f} m  (mínimo {MARGEN:.1f} m)")
            banner.get_bbox_patch().set_facecolor("#E67E22")
        else:
            banner.set_text(f"COLISIÓN ✘   se pasa {-r['restante']:.2f} m del obstáculo")
            banner.get_bbox_patch().set_facecolor(C_NO)

        info.set_text(
            "  RESULTADOS CON LOS VALORES ACTUALES\n"
            f"  v₀ = {v0:5.2f} m/s        t_r = {t_r:4.2f} s\n"
            f"  d_r      = {r['d_r']:6.2f} m    t_f = {r['t_f']:5.2f} s\n"
            f"  d_f      = {r['d_f']:6.2f} m\n"
            f"  d_total  = {r['d_total']:6.2f} m\n"
            f"  restante = {r['restante']:6.2f} m\n"
            f"  F neta   = {r['F']:6.0f} N    K_i = {r['K']:7.0f} J\n"
            + (f"  t_r máx  = {t_r_maximo(a, v0):6.2f} s" if t_r_maximo(a, v0) > 0
               else "  t_r máx  = no existe (ni con t_r = 0 se cumple)"))
        dibujar(0.0)

    def dibujar(t_now):
        t, x, v, r = (estado_sim[k] for k in ("t_arr", "x_arr", "v_arr", "r"))
        xn, vn = np.interp(t_now, t, x), np.interp(t_now, t, v)
        cuerpo.set_x(xn - LARGO); ventana.set_x(xn - 0.9)
        rueda1.set_center((xn - LARGO + 0.6, 0.35)); rueda2.set_center((xn - 0.5, 0.35))
        luz.set_x(xn - LARGO)
        luz.set_facecolor("#FF2B2B" if (t_now > s_tr.val and vn > 0) else "#5A0F1B")
        if t_now <= s_tr.val:
            estado.set_text("Detectando y procesando…")
        elif vn > 0:
            estado.set_text("Frenando")
        else:
            estado.set_text("Detenido")
        estado.set_x(xn - LARGO / 2)
        punto_x.set_data([t_now], [xn]); punto_v.set_data([t_now], [vn])

    def paso(_frame):
        estado_sim["t"] += 0.03
        if estado_sim["t"] > estado_sim["t_arr"][-1] + 0.6:
            estado_sim["t"] = 0.0
        dibujar(min(estado_sim["t"], estado_sim["t_arr"][-1]))

    def reiniciar(_):
        for s in (s_tr, s_a, s_v):
            s.reset()

    for s in (s_tr, s_a, s_v):
        s.on_changed(recalcular)
    boton.on_clicked(reiniciar)
    recalcular()

    if ruta_snapshot:                            # captura fija (para el informe)
        dibujar(2.2)
        fig.savefig(ruta_snapshot, dpi=150, facecolor=fig.get_facecolor())
    if not ruta_snapshot:
        anim = FuncAnimation(fig, paso, frames=itertools.count(), interval=30, cache_frame_data=False)
        fig._anim = anim                         # evita que el recolector de basura detenga la animación
    fig._controles = (s_tr, s_a, s_v, boton)     # mantiene vivos los controles
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6. EJECUCIÓN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tabla_validacion()
    guardar = "--guardar" in sys.argv
    if guardar:
        carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salidas")
        os.makedirs(carpeta, exist_ok=True)
        figura_informe(os.path.join(carpeta, "figura_informe.png"))
        panel_interactivo(os.path.join(carpeta, "panel_interactivo.png"))
        print(f"Figuras guardadas en: {carpeta}")
    else:
        figura_informe()
        panel_interactivo()
        plt.show()