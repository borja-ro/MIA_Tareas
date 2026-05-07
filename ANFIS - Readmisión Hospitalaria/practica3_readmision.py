"""
================================================================================
  PRÁCTICA 3 — ANFIS · Predicción de Riesgo de Readmisión Hospitalaria
  IA Explicable en Entornos Clínicos
  Módulo Inteligencia Artificial · CE IA · Curso 2025-2026
================================================================================

Sistema ANFIS de 3 entradas (Estancia, Comorbilidades, Adherencia) y 4 reglas
difusas, entrenado sobre 500 muestras sintéticas que codifican el criterio
clínico experto del jefe de servicio. Objetivo: estimar el riesgo de readmisión
en 30 días con un modelo auditable regla a regla.

Ejecución:
    python practica3_readmision.py

Salidas:
    convergencia.png         — Curva de convergencia (Apt. 1.4)
    funciones_pertenencia.png — MFs gaussianas aprendidas (extra)
    superficie_riesgo.png    — Heatmap C×A con E fijo (extra)
    Stdout: tabla de reglas, σ promedio, 3 informes, what-if
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Reproducibilidad global
SEED = 7
np.random.seed(SEED)
torch.manual_seed(SEED)
console = Console()


# =============================================================================
# APARTADO 1 — DATASET Y MODELO ANFIS
# =============================================================================

# -----------------------------------------------------------------------------
# 1.1  Generación del dataset sintético
# -----------------------------------------------------------------------------
# Los parámetros de cada gaussiana se eligen para codificar las 4 reglas.
#
# NOTA SOBRE σ² — desviación deliberada de las pistas del enunciado:
# Las pistas sugieren σ² ≈ 0.04–0.06 (gaussianas muy estrechas). Con esos
# valores, las gaussianas solo se activan en los extremos puros (E≈0.85,
# C≈0.80, etc.), pero las distribuciones beta(2,4) y beta(2,3) generan
# muy pocas muestras en esa zona. Resultado: el dataset acaba con < 0.5%
# de pacientes en riesgo > 0.6 y los perfiles de prueba marcados como
# CRÍTICO o MEDIO-ALTO no son distinguibles. El enunciado lo permite
# explícitamente: "no hay una única solución correcta; se evalúa la
# coherencia con las 4 reglas clínicas". Ensancho σ² ≈ 0.12–0.15 para
# que las reglas tengan radios de acción realistas (los pacientes
# pluripatológicos no están solo en C=0.80, también en C=0.60+).
#
#   R1  Estancia MUY LARGA  (E > 0.7)               → riesgo alto
#         pico en E ≈ 0.85  (≈17 días),  σ² = 0.15
#   R2a Comorbilidades ALTAS (C > 0.6)              → riesgo alto
#         pico en C ≈ 0.80,             σ² = 0.15
#   R2b Adherencia BAJA (A < 0.4) actúa como        → multiplicador
#         pico en A ≈ 0.10,             σ² = 0.10   (más estrecha:
#                                                    la "mala adherencia"
#                                                    sí es un extremo)
#   R3  Adherencia ALTA (A > 0.7)                   → protección
#         pico en A ≈ 0.85,             σ² = 0.12
#
#   R4  Pesos de combinación: C(0.50) > E(0.35) >> A(0.15)
#       (la adherencia entra fundamentalmente como modulador de R2 vía
#        proteccion_adher, no como término aditivo dominante)
# -----------------------------------------------------------------------------

N = 500

# Distribuciones beta acordes al enunciado (estancias y comorbilidades sesgadas
# a valores bajos; adherencia sesgada a valores altos).
#
# SEGUNDA DESVIACIÓN DEL ENUNCIADO — muestreo mixto:
# Las betas literales (beta(2,4) para E, beta(2,3) para C) generan ~0
# pacientes en el perfil pluripatológico extremo (E>0.7 ∧ C>0.6 ∧ A<0.4).
# Sin esos casos en train, el modelo no puede aprender a predecir riesgo
# CRÍTICO (>0.8) y los perfiles tipo García Ruiz se subestiman.
# Solución: 90% siguen siendo betas puras (mantiene la distribución
# poblacional realista del enunciado) + 10% son muestras uniformes
# concentradas en la zona crítica, simulando el sesgo de inclusión que
# ocurre en la práctica al recolectar datos hospitalarios (los pacientes
# de alta complejidad están sobrerrepresentados respecto a la población
# general porque generan más episodios de hospitalización).
n_pop  = int(0.9 * N)             # 90% población general (betas)
n_crit = N - n_pop                # 10% pacientes complejos (uniforme zona alta)

E = np.concatenate([np.random.beta(2, 4, n_pop),
                    np.random.uniform(0.50, 1.00, n_crit)])
C = np.concatenate([np.random.beta(2, 3, n_pop),
                    np.random.uniform(0.40, 1.00, n_crit)])
A = np.concatenate([np.random.beta(3, 2, n_pop),
                    np.random.uniform(0.00, 1.00, n_crit)])

# --- Reglas codificadas como gaussianas ---
# R1: pico en E≈0.85 (estancia muy larga ≈ 17 días)
riesgo_estancia    = np.exp(-((E - 0.85) ** 2) / 0.15)

# R2: pico en C≈0.80 (paciente pluripatológico)
riesgo_comorb      = np.exp(-((C - 0.80) ** 2) / 0.15)

# R2 (penalización por mala adherencia): pico en A≈0.10
penaliz_adherencia = np.exp(-((A - 0.10) ** 2) / 0.10)

# R3: pico en A≈0.85 (buena adherencia → protección)
proteccion_adher   = np.exp(-((A - 0.85) ** 2) / 0.12)

# Combinación con los pesos del criterio clínico (R4: C > E >> A)
riesgo = (
    0.35 * riesgo_estancia +
    0.50 * riesgo_comorb * (1 - 0.6 * proteccion_adher) +
    0.15 * penaliz_adherencia +
    0.04 * np.random.randn(N)
).clip(0, 1)

# --- Split 80/20 reproducible ---
idx = np.random.permutation(N)
n_train = int(0.8 * N)
train_idx, test_idx = idx[:n_train], idx[n_train:]

X = np.stack([E, C, A], axis=1).astype(np.float32)
y = riesgo.astype(np.float32)

X_train = torch.tensor(X[train_idx])
y_train = torch.tensor(y[train_idx])
X_test  = torch.tensor(X[test_idx])
y_test  = torch.tensor(y[test_idx])

console.rule("[bold cyan]APARTADO 1 — DATASET Y MODELO ANFIS[/bold cyan]")

tabla_dataset = Table(
    title="Resumen del dataset sintético",
    box=box.ROUNDED,
    show_header=True,
    header_style="bold cyan",
)
tabla_dataset.add_column("Métrica", style="bold")
tabla_dataset.add_column("Valor", justify="right")
tabla_dataset.add_row("Muestras totales", str(N))
tabla_dataset.add_row("Train / Test", f"{len(X_train)} / {len(X_test)}")
tabla_dataset.add_row("Riesgo medio", f"{y.mean():.3f}  (σ={y.std():.3f})")
tabla_dataset.add_row(
    "Riesgo > 0.6 — alerta",
    f"{(y > 0.6).sum()} pacientes ({100*(y > 0.6).mean():.1f}%)",
)
tabla_dataset.add_row(
    "Riesgo > 0.8 — crítico",
    f"{(y > 0.8).sum()} pacientes ({100*(y > 0.8).mean():.1f}%)",
)
console.print(tabla_dataset)


# -----------------------------------------------------------------------------
# 1.2  Clase ANFISLayer  (idéntica al documento de referencia + clamp en σ)
# -----------------------------------------------------------------------------
class ANFISLayer(nn.Module):
    """ANFIS de orden 1 (Takagi-Sugeno) con MFs gaussianas y producto AND."""

    def __init__(self, n_inputs, n_rules):
        super().__init__()
        self.mu    = nn.Parameter(torch.randn(n_rules, n_inputs))
        self.sigma = nn.Parameter(torch.ones(n_rules, n_inputs) * 0.5)
        self.p     = nn.Parameter(torch.randn(n_rules, n_inputs + 1) * 0.1)

    def forward(self, x):
        # Protección numérica: con 3 entradas, sigmas pequeños colapsan w
        sigma = torch.clamp(self.sigma, min=1e-3)

        # Capa 1 — Fuzzificación gaussiana
        diff    = x.unsqueeze(1) - self.mu                  # (B, R, N)
        mu_vals = torch.exp(-diff ** 2 / (2 * sigma ** 2))  # μ_ij(x)

        # Capa 2 — Disparo por regla (AND = producto)
        w = mu_vals.prod(dim=2)                             # (B, R)

        # Capa 3 — Normalización
        w_bar = w / (w.sum(dim=1, keepdim=True) + 1e-8)

        # Capa 4 — Consecuentes lineales (TSK orden 1)
        x_ext = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        y_i   = (x_ext.unsqueeze(1) * self.p).sum(dim=2)    # (B, R)

        # Capa 5 — Salida desfuzzificada
        return (w_bar * y_i).sum(dim=1)

    def get_firing_strengths(self, x):
        """Devuelve los pesos normalizados w_bar (activación por regla)."""
        with torch.no_grad():
            sigma   = torch.clamp(self.sigma, min=1e-3)
            diff    = x.unsqueeze(1) - self.mu
            mu_vals = torch.exp(-diff ** 2 / (2 * sigma ** 2))
            w       = mu_vals.prod(dim=2)
            return (w / (w.sum(dim=1, keepdim=True) + 1e-8)).numpy()


# -----------------------------------------------------------------------------
# 1.3  Configuración + entrenamiento
# -----------------------------------------------------------------------------
N_RULES = 4   # tantas como reglas formuló el clínico

def inicializar_kmeans(model, X_numpy, n_rules):
    """Inicializa los centros de las MFs con K-means (obligatorio con 3 inp)."""
    kmeans = KMeans(n_clusters=n_rules, random_state=42, n_init=10)
    kmeans.fit(X_numpy)
    with torch.no_grad():
        model.mu.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))

# Re-fijamos la semilla aquí para que la inicialización aleatoria de los
# pesos del modelo (mu, sigma, p) sea reproducible, independientemente de
# las operaciones de torch realizadas antes (creación de tensores, etc.)
torch.manual_seed(SEED)
model = ANFISLayer(n_inputs=3, n_rules=N_RULES)
inicializar_kmeans(model, X_train.numpy(), n_rules=N_RULES)

optimizer = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)
criterion = nn.MSELoss()

losses = []
for epoch in range(600):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())


# -----------------------------------------------------------------------------
# 1.4  Evaluación + curva de convergencia
# -----------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train)
    y_pred_test  = model(X_test)
    rmse_train = (criterion(y_pred_train, y_train).item()) ** 0.5
    rmse_test  = (criterion(y_pred_test,  y_test ).item()) ** 0.5

# Baseline para contextualizar el RMSE (extra interpretativo)
lr_baseline = LinearRegression().fit(X[train_idx], y[train_idx])
rmse_lr     = mean_squared_error(y[test_idx],
                                 lr_baseline.predict(X[test_idx])) ** 0.5

tabla_entrenamiento = Table(
    title="Resultados del entrenamiento",
    box=box.ROUNDED,
    show_header=True,
    header_style="bold green",
)
tabla_entrenamiento.add_column("Métrica", style="bold")
tabla_entrenamiento.add_column("Valor", justify="right")
tabla_entrenamiento.add_row("RMSE train", f"{rmse_train:.4f}")
tabla_entrenamiento.add_row(
    "RMSE test (ANFIS)",
    f"{rmse_test:.4f}  {'✓ < 0.12' if rmse_test < 0.12 else '✗ >= 0.12'}",
)
tabla_entrenamiento.add_row("RMSE test baseline lineal", f"{rmse_lr:.4f}")
tabla_entrenamiento.add_row(
    "Mejora ANFIS vs baseline",
    f"{100*(rmse_lr-rmse_test)/rmse_lr:.1f}%",
)
console.print(tabla_entrenamiento)

# Curva de convergencia
plt.figure(figsize=(8, 4.5))
plt.plot(losses, linewidth=1.2)
plt.xlabel('Época')
plt.ylabel('MSE Loss (train)')
plt.title(f'Convergencia ANFIS — RMSE test = {rmse_test:.4f}')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('convergencia.png', dpi=150)
plt.close()
console.print("[green]✓[/green] Curva de convergencia guardada en [bold]convergencia.png[/bold]")


# =============================================================================
# APARTADO 2 — INTERPRETACIÓN DE LAS REGLAS APRENDIDAS
# =============================================================================
console.rule("[bold cyan]APARTADO 2 — INTERPRETACIÓN DE LAS REGLAS APRENDIDAS[/bold cyan]")

mu    = model.mu.detach().numpy()
sigma = torch.clamp(model.sigma, min=1e-3).detach().numpy()

# -----------------------------------------------------------------------------
# 2.1  Tabla de reglas aprendidas
# -----------------------------------------------------------------------------
tabla_reglas = Table(
    title="Tabla de reglas aprendidas",
    box=box.ROUNDED,
    show_header=True,
    header_style="bold magenta",
)
tabla_reglas.add_column("Regla", style="bold")
tabla_reglas.add_column("μ_E", justify="right")
tabla_reglas.add_column("σ_E", justify="right")
tabla_reglas.add_column("μ_C", justify="right")
tabla_reglas.add_column("σ_C", justify="right")
tabla_reglas.add_column("μ_A", justify="right")
tabla_reglas.add_column("σ_A", justify="right")
for i in range(N_RULES):
    tabla_reglas.add_row(
        f"R{i+1}",
        f"{mu[i,0]:.3f}", f"{sigma[i,0]:.3f}",
        f"{mu[i,1]:.3f}", f"{sigma[i,1]:.3f}",
        f"{mu[i,2]:.3f}", f"{sigma[i,2]:.3f}",
    )
console.print(tabla_reglas)


def etiquetar_regla(mu_regla, umbral_alto=0.6, umbral_bajo=0.4):
    """Genera una etiqueta legible a partir de los μ aprendidos.

    Mapea cada μ a un nivel cualitativo (alto / medio / bajo) por variable y
    construye una descripción del perfil clínico que la regla representa.
    """
    nombres   = ['E', 'C', 'A']
    etiquetas = []
    for nombre, val in zip(nombres, mu_regla):
        if val > umbral_alto:
            etiquetas.append(f"{nombre}↑")
        elif val < umbral_bajo:
            etiquetas.append(f"{nombre}↓")
        else:
            etiquetas.append(f"{nombre}=")
    return " ".join(etiquetas)


tabla_etiquetas = Table(
    title="Etiquetas inferidas por regla",
    box=box.SIMPLE_HEAVY,
    show_header=True,
    header_style="bold yellow",
)
tabla_etiquetas.add_column("Regla", style="bold")
tabla_etiquetas.add_column("Perfil clínico inferido")
tabla_etiquetas.add_column("Centro μ", justify="right")
for i in range(N_RULES):
    tabla_etiquetas.add_row(
        f"R{i+1}",
        etiquetar_regla(mu[i]),
        f"({mu[i,0]:.2f}, {mu[i,1]:.2f}, {mu[i,2]:.2f})",
    )
console.print(tabla_etiquetas)

# Identificación de la regla que más se acerca al criterio clínico 2
# (C alto AND A bajo): minimizar |μ_C - 1| + |μ_A - 0|
score_c2 = np.abs(mu[:, 1] - 1.0) + np.abs(mu[:, 2] - 0.0)
regla_c2 = int(np.argmin(score_c2))
console.print(
    Panel.fit(
        f"Regla más alineada con el criterio clínico 2 "
        f"([bold]C alto + A bajo[/bold]): [bold green]R{regla_c2+1}[/bold green]\n"
        f"μ_C = {mu[regla_c2,1]:.3f} · μ_A = {mu[regla_c2,2]:.3f}",
        title="Criterio clínico 2",
        border_style="green",
    )
)


# -----------------------------------------------------------------------------
# 2.2  Variable más determinante (σ promedio mínimo)
# -----------------------------------------------------------------------------
sigma_media = sigma.mean(axis=0)
variables   = ['Estancia (E)', 'Comorbilidades (C)', 'Adherencia (A)']

tabla_sigma = Table(
    title="σ promedio por variable",
    box=box.ROUNDED,
    show_header=True,
    header_style="bold blue",
)
tabla_sigma.add_column("Variable", style="bold")
tabla_sigma.add_column("σ promedio", justify="right")
for nombre, s in zip(variables, sigma_media):
    tabla_sigma.add_row(nombre, f"{s:.3f}")

mas_det = variables[int(sigma_media.argmin())]
tabla_sigma.caption = f"Variable más determinante según el modelo: {mas_det}"
console.print(tabla_sigma)


# -----------------------------------------------------------------------------
# EXTRA: Visualización de las funciones de pertenencia aprendidas
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
xs = np.linspace(0, 1, 200)
colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for k, (ax, nombre) in enumerate(zip(axes, ['Estancia (E)',
                                            'Comorbilidades (C)',
                                            'Adherencia (A)'])):
    for i in range(N_RULES):
        gauss = np.exp(-((xs - mu[i, k]) ** 2) / (2 * sigma[i, k] ** 2))
        ax.plot(xs, gauss, color=colores[i], linewidth=1.8,
                label=f'R{i+1} (μ={mu[i,k]:.2f}, σ={sigma[i,k]:.2f})')
    ax.set_title(nombre)
    ax.set_xlabel('Valor normalizado')
    ax.set_ylabel('μ(x)')
    ax.set_ylim(-0.05, 1.1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='upper right')

plt.suptitle('Funciones de pertenencia gaussianas aprendidas por ANFIS',
             fontweight='bold')
plt.tight_layout()
plt.savefig('funciones_pertenencia.png', dpi=150)
plt.close()
console.print("[green]✓[/green] Funciones de pertenencia guardadas en [bold]funciones_pertenencia.png[/bold]")


# -----------------------------------------------------------------------------
# EXTRA: Superficie de decisión C×A con E fijo en su valor medio
# -----------------------------------------------------------------------------
e_fijo = float(np.mean(E))
grid   = np.linspace(0, 1, 50)
CC, AA = np.meshgrid(grid, grid)
EE     = np.full_like(CC, e_fijo)
puntos = np.stack([EE.ravel(), CC.ravel(), AA.ravel()], axis=1).astype(np.float32)

with torch.no_grad():
    riesgo_grid = model(torch.tensor(puntos)).numpy().reshape(CC.shape)

plt.figure(figsize=(6.5, 5))
cs = plt.contourf(CC, AA, riesgo_grid, levels=20, cmap='RdYlGn_r')
plt.colorbar(cs, label='Riesgo predicho')
plt.contour(CC, AA, riesgo_grid, levels=[0.35, 0.6, 0.8],
            colors='black', linestyles=['--', '-', '-'], linewidths=1)
plt.xlabel('Comorbilidades (C)')
plt.ylabel('Adherencia (A)')
plt.title(f'Superficie de riesgo aprendida — E fijo = {e_fijo:.2f}\n'
          '(líneas: umbrales 0.35 / 0.60 / 0.80)')
plt.tight_layout()
plt.savefig('superficie_riesgo.png', dpi=150)
plt.close()
console.print("[green]✓[/green] Superficie de decisión C×A guardada en [bold]superficie_riesgo.png[/bold]")


# =============================================================================
# APARTADO 3 — SISTEMA EXPLICABLE EN PRODUCCIÓN
# =============================================================================
console.rule("[bold cyan]APARTADO 3 — SISTEMA EXPLICABLE EN PRODUCCIÓN[/bold cyan]")

# -----------------------------------------------------------------------------
# 3.1  Función informe_alta
# -----------------------------------------------------------------------------
def _nivel_y_protocolo(riesgo):
    """Devuelve (nivel, protocolo, recomendación) según los umbrales."""
    if riesgo < 0.35:
        return ("BAJO", "Alta estándar",
                "Control en consulta a los 30 días.")
    elif riesgo < 0.60:
        return ("MEDIO", "Seguimiento telefónico",
                "Llamada de enfermería a los 7 días. Revisar medicación.")
    elif riesgo < 0.80:
        return ("ALTO", "Alerta activa",
                "Visita presencial a los 7 días. Reforzar adherencia.")
    else:
        return ("CRÍTICO", "Protocolo urgente",
                "Valorar hospitalización domiciliaria. "
                "Notificar a especialista.")

def _etiqueta_clinica(mu_regla):
    """Etiqueta clínica humanamente legible a partir de los μ de la regla."""
    e_val, c_val, a_val = mu_regla
    rasgos = []
    if e_val > 0.6:  rasgos.append("estancia larga")
    elif e_val < 0.4: rasgos.append("estancia corta")
    if c_val > 0.6:  rasgos.append("comorbilidades altas")
    elif c_val < 0.4: rasgos.append("comorbilidades bajas")
    if a_val > 0.6:  rasgos.append("adherencia buena")
    elif a_val < 0.4: rasgos.append("adherencia baja")
    return ", ".join(rasgos) if rasgos else "perfil intermedio"

def _barra(peso, ancho_max=20):
    """Devuelve una barra de █ proporcional al peso (0..1)."""
    n = int(round(peso * ancho_max))
    return "█" * max(n, 1 if peso > 0.02 else 0)

def informe_alta(model, E_dias, C_charlson, A_score10, nombre):
    """Imprime un informe de alta explicable para un paciente usando Rich."""
    # Normalización a [0, 1]
    E_n = (E_dias - 1) / 19
    C_n = C_charlson / 8
    A_n = A_score10 / 10

    x = torch.tensor([[E_n, C_n, A_n]], dtype=torch.float32)
    with torch.no_grad():
        riesgo = float(model(x).item())
    riesgo = max(0.0, min(1.0, riesgo))

    pesos = model.get_firing_strengths(x)[0]
    nivel, protocolo, recomendacion = _nivel_y_protocolo(riesgo)
    mu_arr = model.mu.detach().numpy()

    color_nivel = {
        "BAJO": "green",
        "MEDIO": "yellow",
        "ALTO": "orange1",
        "CRÍTICO": "bold red",
    }.get(nivel, "white")

    tabla_paciente = Table.grid(padding=(0, 1))
    tabla_paciente.add_column(style="bold")
    tabla_paciente.add_column(justify="right")
    tabla_paciente.add_row("Paciente", nombre)
    tabla_paciente.add_row("Estancia", f"{E_dias} días  (norm.: {E_n:.2f})")
    tabla_paciente.add_row("Comorbilidades", f"{C_charlson} pts Charlson  (norm.: {C_n:.2f})")
    tabla_paciente.add_row("Adherencia", f"{A_score10}/10  (norm.: {A_n:.2f})")
    tabla_paciente.add_row("Riesgo predicho", f"[{color_nivel}]{riesgo:.3f} → {nivel}[/{color_nivel}]")
    tabla_paciente.add_row("Protocolo", protocolo)

    console.print(
        Panel(
            tabla_paciente,
            title="INFORME DE RIESGO DE READMISIÓN — ClínicaIA",
            border_style=color_nivel.replace("bold ", ""),
            expand=False,
        )
    )

    tabla_factores = Table(
        title=f"Factores determinantes — {nombre}",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    tabla_factores.add_column("Regla", style="bold")
    tabla_factores.add_column("Perfil clínico")
    tabla_factores.add_column("Activación", justify="right")
    tabla_factores.add_column("Barra")

    for i, w in enumerate(pesos):
        etiqueta = _etiqueta_clinica(mu_arr[i])
        barra = _barra(w)
        tabla_factores.add_row(f"R{i+1}", etiqueta, f"{w:.3f}", barra)

    console.print(tabla_factores)
    console.print(
        Panel.fit(
            recomendacion,
            title="Recomendación automática",
            border_style=color_nivel.replace("bold ", ""),
        )
    )


# -----------------------------------------------------------------------------
# 3.2  Tres perfiles de ejemplo
# -----------------------------------------------------------------------------
informe_alta(model, E_dias=16, C_charlson=6, A_score10=2,
             nombre="García Ruiz, M.")
informe_alta(model, E_dias=4,  C_charlson=1, A_score10=9,
             nombre="Martínez López, P.")
informe_alta(model, E_dias=10, C_charlson=4, A_score10=5,
             nombre="Fernández Torres, L.")


# -----------------------------------------------------------------------------
# 3.3  Análisis what-if para Fernández Torres
# -----------------------------------------------------------------------------
console.rule("[bold cyan]Análisis WHAT-IF — Fernández Torres, L.[/bold cyan]")

base = torch.tensor([[0.47, 0.50, 0.50]])
with torch.no_grad():
    riesgo_base = model(base).item()

escenarios = {
    'Alta antes        (E: 0.47 → 0.27)': torch.tensor([[0.27, 0.50, 0.50]]),
    'Tratar comorbil.  (C: 0.50 → 0.30)': torch.tensor([[0.47, 0.30, 0.50]]),
    'Mejorar adherenc. (A: 0.50 → 0.70)': torch.tensor([[0.47, 0.50, 0.70]]),
}

tabla_whatif = Table(
    title=f"Riesgo base: {riesgo_base:.3f}",
    box=box.ROUNDED,
    show_header=True,
    header_style="bold cyan",
)
tabla_whatif.add_column("Escenario")
tabla_whatif.add_column("Riesgo", justify="right")
tabla_whatif.add_column("Δ", justify="right")

mejor_delta, mejor_nombre = 0.0, None
for nombre, x in escenarios.items():
    with torch.no_grad():
        r = model(x).item()
    delta = r - riesgo_base
    estilo_delta = "green" if delta < 0 else "red"
    tabla_whatif.add_row(nombre, f"{r:.3f}", f"[{estilo_delta}]{delta:+.3f}[/{estilo_delta}]")
    if delta < mejor_delta:
        mejor_delta, mejor_nombre = delta, nombre

console.print(tabla_whatif)

if mejor_nombre is not None:
    console.print(
        Panel.fit(
            f"Intervención más eficaz: [bold green]{mejor_nombre.split('(')[0].strip()}[/bold green] "
            f"(Δ = {mejor_delta:+.3f})",
            title="Conclusión what-if",
            border_style="green",
        )
    )
else:
    console.print("[yellow]Ninguna intervención reduce el riesgo en este perfil.[/yellow]")

console.rule("[bold green]Ejecución finalizada[/bold green]")
console.print(
    Panel.fit(
        "Figuras generadas:\n"
        "• convergencia.png\n"
        "• funciones_pertenencia.png\n"
        "• superficie_riesgo.png",
        title="Archivos de salida",
        border_style="green",
    )
)
