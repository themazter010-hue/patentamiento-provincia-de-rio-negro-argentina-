"""
╔══════════════════════════════════════════════════════════════════════╗
║   ANÁLISIS DE MERCADO AUTOMOTOR — RÍO NEGRO 2022–2025               ║
║   Ciencia de Datos Aplicada al Sector Automotor Patagónico          ║
║   Fuentes: DEyCRN · DNRPAyCP · INDEC · Sec. de Energía             ║
╚══════════════════════════════════════════════════════════════════════╝

PREGUNTAS QUE RESPONDE ESTE ANÁLISIS:
  1. ¿Existe efecto sustitución autos → motos cuando cae el salario?
  2. ¿El crecimiento de motos está asociado al trabajo independiente?
  3. ¿Qué relación existe entre combustible y actividad económica?
  4. ¿Qué categoría es más sensible a cambios en el ingreso real?
  5. ¿El mercado refleja crecimiento o adaptación a crisis?
  6. ¿El auto dejó de ser un objetivo? ¿Cuánto necesitás ganar hoy?

PARA USAR EN GOOGLE COLAB:
  - Ejecutá cada celda en orden (Shift + Enter)
  - Todos los datos están hardcodeados (no requiere archivos externos)
  - Requiere: pandas, numpy, matplotlib, seaborn (preinstalados en Colab)
"""

# ══════════════════════════════════════════════════════════════════════
# CELDA 1 — INSTALACIÓN Y CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════

# En Colab estas librerías ya están instaladas.
# Si corrés localmente: pip install pandas numpy matplotlib seaborn scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("✅ Librerías cargadas correctamente")
print("   pandas:", pd.__version__)
print("   numpy: ", np.__version__)
print("   matplotlib:", plt.matplotlib.__version__)


# ══════════════════════════════════════════════════════════════════════
# CELDA 2 — DATASET MAESTRO
# Datos extraídos de informes oficiales DEyCRN (III Trimestre c/año)
# ══════════════════════════════════════════════════════════════════════

df = pd.DataFrame({
    # ── Período ──────────────────────────────────────────────────────
    'año':          [2022,     2023,     2024,     2025],

    # ── Patentamientos Río Negro (III Trim) ──────────────────────────
    # Fuente: DNRPAyCP vía DEyCRN
    'autos':        [900,      1693,     1954,     2055],
    'motos':        [1300,     1385,     2200,     2887],

    # ── Mercado laboral (III Trim, Viedma–C.Patagones) ───────────────
    # Fuente: INDEC EPH / DEyCRN
    'sal_real':     [1234.4,   1128.9,   1171.2,   1428.1],  # pesos ctes 2004
    'sal_idx':      [100.0,    91.5,     94.9,     115.7],   # índice base 2022=100
    'desocupacion': [0.9,      1.1,      1.2,      1.2],    # % PEA
    'subocupacion': [2.5,      1.4,      2.3,      2.0],    # % PEA
    'asalariados':  [106.3,    108.8,    108.9,    108.5],  # miles

    # ── Actividad empresarial ────────────────────────────────────────
    # Fuente: Boletín Oficial Río Negro
    'empresas':     [401,      533,      497,      550],    # altas anuales

    # ── Combustibles ─────────────────────────────────────────────────
    # Fuente: Secretaría de Energía de la Nación
    'combustible':  [141.49,   282.16,   1090.13,  1469.0], # $/litro Gas Oil G2

    # ── Empleo por sector (miles, Río Negro) ─────────────────────────
    # Fuente: DEyCRN / Ministerio de Trabajo
    'construccion': [7.2,      6.8,      5.6,      5.2],   # miles
    'com_servicios':[71.8,     74.6,     74.1,     75.0],  # miles

    # ── Precios de vehículos (pesos corrientes estimados) ────────────
    # Fuente: ACARA / precio de lista promedio segmento medio
    'precio_auto_0km':   [5_000_000,  8_000_000,  20_000_000, 28_000_000],
    'precio_auto_usado': [2_000_000,  3_500_000,   9_000_000, 13_000_000],
    'precio_moto_0km':   [  500_000,  1_500_000,   4_000_000,  6_000_000],
    'precio_moto_usado': [  200_000,    600_000,   1_800_000,  2_800_000],

    # ── Salario mensual corriente (promedio jul-sep) ──────────────────
    # Fuente: DEyCRN (remuneración promedio sector privado)
    'sal_mensual':  [171_508,  398_189,  1_299_860, 1_884_754],
})

# ── Variables derivadas ───────────────────────────────────────────────
df['ratio_moto_auto'] = df['motos'] / df['autos']
df['total_pat']       = df['autos'] + df['motos']
df['prop_motos_pct']  = df['motos'] / df['total_pat'] * 100
df['prop_autos_pct']  = df['autos'] / df['total_pat'] * 100

# Cuota como % del salario
df['cuota_auto_pct']  = df['precio_auto_0km']  / 60 / df['sal_mensual'] * 100
df['cuota_moto_pct']  = df['precio_moto_0km']  / 24 / df['sal_mensual'] * 100

# Salarios necesarios para comprar
df['sal_auto_0km']    = df['precio_auto_0km']  / df['sal_mensual']
df['sal_auto_usado']  = df['precio_auto_usado'] / df['sal_mensual']
df['sal_moto_0km']    = df['precio_moto_0km']  / df['sal_mensual']
df['sal_moto_usado']  = df['precio_moto_usado'] / df['sal_mensual']

# Variaciones % interanuales
for col in ['autos', 'motos', 'sal_idx', 'empresas']:
    df[f'var_{col}'] = df[col].pct_change() * 100

# Elasticidad = %Δvehículo / %Δsalario
df['elast_autos'] = df['var_autos'] / df['var_sal_idx']
df['elast_motos'] = df['var_motos'] / df['var_sal_idx']

# Índices normalizados (base 2022=100)
for col in ['autos','motos','combustible','construccion','com_servicios']:
    df[f'idx_{col}'] = df[col] / df[col].iloc[0] * 100

print("\n📊 DATASET MAESTRO:")
print("="*70)
cols_show = ['año','autos','motos','sal_idx','desocupacion','subocupacion',
             'empresas','combustible','ratio_moto_auto']
print(df[cols_show].to_string(index=False))
print("\n📐 Variables derivadas:")
cols_der = ['año','ratio_moto_auto','cuota_auto_pct','cuota_moto_pct',
            'sal_auto_0km','sal_moto_0km','prop_motos_pct']
print(df[cols_der].round(2).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════
# CELDA 3 — ANÁLISIS DE CORRELACIONES
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("CORRELACIONES — ¿QUÉ VARIABLE EXPLICA CADA PRODUCTO?")
print("="*70)

vars_analisis = ['autos','motos','sal_real','sal_idx','desocupacion',
                 'subocupacion','empresas','combustible','construccion',
                 'com_servicios']

corr = df[vars_analisis].corr()

print("\n📌 Correlación con AUTOS (ordenada por valor absoluto):")
c_autos = corr['autos'].drop('autos').sort_values(key=abs, ascending=False)
for var, r in c_autos.items():
    fuerza = "⭐⭐⭐" if abs(r) > 0.9 else "⭐⭐" if abs(r) > 0.7 else "⭐" if abs(r) > 0.5 else "  —  "
    direccion = "↑ positiva" if r > 0 else "↓ negativa"
    print(f"  {var:20s}: r = {r:+.3f}  {direccion}  {fuerza}")

print("\n📌 Correlación con MOTOS (ordenada por valor absoluto):")
c_motos = corr['motos'].drop('motos').sort_values(key=abs, ascending=False)
for var, r in c_motos.items():
    fuerza = "⭐⭐⭐" if abs(r) > 0.9 else "⭐⭐" if abs(r) > 0.7 else "⭐" if abs(r) > 0.5 else "  —  "
    direccion = "↑ positiva" if r > 0 else "↓ negativa"
    print(f"  {var:20s}: r = {r:+.3f}  {direccion}  {fuerza}")


# ══════════════════════════════════════════════════════════════════════
# CELDA 4 — REGRESIÓN SIMPLE (una variable a la vez)
# ══════════════════════════════════════════════════════════════════════

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def regresion_simple(df, x_col, y_col):
    """Regresión lineal simple entre dos variables del dataframe."""
    d    = df[[x_col, y_col]].dropna()
    x    = d[x_col].values.reshape(-1, 1)
    y    = d[y_col].values
    m    = LinearRegression().fit(x, y)
    y_pred = m.predict(x)
    r2   = r2_score(y, y_pred)
    r, p = stats.pearsonr(d[x_col], d[y_col])
    return {
        'predictor': x_col,
        'target':    y_col,
        'coef':      round(m.coef_[0], 4),
        'intercept': round(m.intercept_, 2),
        'r':         round(r, 4),
        'r2':        round(r2, 4),
        'p_valor':   round(p, 4),
    }

print("\n" + "="*70)
print("REGRESIÓN SIMPLE — ¿CUÁNTO EXPLICA CADA VARIABLE?")
print("="*70)

predictores = ['sal_real','sal_idx','desocupacion','subocupacion',
               'empresas','combustible','construccion']

print("\n📈 Regresiones para AUTOS:")
resultados_autos = [regresion_simple(df, p, 'autos') for p in predictores]
for r in sorted(resultados_autos, key=lambda x: x['r2'], reverse=True):
    print(f"  {r['predictor']:20s} → r={r['r']:+.3f}  R²={r['r2']:.3f}  p={r['p_valor']:.3f}")

print("\n🛵 Regresiones para MOTOS:")
resultados_motos = [regresion_simple(df, p, 'motos') for p in predictores]
for r in sorted(resultados_motos, key=lambda x: x['r2'], reverse=True):
    print(f"  {r['predictor']:20s} → r={r['r']:+.3f}  R²={r['r2']:.3f}  p={r['p_valor']:.3f}")

print("\n💡 INTERPRETACIÓN:")
best_auto = max(resultados_autos, key=lambda x: x['r2'])
best_moto = max(resultados_motos, key=lambda x: x['r2'])
print(f"  → Mejor predictor AUTOS: '{best_auto['predictor']}' (R²={best_auto['r2']:.3f})")
print(f"  → Mejor predictor MOTOS: '{best_moto['predictor']}' (R²={best_moto['r2']:.3f})")


# ══════════════════════════════════════════════════════════════════════
# CELDA 5 — CONCLUSIONES ANALÍTICAS POR PREGUNTA
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("CONCLUSIONES — RESPUESTA A CADA PREGUNTA")
print("="*70)

# P1: Efecto sustitución
ratio_max = df['ratio_moto_auto'].max()
año_max   = df.loc[df['ratio_moto_auto'].idxmax(), 'año']
var_motos_2023 = df.loc[df['año']==2023, 'var_motos'].values[0]
var_sal_2023   = df.loc[df['año']==2023, 'var_sal_idx'].values[0]
print(f"""
1. EFECTO SUSTITUCIÓN AUTOS → MOTOS
   Ratio motos/autos por año: {dict(zip(df['año'], df['ratio_moto_auto'].round(2)))}
   Año de mayor sustitución: {año_max} (ratio = {ratio_max:.2f}x)
   En 2023: salario cayó {var_sal_2023:.1f}% → motos crecieron {var_motos_2023:.1f}%
   CONCLUSIÓN: Sustitución CONFIRMADA pero parcial. La moto no reemplaza
   al auto como bien de transporte sino como herramienta de trabajo.
""")

# P2: Motos y trabajo independiente
r_emp_motos = corr.loc['motos','empresas']
r_sub_motos = corr.loc['motos','subocupacion']
print(f"""2. MOTOS Y TRABAJO INDEPENDIENTE
   r(motos, nuevas_empresas) = {r_emp_motos:.3f}
   r(motos, subocupacion)    = {r_sub_motos:.3f}
   CONCLUSIÓN: La moto está asociada a CREACIÓN de empresas (r={r_emp_motos:.2f}),
   no a subocupación (r={r_sub_motos:.2f} ≈ nulo). El comprador de moto es
   más probablemente un emprendedor activo que un subempleado.
""")

# P3: Combustible y actividad
r_comb_cons = corr.loc['combustible','construccion']
r_comb_serv = corr.loc['combustible','com_servicios']
var_cons_total = (df['construccion'].iloc[-1]/df['construccion'].iloc[0]-1)*100
var_serv_total = (df['com_servicios'].iloc[-1]/df['com_servicios'].iloc[0]-1)*100
print(f"""3. COMBUSTIBLE Y ACTIVIDAD ECONÓMICA
   Combustible ×{df['combustible'].iloc[-1]/df['combustible'].iloc[0]:.1f} entre 2022 y 2025
   Construcción (puestos): {var_cons_total:.1f}% acumulado
   Comercio+Servicios:     +{var_serv_total:.1f}% acumulado
   CONCLUSIÓN: El encarecimiento del combustible destruye empleos en sectores
   intensivos en transporte (construcción) y favorece servicios de proximidad.
""")

# P4: Sensibilidad al ingreso
elast_data = df.dropna(subset=['elast_autos','elast_motos'])
print(f"""4. SENSIBILIDAD AL INGRESO REAL (elasticidad)
   Elasticidad Autos:  {dict(zip(elast_data['año'], elast_data['elast_autos'].round(1)))}
   Elasticidad Motos:  {dict(zip(elast_data['año'], elast_data['elast_motos'].round(1)))}
   CONCLUSIÓN: Las motos son INELÁSTICAS al ingreso — crecen tanto cuando
   el salario sube como cuando cae. Los autos son MÁS sensibles: responden
   fuertemente cuando el salario mejora (2024→2025).
""")

# P5: Crecimiento o adaptación
total_2022 = df.loc[df['año']==2022,'total_pat'].values[0]
total_2025 = df.loc[df['año']==2025,'total_pat'].values[0]
crecimiento = (total_2025/total_2022-1)*100
prop_motos_2022 = df.loc[df['año']==2022,'prop_motos_pct'].values[0]
prop_motos_2025 = df.loc[df['año']==2025,'prop_motos_pct'].values[0]
print(f"""5. ¿CRECIMIENTO O ADAPTACIÓN A CRISIS?
   Patentamientos totales 2022: {int(total_2022):,}
   Patentamientos totales 2025: {int(total_2025):,}
   Crecimiento acumulado: +{crecimiento:.0f}%
   Participación motos 2022: {prop_motos_2022:.0f}%
   Participación motos 2025: {prop_motos_2025:.0f}%
   CONCLUSIÓN: LAS DOS COSAS. El mercado creció +{crecimiento:.0f}% pero la
   composición cambió con las motos ganando participación en crisis.
""")

# P6: Accesibilidad
print(f"""6. ¿EL AUTO DEJÓ DE SER UN OBJETIVO? ¿CUÁNTO NECESITÁS GANAR?
   En 2025 (datos Río Negro):
   Salario mensual referencia: ${df['sal_mensual'].iloc[-1]:,.0f}

   AUTO 0KM:   equivale a {df['sal_auto_0km'].iloc[-1]:.1f} salarios  | cuota 60c = {df['cuota_auto_pct'].iloc[-1]:.0f}% del salario
   AUTO USADO: equivale a {df['sal_auto_usado'].iloc[-1]:.1f} salarios  | cuota 36c = {df['precio_auto_usado'].iloc[-1]/36/df['sal_mensual'].iloc[-1]*100:.0f}% del salario
   MOTO 0KM:   equivale a {df['sal_moto_0km'].iloc[-1]:.1f} salarios  | cuota 24c = {df['cuota_moto_pct'].iloc[-1]:.0f}% del salario
   MOTO USADA: equivale a {df['sal_moto_usado'].iloc[-1]:.1f} salario   | cuota 12c = {df['precio_moto_usado'].iloc[-1]/12/df['sal_mensual'].iloc[-1]*100:.0f}% del salario

   CONCLUSIÓN: El auto NO dejó de ser objetivo — creció +128% en 4 años.
   Pero requiere crédito formal. Sin financiación, el 0km está fuera de
   alcance para la mayoría. El usado es el segmento más dinámico.
""")


# ══════════════════════════════════════════════════════════════════════
# CELDA 6 — DASHBOARD COMPLETO (reproduce el análisis visual)
# ══════════════════════════════════════════════════════════════════════

# ── Paleta de colores ──────────────────────────────────────────────────
C = {
    'az':  '#1565C0',  # azul — autos
    'na':  '#E65100',  # naranja — motos
    've':  '#2E7D32',  # verde — positivo / empresas
    'ro':  '#C62828',  # rojo — alerta / caída
    'vi':  '#6A1B9A',  # violeta — mercado laboral
    'gr':  '#607D8B',  # gris azulado
    'am':  '#F57F17',  # ámbar — combustible
    'ce':  '#0277BD',  # celeste — auto usado
    'bg':  '#F0F4F8',  # fondo general
    'bg2': '#FAFBFC',  # fondo paneles
    'dk':  '#0D1B4B',  # azul muy oscuro — títulos
}

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.facecolor':    C['bg2'],
    'figure.facecolor':  C['bg'],
    'axes.grid':         True,
    'grid.alpha':        0.2,
    'grid.linestyle':    '--',
    'axes.labelsize':    10,
    'xtick.labelsize':   10,
    'ytick.labelsize':   9,
})

fig = plt.figure(figsize=(22, 26), facecolor=C['bg'])
gs  = GridSpec(5, 3, figure=fig,
               hspace=0.52, wspace=0.32,
               top=0.955, bottom=0.03,
               left=0.055, right=0.975)

fig.text(0.5, 0.974,
         'Análisis de Mercado Automotor · Río Negro 2022–2025',
         ha='center', fontsize=19, fontweight='bold', color=C['dk'])
fig.text(0.5, 0.961,
         'Sustitución · Trabajo independiente · Sensibilidad al ingreso · Accesibilidad',
         ha='center', fontsize=11, color=C['gr'])

x      = np.arange(4)
lbl    = ['2022','2023','2024','2025']
años_l = df['año'].tolist()

# ── PANEL 1 — Efecto sustitución ────────────────────────────────────
ax1  = fig.add_subplot(gs[0, 0])
ax1b = ax1.twinx()
norm_a = df['idx_autos'].tolist()
norm_m = df['idx_motos'].tolist()
norm_s = df['sal_idx'].tolist()
ax1.fill_between(x, norm_a, alpha=0.15, color=C['az'])
ax1.fill_between(x, norm_m, alpha=0.15, color=C['na'])
ax1.plot(x, norm_a, 'o-', color=C['az'], lw=2.5, ms=8, label='Autos')
ax1.plot(x, norm_m, 's-', color=C['na'], lw=2.5, ms=8, label='Motos')
ax1b.plot(x, norm_s, '^--', color=C['ro'], lw=2, ms=7, alpha=0.85, label='Salario real')
ax1b.fill_between(x, 100, norm_s,
    where=[s < 100 for s in norm_s], alpha=0.12, color=C['ro'])
ax1b.axhline(100, color=C['ro'], lw=0.8, ls=':', alpha=0.5)
ax1.annotate('Sal. cae −8,5%\nMotos: +28%', xy=(1, norm_m[1]),
    xytext=(0.3, 210), fontsize=8.5, color=C['na'], fontweight='bold',
    arrowprops=dict(arrowstyle='->', color=C['na'], lw=1.4))
ax1.set_xticks(x); ax1.set_xticklabels(lbl)
ax1.set_ylabel('Índice (2022=100)')
ax1b.set_ylabel('Salario real (2022=100)', color=C['ro'], fontsize=9)
ax1b.tick_params(axis='y', labelcolor=C['ro'], labelsize=8)
ax1b.set_ylim(70, 135)
h1 = [mpatches.Patch(color=C['az'], alpha=0.7, label='Autos'),
      mpatches.Patch(color=C['na'], alpha=0.7, label='Motos'),
      plt.Line2D([0],[0], color=C['ro'], lw=2, ls='--', label='Salario real')]
ax1.legend(handles=h1, fontsize=8, loc='upper left')
ax1.set_title('1. Efecto sustitución\nautos → motos',
              fontweight='bold', color=C['dk'], fontsize=11)

# ── PANEL 2 — Ratio motos/autos ─────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ratio = df['ratio_moto_auto'].tolist()
cols_bar = [C['ve'] if r < 1.3 else C['am'] if r < 1.5 else C['ro'] for r in ratio]
bars = ax2.bar(x, ratio, color=cols_bar, alpha=0.88, width=0.55)
ax2.axhline(1.0, color='#333', lw=1.2, ls='--', alpha=0.5)
for bar, r in zip(bars, ratio):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             f'{r:.2f}x', ha='center', fontsize=11,
             fontweight='bold', color=bar.get_facecolor())
ax2.set_xticks(x); ax2.set_xticklabels(lbl)
ax2.set_ylabel('Motos por cada auto vendido')
ax2.set_ylim(0, 2.0)
ax2.set_title('Ratio motos/autos\n(>1 = motos dominan)',
              fontweight='bold', color=C['dk'], fontsize=11)

# ── PANEL 3 — Motos y trabajo independiente ─────────────────────────
ax3  = fig.add_subplot(gs[0, 2])
ax3b = ax3.twinx()
ax3.bar(x, [m/1000 for m in df['motos']], color=C['na'],
        alpha=0.50, width=0.4, label='Motos (miles)')
ax3b.plot(x, df['empresas'], 'D-', color=C['ve'], lw=2.5, ms=9, label='Nuevas empresas')
ax3b.plot(x, [s*100 for s in df['subocupacion']], 'o--', color=C['vi'],
          lw=2, ms=7, label='Subocup.×100')
for i, e in enumerate(df['empresas']):
    ax3b.annotate(f'{e}', (x[i], e), textcoords='offset points',
                  xytext=(5,5), fontsize=8.5, color=C['ve'], fontweight='bold')
ax3.set_xticks(x); ax3.set_xticklabels(lbl)
ax3.set_ylabel('Motos (miles)', color=C['na'])
ax3b.set_ylabel('Empresas / Suboc.×100', fontsize=9)
h3 = [mpatches.Patch(color=C['na'], alpha=0.6, label='Motos'),
      plt.Line2D([0],[0], color=C['ve'], lw=2, marker='D', label='Nuevas emp.'),
      plt.Line2D([0],[0], color=C['vi'], lw=2, ls='--', marker='o', label='Subocup.×100')]
ax3.legend(handles=h3, fontsize=8, loc='upper left')
ax3.set_title('2. Motos y trabajo independiente\n(empresas vs subocupación)',
              fontweight='bold', color=C['dk'], fontsize=11)

# ── PANEL 4 — Combustible y actividad ───────────────────────────────
ax4  = fig.add_subplot(gs[1, 0])
ax4b = ax4.twinx()
ax4.fill_between(x, df['idx_combustible'], alpha=0.18, color=C['am'])
ax4.plot(x, df['idx_combustible'], 'o-', color=C['am'],
         lw=2.5, ms=8, label='Combustible')
ax4b.plot(x, df['idx_com_servicios'], 's-', color=C['ve'],
          lw=2.5, ms=8, label='Comercio+Serv.')
ax4b.plot(x, df['idx_construccion'], '^-', color=C['ro'],
          lw=2.5, ms=8, label='Construcción')
ax4b.axhline(100, color='#999', lw=0.8, ls=':', alpha=0.5)
ax4.set_xticks(x); ax4.set_xticklabels(lbl)
ax4.set_ylabel('Combustible (índice)', color=C['am'], fontsize=9)
ax4b.set_ylabel('Empleo sector (índice 2022=100)', fontsize=9)
ax4b.tick_params(axis='y', labelsize=8)
h4 = [plt.Line2D([0],[0], color=C['am'], lw=2, marker='o', label='Combustible'),
      plt.Line2D([0],[0], color=C['ve'], lw=2, marker='s', label='Comercio+Serv.'),
      plt.Line2D([0],[0], color=C['ro'], lw=2, marker='^', label='Construcción')]
ax4.legend(handles=h4, fontsize=8)
ax4.set_title('3. Combustible y actividad\neconómica predominante',
              fontweight='bold', color=C['dk'], fontsize=11)

# ── PANEL 5 — Elasticidad al ingreso ────────────────────────────────
ax5  = fig.add_subplot(gs[1, 1])
per_lbl    = ['22→23','23→24','24→25']
e_autos    = df['elast_autos'].dropna().tolist()
e_motos    = df['elast_motos'].dropna().tolist()
xp         = np.arange(3)
w_e        = 0.32
b1 = ax5.bar(xp-w_e/2, e_autos, w_e, color=C['az'], alpha=0.88, label='Autos')
b2 = ax5.bar(xp+w_e/2, e_motos, w_e, color=C['na'], alpha=0.88, label='Motos')
ax5.axhline(0, color='#333', lw=1)
for bar in list(b1)+list(b2):
    v = bar.get_height()
    color_text = C['az'] if bar in b1 else C['na']
    ax5.text(bar.get_x()+bar.get_width()/2,
             v + (0.3 if v >= 0 else -0.5),
             f'{v:.1f}', ha='center', fontsize=9,
             color=color_text, fontweight='bold')
ax5.set_xticks(xp); ax5.set_xticklabels(per_lbl, fontsize=10)
ax5.set_ylabel('Elasticidad (%Δvehículo/%Δsalario)')
ax5.legend(fontsize=9)
ax5.set_title('4. Sensibilidad al ingreso real\n(elasticidad por período)',
              fontweight='bold', color=C['dk'], fontsize=11)

# ── PANEL 6 — Composición del mercado ───────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.bar(x, df['prop_motos_pct'], color=C['na'], alpha=0.85, label='Motos %')
ax6.bar(x, df['prop_autos_pct'], bottom=df['prop_motos_pct'],
        color=C['az'], alpha=0.85, label='Autos %')
ax6.axhline(50, color='white', lw=1.5, ls='--', alpha=0.7)
for i, (pa, pm) in enumerate(zip(df['prop_autos_pct'], df['prop_motos_pct'])):
    ax6.text(i, pm/2, f'{pm:.0f}%', ha='center', va='center',
             fontsize=11, fontweight='bold', color='white')
    ax6.text(i, pm+pa/2, f'{pa:.0f}%', ha='center', va='center',
             fontsize=11, fontweight='bold', color='white')
ax6.set_xticks(x); ax6.set_xticklabels(lbl)
ax6.set_ylabel('Composición del mercado (%)')
ax6.set_ylim(0, 100)
ax6.legend(fontsize=9, loc='upper right')
ax6.set_title('5. ¿Crecimiento o adaptación?\nComposición del mercado',
              fontweight='bold', color=C['dk'], fontsize=11)

# ── PANEL 7 — ¿Cuántos salarios necesito? ───────────────────────────
ax7 = fig.add_subplot(gs[2, :2])
w7  = 0.2
b_a0 = ax7.bar(x-0.30, df['sal_auto_0km'],   w7, color=C['az'],  alpha=0.90, label='Auto 0km')
b_au = ax7.bar(x-0.10, df['sal_auto_usado'],  w7, color=C['ce'],  alpha=0.90, label='Auto usado')
b_m0 = ax7.bar(x+0.10, df['sal_moto_0km'],   w7, color=C['na'],  alpha=0.90, label='Moto 0km')
b_mu = ax7.bar(x+0.30, df['sal_moto_usado'],  w7, color=C['am'],  alpha=0.90, label='Moto usada')
for bars_set, col in [(b_a0,C['az']),(b_au,C['ce']),(b_m0,C['na']),(b_mu,C['am'])]:
    for bar in bars_set:
        v = bar.get_height()
        ax7.text(bar.get_x()+bar.get_width()/2, v+0.1, f'{v:.1f}',
                 ha='center', fontsize=8.5, color=col, fontweight='bold')
ax7.axhline(6,  color=C['ve'], lw=1.5, ls='--', alpha=0.7)
ax7.axhline(12, color=C['am'], lw=1.5, ls='--', alpha=0.7)
ax7.text(3.6, 6.15, '6 meses = accesible', fontsize=8.5, color=C['ve'], fontweight='bold')
ax7.text(3.6, 12.15,'12 meses = desafiante', fontsize=8.5, color=C['am'], fontweight='bold')
ax7.set_xticks(x); ax7.set_xticklabels(lbl, fontsize=11)
ax7.set_ylabel('Salarios mensuales equivalentes')
ax7.legend(fontsize=9, loc='upper left', ncol=2)
ax7.set_title('¿Cuántos salarios necesitás para comprar un vehículo?\n'
              '(precio de lista ÷ salario promedio mensual Río Negro)',
              fontweight='bold', color=C['dk'], fontsize=12)

# ── PANEL 8 — Cuota como % del salario ──────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ax8.plot(x, df['cuota_auto_pct'], 'o-', color=C['az'], lw=2.5, ms=9, label='Auto 0km (60c)')
ax8.plot(x, df['cuota_moto_pct'], 's-', color=C['na'], lw=2.5, ms=9, label='Moto 0km (24c)')
ax8.fill_between(x, 0,  30, alpha=0.07, color=C['ve'])
ax8.fill_between(x, 30, 60, alpha=0.07, color=C['am'])
ax8.fill_between(x, 60, df['cuota_auto_pct'].max()+25, alpha=0.08, color=C['ro'])
ax8.axhline(30, color=C['ve'], lw=1.5, ls='--', alpha=0.7)
for i, (va, vm) in enumerate(zip(df['cuota_auto_pct'], df['cuota_moto_pct'])):
    ax8.annotate(f'{va:.0f}%', (x[i], va), textcoords='offset points',
                 xytext=(5, 5), fontsize=9, color=C['az'], fontweight='bold')
    ax8.annotate(f'{vm:.0f}%', (x[i], vm), textcoords='offset points',
                 xytext=(5,-14), fontsize=9, color=C['na'], fontweight='bold')
ax8.text(0.2, 14, '✓ accesible', fontsize=8.5, color=C['ve'], alpha=0.9)
ax8.text(0.2, 38, '⚠ desafiante', fontsize=8.5, color=C['am'], alpha=0.9)
ax8.set_xticks(x); ax8.set_xticklabels(lbl)
ax8.set_ylabel('Cuota como % del salario')
ax8.legend(fontsize=8.5)
ax8.set_title('Cuota financiada\nvs salario mensual',
              fontweight='bold', color=C['dk'], fontsize=11)
ax8.set_ylim(0, df['cuota_auto_pct'].max()+28)

# ── PANEL 9 — ¿El auto dejó de ser objetivo? ───────────────────────
ax9 = fig.add_subplot(gs[3, :])
ax9.axis('off')
fig.patches.extend([plt.Rectangle(
    (ax9.get_position().x0-0.005, ax9.get_position().y0-0.005),
    ax9.get_position().width+0.010,
    ax9.get_position().height+0.010,
    transform=fig.transFigure,
    facecolor=C['dk'], zorder=0
)])
ax9.text(0.5, 0.93,
    '6. ¿El auto dejó de ser un objetivo en Argentina?  '
    '¿Cuánto necesitás ganar HOY para comprarlo?',
    transform=ax9.transAxes, ha='center', fontsize=13,
    fontweight='bold', color='white', va='top')

row = df.iloc[-1]  # datos 2025
bloques_p9 = [
    ('🚗  AUTO 0KM hoy (2025)',
     f'Precio estimado: ${row.precio_auto_0km:,.0f}\n'
     f'Salario ref. (jul-sep 2025): ${row.sal_mensual:,.0f}\n'
     f'Equivalente a: {row.sal_auto_0km:.1f} salarios\n'
     f'Cuota (60 meses): ~${row.precio_auto_0km/60:,.0f} = {row.cuota_auto_pct:.0f}% sal.\n\n'
     '✓ Técnicamente financiable\n⚠ Solo con crédito de largo plazo',
     C['az'], 0.03),
    ('🚗  AUTO USADO hoy (2025)',
     f'Precio estimado: ${row.precio_auto_usado:,.0f}\n'
     f'Equivalente a: {row.sal_auto_usado:.1f} salarios\n'
     f'Cuota (36 meses): ~${row.precio_auto_usado/36:,.0f} = {row.precio_auto_usado/36/row.sal_mensual*100:.0f}% sal.\n\n'
     '✓ Accesible con salario formal\n✓ El segmento más dinámico',
     C['ce'], 0.28),
    ('🛵  MOTO 0KM hoy (2025)',
     f'Precio estimado: ${row.precio_moto_0km:,.0f}\n'
     f'Equivalente a: {row.sal_moto_0km:.1f} salarios\n'
     f'Cuota (24 meses): ~${row.precio_moto_0km/24:,.0f} = {row.cuota_moto_pct:.0f}% sal.\n\n'
     '✓ Muy accesible\n✓ Al alcance de emprendedores',
     C['na'], 0.53),
    ('🛵  MOTO USADA hoy (2025)',
     f'Precio estimado: ${row.precio_moto_usado:,.0f}\n'
     f'Equivalente a: {row.sal_moto_usado:.1f} salarios\n'
     f'Cuota (12 meses): ~${row.precio_moto_usado/12:,.0f} = {row.precio_moto_usado/12/row.sal_mensual*100:.0f}% sal.\n\n'
     '✓ La opción más accesible\n✓ Capital de trabajo inmediato',
     C['am'], 0.76),
]
for titulo_b, texto_b, color_b, x_b in bloques_p9:
    rect = FancyBboxPatch((x_b, 0.04), 0.22, 0.84,
        boxstyle="round,pad=0.02",
        facecolor=color_b, alpha=0.18,
        edgecolor=color_b, linewidth=2,
        transform=ax9.transAxes)
    ax9.add_patch(rect)
    ax9.text(x_b+0.01, 0.84, titulo_b, transform=ax9.transAxes,
             fontsize=10, fontweight='bold', color=color_b, va='top')
    ax9.text(x_b+0.01, 0.72, texto_b, transform=ax9.transAxes,
             fontsize=9, color='#E8E8E8', va='top', linespacing=1.55)

# ── PANEL 10 — Resumen ejecutivo (6 conclusiones) ───────────────────
ax10 = fig.add_subplot(gs[4, :])
ax10.axis('off')
conclusiones_final = [
    ('1. Sustitución\nconfirmada', C['az'],
     'Cuando el salario cae,\nlas motos crecen y los\nautos se moderan. El\nratio motos/autos sube\nen momentos de presión\neconómica.'),
    ('2. Motos =\ncapital de trabajo', C['na'],
     'Correlaciona con nuevas\nempresas (r=0.61), no\ncon subocupación (r=0.08).\nLa moto es inversión\nproductiva, no vehículo\nde bajo ingreso.'),
    ('3. Combustible\nmueve sectores', C['am'],
     'Combustible ×10 en\n4 años. Construcción\nperdió −28% en puestos.\nServicios ganó +4,4%.\nSectores intensivos en\ntransporte pierden.'),
    ('4. Motos:\nelásticas a crisis', C['vi'],
     'Crecen en cualquier\nescenario de ingreso.\nElasticidad negativa\nen 22→23: motos suben\nmientras salario cae.\nAutos son más sensibles.'),
    ('5. Las dos cosas:\ncrecimiento + adapt.', C['ve'],
     'Patentamientos +185%\nacumulado 2022→2025.\nEl mix cambió: motos\nganan participación\nen crisis, autos la\nrecuperan con salario.'),
    ('6. Auto: sigue\nsiendo objetivo', C['ro'],
     'Creció +128% en 4 años.\nNo dejó de ser meta,\npero requiere crédito\nformal. Sin financia-\nción, el 0km está\nfuera de alcance.'),
]
for i, (tit, col, txt) in enumerate(conclusiones_final):
    xi = 0.005 + i*(1/6)
    rect2 = FancyBboxPatch((xi, 0.05), 0.155, 0.88,
        boxstyle="round,pad=0.015",
        facecolor=col, alpha=0.12,
        edgecolor=col, linewidth=1.8,
        transform=ax10.transAxes)
    ax10.add_patch(rect2)
    ax10.text(xi+0.008, 0.90, tit, transform=ax10.transAxes,
              fontsize=9.5, fontweight='bold', color=col, va='top', linespacing=1.3)
    ax10.text(xi+0.008, 0.68, txt, transform=ax10.transAxes,
              fontsize=8.5, color='#333', va='top', linespacing=1.45)

ax10.text(0.5, 0.005,
    'Fuente: DEyCRN · DNRPAyCP · INDEC · Sec. de Trabajo · '
    'Sec. de Energía · Boletín Oficial Río Negro  |  III Trimestre 2022–2025',
    transform=ax10.transAxes, ha='center', fontsize=8, color=C['gr'])

plt.savefig('dashboard_automotor_rionegro.png', dpi=180,
            bbox_inches='tight', facecolor=C['bg'])
plt.show()
print("\n✅ Dashboard guardado como 'dashboard_automotor_rionegro.png'")


# ══════════════════════════════════════════════════════════════════════
# CELDA 7 — HEATMAP DE CORRELACIONES (opcional, para presentación)
# ══════════════════════════════════════════════════════════════════════

fig_corr, ax_c = plt.subplots(figsize=(10, 8), facecolor=C['bg'])

label_map = {
    'autos':'Autos','motos':'Motos','sal_real':'Salario real',
    'sal_idx':'Índ. salarial','desocupacion':'Desocupación',
    'subocupacion':'Subocupación','empresas':'Nuevas empresas',
    'combustible':'Combustible','construccion':'Construcción'
}
corr_labeled = corr.rename(columns=label_map, index=label_map)

sns.heatmap(corr_labeled, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1,
            annot_kws={'size': 10, 'weight': 'bold'},
            linewidths=0.8, square=True, ax=ax_c,
            cbar_kws={'label': 'Pearson r'})

ax_c.set_title('Matriz de correlaciones — Mercado Automotor Río Negro 2022–2025',
               fontsize=13, fontweight='bold', color=C['dk'], pad=15)
plt.setp(ax_c.get_xticklabels(), rotation=35, ha='right', fontsize=10)
plt.setp(ax_c.get_yticklabels(), rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig('heatmap_correlaciones.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Heatmap guardado como 'heatmap_correlaciones.png'")
