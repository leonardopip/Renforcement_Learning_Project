import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

RESULTS_DIR = "results"
csv_path    = os.path.join(RESULTS_DIR, "evaluation_matrix.csv")

# ──────────────────────────────────────────────
# Carica dati
# ──────────────────────────────────────────────
df = pd.read_csv(csv_path)
df = df.set_index("policy")

# Prendi solo le colonne _mean
mean_cols = [c for c in df.columns if c.endswith("_mean")]
df_mean   = df[mean_cols].copy()

# Rinomina colonne per leggibilità
df_mean.columns = [c.replace("_mean", "").replace("Hopper-Target-", "").replace("CustomHopper-target-v0", "Nominal") for c in mean_cols]

# Rinomina righe per leggibilità
df_mean.index = [i.replace("Hopper-", "").replace("-v0", "") for i in df_mean.index]

# ──────────────────────────────────────────────
# Heatmap
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 10))

im = ax.imshow(df_mean.values, aspect="auto", cmap="RdYlGn")

# Assi
ax.set_xticks(range(len(df_mean.columns)))
ax.set_xticklabels(df_mean.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(df_mean.index)))
ax.set_yticklabels(df_mean.index, fontsize=9)

# Valori nelle celle
for i in range(len(df_mean.index)):
    for j in range(len(df_mean.columns)):
        val = df_mean.values[i, j]
        if not np.isnan(val):
            # Testo bianco su celle scure, nero su chiare
            norm_val = (val - df_mean.values.min()) / (df_mean.values.max() - df_mean.values.min())
            color = "white" if norm_val < 0.4 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color=color)

# Linee separatrici per gruppi
# Separa per tipo di randomizzazione (ogni 2 righe: Uni e Gauss)
for y in [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5, 19.5, 21.5]:
    if y < len(df_mean.index):
        ax.axhline(y, color="white", linewidth=0.8, alpha=0.5)

# Separa per tipo di target (ogni 3 colonne)
for x in [0.5, 3.5, 6.5]:
    ax.axvline(x, color="white", linewidth=1.5)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Mean Reward", fontsize=10)

ax.set_title("Policy Transfer Matrix\n(Mean Reward over 50 episodes)", fontsize=13, pad=15)
ax.set_xlabel("Target Environment", fontsize=10)
ax.set_ylabel("Source Policy", fontsize=10)

plt.tight_layout()

# Salva
plot_path = os.path.join(RESULTS_DIR, "heatmap.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"✓ Heatmap salvata in: {plot_path}")


RESULTS_DIR = "results"
# Il file generato dallo script di tuning si chiama hyperparam_results.csv
csv_path = os.path.join(RESULTS_DIR, "hyperparam_results.csv")

# ──────────────────────────────────────────────
# 1. Caricamento e Pulizia Dati
# ──────────────────────────────────────────────
if not os.path.exists(csv_path):
    print(f"❌ Errore: Il file {csv_path} non esiste.")
else:
    df = pd.read_csv(csv_path)

    # Filtriamo eventuali run fallite (avg_reward = -999)
    df = df[df["avg_reward"] != -999]

    # Usiamo 'run_name' come indice per identificare la variante specifica
    df = df.set_index("run_name")

    # Prendiamo solo le colonne che finiscono con _mean (i target di valutazione)
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    df_mean = df[mean_cols].copy()

    # Pulizia nomi Colonne (Target)
    # Trasformiamo "Hopper-Target-Mass-Easy-v0_mean" -> "Mass-Easy"
    df_mean.columns = [
        c.replace("_mean", "")
         .replace("Hopper-Target-", "")
         .replace("CustomHopper-target-v0", "Nominal")
         .replace("-v0", "") 
        for c in mean_cols
    ]

    # ──────────────────────────────────────────────
    # 2. Configurazione Heatmap
    # ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 12)) # Leggermente più grande per contenere più righe

    # Usiamo una colormap divergente o sequenziale (RdYlGn è ottima per i Reward)
    im = ax.imshow(df_mean.values, aspect="auto", cmap="RdYlGn")

    # Assi
    ax.set_xticks(range(len(df_mean.columns)))
    ax.set_xticklabels(df_mean.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(df_mean.index)))
    ax.set_yticklabels(df_mean.index, fontsize=9)

    # Inserimento dei valori nelle celle
    # Calcoliamo min/max globali per il contrasto del testo
    v_min, v_max = df_mean.values.min(), df_mean.values.max()
    
    for i in range(len(df_mean.index)):
        for j in range(len(df_mean.columns)):
            val = df_mean.values[i, j]
            if not np.isnan(val):
                # Determina il colore del testo per leggibilità
                norm_val = (val - v_min) / (v_max - v_min) if v_max > v_min else 0
                text_color = "white" if norm_val < 0.4 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8, color=text_color)

    # ──────────────────────────────────────────────
    # 3. Estetica e Separatori
    # ──────────────────────────────────────────────
    
    # Linea orizzontale per separare i due macro-ambienti (Mass vs Fric)
    # Poiché hai 16 varianti per ambiente, la metà esatta divide i due gruppi
    mid_point = len(df_mean.index) // 2 - 0.5
    ax.axhline(mid_point, color="black", linewidth=2.5)

    # Linee verticali per separare gruppi di Target (Nominal | Mass | Fric | Both)
    # Nominal=1, Mass=3, Fric=3, Both=3 -> posizioni: 0.5, 3.5, 6.5
    for x in [0.5, 3.5, 6.5]:
        ax.axvline(x, color="white", linewidth=2, alpha=0.7)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean Reward", fontsize=12)

    ax.set_title("Hyperparameter Sensitivity & Transfer Matrix\n(Mean Reward over 20 episodes)", fontsize=15, pad=20)
    ax.set_xlabel("Target Test Environments", fontsize=12)
    ax.set_ylabel("Hyperparameter Variants (Source)", fontsize=12)

    plt.tight_layout()

    # Salva
    plot_path = os.path.join(RESULTS_DIR, "hyperparam_heatmap.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    
    print(f"✅ Heatmap salvata in: {plot_path}")