import os
import re
import csv
import sys
import glob
import numpy as np
import gymnasium as gym
import torch

from multiprocessing import Pool, current_process
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# ──────────────────────────────────────────────────────────────────────────────
# Path Resolution
# Il file si trova in: HOPPER_ENV/PPO/results/evaluate_policy_v1.py
# PPO_ROOT  →          HOPPER_ENV/PPO/
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))   # .../PPO/results
PPO_ROOT    = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # .../PPO
MODELS_DIR  = os.path.join(PPO_ROOT, "results_v2", "models")
RESULTS_DIR = os.path.join(PPO_ROOT, "results_v2")

os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, PPO_ROOT)
import env.custom_hopper  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Configuration
# ──────────────────────────────────────────────────────────────────────────────
N_EPISODES   = 50
N_PROCESSES  = 10

VARIATIONS        = [10, 20, 50, 80]
PERTURBATION_TYPES = ["Mass", "Fric", "MassFric"]
DISTRIBUTION_TYPES = ["Uni", "Gauss"]

ENV_IDS = ["CustomHopper-source-v0"] + [
    f"Hopper-{pert}-{dist}-{pct}-v0"
    for pct  in VARIATIONS
    for pert in PERTURBATION_TYPES
    for dist in DISTRIBUTION_TYPES
]

TARGET_IDS = [
    "CustomHopper-target-v0",
    "Hopper-Target-Mass-Easy-v0",
    "Hopper-Target-Mass-Medium-v0",
    "Hopper-Target-Mass-Hard-v0",
    "Hopper-Target-Fric-Easy-v0",
    "Hopper-Target-Fric-Medium-v0",
    "Hopper-Target-Fric-Hard-v0",
    "Hopper-Target-Both-Easy-v0",
    "Hopper-Target-Both-Medium-v0",
    "Hopper-Target-Both-Hard-v0",
]

# Regex per parsing robusto dell'env_id
# Formato: Hopper-{Mass|Fric|MassFric}-{Uni|Gauss}-{percent}-v0
ENV_ID_PATTERN = re.compile(
    r"^Hopper-(Mass|Fric|MassFric)-(Uni|Gauss)-(\d+)-v0$"
)

# ──────────────────────────────────────────────────────────────────────────────
# Path Utilities
# ──────────────────────────────────────────────────────────────────────────────
def get_model_paths(env_id: str) -> list[str]:
    """
    Risolve i path dei modelli salvati per un dato env_id.

    Struttura attesa su disco:
        models/{pert}-{dist}/{percent}/PPO_{env_id}_s*.zip

    Esempio:
        models/MassFric-Uni/10/PPO_Hopper-MassFric-Uni-10-v0_s42.zip
    """
    # Aggiungiamo l'eccezione per l'ambiente source base
    if env_id == "CustomHopper-source-v0":
        model_dir = os.path.join(MODELS_DIR, "source")
        pattern   = os.path.join(model_dir, "*.zip")
        paths     = sorted(glob.glob(pattern))

        # Fallback: Se l'utente ha messo la cartella 'source' direttamente in 'results_v2'
        if not paths:
            model_dir_alt = os.path.join(PPO_ROOT, "results_v2", "source")
            pattern_alt   = os.path.join(model_dir_alt, "*.zip")
            paths         = sorted(glob.glob(pattern_alt))

        if not paths:
            print(f"\n[WARN] Nessun modello '.zip' trovato per il source!")
            print(f"       Ho cercato in: {model_dir}")
            print(f"       E anche in   : {model_dir_alt}\n")

        return paths

    match = ENV_ID_PATTERN.match(env_id)
    if not match:
        print(f"[WARN] env_id non conforme al pattern atteso: '{env_id}'")
        return []

    pert, dist, percent = match.group(1), match.group(2), match.group(3)
    model_dir = os.path.join(MODELS_DIR, f"{pert}-{dist}", percent)
    pattern   = os.path.join(model_dir, f"PPO_{env_id}_s*.zip")
    paths     = sorted(glob.glob(pattern))

    if not paths:
        print(f"[INFO] Nessun modello trovato in: {model_dir}")

    return paths


def extract_seed(model_path: str) -> int | None:
    """Estrae il training seed dal nome file (es. PPO_..._s42.zip → 42)."""
    match = re.search(r"_s(\d+)\.zip$", os.path.basename(model_path))
    return int(match.group(1)) if match else None
    if match:
        return int(match.group(1))
    return 0  # Fallback sicuro se il nome file non contiene "_sNumero"


# ──────────────────────────────────────────────────────────────────────────────
# Worker — eseguito in parallelo da ciascun processo
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_policy_family(env_id: str) -> dict | None:
    """
    Valuta tutti i modelli addestrati per un dato env_id su tutti i TARGET_IDS.

    Eseguito come worker in un processo separato del Pool.
    Ogni chiamata è indipendente: carica i propri modelli e crea i propri ambienti.

    Per ogni modello (seed di training) e ogni ambiente target:
        - esegue N_EPISODES episodi con policy deterministica
        - registra mean e std della reward (variabilità ambientale)

    Le statistiche finali nel dizionario risultante sono:
        - {target_id}_mean : media delle mean-reward tra seed di training
        - {target_id}_std  : media delle std-reward tra seed di training
                             (cattura la variabilità dell'ambiente, non del training)

    Returns:
        dict con i risultati, oppure None se nessun modello è disponibile.
    """
    # Limita i thread torch per evitare oversubscription con multiprocessing
    torch.set_num_threads(1)
    proc_name = current_process().name

    model_paths = get_model_paths(env_id)
    if not model_paths:
        return None

    print(f"[{proc_name}] Inizio: {env_id} ({len(model_paths)} seed trovati)")

    # Pre-popolamento del risultato con NaN per garantire coerenza del CSV
    row: dict = {
        "policy":      env_id,
        "n_seeds":     0,
        "train_seeds": "[]",
        **{f"{tid}_{stat}": np.nan for tid in TARGET_IDS for stat in ("mean", "std", "ep_len")},
    }

    # seed_scores[target_id] = lista di (mean_reward, std_reward) per seed
    seed_scores: dict[str, list[tuple[float, float, float]]] = {
        tid: [] for tid in TARGET_IDS
    }
    used_seeds: list[int] = []

    for model_path in model_paths:
        seed = extract_seed(model_path)

        try:
            model = PPO.load(model_path, device="cpu")
        except Exception as e:
            print(f"[{proc_name}][ERROR] Caricamento fallito '{model_path}': {e}")
            continue

        used_seeds.append(seed)

        # Crea tutti gli ambienti una volta per seed (ottimizzazione)
        envs: dict[str, gym.Env | None] = {}
        for target_id in TARGET_IDS:
            try:
                envs[target_id] = gym.make(target_id)
            except Exception as e:
                print(f"[{proc_name}][WARN] gym.make('{target_id}') fallito: {e}")
                envs[target_id] = None

        for target_id in TARGET_IDS:
            env = envs[target_id]
            if env is None:
                seed_scores[target_id].append((np.nan, np.nan, np.nan))
                continue
            try:
                # Richiedendo return_episode_rewards=True, otteniamo le liste di reward e step per ogni episodio
                ep_rewards, ep_lengths = evaluate_policy(
                    model,
                    env,
                    n_eval_episodes=N_EPISODES,
                    deterministic=True,
                    return_episode_rewards=True,
                )
                
                # Calcoliamo le medie e deviazioni standard
                mean_reward = float(np.mean(ep_rewards))
                std_reward = float(np.std(ep_rewards))
                mean_length = float(np.mean(ep_lengths))
                seed_scores[target_id].append((mean_reward, std_reward, mean_length))

            except Exception as e:
                print(f"[{proc_name}][WARN] Valutazione fallita "
                      f"'{target_id}' seed={seed}: {e}")
                seed_scores[target_id].append((np.nan, np.nan, np.nan))

        # Chiudi tutti gli ambienti dopo aver completato il seed
        for env in envs.values():
            if env is not None:
                env.close()

    if not used_seeds:
        return None

    row["n_seeds"]     = len(used_seeds)
    row["train_seeds"] = str(used_seeds)

    for target_id in TARGET_IDS:
        scores = seed_scores[target_id]
        means  = np.array([s[0] for s in scores], dtype=float)
        stds   = np.array([s[1] for s in scores], dtype=float)
        lens   = np.array([s[2] for s in scores], dtype=float)

        if not np.all(np.isnan(means)):
            row[f"{target_id}_mean"] = round(float(np.nanmean(means)), 2)
            row[f"{target_id}_std"]  = round(float(np.nanmean(stds)),  2)
            row[f"{target_id}_ep_len"] = round(float(np.nanmean(lens)), 1)

    print(f"[{proc_name}] ✓ Completato: {env_id}")
    return row


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Policy da testare:   {len(ENV_IDS)}")
    print(f"Target environments: {len(TARGET_IDS)}")
    print(f"Episodi per coppia:  {N_EPISODES}")
    print(f"Processi paralleli:  {N_PROCESSES}")
    print("═" * 60)

    with Pool(processes=N_PROCESSES) as pool:
        results = pool.map(evaluate_policy_family, ENV_IDS)

    # Rimuovi task falliti
    results = [r for r in results if r is not None]

    if not results:
        print("[ERROR] Nessun risultato prodotto. "
              "Verificare MODELS_DIR e i path dei modelli.")
        return

    fieldnames = ["policy", "n_seeds", "train_seeds"] + [
        f"{tid}_{stat}"
        for tid  in TARGET_IDS
        for stat in ("mean", "std", "ep_len")
    ]

    csv_path = os.path.join(RESULTS_DIR, "evaluation_matrix_v2.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Risultati salvati in: {csv_path}")
    print(f"  Righe scritte: {len(results)}")

    # ── Migliore policy per ogni target ───────────────────────────────────────
    print("\n" + "═" * 60)
    print("MIGLIORE POLICY PER OGNI TARGET")
    print("═" * 60)
    for target_id in TARGET_IDS:
        col  = f"{target_id}_mean"
        col_len = f"{target_id}_ep_len"
        best = max(results, key=lambda r: r.get(col, -np.inf))
        print(f"  {target_id:<40} → {best['policy']:<30} (Reward: {best[col]:.2f} | Ep_Len: {best[col_len]:.1f})")


if __name__ == "__main__":
    main()