import os
import sys
import torch
import numpy as np
import optuna
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# Ottimizzazione thread per CPU Ryzen
torch.set_num_threads(20)
torch.set_num_interop_threads(4)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
import env.custom_hopper  # noqa: E402

# Configurazione
ENV_ID          = "Hopper-MassFric-Gauss-80-v0"
N_ENVS          = 12          
TRAIN_TIMESTEPS = 1000000
EVAL_EPISODES   = 5           # Leggermente aumentati per stabilità
EVAL_FREQ       = 50_000
SEED            = 42          # Ottimizziamo su un singolo seed per velocizzare e permettere più trial
N_TRIALS        = 30          # Aumentato per permettere a Optuna di esplorare meglio
TIMEOUT_SEC     = 3600 * 8
DB_PATH         = f"sqlite:///{SCRIPT_DIR}/optuna_ppo_{ENV_ID}.db"
STUDY_NAME      = f"ppo_{ENV_ID}"

class OptunaPruningCallback(EvalCallback):
    """
    Callback personalizzata per integrare la potatura (pruning) di Optuna
    direttamente DURANTE l'addestramento, interrompendo i trial sfortunati in anticipo.
    """
    def __init__(self, trial: optuna.Trial, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_idx += 1
            # Riporta il risultato a Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            
            # Chiede a Optuna se il trial sta andando troppo male e va interrotto
            if self.trial.should_prune():
                print(f"\n    [Trial {self.trial.number}] PRUNED allo step di valutazione {self.eval_idx} (Reward: {self.last_mean_reward:.2f})")
                self.is_pruned = True
                return False # Interrompe l'addestramento
        
        return continue_training

def sample_params(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 3e-4, log=True),
        "n_steps":       trial.suggest_categorical("n_steps", [512, 1024, 2048]),
        "batch_size":    trial.suggest_categorical("batch_size", [64, 128, 256]),
        "n_epochs":      trial.suggest_int("n_epochs", 5, 15),
        "gamma":         trial.suggest_float("gamma", 0.98, 0.9999),
        "gae_lambda":    trial.suggest_float("gae_lambda", 0.9, 0.99),
        "clip_range":    trial.suggest_float("clip_range", 0.1, 0.2),
        "ent_coef":      trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True),
        "vf_coef":       trial.suggest_float("vf_coef", 0.3, 0.9),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
    }

def objective(trial):
    params  = sample_params(trial)
    print(f"\n>>> Trial {trial.number} iniziato con params: {params}")

    train_env = make_vec_env(ENV_ID, n_envs=N_ENVS, seed=SEED)
    eval_env  = make_vec_env(ENV_ID, n_envs=1,      seed=SEED + 9999)

    # OptunaPruningCallback per potare i trial cattivi
    eval_callback = OptunaPruningCallback(
        trial=trial,
        eval_env=eval_env,
        best_model_save_path=None,
        log_path=None,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        verbose=0,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=params["learning_rate"],
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        n_epochs=params["n_epochs"],
        gamma=params["gamma"],
        gae_lambda=params["gae_lambda"],
        clip_range=params["clip_range"],
        ent_coef=params["ent_coef"],
        vf_coef=params["vf_coef"],
        max_grad_norm=params["max_grad_norm"],
        policy_kwargs=dict(
            log_std_init=-2,
            activation_fn=nn.Tanh,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
        seed=SEED,
        verbose=0, # Log disattivati per non sporcare il terminale
        device="cpu",
    )

    try:
        model.learn(total_timesteps=TRAIN_TIMESTEPS, callback=eval_callback)
        
        if getattr(eval_callback, "is_pruned", False):
            train_env.close()
            eval_env.close()
            raise optuna.exceptions.TrialPruned()
            
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"    [Trial {trial.number}] Errore durante l'addestramento: {e}")
        train_env.close()
        eval_env.close()
        return 0.0

    # Valutazione finale post-addestramento
    mean_reward, _ = evaluate_policy(
        model, eval_env,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
    )

    train_env.close()
    eval_env.close()

    print(f"--- Trial {trial.number} COMPLETATO. Media finale: {mean_reward:.2f}\n")
    return float(mean_reward)

def main():
    # Creazione studio con MedianPruner
    # n_startup_trials=5: non pota nulla per i primi 5 trial per raccogliere statistiche
    # n_warmup_steps=5: aspetta 5 valutazioni (es. 250k step) prima di iniziare a potare
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        storage=DB_PATH,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SEC)
    except KeyboardInterrupt:
        print("\nOttimizzazione interrotta manualmente.")

    # Risultati
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    print("-" * 50)
    print(f"Ottimizzazione terminata.")
    print(f"Trial Completati: {len(completed)}")
    print(f"Trial Pruned: {len(pruned)}")
    
    if len(completed) > 0:
        print(f"Miglior Reward: {study.best_value:.2f}")
        print("Migliori Parametri:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
    else:
        print("Nessun trial completato con successo.")

if __name__ == "__main__":
    main()