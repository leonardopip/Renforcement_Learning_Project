# Progetto Robot learning 
“Idea of yours to further improve the sim-to-real transfer in our simple scenario”


## Environment
### Pusher
#### Reward
he total reward is: ***reward*** *=* *reward_dist + reward_ctrl + reward_near*.

    - *reward_dist*:
    This reward is a measure of how far the object is from the target goal position,
    with a more negative value assigned if the object is further away from the target.
    It is $-w_{dist} \|(P_{object} - P_{target})\|_2$.
    where $w_{dist}$ is the `reward_dist_weight` (default is $1$).
    - *reward_ctrl*:
    A negative reward to penalize the pusher for taking actions that are too large.
    It is measured as the negative squared Euclidean norm of the action, i.e. as $-w_{control} \|action\|_2^2$.
    where $w_{control}$ is the `reward_control_weight` (default is $0.1$).
    - *reward_near*:
    This reward is a measure of how far the *fingertip* of the pusher (the unattached end) is from the object,
    with a more negative value assigned for when the pusher's *fingertip* is further away from the target.
    It is $-w_{near} \|(P_{fingertip} - P_{target})\|_2$.
    where $w_{near}$ is the `reward_near_weight` (default is $0.5$).

 

### hopper

 
## RL API
La differenza principale tra PPO (Proximal Policy Optimization) e SAC (Soft Actor-Critic) risiede nella loro natura on-policy/off-policy e nei meccanismi utilizzati per stabilizzare l'apprendimento:
Natura dell'algoritmo: PPO è un metodo on-policy, il che significa che richiede nuovi campioni di dati per ogni aggiornamento del gradiente, rendendolo meno efficiente nel campionamento. SAC è invece off-policy, permettendo il riutilizzo di esperienze passate tramite un replay buffer, il che lo rende molto più efficiente in termini di campioni necessari.
Obiettivo e Stabilità:
PPO ottimizza una funzione obiettivo "surrogata" e utilizza una tecnica di clipping dei rapporti di probabilità per evitare aggiornamenti troppo grandi e distruttivi della politica.
SAC si basa sul framework della massima entropia: l'agente cerca di massimizzare non solo la ricompensa attesa, ma anche l'entropia della politica (ovvero agire nel modo più casuale possibile pur avendo successo). Questo favorisce l'esplorazione e previene la convergenza prematura.
Prestazioni: SAC ha dimostrato di essere più stabile e performante in compiti di controllo continuo ad alta dimensione rispetto ad altri metodi off-policy, superando spesso anche PPO in termini di velocità di apprendimento e punteggio finale.
Quindi il sac dovrebbe essere migliore nel ambiente dell’hopper dove le masse del corpo variano casualmente, nel caso di un analisi a lungo termine perchè esplora di più così da ottenere massime reward anche in situazioni peggiori. Mentre il PPO impara più velocemente perchè segue una policy precisa senza andare ad esplorare ma basandosi solo sugli ambienti dove ha avuto risultai migliori








## Road Map
- Hopper environment
- Variazione delle masse delle masse
- Determinare quale massa è più rilevante
- Come influisce lo scostarsi dal valore di target 
- Confronto tra PPO e SAC
- test sim-to-real

