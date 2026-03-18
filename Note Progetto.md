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


## TEST HOPPER

PPO 20%  random sulla massa

-test source -> target, source foot 

--- Test summary ---
Mean reward        : 1373.88
Std reward         : 105.37
Min reward         : 1286.15
Max reward         : 1613.32
Mean episode length: 429.32
Min episode length : 404
Max episode length : 500

--- Action statistics ---
Action mean: [ 0.00100593  0.32920462 -0.01002937]
Action std : [0.16379085 0.37209645 0.76105136]
Action min : [-1.         -0.77293354 -1.        ]
Action max : [0.791019 1.       1.      ]

-test source -> target, source leg

--- Test summary ---
Mean reward        : 898.42
Std reward         : 37.38
Min reward         : 806.84
Max reward         : 966.89
Mean episode length: 277.56
Min episode length : 253
Max episode length : 300

--- Action statistics ---
Action mean: [ 0.00184356  0.13132195 -0.06002164]
Action std : [0.6394125  0.40270528 0.8324423 ]
Action min : [-1.        -0.8490269 -1.       ]
Action max : [1. 1. 1.]


-test source -> target, source thigh

--- Test summary ---
Mean reward        : 713.52
Std reward         : 26.24
Min reward         : 659.89
Max reward         : 754.64
Mean episode length: 219.80
Min episode length : 205
Max episode length : 232

--- Action statistics ---
Action mean: [1.9021480e-05 1.8675862e-01 2.0304658e-02]
Action std : [0.1272775  0.36648074 0.8364642 ]
Action min : [-1.        -0.5454902 -1.       ]
Action max : [0.3200751 1.        1.       ]


PPO 50% random sulla massa

-test source -> target, source thigh
--- Test summary ---
Mean reward        : 1329.96
Std reward         : 161.69
Min reward         : 1043.10
Max reward         : 1660.17
Mean episode length: 422.80
Min episode length : 313
Max episode length : 500

--- Action statistics ---
Action mean: [0.00172958 0.05738472 0.14231986]
Action std : [0.2051998  0.34181163 0.804516  ]
Action min : [-1.         -0.99376166 -1.        ]
Action max : [0.77963734 1.         1.        ]

-test source -> target, source leg
--- Test summary ---
Mean reward        : 1435.93
Std reward         : 98.12
Min reward         : 1224.18
Max reward         : 1587.17
Mean episode length: 456.66
Min episode length : 387
Max episode length : 500

--- Action statistics ---
Action mean: [0.00115901 0.29379418 0.18960063]
Action std : [0.5908412 0.3762631 0.8414483]
Action min : [-1. -1. -1.]
Action max : [1. 1. 1.]

-test source -> target, source foot 
--- Test summary ---
Mean reward        : 1340.06
Std reward         : 61.11
Min reward         : 1245.43
Max reward         : 1544.22
Mean episode length: 407.06
Min episode length : 382
Max episode length : 473

--- Action statistics ---
Action mean: [ 0.00300753  0.15166903 -0.01807948]
Action std : [0.34813878 0.39344078 0.74659467]
Action min : [-1. -1. -1.]
Action max : [1. 1. 1.]

PPO 30% random sulla massa

-test source -> target, source thigh
--- Test summary ---
Mean reward        : 1533.49
Std reward         : 2.12
Min reward         : 1527.19
Max reward         : 1537.87
Mean episode length: 500.00
Min episode length : 500
Max episode length : 500

--- Action statistics ---
Action mean: [ 0.0036801   0.29960555 -0.0669262 ]
Action std : [0.27767903 0.4291768  0.61730707]
Action min : [-1.        -0.7839602 -1.       ]
Action max : [1. 1. 1.]


-test source -> target, source leg

--- Test summary ---
Mean reward        : 1454.94
Std reward         : 106.46
Min reward         : 1230.70
Max reward         : 1621.22
Mean episode length: 438.88
Min episode length : 364
Max episode length : 500

--- Action statistics ---
Action mean: [ 0.00146625  0.05490446 -0.06804583]
Action std : [0.16635679 0.35139477 0.73130405]
Action min : [-1. -1. -1.]
Action max : [0.5507138 1.        1.       ]

-test source -> target, source foot 

--- Test summary ---
Mean reward        : 927.91
Std reward         : 26.23
Min reward         : 860.41
Max reward         : 987.75
Mean episode length: 282.58
Min episode length : 264
Max episode length : 300

--- Action statistics ---
Action mean: [0.00504101 0.5775852  0.00462518]
Action std : [0.13338324 0.41172212 0.83499694]
Action min : [-1. -1. -1.]
Action max : [0.59255314 1.         1.        ]

PPO 10% random sulla massa

-test source -> target, source thigh
--- Test summary ---
Mean reward        : 978.43
Std reward         : 23.62
Min reward         : 932.65
Max reward         : 1027.68
Mean episode length: 297.30
Min episode length : 284
Max episode length : 314

--- Action statistics ---
Action mean: [ 0.00385392  0.23905745 -0.21487367]
Action std : [0.43217614 0.4408422  0.7385527 ]
Action min : [-1.        -0.9121574 -1.       ]
Action max : [1. 1. 1.]

-test source -> target, source leg
--- Test summary ---
Mean reward        : 1080.40
Std reward         : 315.62
Min reward         : 812.14
Max reward         : 1690.88
Mean episode length: 328.30
Min episode length : 248
Max episode length : 500

--- Action statistics ---
Action mean: [ 0.00202396  0.4747075  -0.12519617]
Action std : [0.19943601 0.31363326 0.72542036]
Action min : [-1.         -0.80527115 -1.        ]
Action max : [0.6417099 1.        1.       ]

-test source -> target, source foot 

--- Test summary ---
Mean reward        : 987.62
Std reward         : 11.30
Min reward         : 958.57
Max reward         : 1006.83
Mean episode length: 300.60
Min episode length : 291
Max episode length : 309

--- Action statistics ---
Action mean: [ 0.00061681  0.455818   -0.11578556]
Action std : [0.15456648 0.35026082 0.8128668 ]
Action min : [-0.8104982  -0.47427192 -1.        ]
Action max : [0.7961928 1.        1.       ]

PPO 20%  random sulla massa

-test source -> target, source foot 
--- Test summary ---
Mean reward        : 1339.15
Std reward         : 65.10
Min reward         : 1274.10
Max reward         : 1594.97
Mean episode length: 419.56
Min episode length : 403
Max episode length : 500

--- Action statistics ---
Action mean: [ 0.00097925  0.3285141  -0.0118956 ]
Action std : [0.16474676 0.37252212 0.76244384]
Action min : [-1.        -0.7629496 -1.       ]
Action max : [0.78198713 1.         1.        ]

-test source -> target, source leg

--- Test summary ---
Mean reward        : 1299.05
Std reward         : 147.31
Min reward         : 1033.37
Max reward         : 1666.74
Mean episode length: 388.50
Min episode length : 309
Max episode length : 500

--- Action statistics ---
Action mean: [9.6150186e-05 4.9123302e-01 4.5664053e-02]
Action std : [0.1684639  0.43792075 0.8013691 ]
Action min : [-1.         -0.60802186 -1.        ]
Action max : [0.72049934 1.         1.        ]

-test source -> target, source thigh
--- Test summary ---
Mean reward        : 1272.44
Std reward         : 151.50
Min reward         : 1067.66
Max reward         : 1665.64
Mean episode length: 383.32
Min episode length : 319
Max episode length : 500

--- Action statistics ---
Action mean: [ 0.00329574  0.41681793 -0.00152366]
Action std : [0.2111211  0.42217097 0.8167752 ]
Action min : [-1.         -0.76588917 -1.        ]
Action max : [0.8644005 1.        1.       ]

PPO 40%  random sulla massa

-test source -> target, source thigh

--- Test summary ---
Mean reward        : 1121.85
Std reward         : 215.47
Min reward         : 856.00
Max reward         : 1635.09
Mean episode length: 338.86
Min episode length : 262
Max episode length : 500

--- Action statistics ---
Action mean: [ 0.00859731  0.01959575 -0.08688459]
Action std : [0.66821617 0.5273855  0.75271946]
Action min : [-1. -1. -1.]
Action max : [1. 1. 1.]


-test source -> target, source leg
--- Test summary ---
Mean reward        : 1294.10
Std reward         : 89.25
Min reward         : 1201.61
Max reward         : 1465.30
Mean episode length: 409.50
Min episode length : 382
Max episode length : 457

--- Action statistics ---
Action mean: [2.2851939e-04 6.9529498e-01 7.4552954e-03]
Action std : [0.1761939  0.30828884 0.7775076 ]
Action min : [-1.          0.01583662 -1.        ]
Action max : [0.49146506 1.         1.        ]


-test source -> target, source foot
--- Test summary ---
Mean reward        : 1327.17
Std reward         : 108.80
Min reward         : 1164.17
Max reward         : 1497.63
Mean episode length: 408.94
Min episode length : 366
Max episode length : 461

--- Action statistics ---
Action mean: [0.01187841 0.4074271  0.10575613]
Action std : [0.40578318 0.4613509  0.8311204 ]
Action min : [-1. -1. -1.]
Action max : [1. 1. 1.]

 
500k
PPO 20%  random sulla massa

-test source -> target, source thigh
--- Test summary ---
Mean reward        : 717.25
Std reward         : 58.10
Min reward         : 651.79
Max reward         : 854.36
Mean episode length: 219.28
Min episode length : 205
Max episode length : 252

--- Action statistics ---
Action mean: [ 0.00290263  0.14365496 -0.10494966]
Action std : [0.492634   0.36294392 0.79500216]
Action min : [-1. -1. -1.]
Action max : [1. 1. 1.]

-test source -> target, source leg

--- Test summary ---
Mean reward        : 1325.80
Std reward         : 134.98
Min reward         : 1167.32
Max reward         : 1787.64
Mean episode length: 359.98
Min episode length : 318
Max episode length : 500

--- Action statistics ---
Action mean: [-2.7341529e-04  3.6615309e-01 -6.1869003e-02]
Action std : [0.53563166 0.57694995 0.77417487]
Action min : [-1. -1. -1.]
Action max : [1. 1. 1.]

