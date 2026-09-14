# Approfondimento — Perché il sito era lento al risveglio e come il web è diventato Python puro

**Capitolo del diario:** [Capitolo 15](https://ai.briscola.dev/diario) · **Periodo:** 7 luglio 2026

## La diagnosi, con le misure

Dai log di produzione FastAPI Cloud: lo **scale-to-zero scatta dopo ~90 secondi** di
inattività — al traffico attuale il cold start non è l'eccezione, è la norma. Il
risveglio costava **~18.8 s al primo 200**, così composti:

- **10.2 s di warm-up Numba sincrono** nel lifespan (compilazione JIT dei kernel di
  search e solver, pagata PRIMA di servire la prima richiesta);
- **~1–2 s di download dei tre modelli** `.npz` (il filesystem è azzerato a ogni cold
  start, quindi il provisioning riscaricava tutto);
- il resto: scheduling del container, boot, probe di readiness della piattaforma.

## I primi due fix (`1561936`, v0.30.0)

1. **Provisioning e warm-up JIT in un task in background** dopo lo startup: l'app serve
   subito e paga compilazione e download mentre il visitatore carica la home. Sicuro per
   costruzione: i kernel servono solo a search e solver, che entrano in gioco minuti dopo
   l'inizio di una partita. Telemetria per fase nei log (print con i tempi).
2. **I tre `.npz` di runtime committati nel repo (~850 KB totali)**: l'immagine di deploy
   li contiene già e il provisioning degrada a verifica SHA locale (resta come
   fallback/pin di versione). Verificato in locale: primo 200 dopo 0.5 s dall'avvio.

## I 13,7 secondi rimasti dipendono dalla piattaforma

Esito misurato in produzione: **TTFB del risveglio 13.7 s — identico prima e dopo il
de-JIT completo** del passo successivo. Tolto tutto il costo applicativo, resta il prezzo
della piattaforma (scheduling del container + boot + cadenza del probe di readiness). Il
guadagno vero dei fix non è il TTFB: è che la replica è **subito operativa al 100%**
(niente 10 s di compilazione mentre serve i primi giocatori) e che il nuovo default è
~2.4× più capiente. Per abbattere i 13.7 s restano solo leve esterne: un keep-alive con
ping sotto i 90 s (nota pratica: il cron di GitHub Actions ha minimo 5 minuti, servirebbe
un job che pinga ogni 60 s al proprio interno), il piano a pagamento, o un messaggio
"sto svegliando il server" in UI — cosmetico ma onesto.

## Tolto Numba dalla search di produzione (`660e251`, v0.31.0)

Con la promozione della v11 il default UI è diventato `bc_model_pimc_belief_16x8`:
**+3.36 su v11+solver** (CI +3.05..+3.66, 4k seat-fair) a **~15 ms/mossa pensata** in
Python puro (14.9 ms misurati nel gate), contro i ~37 ms del vecchio default JIT 64×10.
La search PIMC di produzione è tornata Python per entrambe le config: il kernel JIT
valeva ~2× di CPU ma costava ~8 s di compilazione a ogni risveglio — un pessimo scambio
con lo scale-to-zero a 90 s. Restava un solo kernel compilato nel runtime: il solver
endgame (~2 s di warm-up in background).

## Tolto Numba anche dal solver: primo tentativo fallito, poi principal variation (`a0e29d8`, v0.31.1)

La sera stessa, la stima diceva che il solver in Python sarebbe costato ~2× per mossa
(0.82 ms/chiamata misurati sul solver di dominio). Il primo tentativo la smentì: il
solver vive nel percorso caldo dei rollout — dentro la search viene richiamato per ogni
carta candidata di ogni determinizzazione — e in Python puro la mossa saliva a
**68.8 ms contro i 17.0 del kernel JIT: 4×**. Disastro certificato, tentativo chiuso.

Il secondo tentativo cambiò la domanda invece della risposta: perché ri-risolvere lo
stesso finale a ogni carta, quando ogni linea ottima raggiunge per definizione lo stesso
delta minimax? Nei rollout si risolve l'endgame **una volta per rollout** e si segue la
**principal variation** fino in fondo: il valore del rollout è matematicamente identico.
E il solver diventa `solve_endgame_fast`, il minimax numerico già in repo da settimane:
**0.09 ms/chiamata** (~10× il solver di dominio, 0 disaccordi su 300 stati). Verifiche:

| Misura | Prima (numba) | Dopo (python puro) |
|---|---|---|
| Costo per mossa pensata (A/B stesso carico) | 17.0 ms | **16.6 ms** (pari) |
| Esiti su 400 partite a parità di seed | 220/168/12 | **identici al bit** |
| RSS per replica | 116 MB | **69 MB** (−47: import numba+llvmlite) |
| Import dell'app | — | **0.23 s**, 0 moduli numba caricati |
| Warm-up JIT al risveglio | ~2 s (solver) | **niente** |

Il dettaglio che rende pulita la rimozione: `ai/endgame/__init__` ora importa i simboli
numba **pigramente** (PEP 562, `__getattr__` di modulo) — il processo web non tocca mai
numba, mentre training e benchmark ottengono i kernel come prima al primo accesso. Il
tavolo online è Python puro da cima a fondo; i kernel veloci restano in palestra, dove i
miliardi di calcoli servono davvero.
