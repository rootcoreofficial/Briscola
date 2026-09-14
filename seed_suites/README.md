## Seed suites (benchmark/regressioni)

Formato:
- **una seed per riga** (intero base 10)
- righe vuote ignorate
- righe che iniziano con `#` ignorate (commenti)

Uso con `scripts/evaluate_agents.py`:
- `--seed-suite small` → usa `small_1000.txt` (1000 seed, pensata per `small=2000` in seat-fair)
- `--seed-suite medium` → usa `medium_5000.txt` (5000 seed, pensata per `medium=10000` in seat-fair)

Nota seat-fair:
- in modalità `--seat-fair` serve **1 seed per coppia** (cioè `num_games/2`)
- quindi `small=2000` richiede 1000 seed, `medium=10000` richiede 5000 seed

Per benchmark “big”:
- invece di versionare un file molto grande (es. 50k seed), puoi generare una suite deterministica via CLI:
  - `--seed-suite-range-start 0` (opzionale: `--seed-suite-range-step 1`)
