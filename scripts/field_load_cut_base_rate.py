#!/usr/bin/env python3
"""
Base-rate di campo: "carico guidato tardi e tagliato" nelle partite perse vs non-perse dall'IA.

Domanda: quando l'IA (da leader) apre un carico non-briscola in late game e l'umano lo taglia
con una briscola, questo evento è un *marcatore causale di sconfitta* o solo un tratto di
sfondo che capita comunque spesso? La risposta orienta se valga la pena un guard/shaping sui
carichi guidati (finora: no — vedi `PLAN.md` e le sonde in simulazione).

Metodo: sull'export live append-only (JSONL, una riga = un'azione), per ogni partita
umano-vs-IA:

1. identifica il posto umano (`actor == "human"`) e l'esito (`metadata.winning_player_index`,
   fallback su `final_points_by_player_index`);
2. conta gli eventi "IA leader apre carico non-briscola con `deck_size <= --deck-max` e
   `points >= --load-min`" e, tramite la chiusura presa, se sono stati *tagliati* (umano vince
   con una briscola);
3. confronta il tasso di partite con >= 1 taglio tra gruppo "perse" e "non-perse", con lift e
   Fisher exact (two-sided, implementato in puro Python — niente dipendenza da scipy).

**Filtro data (importante):** l'export raw può contenere più giornate. Un errore reale
(2026-07-09) è stato mischiare 2026-07-08 e 2026-07-09 in un unico "oggi". Usa `--date` +
`--timezone` per isolare una giornata; la data della partita è `metadata.finished_at` (epoch)
convertito nel fuso indicato.

**Convenzione pareggi:** i pareggi (60-60) NON sono sconfitte per l'IA e vengono inclusi nel
gruppo "non-perse"; il loro numero è riportato a parte per trasparenza.
"""

from __future__ import annotations

import argparse
import datetime
import json
import zoneinfo
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any

_DEFAULT_INPUT = "data/live_exports/2026-07-09/live_actions_completed_raw.jsonl"


def _game_date(records: list[dict[str, Any]], tz: zoneinfo.ZoneInfo) -> str | None:
    """Data locale (YYYY-MM-DD) della partita da `metadata.finished_at` (epoch UTC)."""
    for r in records:
        finished_at = (r.get("metadata") or {}).get("finished_at")
        if finished_at is not None:
            return str(datetime.datetime.fromtimestamp(float(finished_at), tz).date())
    return None


def _game_outcome(records: list[dict[str, Any]]) -> tuple[int, int, str]:
    """Ritorna (human_seat, ai_seat, esito) con esito in {"lost", "won", "draw"} dal lato IA."""
    human_seat = next(r["player_index"] for r in records if r["actor"] == "human")
    ai_seat = 1 - human_seat
    md = records[0].get("metadata") or {}
    win = md.get("winning_player_index")
    if win is None:
        fp = md.get("final_points_by_player_index")
        if fp is None or fp[0] == fp[1]:
            return human_seat, ai_seat, "draw"
        win = 0 if fp[0] > fp[1] else 1
    if win == ai_seat:
        return human_seat, ai_seat, "won"
    if win == human_seat:
        return human_seat, ai_seat, "lost"
    return human_seat, ai_seat, "draw"


def _completion_map(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """trick_index -> record che chiude la presa (`trick.completed == true`, con winner e carte)."""
    comp: dict[int, dict[str, Any]] = {}
    for r in records:
        trick = r.get("trick") or {}
        if trick.get("completed"):
            comp[r["phase"]["trick_index"]] = r
    return comp


def _count_events(
    records: list[dict[str, Any]], *, human_seat: int, deck_max: int, load_min: int
) -> tuple[int, int, int]:
    """
    Conta (eventi, tagli, punti_tagliati) di "IA leader apre carico non-briscola in late game".

    Un evento è un lead dell'IA con carta non-briscola e `points >= load_min` a
    `deck_size <= deck_max`. È un "taglio" se la presa la vince l'umano con una briscola.
    `punti_tagliati` somma il valore delle prese tagliate (ciò che l'umano incassa).
    """
    comp = _completion_map(records)
    events = cuts = cut_points = 0
    for r in records:
        if r["actor"] != "ai":
            continue
        phase = r["phase"]
        if not phase.get("is_lead") or phase.get("table_size", 0) != 0:
            continue
        if phase["deck_size"] > deck_max:
            continue
        card = r["action"]["card"]
        trump_suit = r["observation"]["trump_suit"]
        if card["suit"] == trump_suit or card["points"] < load_min:
            continue
        events += 1
        closing = comp.get(phase["trick_index"])
        if not closing:
            continue
        trick = closing["trick"]
        cards_by_seat = {c["player_index"]: c["card"] for c in trick["cards"]}
        human_card = cards_by_seat.get(human_seat, {})
        if trick.get("winner_index") == human_seat and human_card.get("suit") == trump_suit:
            cuts += 1
            cut_points += sum(c["card"]["points"] for c in trick["cards"])
    return events, cuts, cut_points


def _fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """
    Fisher exact test two-sided per la 2x2 [[a,b],[c,d]], in puro Python.

    Somma le probabilità ipergeometriche di tutte le tabelle con gli stessi marginali la cui
    probabilità è <= a quella osservata. Adeguato per i piccoli n dei dati di campo.
    """
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1, col1 = a + b, a + c

    def hg(x: int) -> float:
        return comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1)

    observed = hg(a)
    total = 0.0
    for x in range(max(0, col1 - (n - row1)), min(row1, col1) + 1):
        prob = hg(x)
        if prob <= observed + 1e-12:
            total += prob
    return min(1.0, total)


def _pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 1) if den else 0.0


def analyze(
    records_by_game: dict[str, list[dict[str, Any]]],
    *,
    deck_max: int,
    load_min: int,
) -> dict[str, Any]:
    """Aggrega gli eventi per gruppo esito (perse vs non-perse) e calcola lift + Fisher."""
    groups = {
        "lost": {"games": 0, "with_event": 0, "with_cut": 0, "events": 0, "cuts": 0, "cut_points": 0},
        "not_lost": {"games": 0, "with_event": 0, "with_cut": 0, "events": 0, "cuts": 0, "cut_points": 0},
    }
    draws = 0
    cut_games_total = cut_games_not_lost = 0

    for records in records_by_game.values():
        human_seat, _ai_seat, outcome = _game_outcome(records)
        if outcome == "draw":
            draws += 1
        grp = "lost" if outcome == "lost" else "not_lost"
        events, cuts, cut_points = _count_events(records, human_seat=human_seat, deck_max=deck_max, load_min=load_min)
        g = groups[grp]
        g["games"] += 1
        g["events"] += events
        g["cuts"] += cuts
        g["cut_points"] += cut_points
        if events:
            g["with_event"] += 1
        if cuts:
            g["with_cut"] += 1
            cut_games_total += 1
            if grp == "not_lost":
                cut_games_not_lost += 1

    lost, not_lost = groups["lost"], groups["not_lost"]
    a, b = lost["with_cut"], lost["games"] - lost["with_cut"]
    c, d = not_lost["with_cut"], not_lost["games"] - not_lost["with_cut"]
    rate_lost = a / lost["games"] if lost["games"] else 0.0
    rate_not_lost = c / not_lost["games"] if not_lost["games"] else 0.0

    return {
        "params": {"deck_max": deck_max, "load_min": load_min},
        "totals": {
            "games": lost["games"] + not_lost["games"],
            "lost": lost["games"],
            "not_lost": not_lost["games"],
            "draws_in_not_lost": draws,
        },
        "groups": {
            grp: {
                **g,
                "with_event_pct": _pct(g["with_event"], g["games"]),
                "with_cut_pct": _pct(g["with_cut"], g["games"]),
                "avg_points_per_cut": round(g["cut_points"] / g["cuts"], 1) if g["cuts"] else 0.0,
            }
            for grp, g in groups.items()
        },
        "cut_rate_lost": round(rate_lost, 4),
        "cut_rate_not_lost": round(rate_not_lost, 4),
        "lift": round(rate_lost / rate_not_lost, 3) if rate_not_lost else None,
        "fisher_p_two_sided": round(_fisher_exact_two_sided(a, b, c, d), 4),
        "cut_games_not_lost_share": {
            "not_lost": cut_games_not_lost,
            "total": cut_games_total,
            "pct": _pct(cut_games_not_lost, cut_games_total),
        },
    }


def _load_games(path: Path, *, date: str | None, tz: zoneinfo.ZoneInfo) -> dict[str, list[dict[str, Any]]]:
    """Carica e raggruppa i record per partita, filtrando per data locale se richiesto."""
    games: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        games[record["game_id"]].append(record)
    for recs in games.values():
        recs.sort(key=lambda r: r["event_id"])
    if date is None:
        return dict(games)
    return {gid: recs for gid, recs in games.items() if _game_date(recs, tz) == date}


def main() -> int:
    parser = argparse.ArgumentParser(description="Base-rate di campo: carichi guidati tagliati (perse vs non-perse).")
    parser.add_argument("--input", default=_DEFAULT_INPUT, help="JSONL append-only delle azioni live.")
    parser.add_argument("--date", default=None, help="Filtra a una data locale YYYY-MM-DD (default: tutte).")
    parser.add_argument("--timezone", default="Europe/Rome", help="Fuso per interpretare finished_at (default Roma).")
    parser.add_argument("--deck-max", type=int, default=8, help="Late game: deck_size <= N (default 8).")
    parser.add_argument("--load-min", type=int, default=10, help="Carico: points >= N (10=Asso/Tre, 4=+Re).")
    parser.add_argument("--out-json", default="", help="Path JSON opzionale.")
    args = parser.parse_args()

    tz = zoneinfo.ZoneInfo(str(args.timezone))
    games = _load_games(Path(args.input), date=args.date, tz=tz)
    result = analyze(games, deck_max=int(args.deck_max), load_min=int(args.load_min))

    scope = args.date if args.date else "tutte le date"
    print(f"=== Base-rate carichi guidati tagliati — {scope} (tz {args.timezone}) ===")
    t = result["totals"]
    print(
        f"  partite: {t['games']} (perse {t['lost']}, non-perse {t['not_lost']}, "
        f"di cui pareggi {t['draws_in_not_lost']}) | deck<={args.deck_max}, load>={args.load_min}"
    )
    for grp in ("lost", "not_lost"):
        g = result["groups"][grp]
        label = "perse   " if grp == "lost" else "non-perse"
        print(
            f"  {label}: {g['games']:3} partite | con >=1 carico-late: {g['with_event']:2} "
            f"({g['with_event_pct']}%) | con >=1 TAGLIATO: {g['with_cut']:2} ({g['with_cut_pct']}%) | "
            f"eventi={g['events']} tagli={g['cuts']} pt_medi/taglio={g['avg_points_per_cut']}"
        )
    print(f"  lift(perse/non-perse) = {result['lift']} | Fisher two-sided p = {result['fisher_p_two_sided']}")
    share = result["cut_games_not_lost_share"]
    print(f"  partite con >=1 taglio in cui l'IA NON ha perso: {share['not_lost']}/{share['total']} ({share['pct']}%)")

    if str(args.out_json).strip():
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": {"input": str(args.input), "date": args.date, "timezone": str(args.timezone)}, **result}
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON salvato in: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
