"""
Regolarizzazioni per policy discrete (didattico).

Contesto
--------
Durante il fine-tuning RL può succedere che una policy "dimentichi" alcuni comportamenti
desiderati imparati via Behavior Cloning (BC). Un trucco semplice per mitigare questo effetto è
aggiungere un termine di loss che mantiene la policy vicina a un teacher fisso.

In questo progetto lo usiamo per stabilizzare la "gestione briscole" (anti-overkill) senza
ricorrere a post-processing a inference-time (guard).

Matematica minima (cross-entropy come vincolo morbido)
------------------------------------------------------
Dato:
- `p` = distribuzione della policy corrente `π_policy(·|s)` (dopo action mask)
- `q` = distribuzione di un anchor/teacher `π_anchor(·|s)` (stesso spazio azioni e stessa mask)

Definiamo la loss additiva (da minimizzare):

  L_anchor = beta * CE(q, p) = beta * Σ_a q[a] * (-log p[a])

Gradiente rispetto ai logits della policy (prima della softmax):

  ∂L_anchor/∂logits = beta * (p - q)

Nota:
- questa formula assume che `p` e `q` siano distribuzioni valide (somma 1) e che l'action mask
  abbia già "eliminato" le azioni non valide (probabilità ~0).
"""

from __future__ import annotations

import numpy as np


def cross_entropy_from_probs(*, target_probs: np.ndarray, pred_probs: np.ndarray) -> float:
    """
    Cross-entropy CE(target, pred) per distribuzioni discrete.

    Argomenti:
        target_probs: q, distribuzione target (anchor/teacher), shape (A,)
        pred_probs: p, distribuzione predetta (policy), shape (A,)

    Ritorna:
        CE(q, p) = -Σ q[a] log(p[a]) (float).
    """
    if target_probs.shape != pred_probs.shape:
        raise ValueError(f"Shape mismatch: target={target_probs.shape} pred={pred_probs.shape}")
    p = pred_probs.astype(np.float64, copy=False) + 1e-12
    q = target_probs.astype(np.float64, copy=False)
    return float(-np.sum(q * np.log(p)))


def grad_ce_wrt_logits_from_probs(
    *, pred_probs: np.ndarray, target_probs: np.ndarray, action_mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Gradiente della cross-entropy CE(target, pred) rispetto ai logits che generano `pred_probs`.

    Formula:
        dL/dlogits = pred_probs - target_probs

    Argomenti:
        pred_probs: p, distribuzione policy corrente (softmax *dopo* mask), shape (A,)
        target_probs: q, distribuzione anchor/teacher (softmax *dopo* mask), shape (A,)
        action_mask: opzionale, bool shape (A,). Se fornita, azzera il gradiente sulle azioni non valide.

    Ritorna:
        Gradiente (A,) float64 (comodo da sommare a gradienti RL in float64).
    """
    if pred_probs.shape != target_probs.shape:
        raise ValueError(f"Shape mismatch: pred={pred_probs.shape} target={target_probs.shape}")
    grad = pred_probs.astype(np.float64, copy=False) - target_probs.astype(np.float64, copy=False)
    if action_mask is not None:
        if action_mask.shape != pred_probs.shape:
            raise ValueError(f"Mask shape mismatch: mask={action_mask.shape} probs={pred_probs.shape}")
        grad = grad.copy()
        grad[~action_mask.astype(bool, copy=False)] = 0.0
    return grad
