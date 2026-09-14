import numpy as np
import pytest

from briscola_ai.ai.training.policy_regularization import cross_entropy_from_probs, grad_ce_wrt_logits_from_probs


def test_cross_entropy_is_zero_for_perfect_one_hot() -> None:
    """CE(one_hot, same_one_hot) = 0 (entro la tolleranza numerica)."""
    q = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    p = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    ce = cross_entropy_from_probs(target_probs=q, pred_probs=p)
    assert ce == pytest.approx(0.0, abs=1e-9)


def test_grad_ce_wrt_logits_matches_pred_minus_target() -> None:
    """∂CE/∂logits = p - q per distribuzioni discrete."""
    q = np.asarray([0.2, 0.5, 0.3], dtype=np.float64)
    p = np.asarray([0.1, 0.6, 0.3], dtype=np.float64)
    grad = grad_ce_wrt_logits_from_probs(pred_probs=p, target_probs=q)
    expected = p - q
    assert np.allclose(grad, expected)


def test_grad_ce_respects_action_mask() -> None:
    """Con mask, il gradiente sulle azioni non valide deve essere zero."""
    q = np.asarray([0.2, 0.5, 0.3], dtype=np.float64)
    p = np.asarray([0.1, 0.6, 0.3], dtype=np.float64)
    mask = np.asarray([True, False, True], dtype=bool)
    grad = grad_ce_wrt_logits_from_probs(pred_probs=p, target_probs=q, action_mask=mask)
    assert grad[1] == 0.0
    assert grad[0] == pytest.approx(p[0] - q[0])
    assert grad[2] == pytest.approx(p[2] - q[2])
