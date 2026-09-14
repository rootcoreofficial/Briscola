"""Fast path 2-player in Python/NumPy per self-play e valutazione."""

from .state_2p import Fast2PState, new_fast_2p_state, play_random_fast_2p, step_fast_2p

__all__ = ["Fast2PState", "new_fast_2p_state", "play_random_fast_2p", "step_fast_2p"]
