"""Test dello storico O(1) usato dai training multi-milione."""

from __future__ import annotations

from briscola_ai.ai.training.streaming_history import StreamingHistory


def test_summary_retains_only_first_and_last_across_many_rows() -> None:
    """Il numero di oggetti trattenuti non deve crescere col numero di update."""
    history = StreamingHistory[dict[str, int]](mode="summary")

    for index in range(100_000):
        history.append({"index": index})

    assert history.count == 100_000
    assert history.retained_rows == 2
    assert history.metadata(full_key="rows", summary_key="summary") == {
        "summary": {
            "mode": "summary",
            "count": 100_000,
            "first": {"index": 0},
            "last": {"index": 99_999},
        }
    }


def test_summary_resume_preserves_count_without_materializing_history() -> None:
    """Un segmento ripreso deve estendere conteggio e ultima riga in memoria costante."""
    initial = StreamingHistory[dict[str, int]](mode="summary")
    initial.append({"index": 1})
    initial.append({"index": 2})

    resumed = StreamingHistory[dict[str, int]].from_resume_state(
        initial.resume_state(),
        expected_mode="summary",
    )
    resumed.append({"index": 3})

    assert resumed.count == 3
    assert resumed.first == {"index": 1}
    assert resumed.last == {"index": 3}
    assert resumed.retained_rows == 2


def test_full_mode_keeps_backward_compatible_rows() -> None:
    """I run piccoli devono continuare a serializzare tutte le metriche."""
    history = StreamingHistory[dict[str, int]](mode="full")
    history.append({"index": 1})
    history.append({"index": 2})

    assert history.metadata(full_key="rows", summary_key="summary") == {"rows": [{"index": 1}, {"index": 2}]}
    assert history.retained_rows == 2
