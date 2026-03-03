import os


def _env_truthy(name: str) -> bool:
    raw = os.getenv(name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


USING_CYTHON_ENGINE_KERNELS = False

if not _env_truthy("PLURIBUS_DISABLE_CYTHON"):
    try:
        from ._engine_kernels import (  # type: ignore[attr-defined]
            all_live_players_all_in,
            eligible_seats,
            filter_pending_eligible,
            live_player_count_and_last,
            next_pending_from,
            total_pot,
        )

        USING_CYTHON_ENGINE_KERNELS = True
    except Exception:
        from ._py_engine_kernels import (
            all_live_players_all_in,
            eligible_seats,
            filter_pending_eligible,
            live_player_count_and_last,
            next_pending_from,
            total_pot,
        )
else:
    from ._py_engine_kernels import (
        all_live_players_all_in,
        eligible_seats,
        filter_pending_eligible,
        live_player_count_and_last,
        next_pending_from,
        total_pot,
    )


__all__ = [
    "USING_CYTHON_ENGINE_KERNELS",
    "all_live_players_all_in",
    "eligible_seats",
    "filter_pending_eligible",
    "live_player_count_and_last",
    "next_pending_from",
    "total_pot",
]
