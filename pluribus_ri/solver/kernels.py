import os


def _env_truthy(name: str) -> bool:
    raw = os.getenv(name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


USING_CYTHON_KERNELS = False

if not _env_truthy("PLURIBUS_DISABLE_CYTHON"):
    try:
        from ._cykernels import (  # type: ignore[attr-defined]
            accumulate_strategy_sums_inplace,
            current_strategy_from_regret_array,
        )

        USING_CYTHON_KERNELS = True
    except Exception:
        from ._pykernels import accumulate_strategy_sums_inplace, current_strategy_from_regret_array
else:
    from ._pykernels import accumulate_strategy_sums_inplace, current_strategy_from_regret_array


__all__ = [
    "USING_CYTHON_KERNELS",
    "accumulate_strategy_sums_inplace",
    "current_strategy_from_regret_array",
]
