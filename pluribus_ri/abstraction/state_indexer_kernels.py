import os


def _env_truthy(name: str) -> bool:
    raw = os.getenv(name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


USING_CYTHON_STATE_INDEXER_KERNELS = False

if not _env_truthy("PLURIBUS_DISABLE_CYTHON"):
    try:
        from ._state_indexer_kernels import build_public_state_token  # type: ignore[attr-defined]

        USING_CYTHON_STATE_INDEXER_KERNELS = True
    except Exception:
        from ._py_state_indexer_kernels import build_public_state_token
else:
    from ._py_state_indexer_kernels import build_public_state_token


__all__ = [
    "USING_CYTHON_STATE_INDEXER_KERNELS",
    "build_public_state_token",
]
