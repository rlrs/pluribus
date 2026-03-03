import os


def _env_truthy(name: str) -> bool:
    raw = os.getenv(name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


USING_CYTHON_ACTION_KERNELS = False

if not _env_truthy("PLURIBUS_DISABLE_CYTHON"):
    try:
        from ._action_kernels import legal_action_specs  # type: ignore[attr-defined]

        USING_CYTHON_ACTION_KERNELS = True
    except Exception:
        from ._py_action_kernels import legal_action_specs
else:
    from ._py_action_kernels import legal_action_specs


__all__ = [
    "USING_CYTHON_ACTION_KERNELS",
    "legal_action_specs",
]
