from setuptools import Extension, find_packages, setup


def _build_ext_modules() -> list[Extension]:
    try:
        from Cython.Build import cythonize
    except Exception:
        return []

    extensions = [
        Extension(
            name="pluribus_ri.solver._cykernels",
            sources=["pluribus_ri/solver/_cykernels.pyx"],
        ),
        Extension(
            name="pluribus_ri.core._engine_kernels",
            sources=["pluribus_ri/core/_engine_kernels.pyx"],
        ),
        Extension(
            name="pluribus_ri.abstraction._state_indexer_kernels",
            sources=["pluribus_ri/abstraction/_state_indexer_kernels.pyx"],
        ),
        Extension(
            name="pluribus_ri.abstraction._action_kernels",
            sources=["pluribus_ri/abstraction/_action_kernels.pyx"],
        ),
    ]
    return cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )


setup(
    packages=find_packages(include=["pluribus_ri*"]),
    ext_modules=_build_ext_modules(),
)
