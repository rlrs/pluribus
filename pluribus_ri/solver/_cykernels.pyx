# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list current_strategy_from_regret_array(object regrets):
    cdef Py_ssize_t num_actions = len(regrets)
    cdef list positive = [0.0] * num_actions
    cdef double normalizer = 0.0
    cdef Py_ssize_t i
    cdef double value
    cdef double uniform

    for i in range(num_actions):
        value = float(regrets[i])
        if value < 0.0:
            value = 0.0
        positive[i] = value
        normalizer += value

    if normalizer <= 0.0:
        uniform = 1.0 / num_actions
        return [uniform] * num_actions

    for i in range(num_actions):
        positive[i] = positive[i] / normalizer

    return positive


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void accumulate_strategy_sums_inplace(
    list sums,
    object strategy,
    double weight,
):
    cdef Py_ssize_t n = len(sums)
    cdef Py_ssize_t i
    cdef double current
    cdef double prob

    if len(strategy) != n:
        raise ValueError("strategy length mismatch")

    for i in range(n):
        current = <double>sums[i]
        prob = float(strategy[i])
        sums[i] = current + (weight * prob)
