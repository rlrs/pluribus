import unittest

from pluribus_ri.solver import kernels
from pluribus_ri.solver import _pykernels


class SolverKernelParityTests(unittest.TestCase):
    def test_current_strategy_matches_python_reference(self) -> None:
        cases = [
            [0, 0, 0],
            [10, 0, -1],
            [-10, -2, -7, 0],
            [3, 5, 11, 0, 2],
        ]
        for regrets in cases:
            got = kernels.current_strategy_from_regret_array(regrets)
            expected = _pykernels.current_strategy_from_regret_array(regrets)
            self.assertEqual(len(got), len(expected))
            for left, right in zip(got, expected):
                self.assertAlmostEqual(left, right, places=12)

    def test_accumulate_matches_python_reference(self) -> None:
        sums_a = [1.0, 2.0, 3.0]
        sums_b = [1.0, 2.0, 3.0]
        strategy = [0.2, 0.3, 0.5]
        weight = 17.0

        kernels.accumulate_strategy_sums_inplace(sums_a, strategy, weight)
        _pykernels.accumulate_strategy_sums_inplace(sums_b, strategy, weight)

        for left, right in zip(sums_a, sums_b):
            self.assertAlmostEqual(left, right, places=12)

    def test_accumulate_raises_for_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            kernels.accumulate_strategy_sums_inplace([0.0, 1.0], [1.0], 1.0)

    def test_kernel_loader_exports_flag(self) -> None:
        self.assertIsInstance(kernels.USING_CYTHON_KERNELS, bool)


if __name__ == "__main__":
    unittest.main()
