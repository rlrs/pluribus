import tempfile
import unittest
from pathlib import Path

from pluribus_ri.abstraction import (
    AbstractionTablesConfig,
    ActionAbstractionConfig,
    load_abstraction_tables_config,
    write_abstraction_tables_config,
)


class AbstractionTablesConfigTests(unittest.TestCase):
    def test_config_roundtrip(self) -> None:
        config = AbstractionTablesConfig(
            history_scope="all",
            action=ActionAbstractionConfig(
                preflop_raise_multipliers=(2.0, 2.5, 4.0, 8.0),
                postflop_pot_raise_fractions=(0.4, 0.8, 1.6),
                flop_pot_raise_fractions=(0.25, 0.75, 1.25),
                turn_pot_raise_fractions=(0.5, 1.0),
                river_pot_raise_fractions=(0.5,),
                include_all_in=False,
                max_raise_actions=5,
                preflop_bucket_policy="canonical169",
                postflop_bucket_policy="texture_v1",
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "abstraction_tables.json"
            written = write_abstraction_tables_config(config, path)
            self.assertTrue(written.exists())

            loaded = load_abstraction_tables_config(path)
            self.assertEqual(loaded, config)

    def test_from_dict_defaults(self) -> None:
        loaded = AbstractionTablesConfig.from_dict({})
        self.assertEqual(loaded.history_scope, "street")
        self.assertEqual(loaded.action.preflop_raise_multipliers, (2.0, 3.0, 5.0, 10.0))
        self.assertEqual(loaded.action.preflop_bucket_policy, "legacy")
        self.assertEqual(loaded.action.postflop_bucket_policy, "legacy")


if __name__ == "__main__":
    unittest.main()
