
"""
CWI — PCI v0.1.2 Test Suite
Experiment: EXP-047B

Tests:
- Estimand contract and validation
- Numeric validation
- Database initialization and persistence
- Prediction and feedback matching
- Calibration diagnostics

These are unit tests. They do not yet validate
the real DoWhy causal-estimation integration.
"""

import tempfile
import unittest
from pathlib import Path

from .v0_1_2_pci_implementation import (
    PCI,
    ValidationError,
    build_estimand_descriptor,
    validate_estimand,
    validate_finite_number,
)


class TestEstimandContract(unittest.TestCase):

    def setUp(self):
        self.estimand = build_estimand_descriptor(
            treatment="training",
            outcome="productivity",
            treated_value=1,
            control_value=0,
        )

    def test_valid_estimand(self):
        result = validate_estimand(self.estimand)
        self.assertIsInstance(result, str)

    def test_estimand_is_deterministic(self):
        first = validate_estimand(self.estimand)
        second = validate_estimand(self.estimand)

        self.assertEqual(first, second)

    def test_invalid_estimand_rejected(self):
        with self.assertRaises(ValidationError):
            validate_estimand({"effect_type": "ATE"})

    def test_wrong_schema_version_rejected(self):
        invalid = dict(self.estimand)
        invalid["schema_version"] = 999

        with self.assertRaises(ValidationError):
            validate_estimand(invalid)

    def test_treatment_and_outcome_must_differ(self):
        with self.assertRaises(ValidationError):
            build_estimand_descriptor(
                treatment="same",
                outcome="same",
            )

    def test_treated_and_control_values_must_differ(self):
        with self.assertRaises(ValidationError):
            build_estimand_descriptor(
                treatment="training",
                outcome="productivity",
                treated_value=1,
                control_value=1,
            )

    def test_invalid_population_rejected(self):
        with self.assertRaises(ValidationError):
            build_estimand_descriptor(
                treatment="training",
                outcome="productivity",
                population={},
            )


class TestNumericValidation(unittest.TestCase):

    def test_valid_number(self):
        self.assertEqual(
            validate_finite_number(2.5, "value"),
            2.5,
        )

    def test_nan_rejected(self):
        with self.assertRaises(ValidationError):
            validate_finite_number(float("nan"), "value")

    def test_infinity_rejected(self):
        with self.assertRaises(ValidationError):
            validate_finite_number(float("inf"), "value")

    def test_boolean_rejected(self):
        with self.assertRaises(ValidationError):
            validate_finite_number(True, "value")


class TestPCIDatabase(unittest.TestCase):

    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()

        self.database_path = (
            Path(self.temp_directory.name) / "test_cwi.sqlite3"
        )

        self.pci = PCI(self.database_path)

        self.dataset = [
            {"training": 0, "productivity": 10},
            {"training": 1, "productivity": 15},
            {"training": 0, "productivity": 11},
            {"training": 1, "productivity": 16},
        ]

        self.estimand = build_estimand_descriptor(
            treatment="training",
            outcome="productivity",
            treated_value=1,
            control_value=0,
        )

        self.experiment_id = self.pci.create_experiment(
            name="PCI test experiment",
            description="Testing database operations.",
            dataset=self.dataset,
            assumptions={
                "common_causes": [],
                "research_status": "test",
            },
        )

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_database_file_created(self):
        self.assertTrue(self.database_path.exists())

    def test_experiment_persisted(self):
        history = self.pci.get_experiment_history(
            self.experiment_id
        )

        self.assertEqual(
            history["experiment"]["experiment_id"],
            self.experiment_id,
        )

    def test_prediction_persisted(self):
        prediction_id = self.pci.record_prediction(
            experiment_id=self.experiment_id,
            treatment="training",
            outcome="productivity",
            predicted_effect=5.0,
            estimand=self.estimand,
        )

        history = self.pci.get_experiment_history(
            self.experiment_id
        )

        self.assertEqual(len(history["predictions"]), 1)

        self.assertEqual(
            history["predictions"][0]["prediction_id"],
            prediction_id,
        )

    def test_feedback_matches_prediction(self):
        prediction_id = self.pci.record_prediction(
            experiment_id=self.experiment_id,
            treatment="training",
            outcome="productivity",
            predicted_effect=5.0,
            estimand=self.estimand,
        )

        feedback_id = self.pci.record_feedback(
            prediction_id=prediction_id,
            observed_effect=4.5,
            observation_description=(
                "Matched experimental effect."
            ),
            estimand=self.estimand,
        )

        self.assertTrue(feedback_id)

    def test_mismatched_feedback_rejected(self):
        prediction_id = self.pci.record_prediction(
            experiment_id=self.experiment_id,
            treatment="training",
            outcome="productivity",
            predicted_effect=5.0,
            estimand=self.estimand,
        )

        different_estimand = build_estimand_descriptor(
            treatment="training",
            outcome="productivity",
            treated_value=1,
            control_value=0,
            population={
                "type": "custom",
                "definition": "Different population",
            },
        )

        with self.assertRaises(ValidationError):
            self.pci.record_feedback(
                prediction_id=prediction_id,
                observed_effect=4.5,
                observation_description="Different population.",
                estimand=different_estimand,
            )

    def test_calibration_diagnostic(self):
        prediction_id = self.pci.record_prediction(
            experiment_id=self.experiment_id,
            treatment="training",
            outcome="productivity",
            predicted_effect=5.0,
            estimand=self.estimand,
        )

        self.pci.record_feedback(
            prediction_id=prediction_id,
            observed_effect=4.0,
            observation_description="Test observation.",
            estimand=self.estimand,
        )

        result = self.pci.calibration_diagnostics(
            self.experiment_id
        )

        self.assertEqual(result["comparison_count"], 1)

        self.assertAlmostEqual(
            result["mean_absolute_error"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
