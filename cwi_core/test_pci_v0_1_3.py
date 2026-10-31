"""
CWI PCI v0.1.3 — Test Suite
"""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from . import v0_1_3_pci_implementation as pci


def make_dataset():
    return pd.DataFrame({
        "treatment": [0, 1, 0, 1, 0, 1],
        "outcome": [10, 14, 12, 16, 11, 15],
        "age": [20, 20, 30, 30, 40, 40],
        "income": [100, 100, 200, 200, 300, 300],
    })


def make_estimand(
    treatment="treatment",
    outcome="outcome",
    treated_value=1,
    control_value=0,
    population=None,
):
    return pci.build_estimand_descriptor(
        treatment=treatment,
        outcome=outcome,
        treated_value=treated_value,
        control_value=control_value,
        population=population,
    )


class FakeEstimate:
    def __init__(self, value):
        self.value = value


class FakeRefutation:
    def __str__(self):
        return "Mock refutation result"


class FakeCausalModel:
    instances = []

    def __init__(self, data, treatment, outcome, common_causes):
        self.data = data.copy()
        self.treatment = treatment
        self.outcome = outcome
        self.common_causes = common_causes
        self.estimate_arguments = None
        FakeCausalModel.instances.append(self)

    def identify_effect(self, proceed_when_unidentifiable=False):
        return "mock_identified_estimand"

    def estimate_effect(
        self,
        identified_estimand,
        method_name=None,
        treatment_value=1,
        control_value=0,
    ):
        self.estimate_arguments = {
            "identified_estimand": identified_estimand,
            "method_name": method_name,
            "treatment_value": treatment_value,
            "control_value": control_value,
        }
        return FakeEstimate(4.0)

    def refute_estimate(
        self,
        identified_estimand,
        estimate,
        method_name=None,
    ):
        return FakeRefutation()


class TestEstimandContract(unittest.TestCase):

    def test_valid_estimand(self):
        estimand = make_estimand()
        serialized = pci.validate_estimand(estimand)
        self.assertIsInstance(serialized, str)
        self.assertEqual(json.loads(serialized)["effect_type"], "ATE")

    def test_estimand_is_deterministic(self):
        first = make_estimand()
        second = dict(reversed(list(first.items())))
        self.assertEqual(
            pci.validate_estimand(first),
            pci.validate_estimand(second),
        )

    def test_same_treatment_and_outcome_rejected(self):
        with self.assertRaises(pci.ValidationError):
            make_estimand(treatment="value", outcome="value")

    def test_equal_treatment_values_rejected(self):
        with self.assertRaises(pci.ValidationError):
            make_estimand(treated_value=1, control_value=1)

    def test_wrong_schema_version_rejected(self):
        estimand = make_estimand()
        estimand["schema_version"] = 999
        with self.assertRaises(pci.ValidationError):
            pci.validate_estimand(estimand)

    def test_unknown_effect_type_rejected(self):
        estimand = make_estimand()
        estimand["effect_type"] = "CATE"
        with self.assertRaises(pci.ValidationError):
            pci.validate_estimand(estimand)

    def test_missing_estimand_field_rejected(self):
        estimand = make_estimand()
        del estimand["population"]
        with self.assertRaises(pci.ValidationError):
            pci.validate_estimand(estimand)

    def test_invalid_population_rejected(self):
        estimand = make_estimand()
        estimand["population"] = {}
        with self.assertRaises(pci.ValidationError):
            pci.validate_estimand(estimand)


class TestPopulationValidation(unittest.TestCase):

    def test_observed_data_population(self):
        population = {
            "type": "observed_data",
            "definition": "All supplied rows",
        }
        result = pci.Foundations._validate_population(population)
        self.assertEqual(result, population)

    def test_filter_population_validated(self):
        population = {
            "type": "filters",
            "logic": "AND",
            "conditions": [{
                "column": "age",
                "operator": "gte",
                "value": 30,
            }],
        }
        result = pci.Foundations._validate_population(population)
        self.assertEqual(result["type"], "filters")
        self.assertEqual(result["logic"], "AND")

    def test_unsupported_population_type_rejected(self):
        population = {
            "type": "custom_population",
            "definition": "Unknown",
        }
        with self.assertRaises(pci.ValidationError):
            pci.Foundations._validate_population(population)

    def test_invalid_filter_logic_rejected(self):
        population = {
            "type": "filters",
            "logic": "XOR",
            "conditions": [{
                "column": "age",
                "operator": "gte",
                "value": 30,
            }],
        }
        with self.assertRaises(pci.ValidationError):
            pci.Foundations._validate_population(population)

    def test_empty_filter_conditions_rejected(self):
        population = {
            "type": "filters",
            "logic": "AND",
            "conditions": [],
        }
        with self.assertRaises(pci.ValidationError):
            pci.Foundations._validate_population(population)

    def test_unknown_filter_operator_rejected(self):
        population = {
            "type": "filters",
            "logic": "AND",
            "conditions": [{
                "column": "age",
                "operator": "contains",
                "value": 30,
            }],
        }
        with self.assertRaises(pci.ValidationError):
            pci.Foundations._validate_population(population)

    def test_filter_missing_column_rejected(self):
        population = {
            "type": "filters",
            "logic": "AND",
            "conditions": [{
                "column": "unknown_column",
                "operator": "gte",
                "value": 30,
            }],
        }
        with self.assertRaises(pci.ValidationError):
            pci.Foundations._apply_population(make_dataset(), population)

    def test_filter_selects_expected_rows(self):
        population = {
            "type": "filters",
            "logic": "AND",
            "conditions": [{
                "column": "age",
                "operator": "gte",
                "value": 30,
            }],
        }
        selected, normalized = pci.Foundations._apply_population(
            make_dataset(), population
        )
        self.assertEqual(len(selected), 4)
        self.assertTrue((selected["age"] >= 30).all())
        self.assertEqual(normalized["logic"], "AND")

    def test_filter_with_no_rows_rejected(self):
        population = {
            "type": "filters",
            "logic": "AND",
            "conditions": [{
                "column": "age",
                "operator": "gt",
                "value": 1000,
            }],
        }
        with self.assertRaises(pci.ValidationError):
            pci.Foundations._apply_population(make_dataset(), population)


class TestDataValidation(unittest.TestCase):

    def test_valid_binary_treatment(self):
        pci.Foundations._validate_binary_treatment(
            make_dataset(), "treatment", 1, 0
        )

    def test_third_treatment_value_rejected(self):
        data = make_dataset()
        data.loc[0, "treatment"] = 2
        with self.assertRaises(pci.ValidationError):
            pci.Foundations._validate_binary_treatment(
                data, "treatment", 1, 0
            )

    def test_missing_treatment_group_rejected(self):
        data = make_dataset()
        data["treatment"] = 1
        with self.assertRaises(pci.ValidationError):
            pci.Foundations._validate_binary_treatment(
                data, "treatment", 1, 0
            )

    def test_treatment_and_outcome_cannot_be_common_causes(self):
        with self.assertRaises(pci.ValidationError):
            pci.Foundations._validate_dataframe(
                make_dataset(),
                treatment="treatment",
                outcome="outcome",
                common_causes=["treatment"],
            )

    def test_duplicate_common_causes_rejected(self):
        with self.assertRaises(pci.ValidationError):
            pci.Foundations._validate_common_causes(["age", "age"])

    def test_missing_required_column_rejected(self):
        data = make_dataset().drop(columns=["age"])
        with self.assertRaises(pci.ValidationError):
            pci.Foundations._validate_dataframe(
                data, "treatment", "outcome", ["age"]
            )

    def test_missing_required_values_rejected(self):
        data = make_dataset()
        data.loc[0, "outcome"] = None
        with self.assertRaises(pci.ValidationError):
            pci.Foundations._validate_dataframe(
                data, "treatment", "outcome", ["age"]
            )


class TestDoWhyInterface(unittest.TestCase):

    def setUp(self):
        FakeCausalModel.instances = []

    @patch.object(
        pci.Foundations,
        "_load_dowhy",
        return_value=FakeCausalModel,
    )
    def test_declared_contrast_is_passed_to_dowhy(self, mocked):
        result = pci.Foundations.identify_and_estimate(
            data=make_dataset(),
            treatment="treatment",
            outcome="outcome",
            common_causes=["age"],
            treated_value=1,
            control_value=0,
        )
        model = FakeCausalModel.instances[-1]
        self.assertEqual(result["estimate"], 4.0)
        self.assertEqual(model.estimate_arguments["treatment_value"], 1)
        self.assertEqual(model.estimate_arguments["control_value"], 0)

    @patch.object(
        pci.Foundations,
        "_load_dowhy",
        return_value=FakeCausalModel,
    )
    def test_non_default_contrast_is_preserved(self, mocked):
        data = pd.DataFrame({
            "dose": [2, 5, 2, 5, 2, 5],
            "outcome": [10, 14, 12, 16, 11, 15],
            "age": [20, 20, 30, 30, 40, 40],
        })
        result = pci.Foundations.identify_and_estimate(
            data=data,
            treatment="dose",
            outcome="outcome",
            common_causes=["age"],
            treated_value=5,
            control_value=2,
        )
        model = FakeCausalModel.instances[-1]
        self.assertEqual(result["estimate"], 4.0)
        self.assertEqual(model.estimate_arguments["treatment_value"], 5)
        self.assertEqual(model.estimate_arguments["control_value"], 2)

    @patch.object(
        pci.Foundations,
        "_load_dowhy",
        return_value=FakeCausalModel,
    )
    def test_population_is_applied_before_model_creation(self, mocked):
        population = {
            "type": "filters",
            "logic": "AND",
            "conditions": [{
                "column": "age",
                "operator": "gte",
                "value": 30,
            }],
        }
        result = pci.Foundations.identify_and_estimate(
            data=make_dataset(),
            treatment="treatment",
            outcome="outcome",
            common_causes=["age"],
            population=population,
        )
        model = FakeCausalModel.instances[-1]
        self.assertEqual(len(model.data), 4)
        self.assertEqual(result["sample_size"], 4)
        self.assertEqual(result["source_sample_size"], 6)

    @patch.object(
        pci.Foundations,
        "_load_dowhy",
        return_value=FakeCausalModel,
    )
    def test_refutation_uses_declared_contrast(self, mocked):
        result = pci.Foundations.refute_estimate(
            data=make_dataset(),
            treatment="treatment",
            outcome="outcome",
            common_causes=["age"],
            treated_value=1,
            control_value=0,
        )
        model = FakeCausalModel.instances[-1]
        self.assertEqual(model.estimate_arguments["treatment_value"], 1)
        self.assertEqual(model.estimate_arguments["control_value"], 0)
        self.assertIn("result", result)
        self.assertEqual(result["sample_size"], 6)


class TestDatasetFingerprint(unittest.TestCase):

    def test_dataframe_fingerprint_is_deterministic(self):
        data = make_dataset()
        self.assertEqual(
            pci.fingerprint_dataset(data),
            pci.fingerprint_dataset(data.copy()),
        )

    def test_changed_data_changes_fingerprint(self):
        first = make_dataset()
        second = make_dataset()
        second.loc[0, "outcome"] = 999
        self.assertNotEqual(
            pci.fingerprint_dataset(first),
            pci.fingerprint_dataset(second),
        )

    def test_json_data_fingerprint_is_deterministic(self):
        self.assertEqual(
            pci.fingerprint_dataset({"a": 1, "b": 2}),
            pci.fingerprint_dataset({"b": 2, "a": 1}),
        )


class TestPCIDatabase(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "test_cwi.sqlite3"
        self.pci = pci.PCI(self.database_path)
        self.experiment_id = self.pci.create_experiment(
            name="PCI v0.1.3 test",
            description="Automated test experiment",
            dataset=make_dataset(),
            assumptions={
                "consistency": True,
                "exchangeability": "assumed for test",
            },
        )
        self.estimand = make_estimand()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_database_file_created(self):
        self.assertTrue(self.database_path.exists())

    def test_experiment_persisted(self):
        history = self.pci.get_experiment_history(self.experiment_id)
        self.assertEqual(
            history["experiment"]["experiment_id"],
            self.experiment_id,
        )

    def test_prediction_persisted(self):
        prediction_id = self.pci.record_prediction(
            experiment_id=self.experiment_id,
            treatment="treatment",
            outcome="outcome",
            predicted_effect=4.0,
            estimand=self.estimand,
        )
        history = self.pci.get_experiment_history(self.experiment_id)
        self.assertEqual(len(history["predictions"]), 1)
        self.assertEqual(
            history["predictions"][0]["prediction_id"],
            prediction_id,
        )

    def test_feedback_matches_prediction(self):
        prediction_id = self.pci.record_prediction(
            self.experiment_id, "treatment", "outcome", 4.0, self.estimand
        )
        feedback_id = self.pci.record_feedback(
            prediction_id=prediction_id,
            observed_effect=4.2,
            observation_description="Matched observation",
            estimand=self.estimand,
        )
        history = self.pci.get_experiment_history(self.experiment_id)
        self.assertTrue(feedback_id)
        self.assertEqual(len(history["feedback"]), 1)

    def test_mismatched_feedback_rejected(self):
        prediction_id = self.pci.record_prediction(
            self.experiment_id, "treatment", "outcome", 4.0, self.estimand
        )
        different_estimand = make_estimand(treated_value=2, control_value=0)
        with self.assertRaises(pci.ValidationError):
            self.pci.record_feedback(
                prediction_id=prediction_id,
                observed_effect=4.2,
                observation_description="Wrong estimand",
                estimand=different_estimand,
            )

    def test_unknown_experiment_rejected(self):
        with self.assertRaises(pci.RecordNotFoundError):
            self.pci.record_prediction(
                "does-not-exist",
                "treatment",
                "outcome",
                4.0,
                self.estimand,
            )

    def test_unknown_prediction_rejected(self):
        with self.assertRaises(pci.RecordNotFoundError):
            self.pci.record_feedback(
                "does-not-exist",
                4.0,
                "Unknown prediction",
                self.estimand,
            )

    def test_boolean_prediction_rejected(self):
        with self.assertRaises(pci.ValidationError):
            self.pci.record_prediction(
                self.experiment_id,
                "treatment",
                "outcome",
                True,
                self.estimand,
            )

    def test_infinite_feedback_rejected(self):
        prediction_id = self.pci.record_prediction(
            self.experiment_id, "treatment", "outcome", 4.0, self.estimand
        )
        with self.assertRaises(pci.ValidationError):
            self.pci.record_feedback(
                prediction_id,
                float("inf"),
                "Invalid feedback",
                self.estimand,
            )

    def test_single_estimand_calibration(self):
        prediction_id = self.pci.record_prediction(
            self.experiment_id, "treatment", "outcome", 4.0, self.estimand
        )
        self.pci.record_feedback(
            prediction_id, 5.0, "Observed effect", self.estimand
        )
        result = self.pci.calibration_diagnostics(self.experiment_id)
        self.assertEqual(result["comparison_count"], 1)
        self.assertAlmostEqual(result["mean_absolute_error"], 1.0)
        self.assertEqual(len(result["estimand_groups"]), 1)

    def test_different_estimands_are_not_pooled(self):
        second_estimand = make_estimand(
            treated_value=2,
            control_value=0,
            population={
                "type": "observed_data",
                "definition": "A separately defined population",
            },
        )
        first_prediction = self.pci.record_prediction(
            self.experiment_id, "treatment", "outcome", 4.0, self.estimand
        )
        self.pci.record_feedback(
            first_prediction, 5.0, "First matched feedback", self.estimand
        )
        second_prediction = self.pci.record_prediction(
            self.experiment_id,
            "treatment",
            "outcome",
            10.0,
            second_estimand,
        )
        self.pci.record_feedback(
            second_prediction,
            12.0,
            "Second matched feedback",
            second_estimand,
        )
        result = self.pci.calibration_diagnostics(self.experiment_id)
        self.assertEqual(result["comparison_count"], 2)
        self.assertIsNone(result["mean_absolute_error"])
        self.assertIsNone(result.get("mae"))
        self.assertEqual(len(result["estimand_groups"]), 2)


class TestDatabaseVersion(unittest.TestCase):

    def test_incompatible_schema_version_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "old.sqlite3"
            connection = sqlite3.connect(database_path)
            connection.execute("""
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            connection.execute(
                """
                INSERT INTO metadata(key, value)
                VALUES (?, ?)
                """,
                ("schema_version", "999"),
            )
            connection.commit()
            connection.close()
            with self.assertRaises(pci.DatabaseVersionError):
                pci.PCI(database_path)


class TestSystemInformation(unittest.TestCase):

    def test_version_is_013(self):
        information = pci.system_information()
        self.assertEqual(
            information["architecture_version"],
            "0.1.3",
        )

    def test_supported_effect_is_ate(self):
        information = pci.system_information()
        self.assertEqual(information["supported_effect"], "ATE")

    def test_supported_treatment_is_binary(self):
        information = pci.system_information()
        self.assertEqual(information["supported_treatment"], "Binary")


if __name__ == "__main__":
    unittest.main(verbosity=2)
