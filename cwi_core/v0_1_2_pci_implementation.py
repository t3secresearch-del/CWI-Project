
"""
CWI — Predictive Causal Intelligence (PCI)
Version: 0.1.2
Experiment: EXP-047B

Research implementation for traceable causal experiments,
predictions, and matched feedback.

Scientific boundary:
    Estimates depend on supplied data, assumptions,
    identification, and estimation methods.
    Refutation tests do not prove causal truth.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_NAME = "Causal World Intelligence"
ARCHITECTURE_NAME = "Predictive Causal Intelligence"

ARCHITECTURE_VERSION = "0.1.2"
EXPERIMENT_ID = "EXP-047B"

DATABASE_SCHEMA_VERSION = 1
ESTIMAND_SCHEMA_VERSION = 1
FINGERPRINT_VERSION = "1"

DEFAULT_DATA_DIRECTORY = Path.home() / ".cwi"
DEFAULT_DATABASE_PATH = (
    DEFAULT_DATA_DIRECTORY / "cwi_history.sqlite3"
)


# ============================================================
# 2. EXCEPTIONS
# ============================================================

class CWIError(Exception):
    """Base exception for CWI."""


class ValidationError(CWIError):
    """Invalid input or incompatible data."""


class RecordNotFoundError(CWIError):
    """Requested record does not exist."""


class DatabaseVersionError(CWIError):
    """Database schema is incompatible."""


# ============================================================
# 3. GENERAL UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_id() -> str:
    return str(uuid.uuid4())


def validate_name(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            f"{field_name} must be a non-empty string."
        )

    return value.strip()


def canonical_json(data: Any) -> str:
    """Serialize JSON deterministically."""

    try:
        return json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"Data cannot be serialized: {exc}"
        ) from exc


def validate_finite_number(
    value: Any,
    field_name: str,
) -> float:

    if isinstance(value, bool):
        raise ValidationError(
            f"{field_name} must be numeric, not boolean."
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{field_name} must be numeric."
        ) from exc

    if not math.isfinite(number):
        raise ValidationError(
            f"{field_name} must be finite."
        )

    return number


def validate_json_value(value: Any, field_name: str) -> Any:
    """Validate that a value can safely enter canonical JSON."""

    canonical_json(value)
    return value


# ============================================================
# 4. ESTIMAND CONTRACT
# ============================================================

def build_estimand_descriptor(
    treatment: str,
    outcome: str,
    treated_value: Any = 1,
    control_value: Any = 0,
    population: Optional[dict] = None,
) -> dict:
    """
    Create PCI's canonical estimand identity.

    Version 0.1.2 supports binary-treatment ATE only.
    The descriptor is deliberately separate from DoWhy's
    internal estimand representation.
    """

    treatment = validate_name(treatment, "treatment")
    outcome = validate_name(outcome, "outcome")

    if treatment == outcome:
        raise ValidationError(
            "Treatment and outcome must differ."
        )

    validate_json_value(treated_value, "treated_value")
    validate_json_value(control_value, "control_value")

    if treated_value == control_value:
        raise ValidationError(
            "Treated and control values must differ."
        )

    if population is None:
        population = {
            "type": "observed_data",
            "definition": (
                "Rows in the supplied dataset used for estimation."
            ),
        }

    if not isinstance(population, dict) or not population:
        raise ValidationError(
            "population must be a non-empty dictionary."
        )

    return {
        "schema_version": ESTIMAND_SCHEMA_VERSION,
        "effect_type": "ATE",
        "treatment": {
            "variable": treatment,
            "treated_value": treated_value,
            "control_value": control_value,
        },
        "outcome": {
            "variable": outcome,
        },
        "population": population,
    }


def validate_estimand(estimand: Any) -> str:
    """Validate and serialize the canonical PCI estimand."""

    if not isinstance(estimand, dict):
        raise ValidationError(
            "Estimand must be a dictionary."
        )

    required = {
        "schema_version",
        "effect_type",
        "treatment",
        "outcome",
        "population",
    }

    if set(estimand.keys()) != required:
        raise ValidationError(
            "Estimand does not match the PCI estimand contract."
        )

    if estimand["schema_version"] != ESTIMAND_SCHEMA_VERSION:
        raise ValidationError(
            "Unsupported estimand schema version."
        )

    if estimand["effect_type"] != "ATE":
        raise ValidationError(
            "PCI v0.1.2 supports ATE only."
        )

    treatment = estimand["treatment"]
    outcome = estimand["outcome"]
    population = estimand["population"]

    if not isinstance(treatment, dict):
        raise ValidationError(
            "Estimand treatment must be a dictionary."
        )

    if set(treatment.keys()) != {
        "variable",
        "treated_value",
        "control_value",
    }:
        raise ValidationError(
            "Invalid treatment descriptor."
        )

    if not isinstance(outcome, dict):
        raise ValidationError(
            "Estimand outcome must be a dictionary."
        )

    if set(outcome.keys()) != {"variable"}:
        raise ValidationError(
            "Invalid outcome descriptor."
        )

    treatment_name = validate_name(
        treatment["variable"],
        "estimand treatment",
    )

    outcome_name = validate_name(
        outcome["variable"],
        "estimand outcome",
    )

    if treatment_name == outcome_name:
        raise ValidationError(
            "Treatment and outcome must differ."
        )

    if treatment["treated_value"] == treatment["control_value"]:
        raise ValidationError(
            "Treated and control values must differ."
        )

    if not isinstance(population, dict) or not population:
        raise ValidationError(
            "Estimand population must be a non-empty dictionary."
        )

    return canonical_json(estimand)


# ============================================================
# 5. DATASET FINGERPRINTING
# ============================================================

def fingerprint_dataset(data: Any) -> str:
    """
    Create a SHA-256 dataset fingerprint.

    Pandas DataFrames use pandas row hashing plus schema metadata.
    Other JSON-compatible objects use deterministic serialization.
    """

    digest = hashlib.sha256()

    is_dataframe = (
        data.__class__.__name__ == "DataFrame"
        and data.__class__.__module__.startswith("pandas")
    )

    if is_dataframe:
        try:
            import pandas as pd

            schema = {
                "fingerprint_version": FINGERPRINT_VERSION,
                "kind": "pandas_dataframe",
                "columns": [str(column) for column in data.columns],
                "dtypes": [str(dtype) for dtype in data.dtypes],
                "index_names": [
                    str(name) if name is not None else None
                    for name in data.index.names
                ],
                "shape": list(data.shape),
            }

            digest.update(
                canonical_json(schema).encode("utf-8")
            )

            row_hashes = pd.util.hash_pandas_object(
                data,
                index=True,
                categorize=True,
            )

            digest.update(row_hashes.to_numpy().tobytes())

            return digest.hexdigest()

        except Exception as exc:
            raise ValidationError(
                f"Could not fingerprint DataFrame: {exc}"
            ) from exc

    serialized = canonical_json({
        "fingerprint_version": FINGERPRINT_VERSION,
        "data": data,
    })

    digest.update(serialized.encode("utf-8"))

    return digest.hexdigest()


# ============================================================
# 6. FOUNDATIONS — DOWHY INTERFACE
# ============================================================

class Foundations:
    """
    Interface to established causal inference technology.

    DoWhy is imported only when a causal operation is requested.
    """

    @staticmethod
    def _load_dowhy():
        try:
            from dowhy import CausalModel
            return CausalModel
        except ImportError as exc:
            raise CWIError(
                "DoWhy is not installed. "
                "Install the project requirements first."
            ) from exc

    @staticmethod
    def _validate_dataframe(
        data: Any,
        treatment: str,
        outcome: str,
        common_causes: list[str],
    ) -> None:

        if not hasattr(data, "columns"):
            raise ValidationError(
                "Data must provide DataFrame-style columns."
            )

        required = [treatment, outcome] + common_causes

        missing = [
            column for column in required
            if column not in data.columns
        ]

        if missing:
            raise ValidationError(
                f"Missing required columns: {missing}"
            )

        if data.empty:
            raise ValidationError(
                "Dataset must not be empty."
            )

    @staticmethod
    def _validate_binary_treatment(
        data: Any,
        treatment: str,
        treated_value: Any,
        control_value: Any,
    ) -> None:
        """Require both specified treatment groups."""

        values = data[treatment]

        if values.isna().any():
            raise ValidationError(
                "Treatment column contains missing values."
            )

        unique_values = list(values.unique())

        if len(unique_values) != 2:
            raise ValidationError(
                "ATE in PCI v0.1.2 requires exactly two "
                "treatment values."
            )

        if not (
            all(value in (treated_value, control_value)
                for value in unique_values)
            and treated_value in unique_values
            and control_value in unique_values
        ):
            raise ValidationError(
                "Treatment values do not match the specified "
                "treated and control values."
            )

    @staticmethod
    def _validate_common_causes(
        common_causes: Optional[list[str]],
    ) -> list[str]:

        if common_causes is None:
            raise ValidationError(
                "Explicit causal assumptions are required."
            )

        if not isinstance(common_causes, list):
            raise ValidationError(
                "common_causes must be a list."
            )

        validated = []

        for cause in common_causes:
            validated.append(
                validate_name(cause, "common cause")
            )

        if len(set(validated)) != len(validated):
            raise ValidationError(
                "Duplicate common causes are not allowed."
            )

        return validated

    @staticmethod
    def identify_and_estimate(
        data: Any,
        treatment: str,
        outcome: str,
        common_causes: Optional[list[str]] = None,
        effect_method: str = "backdoor.linear_regression",
        treated_value: Any = 1,
        control_value: Any = 0,
        population: Optional[dict] = None,
    ) -> dict:
        """
        Identify and estimate an ATE.

        Returns:
            estimand:
                Canonical PCI estimand descriptor.
            dowhy_estimand_description:
                Human-readable description from DoWhy.
            estimate:
                Numeric effect estimate.
        """

        treatment = validate_name(treatment, "treatment")
        outcome = validate_name(outcome, "outcome")
        effect_method = validate_name(
            effect_method,
            "effect_method",
        )

        causes = Foundations._validate_common_causes(
            common_causes
        )

        Foundations._validate_dataframe(
            data,
            treatment,
            outcome,
            causes,
        )

        Foundations._validate_binary_treatment(
            data,
            treatment,
            treated_value,
            control_value,
        )

        descriptor = build_estimand_descriptor(
            treatment=treatment,
            outcome=outcome,
            treated_value=treated_value,
            control_value=control_value,
            population=population,
        )

        CausalModel = Foundations._load_dowhy()

        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=causes,
        )

        dowhy_estimand = model.identify_effect(
            proceed_when_unidentifiable=False
        )

        if dowhy_estimand is None:
            raise CWIError(
                "DoWhy did not return an identified estimand."
            )

        estimate = model.estimate_effect(
            dowhy_estimand,
            method_name=effect_method,
        )

        return {
            "treatment": treatment,
            "outcome": outcome,
            "common_causes": causes,
            "effect_method": effect_method,
            "estimand": descriptor,
            "dowhy_estimand_description": str(dowhy_estimand),
            "estimate": validate_finite_number(
                estimate.value,
                "estimated effect",
            ),
            "interpretation": (
                "Conditional ATE estimate based on supplied data "
                "and assumptions; not proof of causal truth."
            ),
        }

    @staticmethod
    def refute_estimate(
        data: Any,
        treatment: str,
        outcome: str,
        common_causes: list[str],
        effect_method: str = "backdoor.linear_regression",
        refutation_method: str = "random_common_cause",
        treated_value: Any = 1,
        control_value: Any = 0,
    ) -> dict:

        treatment = validate_name(treatment, "treatment")
        outcome = validate_name(outcome, "outcome")

        causes = Foundations._validate_common_causes(
            common_causes
        )

        Foundations._validate_dataframe(
            data,
            treatment,
            outcome,
            causes,
        )

        Foundations._validate_binary_treatment(
            data,
            treatment,
            treated_value,
            control_value,
        )

        descriptor = build_estimand_descriptor(
            treatment=treatment,
            outcome=outcome,
            treated_value=treated_value,
            control_value=control_value,
        )

        CausalModel = Foundations._load_dowhy()

        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=causes,
        )

        dowhy_estimand = model.identify_effect(
            proceed_when_unidentifiable=False
        )

        if dowhy_estimand is None:
            raise CWIError(
                "DoWhy did not return an identified estimand."
            )

        estimate = model.estimate_effect(
            dowhy_estimand,
            method_name=effect_method,
        )

        refutation = model.refute_estimate(
            dowhy_estimand,
            estimate,
            method_name=refutation_method,
        )

        return {
            "estimand": descriptor,
            "refutation_method": refutation_method,
            "result": str(refutation),
            "scientific_boundary": (
                "Refutation tests do not prove causal truth."
            ),
        }


# ============================================================
# 7. PCI — PERSISTENT DATABASE
# ============================================================

class PCI:
    """
    Persistent experiment, prediction and feedback manager.

    Runtime data is stored outside the source repository.
    """

    def __init__(
        self,
        database_path: Optional[str | Path] = None,
    ):

        self.database_path = (
            Path(database_path).expanduser()
            if database_path is not None
            else DEFAULT_DATABASE_PATH
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    @contextmanager
    def _connection(self):
        """
        Manage transaction and connection lifecycle.

        Unlike sqlite3.Connection's context manager alone,
        this explicitly closes the connection.
        """

        connection = sqlite3.connect(
            str(self.database_path),
            timeout=30,
        )

        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")

        try:
            yield connection
            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    # --------------------------------------------------------
    # 7.1 DATABASE INITIALIZATION
    # --------------------------------------------------------

    def _initialize_database(self) -> None:

        with self._connection() as connection:

            connection.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            existing = connection.execute(
                """
                SELECT value
                FROM metadata
                WHERE key = ?
                """,
                ("schema_version",),
            ).fetchone()

            other_tables = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                AND name NOT LIKE 'sqlite_%'
                AND name != 'metadata'
                """
            ).fetchall()

            if existing is None:

                if other_tables:
                    raise DatabaseVersionError(
                        "Database has application tables but no "
                        "schema version. Manual review is required."
                    )

                connection.execute(
                    """
                    INSERT INTO metadata(key, value)
                    VALUES (?, ?)
                    """,
                    (
                        "schema_version",
                        str(DATABASE_SCHEMA_VERSION),
                    ),
                )

            elif int(existing["value"]) != DATABASE_SCHEMA_VERSION:

                raise DatabaseVersionError(
                    "Database schema version mismatch. "
                    "A reviewed migration is required."
                )

            connection.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    dataset_fingerprint TEXT NOT NULL,
                    assumptions_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    treatment TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    predicted_effect REAL NOT NULL,
                    estimand_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,

                    FOREIGN KEY (experiment_id)
                    REFERENCES experiments(experiment_id)
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    prediction_id TEXT NOT NULL UNIQUE,
                    experiment_id TEXT NOT NULL,
                    observed_effect REAL NOT NULL,
                    estimand_json TEXT NOT NULL,
                    observation_description TEXT NOT NULL,
                    created_at TEXT NOT NULL,

                    FOREIGN KEY (prediction_id)
                    REFERENCES predictions(prediction_id),

                    FOREIGN KEY (experiment_id)
                    REFERENCES experiments(experiment_id)
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_predictions_experiment
                ON predictions(experiment_id)
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_feedback_experiment
                ON feedback(experiment_id)
            """)

    # --------------------------------------------------------
    # 7.2 CREATE EXPERIMENT
    # --------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        description: str,
        dataset: Any,
        assumptions: dict,
    ) -> str:

        name = validate_name(name, "experiment name")

        if not isinstance(description, str):
            raise ValidationError(
                "description must be a string."
            )

        if not isinstance(assumptions, dict):
            raise ValidationError(
                "assumptions must be a dictionary."
            )

        dataset_hash = fingerprint_dataset(dataset)
        assumptions_json = canonical_json(assumptions)

        experiment_id = create_id()

        with self._connection() as connection:

            connection.execute(
                """
                INSERT INTO experiments (
                    experiment_id,
                    name,
                    description,
                    dataset_fingerprint,
                    assumptions_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    name,
                    description,
                    dataset_hash,
                    assumptions_json,
                    utc_now(),
                ),
            )

        return experiment_id

    # --------------------------------------------------------
    # 7.3 RECORD PREDICTION
    # --------------------------------------------------------

    def record_prediction(
        self,
        experiment_id: str,
        treatment: str,
        outcome: str,
        predicted_effect: float,
        estimand: dict,
    ) -> str:

        experiment_id = validate_name(
            experiment_id,
            "experiment_id",
        )

        treatment = validate_name(treatment, "treatment")
        outcome = validate_name(outcome, "outcome")

        if treatment == outcome:
            raise ValidationError(
                "Treatment and outcome must differ."
            )

        predicted_effect = validate_finite_number(
            predicted_effect,
            "predicted_effect",
        )

        estimand_json = validate_estimand(estimand)

        if estimand["treatment"]["variable"] != treatment:
            raise ValidationError(
                "Estimand treatment does not match prediction."
            )

        if estimand["outcome"]["variable"] != outcome:
            raise ValidationError(
                "Estimand outcome does not match prediction."
            )

        prediction_id = create_id()

        with self._connection() as connection:

            experiment = connection.execute(
                """
                SELECT experiment_id
                FROM experiments
                WHERE experiment_id = ?
                """,
                (experiment_id,),
            ).fetchone()

            if experiment is None:
                raise RecordNotFoundError(
                    f"Experiment '{experiment_id}' does not exist."
                )

            connection.execute(
                """
                INSERT INTO predictions (
                    prediction_id,
                    experiment_id,
                    treatment,
                    outcome,
                    predicted_effect,
                    estimand_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    experiment_id,
                    treatment,
                    outcome,
                    predicted_effect,
                    estimand_json,
                    utc_now(),
                ),
            )

        return prediction_id

    # --------------------------------------------------------
    # 7.4 RECORD MATCHED FEEDBACK
    # --------------------------------------------------------

    def record_feedback(
        self,
        prediction_id: str,
        observed_effect: float,
        observation_description: str,
        estimand: dict,
    ) -> str:

        prediction_id = validate_name(
            prediction_id,
            "prediction_id",
        )

        observed_effect = validate_finite_number(
            observed_effect,
            "observed_effect",
        )

        if not isinstance(observation_description, str):
            raise ValidationError(
                "observation_description must be a string."
            )

        feedback_estimand_json = validate_estimand(estimand)

        feedback_id = create_id()

        with self._connection() as connection:

            prediction = connection.execute(
                """
                SELECT prediction_id, experiment_id, estimand_json
                FROM predictions
                WHERE prediction_id = ?
                """,
                (prediction_id,),
            ).fetchone()

            if prediction is None:
                raise RecordNotFoundError(
                    f"Prediction '{prediction_id}' does not exist."
                )

            if prediction["estimand_json"] != feedback_estimand_json:
                raise ValidationError(
                    "Feedback estimand does not match prediction. "
                    "Comparison is not scientifically valid."
                )

            connection.execute(
                """
                INSERT INTO feedback (
                    feedback_id,
                    prediction_id,
                    experiment_id,
                    observed_effect,
                    estimand_json,
                    observation_description,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    prediction_id,
                    prediction["experiment_id"],
                    observed_effect,
                    feedback_estimand_json,
                    observation_description,
                    utc_now(),
                ),
            )

        return feedback_id

    # --------------------------------------------------------
    # 7.5 READ EXPERIMENT HISTORY
    # --------------------------------------------------------

    def get_experiment_history(
        self,
        experiment_id: str,
    ) -> dict:

        experiment_id = validate_name(
            experiment_id,
            "experiment_id",
        )

        with self._connection() as connection:

            experiment = connection.execute(
                """
                SELECT *
                FROM experiments
                WHERE experiment_id = ?
                """,
                (experiment_id,),
            ).fetchone()

            if experiment is None:
                raise RecordNotFoundError(
                    f"Experiment '{experiment_id}' does not exist."
                )

            predictions = connection.execute(
                """
                SELECT *
                FROM predictions
                WHERE experiment_id = ?
                ORDER BY created_at
                """,
                (experiment_id,),
            ).fetchall()

            feedback = connection.execute(
                """
                SELECT *
                FROM feedback
                WHERE experiment_id = ?
                ORDER BY created_at
                """,
                (experiment_id,),
            ).fetchall()

        return {
            "experiment": dict(experiment),
            "predictions": [
                dict(row) for row in predictions
            ],
            "feedback": [
                dict(row) for row in feedback
            ],
        }

    # --------------------------------------------------------
    # 7.6 CALIBRATION DIAGNOSTICS
    # --------------------------------------------------------

    def calibration_diagnostics(
        self,
        experiment_id: str,
    ) -> dict:

        history = self.get_experiment_history(experiment_id)

        predictions = {
            item["prediction_id"]: item
            for item in history["predictions"]
        }

        comparisons = []

        for item in history["feedback"]:

            prediction = predictions.get(
                item["prediction_id"]
            )

            if prediction is None:
                continue

            if (
                prediction["estimand_json"]
                != item["estimand_json"]
            ):
                continue

            absolute_error = abs(
                prediction["predicted_effect"]
                - item["observed_effect"]
            )

            comparisons.append({
                "prediction_id": item["prediction_id"],
                "predicted_effect": prediction["predicted_effect"],
                "observed_effect": item["observed_effect"],
                "absolute_error": absolute_error,
            })

        if not comparisons:
            return {
                "experiment_id": experiment_id,
                "comparison_count": 0,
                "mean_absolute_error": None,
                "comparisons": [],
                "status": "insufficient_matched_feedback",
            }

        mean_absolute_error = sum(
            item["absolute_error"]
            for item in comparisons
        ) / len(comparisons)

        return {
            "experiment_id": experiment_id,
            "comparison_count": len(comparisons),
            "mean_absolute_error": mean_absolute_error,
            "comparisons": comparisons,
            "status": "diagnostic_only",
        }


# ============================================================
# 8. SYSTEM INFORMATION
# ============================================================

def system_information() -> dict:

    return {
        "project": PROJECT_NAME,
        "architecture": ARCHITECTURE_NAME,
        "architecture_version": ARCHITECTURE_VERSION,
        "experiment": EXPERIMENT_ID,
        "database": str(DEFAULT_DATABASE_PATH),
        "database_schema_version": DATABASE_SCHEMA_VERSION,
        "estimand_schema_version": ESTIMAND_SCHEMA_VERSION,
        "fingerprint_version": FINGERPRINT_VERSION,
        "runtime_storage": "SQLite",
        "supported_effect": "ATE",
        "supported_treatment": "Binary",
        "scientific_status": (
            "Research implementation; not independently validated."
        ),
    }


# ============================================================
# 9. DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    print(
        json.dumps(
            system_information(),
            indent=2,
            ensure_ascii=False,
        )
    )
