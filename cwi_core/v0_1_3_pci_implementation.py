"""
CWI — Predictive Causal Intelligence (PCI)
Version: 0.1.3
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

ARCHITECTURE_VERSION = "0.1.3"
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
    """Serialize data deterministically."""

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

    Version 0.1.3 supports binary-treatment ATE with declared contrast.
    The descriptor is separate from DoWhy's internal estimand.
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
            "PCI v0.1.3 supports ATE only."
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

    validate_json_value(
        treatment["treated_value"],
        "treated_value",
    )

    validate_json_value(
        treatment["control_value"],
        "control_value",
    )

    if treatment["treated_value"] == treatment["control_value"]:
        raise ValidationError(
            "Treated and control values must differ."
        )

    if not isinstance(population, dict) or not population:
        raise ValidationError(
            "Estimand population must be a non-empty dictionary."
        )

    normalized_population = Foundations._validate_population(
        population
    )

    normalized_estimand = dict(estimand)
    normalized_estimand["population"] = normalized_population

    return canonical_json(normalized_estimand)


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
    """Validated interface to DoWhy causal inference."""

    FILTER_OPERATORS = {
        "eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in"
    }

    @staticmethod
    def _load_dowhy():
        try:
            from dowhy import CausalModel
            return CausalModel
        except ImportError as exc:
            raise CWIError(
                "DoWhy is not installed. Install project requirements first."
            ) from exc

    @staticmethod
    def _validate_common_causes(common_causes):
        if common_causes is None or not isinstance(common_causes, list):
            raise ValidationError(
                "Explicit common_causes list is required."
            )

        validated = [
            validate_name(cause, "common cause")
            for cause in common_causes
        ]

        if len(set(validated)) != len(validated):
            raise ValidationError(
                "Duplicate common causes are not allowed."
            )

        return validated

    @staticmethod
    def _validate_population(population):
        if population is None:
            return {
                "type": "observed_data",
                "definition": "All rows in the supplied dataset."
            }

        if not isinstance(population, dict) or not population:
            raise ValidationError(
                "population must be a non-empty dictionary."
            )

        kind = population.get("type")

        if kind == "observed_data":
            if set(population) - {"type", "definition"}:
                raise ValidationError(
                    "Invalid observed_data population fields."
                )
            return population

        if kind != "filters":
            raise ValidationError(
                "Causal operations support population type "
                "'observed_data' or 'filters'. Custom descriptors "
                "cannot be used for estimation."
            )

        if set(population) != {"type", "logic", "conditions"}:
            raise ValidationError(
                "Invalid filters population structure."
            )

        if population["logic"] not in {"AND", "OR"}:
            raise ValidationError(
                "Filter logic must be AND or OR."
            )

        conditions = population["conditions"]

        if not isinstance(conditions, list) or not conditions:
            raise ValidationError(
                "Filter conditions must be a non-empty list."
            )

        normalized = []

        for condition in conditions:
            if not isinstance(condition, dict) or set(condition) != {
                "column", "operator", "value"
            }:
                raise ValidationError(
                    "Each filter needs column, operator, value."
                )

            column = validate_name(
                condition["column"],
                "filter column",
            )

            operator = condition["operator"]

            if operator not in Foundations.FILTER_OPERATORS:
                raise ValidationError(
                    f"Unsupported filter operator: {operator}"
                )

            value = condition["value"]

            validate_json_value(
                value,
                f"filter value for {column}",
            )

            if operator in {"in", "not_in"}:
                if not isinstance(value, list) or not value:
                    raise ValidationError(
                        f"Operator {operator} requires a non-empty list."
                    )

            normalized.append({
                "column": column,
                "operator": operator,
                "value": value,
            })

        return {
            "type": "filters",
            "logic": population["logic"],
            "conditions": normalized,
        }

    @staticmethod
    def _apply_population(data, population):
        population = Foundations._validate_population(population)

        if population["type"] == "observed_data":
            return data.copy(), population

        filtered = data.copy()
        masks = []

        for condition in population["conditions"]:
            column = condition["column"]

            if column not in filtered.columns:
                raise ValidationError(
                    f"Population column not found: {column}"
                )

            op = condition["operator"]
            value = condition["value"]
            series = filtered[column]

            try:
                if op == "eq":
                    mask = series == value
                elif op == "ne":
                    mask = series != value
                elif op == "gt":
                    mask = series > value
                elif op == "gte":
                    mask = series >= value
                elif op == "lt":
                    mask = series < value
                elif op == "lte":
                    mask = series <= value
                elif op == "in":
                    mask = series.isin(value)
                else:
                    mask = ~series.isin(value)

            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"Filter cannot be applied to column '{column}': {exc}"
                ) from exc

            masks.append(mask.fillna(False))

        combined = masks[0]

        for mask in masks[1:]:
            if population["logic"] == "AND":
                combined = combined & mask
            else:
                combined = combined | mask

        filtered = filtered.loc[combined].copy()

        if filtered.empty:
            raise ValidationError(
                "Population filter selected no rows."
            )

        return filtered, population

    @staticmethod
    def _validate_dataframe(
        data,
        treatment,
        outcome,
        common_causes,
    ):
        if not hasattr(data, "columns") or not hasattr(data, "copy"):
            raise ValidationError(
                "data must be a pandas DataFrame."
            )

        required = [treatment, outcome] + common_causes

        missing = [
            column
            for column in required
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

        if data[required].isna().any().any():
            raise ValidationError(
                "Required treatment, outcome, or common-cause "
                "columns contain missing values."
            )

        if treatment in common_causes or outcome in common_causes:
            raise ValidationError(
                "Treatment and outcome cannot also be common causes."
            )

    @staticmethod
    def _validate_binary_treatment(
        data,
        treatment,
        treated_value,
        control_value,
    ):
        values = list(data[treatment].unique())

        if len(values) != 2 or not (
            all(
                value in (treated_value, control_value)
                for value in values
            )
            and treated_value in values
            and control_value in values
        ):
            raise ValidationError(
                "Binary ATE requires exactly the declared treated "
                "and control values, with both groups present."
            )

    @staticmethod
    def _prepare(
        data,
        treatment,
        outcome,
        common_causes,
        treated_value,
        control_value,
        population,
    ):
        treatment = validate_name(treatment, "treatment")
        outcome = validate_name(outcome, "outcome")

        if treatment == outcome:
            raise ValidationError(
                "Treatment and outcome must differ."
            )

        causes = Foundations._validate_common_causes(
            common_causes
        )

        selected, normalized_population = (
            Foundations._apply_population(data, population)
        )

        Foundations._validate_dataframe(
            selected,
            treatment,
            outcome,
            causes,
        )

        Foundations._validate_binary_treatment(
            selected,
            treatment,
            treated_value,
            control_value,
        )

        return (
            treatment,
            outcome,
            causes,
            selected,
            normalized_population,
        )

    @staticmethod
    def identify_and_estimate(
        data,
        treatment,
        outcome,
        common_causes=None,
        effect_method="backdoor.linear_regression",
        treated_value=1,
        control_value=0,
        population=None,
    ):
        (
            treatment,
            outcome,
            causes,
            selected,
            population,
        ) = Foundations._prepare(
            data,
            treatment,
            outcome,
            common_causes,
            treated_value,
            control_value,
            population,
        )

        descriptor = build_estimand_descriptor(
            treatment,
            outcome,
            treated_value,
            control_value,
            population,
        )

        CausalModel = Foundations._load_dowhy()

        model = CausalModel(
            data=selected,
            treatment=treatment,
            outcome=outcome,
            common_causes=causes,
        )

        identified = model.identify_effect(
            proceed_when_unidentifiable=False
        )

        if identified is None:
            raise CWIError(
                "DoWhy did not return an identified estimand."
            )

        estimate = model.estimate_effect(
            identified,
            method_name=validate_name(
                effect_method,
                "effect_method",
            ),
            treatment_value=treated_value,
            control_value=control_value,
        )

        return {
            "treatment": treatment,
            "outcome": outcome,
            "common_causes": causes,
            "effect_method": effect_method,
            "estimand": descriptor,
            "dowhy_estimand_description": str(identified),
            "estimate": validate_finite_number(
                estimate.value,
                "estimated effect",
            ),
            "sample_size": int(len(selected)),
            "source_sample_size": int(len(data)),
            "interpretation": (
                "ATE estimate for the declared treatment contrast "
                "and selected population, conditional on supplied "
                "data and assumptions; not proof of causal truth."
            ),
        }

    @staticmethod
    def refute_estimate(
        data,
        treatment,
        outcome,
        common_causes,
        effect_method="backdoor.linear_regression",
        refutation_method="random_common_cause",
        treated_value=1,
        control_value=0,
        population=None,
    ):
        (
            treatment,
            outcome,
            causes,
            selected,
            population,
        ) = Foundations._prepare(
            data,
            treatment,
            outcome,
            common_causes,
            treated_value,
            control_value,
            population,
        )

        descriptor = build_estimand_descriptor(
            treatment,
            outcome,
            treated_value,
            control_value,
            population,
        )

        CausalModel = Foundations._load_dowhy()

        model = CausalModel(
            data=selected,
            treatment=treatment,
            outcome=outcome,
            common_causes=causes,
        )

        identified = model.identify_effect(
            proceed_when_unidentifiable=False
        )

        if identified is None:
            raise CWIError(
                "DoWhy did not return an identified estimand."
            )

        estimate = model.estimate_effect(
            identified,
            method_name=validate_name(
                effect_method,
                "effect_method",
            ),
            treatment_value=treated_value,
            control_value=control_value,
        )

        refutation = model.refute_estimate(
            identified,
            estimate,
            method_name=validate_name(
                refutation_method,
                "refutation_method",
            ),
        )

        return {
            "estimand": descriptor,
            "refutation_method": refutation_method,
            "result": str(refutation),
            "sample_size": int(len(selected)),
            "scientific_boundary": (
                "Refutation tests assess robustness under a specified "
                "test; they do not prove causal truth."
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

        This explicitly closes the SQLite connection.
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
        """
        Return separate diagnostic summaries for each exact estimand.

        Errors from different estimands are never pooled.
        """

        history = self.get_experiment_history(experiment_id)

        predictions = {
            item["prediction_id"]: item
            for item in history["predictions"]
        }

        groups = {}

        for feedback in history["feedback"]:
            prediction = predictions.get(
                feedback["prediction_id"]
            )

            if prediction is None:
                continue

            if (
                prediction["estimand_json"]
                != feedback["estimand_json"]
            ):
                continue

            key = prediction["estimand_json"]

            group = groups.setdefault(key, [])

            error = abs(
                prediction["predicted_effect"]
                - feedback["observed_effect"]
            )

            group.append({
                "prediction_id": feedback["prediction_id"],
                "predicted_effect": prediction["predicted_effect"],
                "observed_effect": feedback["observed_effect"],
                "absolute_error": error,
            })

        estimand_groups = []
        all_comparisons = []

        for estimand_json, comparisons in sorted(groups.items()):
            mae = (
                sum(
                    item["absolute_error"]
                    for item in comparisons
                )
                / len(comparisons)
            )

            estimand_groups.append({
                "estimand": json.loads(estimand_json),
                "comparison_count": len(comparisons),
                "mean_absolute_error": mae,
                "comparisons": comparisons,
                "status": "diagnostic_only",
            })

            all_comparisons.extend(comparisons)

        if not estimand_groups:
            return {
                "experiment_id": experiment_id,
                "comparison_count": 0,
                "mean_absolute_error": None,
                "comparisons": [],
                "estimand_groups": [],
                "status": "insufficient_matched_feedback",
            }

        # Preserve the v0.1.2 response contract when exactly one
        # estimand is present.
        if len(estimand_groups) == 1:
            only_group = estimand_groups[0]

            return {
                "experiment_id": experiment_id,
                "comparison_count": only_group["comparison_count"],
                "mean_absolute_error": only_group["mean_absolute_error"],
                "comparisons": only_group["comparisons"],
                "estimand_groups": estimand_groups,
                "status": "diagnostic_only",
            }

        # Do not report a pooled MAE across different estimands.
        return {
            "experiment_id": experiment_id,
            "comparison_count": len(all_comparisons),
            "mean_absolute_error": None,
            "comparisons": [],
            "estimand_groups": estimand_groups,
            "status": "grouped_diagnostic_only",
            "pooling_policy": (
                "No pooled metric across distinct estimands."
            ),
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
            "Research implementation; validate assumptions and integrations."
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
