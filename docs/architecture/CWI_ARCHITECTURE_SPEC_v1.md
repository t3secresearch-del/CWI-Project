# CWI Architecture Specification v1

Status: DESIGN BASELINE
Version: 1.0.0

## 1. Mission

Causal World Intelligence (CWI) is a modular research system designed
to build, represent, test, and revise models of the world.

CWI is not defined as a chatbot or as a single machine-learning model.
It coordinates specialized components through shared data contracts.

## 2. Architectural principles

1. Evidence must be traceable to its source.
2. Observations, assumptions, simulations, and derived claims must
   remain distinguishable.
3. Uncertainty must be represented explicitly.
4. Modules must communicate through documented contracts.
5. Predictions must be distinguishable from observations.
6. Model revisions must be testable and versioned.
7. Existing technologies should be integrated when they satisfy
   demonstrated requirements.
8. New methods should be developed where existing methods are
   insufficient.
9. A successful test supports only the scope it actually tested.

## 3. Core engines

- CWI Core: orchestration, task routing, and system coordination.
- Perception Engine: converts supported inputs into structured data.
- Reasoning Engine: hypotheses, logical relations, and planning.
- World Model: represents states, entities, and state transitions.
- Causal Engine: causal identification, estimation, and reasoning.
- Experiment Engine: selects informative tests and interventions.
- Memory and Knowledge: stores claims, evidence, and provenance.
- Learning and Evaluation: compares predictions with observations,
  evaluates errors, and controls model revisions.

## 4. Shared World State

Engines exchange structured records through a shared representation.

A record should preserve:
- identity and version;
- subject, relation, and object;
- claim or record type;
- temporal context;
- evidence references;
- uncertainty;
- provenance and status.

The shared representation is a design contract, not yet a complete
world model implementation.

## 5. Evidence lifecycle

Input -> validation -> representation -> prediction or hypothesis
-> experiment or observation -> evaluation -> controlled revision.

Synthetic and simulated evidence must never be silently presented
as real-world observations.

## 6. Module protocol

Modules should declare:
- module name and version;
- accepted input contract;
- output contract;
- assumptions;
- evidence references;
- uncertainty and warnings;
- execution status.

## 7. Initial integration strategy

The initial implementation will use Python as the integration layer.
Candidate technologies include:

- SQLite for persistent local records.
- NetworkX for graph representations.
- DoWhy for causal inference.
- NumPy and pandas for numerical and tabular data.
- PyTorch or probabilistic tools when justified by a specific task.

These are candidates, not mandatory dependencies. Each integration
must have a defined purpose and tests.

## 8. Implementation sequence

1. Freeze and validate shared data contracts.
2. Implement shared record storage and validation.
3. Define a module protocol and registry.
4. Connect the existing PCI implementation through an adapter.
5. Integrate graph and evidence representations.
6. Build hypothesis and evidence management.
7. Implement world-state transition models.
8. Implement experiment selection.
9. Add controlled learning and model revision.
10. Add perception modalities and system-level evaluation.

## 9. Current limitations

This specification does not claim that CWI already possesses
general intelligence, autonomous scientific discovery, or a
complete world model.

Those capabilities require implementations and independent tests.

## 10. Change control

Changes to these contracts must be versioned. Existing records must
not be silently reinterpreted after a schema change.
