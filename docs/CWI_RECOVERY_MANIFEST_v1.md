# CWI Recovery Manifest v1

## Purpose
Record the verified state of CWI after repository recovery.
This document distinguishes verified evidence from missing artifacts.

## Repository baseline
- Repository: https://github.com/t3secresearch-del/CWI-Project
- Branch: main
- Baseline HEAD before this manifest: c6aaea23dd43760cad54c0bc70b54e515431fe11
- Working tree before this manifest: CLEAN
- Inventory count: 17 files
- PCI implementation files: 3
- PCI test files: 2
- FCI experiment evidence files: 3

## Verified PCI tests
- PCI v0.1.2: 17/17 tests passed.
- PCI v0.1.3: 46/46 tests passed.
- Total: 63/63 tests passed.
- Invocation: Python package mode (`python -m`).
- Scope: existing unit tests only.
- Limitation: unit-test success does not establish universal correctness.
- DoWhy interface tests include mocked interface behavior.

## FCI experiment evidence

### FCI Stability Challenge v1
- Status: COMPLETED — SCOPED.
- Evidence SHA256:
  `7f1deea0f7cf04adaae9dbdf62c123b86e52b120ceeb962c024811fffee12412`

### FCI Nonlinear Challenge v2
- Status: COMPLETED — SCOPED.
- Evidence SHA256:
  `e6bcee349be7c327f9f813b3ac1e92dc3df7afb584cc8eaeb3d9ec1f24ed51c5`

### FCI Non-Gaussian Challenge v3
- Status: FAILED — SCOPED.
- Evidence SHA256:
  `96652023dae284014de33e566e7ebddd70d2762f27ca1e93642932282f89b0d2`
- Result: 3 expected adjacencies found out of 4.
- Missing adjacency: A-B.
- Original experiment source code and raw dataset were not found
  in the recovered repository.

## Known gaps
1. Original FCI v1, v2 and v3 experiment source scripts are absent
   from the recovered repository inventory.
2. Raw datasets and detailed KCI independence-test diagnostics
   for FCI v3 are not present in its saved JSON evidence.
3. Some previously developed CWI modules are not present in this
   recovered repository snapshot.
4. No claim is made that missing files never existed elsewhere.

## Current work status
- Repository recovery inventory: COMPLETED — SCOPED.
- PCI v0.1.2 baseline: PASS.
- PCI v0.1.3 baseline: PASS.
- FCI v3 root-cause diagnosis: IN PROGRESS.
- CWI causal discovery roadmap: IN PROGRESS.

## Preservation policy
Every future experiment must preserve:
1. Source code.
2. Raw data or deterministic data-generation code.
3. Configuration and dependency versions.
4. Results, including failures.
5. Reproduction instructions.
6. SHA256 hashes.
7. Git commit and remote verification.

No experiment is considered preserved until its required artifacts
are committed and verified on the remote repository.
