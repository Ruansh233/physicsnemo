# Codex Workspace Instruction Design

## Goal

Create a repo-root `AGENTS.md` that gives Codex clear, durable workspace
instructions for contributing safely to the PhysicsNeMo repository.

## Scope

The file should be concise and operational. It should help Codex:

- Understand what kind of repository this is.
- Follow the repository's coding standards when programming.
- Respect dependency and optional-import rules.
- Choose reasonable verification commands.
- Avoid touching unrelated generated or experimental workspace artifacts.

It should not become a full contributor guide or duplicate large sections of
existing documentation.

## Recommended Structure

### 1. Repository Overview

Summarize that PhysicsNeMo is a large Python/PyTorch framework for physics ML,
with core library code under `physicsnemo/`, examples under `examples/`, and
specialized standards under `CODING_STANDARDS/`.

### 2. Required Coding Guidance

State that, before implementing or modifying code, Codex must review and follow
the relevant standards in:

- `CODING_STANDARDS/FUNCTIONAL_APIS.md`
- `CODING_STANDARDS/MODELS_IMPLEMENTATION.md`
- `CODING_STANDARDS/EXTERNAL_IMPORTS.md`

The instruction should make these standards authoritative for programming work.
It should also tell Codex to preserve existing local style and repository
patterns where the standards do not speak directly.

### 3. Environment And Dependency Guidance

Capture the repo facts that are most useful for agent work:

- `README.md` and `install.md` use `uv` workflows for local installation.
- `pyproject.toml` is the source of truth for package dependencies and dev
  dependencies.
- Optional dependencies must stay optional and must be guarded appropriately.
- Example-specific requirements belong with examples rather than leaking into
  the core package.

### 4. Validation Guidance

Recommend verification in increasing scope:

- Prefer targeted tests for the touched area first.
- Use relevant repo commands when needed, such as:
  - `make pytest`
  - `make doctest`
  - `make lint`
  - `make black`
- Avoid claiming broad validation without actually running the relevant command.

### 5. Editing Guardrails

Tell Codex to:

- Keep changes scoped to the request.
- Avoid modifying unrelated generated datasets, outputs, or workspace-only files.
- Avoid broad refactors unless they directly support the task.
- Preserve public APIs unless the task explicitly requires changing them and the
  relevant coding standards permit it.

## Expected Result

The resulting `AGENTS.md` should be:

- Short enough to remain maintainable.
- Specific enough to materially improve Codex behavior in this workspace.
- Aligned with existing repo documentation and standards rather than inventing
  new policy.

## Self-Review

- No placeholder text remains.
- The structure is focused on workspace instructions rather than general docs.
- `CODING_STANDARDS/` is treated as the central programming authority.
- Validation guidance matches commands that exist in this repository.
