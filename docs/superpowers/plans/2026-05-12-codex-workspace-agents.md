# Codex Workspace AGENTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a workspace-local repo-root `AGENTS.md` that gives Codex practical contribution guidance for PhysicsNeMo.

**Architecture:** Create one concise instruction file at the repository root. The file will synthesize repo context, mandatory use of `CODING_STANDARDS/`, dependency discipline, validation commands, and editing guardrails.

**Tech Stack:** Markdown documentation for Codex workspace instructions.

---

### Task 1: Create `AGENTS.md`

**Files:**
- Create: `AGENTS.md`

- [ ] **Step 1: Write the instruction file**

```markdown
# AGENTS.md

## Workspace Scope

These instructions apply to this PhysicsNeMo workspace only.

## Repository Context

- PhysicsNeMo is a large Python/PyTorch framework for physics ML and scientific computing.
- Core package code lives under `physicsnemo/`.
- End-to-end examples and workflow-specific code live under `examples/`.
- Benchmarks, docs, and tests are maintained separately; keep edits scoped to the task.

## Required Coding Standards

- Before implementing or modifying code, review and follow the relevant guidance in:
  - `CODING_STANDARDS/FUNCTIONAL_APIS.md`
  - `CODING_STANDARDS/MODELS_IMPLEMENTATION.md`
  - `CODING_STANDARDS/EXTERNAL_IMPORTS.md`
- Treat `CODING_STANDARDS/` as authoritative for programming work in this repository.
- Where those standards do not prescribe a choice, preserve existing local style and nearby repository patterns.

## Environment And Dependency Guidance

- Prefer the repo's documented `uv` workflows when working with local environments or installs.
- `pyproject.toml` is the source of truth for package dependencies, optional extras, and development dependencies.
- Keep optional dependencies optional. Guard optional imports and accelerated paths according to `CODING_STANDARDS/EXTERNAL_IMPORTS.md`.
- Example-specific requirements should stay with the relevant example rather than leaking into the core package.

## Validation Guidance

- Prefer targeted tests or checks for the touched area first.
- When broader validation is warranted, use the repository's existing commands, such as:
  - `make pytest`
  - `make doctest`
  - `make lint`
  - `make black`
- Do not claim broad validation without actually running the relevant command.

## Editing Guardrails

- Keep changes focused on the user's request.
- Avoid modifying unrelated generated datasets, outputs, or workspace-only artifacts.
- Avoid broad refactors unless they directly support the task.
- Preserve public APIs unless the task explicitly requires a change and the relevant coding standards allow it.
```

- [ ] **Step 2: Verify the file content**

Run: `sed -n '1,220p' AGENTS.md`

Expected: the file contains the workspace-local Codex instructions shown above.
