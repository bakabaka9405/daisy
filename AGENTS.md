# AGENTS.md

Repository guidance for coding agents working in `E:\Working\proj\daisy`.

## Scope
- This repo is a Python 3.12+ experiment library centered on `daisy/`.
- The main user-facing entrypoint is the task CLI in `daisy/__main__.py`.
- Formal experiments are configuration-driven and run from TOML task files.
- Reusable computation belongs in `daisy/`.
- Project-specific docs, task instances, and batch launchers live under `proj/mae/`.

## Rule File Status
- No existing `AGENTS.md` was present when this file was created.
- No `.cursorrules`, `.cursor/rules/`, or `.github/copilot-instructions.md` files were found.
- The strongest repo-specific instructions come from `docs/task 规范.md` and `proj/mae/docs/README.md`.

## Repository Layout
- `daisy/`: reusable package code for datasets, models, task configs, runners, protocols, and utilities.
- `daisy/task/`: registry-driven task system.
- `daisy/task/tasks/<task_name>/`: each task type has a `config.py` and a `runner.py`.
- `daisy/protocol/`: split manifests, leakage checks, and other reproducibility helpers.
- `tasks/`: example root-level TOML task files.
- `docs/`: repository-wide task and workflow rules.
- `proj/mae/`: MAE project workspace for docs, task instances, and batch orchestration.
- `tmp/`: legacy and exploratory scripts; do not treat this as the mainline workflow.
- `outputs/`: default output location for generated experiment artifacts.

## Architecture Notes
- CLI commands live in `daisy/__main__.py`: `run`, `list`, and `ui`.
- Config loading and task execution live in `daisy/task/runner.py`.
- Task registration happens through `TaskRegistry` in `daisy/task/registry.py`.
- Import side effects in `daisy/task/tasks/__init__.py` are required so runners register.
- Runtime helpers such as output-path resolution and JSON snapshots live in `daisy/task/runtime.py`.
- Formal task config models are defined with Pydantic under `daisy/task/tasks/*/config.py`.
- Formal runners are under `daisy/task/tasks/*/runner.py`.

## Core Commands
Use direct Python tooling; there is no custom script runner in `pyproject.toml`.
Use uv as virtual env manager. There's a alias `uvac` for activating the uv environment store in `~/.venv/`
- Environment activation (must use before any python commands): `uvac pytorch`
- Optional uv setup: `uv pip install -e .`
- Build package: `python -m build`
- Lint: `ruff check .`
- Format: `ruff format .`
- Type check: `pyright`
- List tasks: `python -m daisy list`
- Run a task: `python -m daisy run tasks/example.toml`
- Force CPU for a smoke run: `python -m daisy run tasks/example.toml --device cpu`
- Start the UI: `python -m daisy ui`

## Test Guidance
- There is currently no formal `tests/` suite and no pytest config in the repo.
- There is no endorsed single-test command for existing code because no real test suite exists yet.
- Do not treat `tmp/*test.py` scripts as the preferred validation path; they are ad hoc and often machine-specific.
- For current changes, validate through the smallest relevant CLI or module-level smoke command.
- If you add formal pytest tests, use `python -m pytest`.
- For a single pytest test, use `python -m pytest path/to/test_file.py::test_name -q`.
- Prefer adding new automated checks as real pytest tests rather than more `tmp/` scripts.

## Change Placement
- Put reusable logic in `daisy/`.
- Keep project orchestration in task files or batch launchers.
- Keep project docs and concrete experiment assets in `proj/mae/`.
- Avoid adding new mainline workflows under `tmp/`.
- Do not hard-code machine-local paths in library code.
- Preserve reproducibility metadata whenever you touch experiment flows.

## Formatting Conventions
- Follow Ruff settings from `pyproject.toml`.
- Use tabs for indentation in Python files.
- Use single quotes by default.
- Keep line length within 150 characters.
- Existing code uses short docstrings; match nearby style.
- Existing docstrings and comments may be Chinese; preserve local language/style in touched files.
- Do not add comments for obvious code.
- Add a short docstring only when the module, class, or function benefits from it.

## Import Conventions
- Group imports as standard library, third-party packages, then local imports.
- Separate import groups with blank lines.
- Prefer absolute imports for cross-package references such as `import daisy` or `from daisy.protocol import ...`.
- Prefer relative imports for close intra-package references inside task modules.
- If an import is needed only for typing, prefer `TYPE_CHECKING` blocks when that avoids runtime imports.
- Newer core files often use `from __future__ import annotations`; follow the style of the file you edit.

## Type And Data Modeling
- Use modern Python typing such as `list[str]`, `dict[str, Any]`, and `A | B` unions.
- Prefer `Path` over raw path strings once values enter runtime code.
- Use `Literal[...]` for finite string options in configs.
- Task configs are modeled with nested Pydantic `BaseModel` classes.
- Use `Field(default_factory=...)` for mutable defaults.
- Root task config objects should keep `extra = 'forbid'` behavior.
- Use `model_validate()` when loading structured config data.
- Use `model_dump(..., exclude_none=True)` when persisting config snapshots.
- For small structured runtime containers, `@dataclass(slots=True)` is already used and is a good fit.

## Naming Conventions
- Use `snake_case` for modules, files, functions, methods, and variables.
- Use `PascalCase` for classes.
- Use `UPPER_SNAKE_CASE` for module-level constants.
- Config classes should end with `Config`.
- Runner classes should end with `Runner`.
- Task type strings are lowercase snake_case literals such as `classification`, `mae_pretrain`, and `predict_export`.
- Keep naming stable when it affects task IDs, output paths, or registry keys.

## Error Handling And Runtime Behavior
- Fail fast on invalid config or protocol states.
- Prefer explicit exceptions such as `ValueError`, `FileNotFoundError`, and `RuntimeError`.
- Keep error messages concrete and parameterized with the invalid value when helpful.
- Avoid broad `except Exception` unless you are at a user-facing boundary such as CLI or UI glue.
- In runners, concise `print()` progress output is the established pattern; there is no logging framework standard here.
- When a task runs, preserve or extend the pattern of writing JSON snapshots and protocol metadata.

## Task System Rules
- The task registry is the extension point.
- New task types belong in `daisy/task/tasks/<task_name>/`.
- Each task type should provide `config.py` and `runner.py`.
- Register runners with `@TaskRegistry.register`.
- Ensure the task module is imported from `daisy/task/tasks/__init__.py` so registration happens.
- Keep `get_task_type()`, the registry key, and the config `Literal[...]` value in sync.
- Prefer reusing shared helpers in `daisy/task/runtime.py`, `daisy/task/runner.py`, and `daisy/task/tasks/inference_common.py`.

## Task File Rules
- Formal task files use TOML, not YAML.
- `task_id` is derived from the task filename stem, so filenames must be stable and readable.
- A formal task should explicitly declare `task_type`, `[meta]`, task-specific config, and `[output]`.
- Formal task metadata should include `title`, `description`, `created_at`, and `commit`; `creator` is strongly recommended.
- Once a formal task has produced official outputs, do not edit it in place; create a new task file instead.
- Prefer explicit seeds, manifests, and protocol parameters over hidden defaults.
- Avoid machine-specific notes or assumptions inside task files.

## Formal Experiment Rules
- Formal experiments should be reproducible, reviewable, and configuration-driven.
- Prefer explicit seeds, manifests, fixed splits, and recorded protocol parameters over hidden defaults.
- Output directories should stay traceable to the task file and commit.
- Preserve task snapshots, protocol snapshots, metrics, and related JSON metadata.
- Mainline experiment execution should be callable through `python -m daisy run <task.toml>`.
- Do not keep paper-grade or project-grade experiments as one-off scripts.

## `proj/mae` Rules
- `proj/mae/` is for docs, tasks, batch scripts, and archived assets; not reusable training logic.
- Formal MAE experiments should run through `python -m daisy run <task.toml>`.
- Keep `proj/mae/tasks/` task instances flat rather than deeply nested.
- Keep templates separate from concrete task instances.
- Phase-based naming is the documented convention: `p{phase}-{topic}-{variant}.toml`.
- `outputs/mae/{phase}/{task_id}` is the documented project-specific output pattern.
- Batch scripts should generate or fill TOML and invoke the CLI; they should not duplicate training logic.
- Root `tmp/` is not the mainline entrypoint for MAE work.

## Verification Checklist
- Run `ruff check .` if you changed Python code.
- Run `ruff format .` if formatting drift is likely.
- Run `pyright` if you changed typed library or task code.
- Run the smallest relevant `python -m daisy ...` command for task, runner, or CLI changes.
- If you add pytest tests, run the narrowest target first, then the full suite if practical.
- Do not treat legacy `tmp/` scripts as proof that a new change is production-ready.

## Practical Defaults For Agents
- Read `pyproject.toml` first when you need tooling truth.
- Read `docs/task 规范.md` before changing task-file structure or experiment workflow.
- Read `proj/mae/docs/README.md` before reorganizing MAE project assets.
- Prefer minimal, local edits that match the touched file's style.
- When in doubt, choose reproducibility and explicit configuration over convenience.
