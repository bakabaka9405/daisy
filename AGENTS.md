# AGENTS.md

Repository guidance for coding agents working in `daisy`.

## Scope
- This repo is a Python 3.13+ experiment library centered on `daisy/`.
- The main user-facing entrypoint is the task CLI in `daisy/__main__.py`.
- Formal experiments are configuration-driven and run from TOML task files.
- Reusable computation belongs in `daisy/`.
- Project-specific docs, task instances, and batch launchers live under `proj/mae/`.

## Repository Layout
- `daisy/`: reusable package code for datasets, models, task configs, runners, data helpers, and utilities.
- `daisy/analysis/`: analysis utilities for interpretability, statistics, and visualization.
- `daisy/analysis/mae/`: MAE project-specific analysis tools including ViT Grad Rollout interpretability.
- `daisy/task/`: registry-driven task system.
- `daisy/task/tasks/<task_name>/`: task-specific configs and runners.
- `daisy/task/shared/`: shared task config models and reusable task helpers.
- `daisy/task/data/`: shared labeled-data loading and split selection helpers.
- `tasks/`: example root-level TOML task files.
- `docs/`: repository-wide task and workflow rules.
- `proj/mae/`: MAE project workspace for docs, task instances, and batch orchestration.
- `tmp/`: legacy and exploratory scripts; do not treat this as the mainline workflow.

## Architecture Notes
- CLI commands live in `daisy/__main__.py`: `run`, `list`, and `ui`.
- Config loading and task execution live in `daisy/task/runner.py`.
- Task registration happens through `TaskRegistry` in `daisy/task/registry.py`.
- Task discovery is handled by `discover_tasks()` in `daisy/task/tasks/__init__.py`; do not maintain a manual import list there.
- Runtime helpers such as output-path resolution, run context, and JSON snapshots live in `daisy/task/runtime.py`.
- Task serialization lives in `daisy/task/serialization.py`.
- Shared config models and transforms live in `daisy/task/shared/`.
- Shared labeled-data and split helpers live in `daisy/task/data/`.
- Formal task config models are defined either in `daisy/task/shared/` or `daisy/task/tasks/*/config.py`.
- Formal runners are under `daisy/task/tasks/*/runner.py`.
- UI field metadata is centralized in `daisy/task/ui_config.py` and consumed by `daisy/task/ui_builder.py`.
- ViT interpretability (Grad Rollout) is implemented in `daisy/analysis/mae/interpretability.py` for avgpool-based ViT models.

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

## Change Placement
- Put reusable logic in `daisy/`.
- Keep project orchestration in task files or batch launchers.
- Avoid adding new mainline workflows under `tmp/`.
- Do not hard-code machine-local paths in library code.
- Preserve reproducibility metadata whenever you touch experiment flows.
- Do not place privacy-sensitive information in the repo, especially in version control; use environment variables or secure vaults instead.

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
- Prefer absolute imports for cross-package references such as `import daisy` or `from daisy.task.data import ...`.
- Prefer relative imports for close intra-package references inside task modules.
- If an import is needed only for typing, prefer `TYPE_CHECKING` blocks when that avoids runtime imports.
- Newer core files often use `from __future__ import annotations`; follow the style of the file you edit.
- Public task APIs should be imported from `daisy.task`; do not reintroduce removed compatibility modules like `daisy.task.config` or `daisy.task.compat`.

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
- For task UI metadata, use `UIFieldConfig`; do not use `dict` or `json_schema_extra` as a parallel schema source.
- Use `dataclass` instead of `dict` for structured data.

## Error Handling And Runtime Behavior
- Fail fast on invalid config or split states.
- Prefer explicit exceptions such as `ValueError`, `FileNotFoundError`, and `RuntimeError`.
- Keep error messages concrete and parameterized with the invalid value when helpful.
- Avoid broad `except Exception` unless you are at a user-facing boundary such as CLI or UI glue.
- In runners, concise `print()` progress output is the established pattern; there is no logging framework standard here.
- When a task runs, preserve or extend the pattern of writing JSON snapshots and output metadata.

## Task System Rules
- The task registry is the extension point.
- New task types belong in `daisy/task/tasks/<task_name>/`.
- Each task type should provide `config.py` and `runner.py`.
- Register runners with `@TaskRegistry.register`.
- New task packages are auto-discovered; do not edit `daisy/task/tasks/__init__.py` to add manual imports.
- Keep `get_task_type()`, the registry key, and the config `Literal[...]` value in sync.
- Put cross-task config models in `daisy/task/shared/`, not inside another task's `config.py`.
- Put reusable labeled-data split logic in `daisy/task/data/`.
- Prefer reusing shared helpers in `daisy/task/runtime.py`, `daisy/task/runner.py`, `daisy/task/data/`, `daisy/task/shared/`, and `daisy/task/tasks/inference_common.py`.
- Runner UI configuration should go through `get_ui_field_overrides()` and `UIFieldConfig`.

## Task File Rules
- Formal task files use TOML, not YAML.
- `task_id` is derived from the task filename stem, so filenames must be stable and readable.
- A formal task should explicitly declare `task_type`, `[meta]`, task-specific config, and `[output]`.
- Formal task metadata should include `title`, `description`, `created_at`, and `commit`; `creator` is strongly recommended.
- Once a formal task has produced official outputs, do not edit it in place; create a new task file instead.
- Prefer explicit seeds and split parameters over hidden defaults.
- Avoid machine-specific notes or assumptions inside task files.
- Only legacy classification-style task files may omit `task_type`; new task files must declare it explicitly.

## Formal Experiment Rules
- Formal experiments should be reproducible, reviewable, and configuration-driven.
- Prefer explicit seeds, fixed splits, and recorded split parameters over hidden defaults.
- Output directories should stay traceable to the task file and commit.
- Preserve task snapshots, metrics, and related JSON metadata.
- Mainline experiment execution should be callable through `python -m daisy run <task.toml>`.
- Do not keep paper-grade or project-grade experiments as one-off scripts.


## Verification Checklist
- Run `ruff check .` if you changed Python code.
- Run `ruff format .` if formatting drift is likely.
- Run `pyright daisy` if you changed typed library or task code.
- Run the smallest relevant `python -m daisy ...` command for task, runner, or CLI changes.
- If you add pytest tests, run the narrowest target first, then the full suite if practical.
- Do not treat legacy `tmp/` scripts as proof that a new change is production-ready.

## Practical Defaults For Agents
- Read `pyproject.toml` first when you need tooling truth.
- Read `docs/task 规范.md` before changing task-file structure, task module architecture, or experiment workflow.
- Read `proj/mae/docs/README.md` before reorganizing MAE project assets.
- Prefer minimal, local edits that match the touched file's style.
- When in doubt, choose reproducibility and explicit configuration over convenience.
