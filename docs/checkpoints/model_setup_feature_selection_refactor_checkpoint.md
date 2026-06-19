# Model Setup Feature Selection Refactor Checkpoint

Purpose: checkpoint commit before refactoring the Model Setup feature-selection UI.

Scope after this checkpoint:
- Refactor `utils/model_lab_feature_selection.py` UI flow.
- Preserve returned payload shape.
- Preserve feature registry, unsafe filtering, save, validation, and persistence behavior.

Rollback note:
- Revert to this checkpoint commit if the feature-selection UX refactor needs to be backed out.
