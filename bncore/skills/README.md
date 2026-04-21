# PyBNCore GUI Skills

These SKILL files encode the rules, patterns, and non-obvious decisions for building the PyBNCore desktop GUI. They are loaded progressively by the agent based on the task at hand.

## Skills

| Skill | Scope | Read it when… |
|---|---|---|
| [gui-architecture](gui-architecture/SKILL.md) | Package layout, layering, data flow | creating/moving any module, reviewing structure |
| [pyside-gui](pyside-gui/SKILL.md) | PySide6 widgets, layouts, shortcuts | writing anything in `views/` |
| [pybncore-integration](pybncore-integration/SKILL.md) | Wrapper API, DTOs, service template | writing anything in `services/` |
| [qt-workers](qt-workers/SKILL.md) | Background threads, cancellation, locking | any call > ~100 ms |
| [graph-editor](graph-editor/SKILL.md) | `QGraphicsScene`, node/edge items, commands | touching `views/graph_canvas/` |
| [hybrid-continuous](hybrid-continuous/SKILL.md) | Continuous nodes, DD, `ContinuousPosterior` | any hybrid-feature UI |
| [code-standards](code-standards/SKILL.md) | Python style, typing, tests, logging | always active |

## Using them
- Each SKILL.md is a standalone contract; they can be read independently.
- Rules are prescriptive: if the agent breaks one, the review should reject the change.
- Update the skill file **in the same PR** that changes a rule — don't let them drift.
