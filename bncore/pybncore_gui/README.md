# PyBNCore GUI

Desktop application for authoring, inspecting, and running inference on
PyBNCore Bayesian networks. Built on PySide6 with a layered architecture
(views → viewmodels → services → workers → domain).

## Install

```bash
pip install "pybncore[gui]"
```

Or, from a working copy:

```bash
pip install -e ".[gui,dev]"
```

## Launch

```bash
pybncore-gui                        # start empty
pybncore-gui path/to/model.xdsl     # auto-open an XDSL
```

Or `python -m pybncore_gui`.

## Features (Phases 1–5 of
[docs/GUI_IMPLEMENTATION_PLAN.md](../docs/GUI_IMPLEMENTATION_PLAN.md))

- **Authoring** — discrete / Noisy-MAX / equation / continuous node
  creation, CPT editing, editable state lists, renames, deletes, edges
  via drag-to-connect or Shift+drag. Full undo/redo.
- **Sub-models** — GeNIe-compatible `<extensions><genie>` round-trip,
  nested tree in the model explorer, breadcrumb navigation, drag-to-
  reparent, ghost stubs for edges crossing into hidden sub-trees.
- **Inference** — exact (junction tree) single queries, batch queries,
  MAP/MPE, soft/virtual evidence, loopy-BP fallback, rare-event
  thresholds and hybrid (DD) inference for continuous models.
- **Analysis** — parameter sensitivity ranking, value-of-information,
  performance benchmark sweep, Monte-Carlo aggregation.
- **Persistence** — XDSL read/write, BIF import, `.pbnproj` project
  sidecar (positions, scenarios, settings, sub-model layout,
  descriptions).

## Keyboard highlights

| Action | Shortcut |
|---|---|
| Open XDSL / Save / Save As | `Ctrl+O` / `Ctrl+S` / `Ctrl+Shift+S` |
| Open / Save Project | `Ctrl+Alt+O` / `Ctrl+Alt+S` |
| Undo / Redo | `Ctrl+Z` / `Ctrl+Y` |
| Add Discrete / Noisy-MAX / Equation / Continuous Node | `Ctrl+Shift+N` / `M` / `E` / `C` |
| Add Threshold | `Ctrl+Shift+T` |
| Add Sub-Model | `Ctrl+G` |
| Exit Sub-Model | `Alt+↑` |
| Rename / Delete | `F2` / `Del` |
| Validate Model | `F7` |
| Compile / Query / MAP / Hybrid | `F5` / `Ctrl+⏎` / `Ctrl+M` / `Ctrl+H` |
| Engine Settings | `Ctrl+,` |

## Tests

```bash
pip install -e ".[gui,dev]"
pytest tests/gui
```

The service tests run against [tests/gui/fixtures/asia_mini.xdsl](../tests/gui/fixtures/asia_mini.xdsl) —
a minimal 3-node Bayesian network with a `<genie>` extension block.

## Packaging

A barebones PyInstaller spec ships at
[`packaging/pybncore_gui.spec`](../packaging/pybncore_gui.spec):

```bash
pip install pyinstaller
pyinstaller packaging/pybncore_gui.spec
```

The resulting single-file binary lives under `dist/`.
