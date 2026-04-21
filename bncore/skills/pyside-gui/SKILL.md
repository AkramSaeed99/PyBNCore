---
name: pyside-gui
description: Build responsive, modular PySide6 interfaces — main window shell, dock panels, dialogs, layouts, signals/slots — for the PyBNCore desktop GUI. Use whenever creating or modifying any `views/` file.
---

# When to use
- Creating a new widget, dialog, dock, or menu item.
- Refactoring layouts or styling.
- Adding keyboard shortcuts or a command palette entry.

# Framework
- PySide6 (Qt 6). No Tkinter, no web.
- Minimum PySide6 version: 6.6+.

# Main window shell
- `QMainWindow` with:
  - `QMenuBar` (File, Edit, Model, Inference, Analysis, View, Help)
  - `QToolBar` (New, Open XDSL, Save, Compile, Run Query, Reset Evidence)
  - Left `QDockWidget` → model explorer (`QTreeView`)
  - Center → graph canvas (`QGraphicsView` on a custom `QGraphicsScene`)
  - Right `QDockWidget` → node property inspector (stacked pages per node kind)
  - Bottom `QDockWidget` → tabbed: Results, Logs, Validation
  - `QStatusBar` showing compile state, node count, treewidth, active engine

# Layout rules
- Always use layouts (`QVBoxLayout`, `QHBoxLayout`, `QGridLayout`, `QFormLayout`, `QSplitter`). No absolute positioning.
- Dock widgets must support tabbing, floating, and state save via `saveState()` / `restoreState()`.
- Every dialog is modal unless it edits live state (inspector is non-modal and dockable).

# Signal / slot rules
- Slots are one-liners that call a viewmodel method.
- Never perform IO, inference, or CPT math inside a slot.
- Name signals `<noun>_<past-tense>`: `node_selected`, `evidence_changed`, `query_completed`.
- Disconnect signals explicitly in `closeEvent` for custom dialogs.

# Graph canvas requirements
- `QGraphicsScene` + custom `QGraphicsView` with zoom (Ctrl+wheel), pan (middle-drag), rubber-band select.
- `NodeItem(QGraphicsObject)` per node — supports selection, move, label, state count badge, evidence indicator, type glyph (discrete/noisy-max/equation/continuous).
- `EdgeItem(QGraphicsPathItem)` — bezier or orthogonal routing, arrowhead, updates on node move.
- Hit-test via `shape()` override, not `boundingRect()`.
- Multi-select returns a list; bulk actions run through a single undo command.

# Progressive disclosure
- Beginner surface: Model, Inference, Results tabs.
- Advanced drawer (collapsible or menu): MAP/MPE, Sensitivity, VOI, Loopy BP, Triangulation heuristic, JT stats, Batch, Hybrid/Continuous.
- Never show advanced panels by default on first launch.

# Empty / loading / error states (mandatory for every panel)
- Empty: single-line hint + primary action button.
- Loading: `QProgressBar` busy indicator + cancel button if the worker is cancellable.
- Error: red banner at panel top with the message + "Details" expander containing the traceback.
- Success: populated content; no trailing toast unless the action was fire-and-forget.

# Styling
- One `resources/app.qss` stylesheet. No per-widget inline stylesheets except one-offs (e.g. error banners).
- Use `objectName` for targeted QSS rules; avoid class selectors that break on subclassing.

# Keyboard & command palette
- Standard: Ctrl+N (new), Ctrl+O (open), Ctrl+S (save), Ctrl+Z/Y (undo/redo), F5 (compile), Ctrl+Enter (run query), Del (delete selection).
- Command palette (Ctrl+Shift+P) lists every `QAction` registered on the main window — build it from `findChildren(QAction)`.

# Performance rules
- Never call `scene.update()` in a tight loop; batch with `QGraphicsScene.invalidate(rect)`.
- For > 200 nodes, enable item caching: `setCacheMode(DeviceCoordinateCache)`.
- Defer heavy repaints by using `QTimer.singleShot(0, ...)`.

# Bans
- No blocking calls in the UI thread (no `time.sleep`, no synchronous file IO > 10 MB, no `wrapper.update_beliefs()`).
- No `QWidget` subclass > ~300 lines — extract sub-widgets.
- No mixing of layout wiring and data binding in the same method — separate `_build_ui()` from `_bind_viewmodel()`.
