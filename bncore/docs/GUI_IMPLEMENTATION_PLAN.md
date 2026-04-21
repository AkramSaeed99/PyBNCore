# PyBNCore Desktop GUI — Implementation Plan

Status: Draft (2026-04-20)
Owner: @akram
Backend: `pybncore.PyBNCoreWrapper` (see `pybncore/wrapper.py`)
Framework: PySide6 (Qt 6.6+)
Package root: `pybncore_gui/`

---

## 1. Goal

Build a professional desktop application in the spirit of GeNIe Fusion, tailored to PyBNCore's real capabilities — exact JT inference, batched inference, XDSL/BIF IO, soft evidence, MAP/MPE, sensitivity, VOI, Noisy-MAX, equation nodes, loopy BP, and hybrid continuous support.

Non-goals (for now): web deployment, mobile, multi-user collaboration, cloud sync.

## 2. Architecture (summary — see `skills/gui-architecture/SKILL.md`)

Layered package:

```
pybncore_gui/
├── app.py
├── domain/        # dataclasses, errors, session
├── services/      # facade over PyBNCoreWrapper
├── viewmodels/    # QObject + signals, own UI state
├── views/         # QMainWindow, panels, dialogs, graph canvas
├── workers/       # QThread wrappers for heavy ops
├── commands/      # QUndoCommand subclasses
└── resources/     # qss, icons
```

Layering rules (strict):
- `views/` may **not** import `pybncore` or `services/`.
- `viewmodels/` talks to `services/`; `workers/` also talk to `services/`.
- `services/` is the only code that imports `PyBNCoreWrapper`.

## 3. Tech stack decisions

| Concern | Choice | Why |
|---|---|---|
| UI toolkit | PySide6 | Best Python fit for GeNIe-like desktop UX |
| Graph canvas | `QGraphicsScene`/`QGraphicsView` | Mature, scales to 200+ nodes |
| Plots | `pyqtgraph` | Fast CDF/PDF rendering in a Qt-native widget |
| Threading | `QObject.moveToThread(QThread)` | Clean signal-based results |
| Undo/redo | `QUndoStack` + `QUndoCommand` | Built-in merge, labels, UI |
| Project file | JSON sidecar + original XDSL | Keeps XDSL canonical, adds GUI metadata (layout, scenarios) |
| Packaging | PyInstaller per-OS | Single-file distributable for end users |

## 4. Dependencies to add (as optional extra `gui`)

```toml
[project.optional-dependencies]
gui = [
  "PySide6>=6.6",
  "pyqtgraph>=0.13",
  "networkx>=3.0",   # auto-layout (phase 2+)
]
dev = ["pytest", "pytest-qt", "pytest-cov", "ruff", "mypy"]
```

## 5. Phased delivery

Each phase is a vertical slice shippable on its own. Phases are sized so one SKILL-guided implementation prompt can cover them.

### Phase 1 — Thin vertical slice (foundation)
**Goal:** launch the app, open a model, run a single query end-to-end.

Scope:
- `app.py` bootstrap (QApplication, main window, DI of `ModelSession` + services).
- Main window shell: menu, toolbar, dock layout, status bar.
- `io_service.open_xdsl(path)` / `new_model()` using `read_xdsl`.
- Model explorer (`QTreeView`) listing nodes grouped by kind.
- Graph canvas with read-only `NodeItem` + `EdgeItem` (no editing yet).
- Node property inspector — discrete nodes only: name, states, shaped CPT as a `QTableView`.
- Hard-evidence editor (one `QComboBox` per evidence-marked node).
- `inference_service.compile()` via `CompileWorker`.
- Single-node posterior query via `QueryWorker`.
- Results panel: posterior bar chart (`pyqtgraph`).
- Logs panel.
- Minimal error handling: domain errors surface as red banners.

**Acceptance:**
- App launches on macOS, Linux, Windows.
- Open `tests/fixtures/asia.xdsl` (or equivalent), see 8 nodes and edges.
- Click a node → inspector shows states + CPT.
- Set evidence for 1–2 nodes → hit Run Query → bar chart updates.
- UI never freezes during compile/query.

**Exit criteria for the codebase:**
- No `pybncore` imports in `views/`.
- `services/` covered by pytest at ≥ 70 %.
- `ruff` and `mypy` pass on `domain/` and `services/`.

---

### Phase 2 — Authoring & usability
**Goal:** actually edit a network.

Scope:
- Drag-to-add nodes from a palette; drag-to-connect edges (see `skills/graph-editor`).
- Undo/redo for every structural change via `QUndoStack`.
- CPT editing in the inspector with validation (row-stochastic + shape match).
- Rename/delete nodes and edges.
- Import XDSL + BIF; save XDSL; save project sidecar (node positions, scenarios).
- Validation panel: lists structural issues (cycles, missing CPTs, orphan nodes).
- Keyboard shortcuts + command palette.
- State save/restore (dock layout, window geometry).

**Acceptance:**
- Build a 5-node network from scratch, compile, query, save, re-open — identical state.
- All edits undoable/redoable.
- Invalid CPT entry shows inline validation error; OK button disabled.

---

### Phase 3 — Inference power features
**Goal:** match GeNIe's analysis depth on the discrete side.

Scope:
- Soft / virtual evidence editor per node (likelihood vector with row-stochastic check).
- MAP/MPE runner → `MAPResult` view (assignment table + log-probability).
- Scenario manager: named evidence sets, duplicate, diff view.
- Batch evidence table (`QTableView` backed by a numpy matrix) → `BatchQueryWorker` → results matrix viewer.
- Engine-settings dialog: loopy BP toggle + iterations/damping; triangulation heuristic selector; JT stats panel (cliques, treewidth, memory).
- Result comparison tab: two posteriors side-by-side.

**Acceptance:**
- Run 100-row batch in < 1 s on the Alarm network.
- Switching to loopy BP returns approximate marginals without crashing on a dense graph.
- MAP result matches `query_map` called directly.

---

### Phase 4 — PyBNCore differentiators
**Goal:** ship features the competitors don't have (or hide).

Scope:
- Sensitivity analysis panel (`sensitivity_ranking`) with sortable table + bar chart.
- VOI panel (`value_of_information`) ranked list + candidate selector.
- Noisy-MAX wizard: parent/state pickers, inhibitor probabilities, preview.
- Equation node editor: parent binding + expression field + validation against `set_equation`.
- Performance/benchmark panel: run `batch_query_marginals` timing with varying row counts, plot throughput.
- Monte-Carlo batch workflow: generate evidence rows from priors, run, aggregate.

**Acceptance:**
- Sensitivity ranking for Asia completes < 2 s, top entry matches notebook reference.
- VOI ranking surfaces the node with highest information gain given a query node.
- Noisy-MAX wizard produces a node whose CPT matches a known analytical case.

---

### Phase 5 — Hybrid / continuous experience
**Goal:** first-class continuous support (see `skills/hybrid-continuous`).

Scope:
- Continuous node dialogs (normal, lognormal, uniform, exponential, deterministic, threshold) with live PDF preview.
- Continuous evidence editor (hard value + likelihood-curve mode).
- Continuous-posterior panel: PDF/CDF plot, cursor readout, quantile strip, tail-probability inputs — backed by `ContinuousPosteriorDTO`.
- DD convergence diagnostics (iteration, KL, bin count per node) streamed from `HybridQueryWorker.progress`.
- Rare-event mode: threshold-seeding dialog with "suggest from prior".

**Acceptance:**
- Create a Normal → Deterministic network, query, see a continuous posterior with interactive quantiles.
- Cancel a long DD run cleanly — UI returns to idle within one iteration.

---

## 6. Risk register

| Risk | Mitigation |
|---|---|
| Wrapper is not thread-safe across simultaneous writers | `ModelSession.QMutex` + queue all wrapper work on a single worker thread at a time |
| `update_beliefs()` cost blows up on high-treewidth nets | Expose triangulation heuristic selector; surface treewidth in status bar before compile |
| Graph canvas perf at 500+ nodes | `DeviceCoordinateCache` item caching + `NoIndex` during drag |
| Continuous posteriors require many bins → slow DD | Surface bin count + convergence; let user cap iterations |
| XDSL round-trip losing GUI-only metadata | Keep GUI metadata (positions, scenarios) in a sidecar JSON next to the XDSL |
| Undo stack divergence from service state | All mutations go through commands; commands call services; services are the only mutators |
| PyQt vs PySide licensing confusion | Use PySide6 (LGPL) consistently |
| Slow first open on large nets | Show loading state with cancel; stream progress |

## 7. Milestones (indicative, single-engineer, calendar weeks)

| Phase | Calendar weeks | Gate |
|---|---|---|
| 1 — thin slice | 2 | acceptance + review |
| 2 — authoring | 2 | acceptance + manual QA |
| 3 — inference power | 2 | acceptance + regression tests |
| 4 — differentiators | 2 | acceptance + benchmark parity |
| 5 — hybrid/continuous | 2 | acceptance + real-world model walkthrough |

Total: ~10 weeks to feature-complete v1.

## 8. Definition of done (v1)

- Every phase's acceptance criteria met.
- `ruff` clean, `mypy --strict` clean on `domain/` + `services/`.
- `pytest` green; services coverage ≥ 90 %, viewmodels ≥ 75 %.
- A single PyInstaller build per OS runs with no Python preinstalled.
- README in `pybncore_gui/` with a screenshot and run instructions.

## 9. Next action

Implement **Phase 1** per [pybncore-integration](../skills/pybncore-integration/SKILL.md), [pyside-gui](../skills/pyside-gui/SKILL.md), [qt-workers](../skills/qt-workers/SKILL.md), and [code-standards](../skills/code-standards/SKILL.md). The prompt should quote the Phase 1 acceptance criteria verbatim.
