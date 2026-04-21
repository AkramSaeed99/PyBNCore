---
name: graph-editor
description: Implement the interactive Bayesian-network graph canvas on `QGraphicsScene` — node items, edge routing, selection, drag/drop, zoom/pan, and command-based edits. Use whenever working in `views/graph_canvas/` or on canvas-related viewmodels.
---

# When to use
- Adding a new node visual type (discrete, noisy-max, equation, continuous, threshold).
- Changing edge routing, hit-testing, or selection.
- Implementing drag-to-connect, rubber-band select, or multi-select bulk edits.
- Adding context menus or in-canvas inline editing.

# Core classes
```
views/graph_canvas/
├── scene.py             # GraphScene(QGraphicsScene)
├── view.py              # GraphView(QGraphicsView)   — zoom/pan/shortcuts
├── node_item.py         # NodeItem(QGraphicsObject)
├── edge_item.py         # EdgeItem(QGraphicsPathItem)
├── port_item.py         # PortItem(QGraphicsItem)    — attach points
├── pending_edge.py      # PendingEdge — rubber-banded edge during drag-to-connect
└── style.py             # colors, pens, brushes, typography
```

# Node visual contract
- `NodeItem` subclasses `QGraphicsObject` (needs signals).
- Emits: `moved(node_id, QPointF)`, `selected(node_id)`, `double_clicked(node_id)`, `context_menu_requested(node_id, QPointF)`.
- Geometry: rounded rect, 160×72 default, grows to fit label + state count.
- Glyph in top-left corner indicates kind:
  - D = discrete
  - NM = noisy-max
  - EQ = equation
  - C = continuous (normal/lognormal/uniform/exponential)
  - T = threshold
  - Δ = deterministic-continuous
- Evidence indicator: small filled pill in the top-right showing observed state or numeric value.
- Query target indicator: dashed outline when marked as a query node.

# Edge visual contract
- `EdgeItem` paints a cubic bezier with an arrowhead at the child end.
- Recomputes path in `updatePath()` whenever either endpoint moves — connect `NodeItem.moved` to the scene, which calls back into the edge.
- `shape()` returns a stroked path 10 px wide for comfortable hit-testing.
- Selected edges draw in accent color with a thicker pen.

# Scene coordinates & layout
- 1 scene unit = 1 pixel at zoom 1.0.
- Default grid: 20-px snap; toggleable.
- New nodes drop at the view center unless the user drags from the palette.
- Auto-layout (later phase): use Sugiyama layered layout via `networkx` → assign y by topological rank, x by barycenter.

# Interactions
| Gesture | Result |
|---|---|
| Left-click empty | Clear selection |
| Left-click node | Select node, focus inspector |
| Shift/Ctrl-click | Extend selection |
| Left-drag on empty | Rubber-band select |
| Left-drag node | Move selected; snap to grid |
| Left-drag on port | Start drag-to-connect; live `PendingEdge` follows cursor |
| Release on another node's port | Commit edge via `AddEdgeCommand` |
| Double-click node | Open inspector tab for that node |
| Right-click node | Context menu: Set as query, Set evidence, Edit CPT, Delete |
| Right-click empty | Context menu: Add node (submenu by kind) |
| Delete key | `RemoveSelectionCommand` |
| Ctrl+wheel | Zoom (0.1× – 8×), anchored under cursor |
| Middle-drag | Pan |
| Space+drag | Pan alternative |

# Command integration
- Every canvas edit creates a `QUndoCommand` pushed to the app's undo stack. Never mutate the `ModelSession` directly from the scene.
- Commands: `AddNodeCommand`, `RemoveNodeCommand`, `AddEdgeCommand`, `RemoveEdgeCommand`, `MoveNodesCommand` (coalesces a drag into one undoable step), `RenameNodeCommand`.
- Moving the same selection within 500 ms coalesces via `QUndoCommand.mergeWith`.

# Performance
- For > 200 nodes enable `setCacheMode(DeviceCoordinateCache)` on `NodeItem`.
- Use `QGraphicsScene.setItemIndexMethod(NoIndex)` when items move frequently (dragging); re-enable `BspTreeIndex` after drag end.
- Batch scene updates: wrap multi-node moves in `scene.blockSignals(True)` and a single `scene.update(rect)`.

# Binding to the viewmodel
- `GraphViewModel` exposes `nodes: Mapping[str, NodeModel]`, `edges: Mapping[str, EdgeModel]`, and signals `nodes_changed`, `edges_changed`, `selection_changed`.
- `GraphScene` subscribes to those signals and reconciles: create/update/remove items — never rebuild the whole scene.
- Selection is owned by the scene; changes are forwarded to the viewmodel via `selectionChanged`.

# Bans
- No custom painting that bypasses `paint()`; use Qt's rendering.
- No direct access to `PyBNCoreWrapper` from any canvas file.
- No geometry math inside `paint()` — precompute in `updatePath()` / `updateGeometry()`.
- No blocking calls inside `mouseMoveEvent`.
