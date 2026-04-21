---
name: qt-workers
description: Offload long-running PyBNCore operations (compile, query, batch, sensitivity, VOI, XDSL IO) onto background threads using PySide6. Use whenever a service call can take more than ~100 ms.
---

# When to use
- Any service method that calls `update_beliefs`, `batch_query_marginals`, `sensitivity*`, `value_of_information`, `hybrid_query`, or large file IO.
- Any operation whose latency scales with node count, batch size, or file size.

# Pattern: `QObject` + worker thread (preferred over subclassing `QThread`)
```python
class BaseWorker(QObject):
    finished = Signal(object)       # payload is a DTO
    failed   = Signal(str, str)     # user_message, traceback_text
    progress = Signal(int, str)     # percent, label

    @Slot()
    def run(self) -> None:
        try:
            result = self._execute()
            self.finished.emit(result)
        except DomainError as e:
            self.failed.emit(e.user_message, traceback.format_exc())
        except Exception as e:                       # noqa: BLE001 — last-resort barrier
            self.failed.emit(f"Unexpected error: {e}", traceback.format_exc())

    def _execute(self) -> object:
        raise NotImplementedError
```

Start it from the viewmodel:
```python
thread = QThread(parent=self)
worker = CompileWorker(self._inference_service)
worker.moveToThread(thread)
thread.started.connect(worker.run)
worker.finished.connect(thread.quit)
worker.failed.connect(thread.quit)
thread.finished.connect(worker.deleteLater)
thread.finished.connect(thread.deleteLater)
worker.finished.connect(self._on_compiled)
worker.failed.connect(self._on_failed)
thread.start()
```

# Rules
- **Never** touch a `QWidget` from a worker thread. Communicate only via signals.
- **Never** pass the live `PyBNCoreWrapper` across threads if another thread is writing to it. Serialize access at the `ModelSession` level with a `QMutex` (or simply queue all wrapper work onto the same worker thread).
- Workers receive **service instances**, not views or viewmodels.
- Every worker emits exactly one terminal signal — `finished` or `failed` — never both.
- Long workers must emit `progress` at least every ~500 ms.

# Cancellation
- Workers that support cancel expose a `cancel()` slot that sets an atomic flag.
- The `_execute` loop checks the flag between chunks (e.g. between batch rows).
- Cancelled workers emit `failed("Cancelled by user", "")` so the UI consistently clears its loading state.

# Thread-safety for the session
- The `ModelSession` owns the wrapper and a single `QMutex`.
- Services acquire the lock inside each public method: `with self._session.locked(): ...`.
- Only one worker thread may hold the lock at a time; queue further work at the viewmodel level.

# Patterns per operation
| Operation | Worker class | Typical cost |
|---|---|---|
| Open XDSL | `LoadModelWorker` | IO-bound, 50 ms – 5 s |
| Compile (`update_beliefs`) | `CompileWorker` | CPU, treewidth-dependent |
| Single query | usually sync (< 20 ms after compile); fall back to worker if cold |
| Batch marginals | `BatchQueryWorker` | CPU, scales with rows |
| Sensitivity ranking | `SensitivityWorker` | CPU, scales with parameters |
| VOI | `VOIWorker` | CPU, scales with candidates |
| Hybrid query | `HybridQueryWorker` | CPU, iterative DD |

# Bans
- No `time.sleep` inside the UI thread — ever.
- No shared mutable state between worker and UI except DTOs passed via signals.
- No `QApplication.processEvents()` loops as a cancellation substitute.
- No spawning a new `QThread` per click when the previous one is still running — cancel or queue.
