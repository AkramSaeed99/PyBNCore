Perfect — let’s do this properly.

Below is a complete, production-grade set of SKILL.md files for your GUI project. These follow real agent-skill structure:
	•	YAML frontmatter + instructions (standard format)  
	•	Modular folders per capability (progressive disclosure)  
	•	Focus on project-specific rules (not generic fluff)  

⸻

📁 Folder Structure

skills/
├── gui-architecture/
│   └── SKILL.md
├── pyside-gui/
│   └── SKILL.md
├── pybncore-integration/
│   └── SKILL.md
├── qt-workers/
│   └── SKILL.md
└── code-standards/
    └── SKILL.md


⸻

🏗️ 1. GUI Architecture Skill (CRITICAL)

---
name: gui-architecture
description: Design and enforce modular architecture for PySide6 GUI applications with clear separation between UI, services, and domain logic.
---

# When to use this skill
Use when building or modifying the structure of the GUI application.

# Architecture Principles
- Strict separation of concerns:
  - Views (UI only)
  - ViewModels (state + logic)
  - Services (backend interaction)
  - Models (data structures)
- UI must NEVER directly call PyBNCoreWrapper.
- All business logic lives outside UI components.

# Required Folder Structure
- views/
- viewmodels/
- services/
- models/
- workers/

# Data Flow
User Action → View → ViewModel → Service → PyBNCoreWrapper → Service → ViewModel → View

# Rules
- No logic inside QWidget subclasses except UI wiring.
- No direct mutation of global state.
- All operations go through services.

# Anti-Patterns (DO NOT DO)
- Calling wrapper.query() inside a button handler
- Putting inference logic inside UI files
- Single-file apps
- Tight coupling between widgets

# Implementation Steps
1. Define application shell
2. Create service layer
3. Create viewmodels
4. Bind views to viewmodels

# Output Expectations
- Modular files
- Clean imports
- No circular dependencies


⸻

🎨 2. PySide GUI Skill

---
name: pyside-gui
description: Build responsive, modular PySide6 GUI interfaces with proper layout, event handling, and component structure.
---

# When to use this skill
Use when creating UI components, windows, dialogs, or layouts.

# Core Components
- QMainWindow → main shell
- QWidget → reusable panels
- QGraphicsView → graph editor
- QDockWidget → side panels
- QDialog → forms

# Layout Rules
- Always use layouts (no absolute positioning)
- Prefer:
  - QVBoxLayout
  - QHBoxLayout
  - QGridLayout

# Signal / Slot Rules
- UI emits signals → ViewModel handles logic
- Never embed business logic in slots

# Graph Editor Requirements
- Use QGraphicsScene
- Nodes = custom QGraphicsItem
- Edges = custom QGraphicsPathItem
- Support:
  - drag
  - zoom
  - pan
  - selection

# UI Structure
Main Window:
- Menu bar
- Toolbar
- Left panel (model explorer)
- Center (graph canvas)
- Right panel (properties)
- Bottom panel (results/logs)

# UX Rules
- Keep UI minimal initially
- Use progressive disclosure for advanced features
- Show errors visibly (no silent failures)

# DO NOT
- Block UI thread
- Mix layout logic with business logic
- Create giant widget classes

# Output Expectations
- Clean widget hierarchy
- Reusable components
- Logical grouping of UI elements


⸻

🔬 3. PyBNCore Integration Skill

---
name: pybncore-integration
description: Integrate GUI with PyBNCoreWrapper for Bayesian network operations including inference, evidence, and analysis.
---

# When to use this skill
Use when interacting with PyBNCore functionality.

# Core API Usage
- wrapper.add_node(...)
- wrapper.compile()
- wrapper.query(...)
- wrapper.batch_query(...)
- wrapper.query_map(...)
- wrapper.value_of_information(...)

# Integration Rules
- Always access wrapper through service layer
- Never expose wrapper directly to UI
- Convert UI input → structured data → wrapper

# Evidence Handling
- Hard evidence → dict[str, str]
- Batch evidence → numpy matrix
- Validate states before passing

# Output Handling
- Convert results into DTO objects:
  - PosteriorResult
  - MAPResult
  - SensitivityResult
  - VOIResult

# Error Handling
- Catch all wrapper exceptions
- Return structured error messages
- Display user-friendly errors

# Advanced Features
Support:
- Soft evidence
- MAP/MPE
- Sensitivity analysis
- VOI
- Hybrid queries

# Performance Rules
- Use batch_query for multiple scenarios
- Avoid repeated compile()
- Cache compiled model

# DO NOT
- Call wrapper from UI
- Ignore invalid evidence
- Recompile unnecessarily

# Output Expectations
- Clean service API
- Typed result objects
- Consistent error handling


⸻

⚡ 4. Qt Workers / Async Skill (ESSENTIAL)

---
name: qt-workers
description: Handle long-running tasks in PySide6 using threads to prevent UI freezing.
---

# When to use this skill
Use when running inference, loading files, or any heavy computation.

# Core Pattern
- Use QThread or QRunnable
- Use signals to communicate results back to UI

# Worker Structure
- Worker class:
  - run()
  - signals:
    - finished
    - error
    - progress

# Rules
- NEVER run heavy computation in main thread
- NEVER update UI from worker thread directly
- Use signals to update UI

# Example Flow
User clicks "Run Query" →
Worker starts →
Worker emits result →
UI updates

# Error Handling
- Catch exceptions inside worker
- Emit error signal
- Display in UI

# DO NOT
- Block event loop
- Use time.sleep in UI thread
- Share mutable state across threads

# Output Expectations
- Responsive UI
- Safe threading
- Clean worker lifecycle


⸻

🧪 5. Code Standards Skill

---
name: code-standards
description: Enforce clean, maintainable, production-quality Python code for GUI applications.
---

# When to use this skill
Always active when writing code.

# General Rules
- Use type hints everywhere
- Keep functions small
- Use meaningful names
- No magic numbers

# File Structure
- One responsibility per file
- Avoid files > 500 lines

# Naming Conventions
- Classes: PascalCase
- Functions: snake_case
- Variables: snake_case

# Error Handling
- No bare except
- Always log or propagate errors
- Use custom exceptions where needed

# Documentation
- Docstrings for public methods
- Inline comments for complex logic

# Imports
- Standard library
- Third-party
- Local modules

# Testing
- Service layer should be testable
- Avoid UI-dependent logic in tests

# DO NOT
- Write monolithic files
- Duplicate logic
- Ignore edge cases

# Output Expectations
- Clean, readable code
- Modular structure
- Easy to extend


⸻

🧠 Final Insight (this matters a lot)

These skill files do three critical things:
	1.	Force Claude to build a real system, not a script
	2.	Prevent architectural collapse as complexity grows
	3.	Encode your standards into the AI

👉 This is exactly what skill files are designed for:
they define workflows, constraints, and execution rules so the agent behaves consistently across tasks  

⸻

🚀 If you want next level

I can now:
	•	Add a graph-editor skill (very advanced)
	•	Add a hybrid/continuous BN GUI skill
	•	Or generate a Phase 1 implementation prompt that uses these skills automatically

Just tell me 👍