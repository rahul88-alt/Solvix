# Solvix — Master Project Document
### An AI agent that understands a codebase, reasons about requirements, and makes code changes autonomously

**Version:** 0.1 (Draft)
**Owner:** [Your name]
**Status:** Discovery / Pre-build

---

## 1. Vision

Solvix is an AI agent that takes a plain-language requirement or bug report, understands the relevant parts of a codebase, proposes and applies code changes, verifies them against tests, and iterates until the change is correct — with a human in the loop for review and approval.

The vision is not "an AI that answers anything." It's a **reliable, narrow, verifiable** system: given a bounded task and a real codebase, it gets from requirement to working, tested code with minimal human babysitting.

## 2. Mission

To reduce the manual, tedious portion of software development — implementing well-specified features, fixing reproducible bugs, keeping code in sync with changing requirements — by giving developers an agent that can safely read, reason about, and modify their code, while keeping humans firmly in control of what actually merges.

## 3. Problem Statement

Developers spend a large share of their time on tasks that are well-specified but tedious to execute: implementing a described feature, fixing a reproducible bug, updating code to match a changed API, writing tests for existing code. These tasks require understanding the existing codebase, making a coherent change, and confirming it works — a loop that today is entirely manual.

Existing tools help pieces of this (autocomplete, single-file chat assistants) but few close the full loop: **understand → plan → edit → verify → retry.**

## 4. Target Users / Personas

| Persona | Description | Needs |
|---|---|---|
| **Solo developer / indie hacker** | Working alone on a side project or small product | Fast turnaround on small features and bug fixes without context-switching |
| **Team engineer** | Works in a larger codebase with conventions, tests, CI | Changes that respect existing patterns, pass CI, and are safe to review |
| **Engineering manager / tech lead** | Assigns and reviews work | Visibility into what the agent changed and why; confidence it didn't break anything |
| **QA / maintainer** | Verifies correctness | Clear diffs, test coverage for the change, explainable reasoning |

## 5. Goals & Success Metrics

| Goal | Metric |
|---|---|
| Correctly resolve well-specified tasks | % of tasks resolved without human code edits after agent's first PR |
| Reduce time-to-resolution | Median time from task creation to mergeable PR |
| Maintain code quality | % of agent PRs that pass existing lint/test/CI without modification |
| Build user trust | % of proposed changes accepted vs. rejected/reverted |
| Avoid unsafe autonomy | 0 destructive actions taken without explicit user confirmation (deletes, force pushes, prod deploys) |

---

## 6. High-Level Architecture

Five stages, looping between reasoning and execution until verification passes or a retry limit is hit:

```
 Task input
     |
     v
 Context retrieval  (repo index + embeddings)
     |
     v
 Reasoning engine   (LLM plans the change)
     |
     v
 Execute & verify   (edit code, run tests) <---+
     |                                          |
     |---- fails: error fed back to reasoning --+
     |
     v (passes)
 Deliver output     (PR, diff, or report)
```

1. **Task input** — a requirement, ticket, or bug report enters the system.
2. **Context retrieval** — the relevant slice of the codebase is retrieved via a repo index (embeddings + symbol/file search), not the whole repo dumped into context.
3. **Reasoning engine** — an LLM (via API, e.g. Claude) plans the change: which files, what edits, what tests to add or run.
4. **Execute and verify** — the agent applies the diff, runs tests/linters, and captures the result. On failure, it loops back to the reasoning engine with the error output, retrying up to a capped limit.
5. **Deliver output** — a PR, diff, or direct answer is produced for human review.

### Core components

- **Repo indexer** — chunks and embeds the codebase; supports semantic + symbol-based search.
- **Context assembler** — decides what to actually put in the LLM's context window for a given task (retrieved files, related tests, project conventions).
- **Planning/reasoning module** — turns a requirement into a concrete, ordered task list and code diff proposals.
- **Tool executor** — a sandboxed environment where the agent can run shell commands, apply patches, run tests, and read output.
- **Verification layer** — runs the test suite/linter/build and reports pass/fail with details fed back to the reasoning module.
- **Memory/state tracker** — remembers what's been tried in the current task (avoids repeating failed approaches).
- **Human review interface** — surfaces the diff, reasoning trace, and test results for approval before merge.

---

## 7. Technical Design

### 7.1 Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Agent runtime | Python 3.11+ | Best ecosystem for LLM tooling, embeddings, subprocess/sandbox control |
| LLM | Local, via Ollama (Qwen2.5-Coder 14B default; Qwen3.6 27B optional for harder tasks) | No external API key/cost; runs fully offline; swappable behind an interface. Trade-off: noticeably weaker multi-step reasoning than a hosted frontier model — expect more retries/correction on complex tasks |
| Embeddings | Any embedding API (Voyage, OpenAI, or local `sentence-transformers` for offline mode) | Needed for semantic repo search |
| Vector store | Local — Chroma or a flat FAISS index | No need for a hosted DB at MVP scale (single repo, thousands of files at most) |
| Symbol search | `ripgrep` + language-aware parsing via `tree-sitter` | Fast exact/structural search to complement embeddings |
| Sandbox | Docker container per task | Isolation from host; disposable, reproducible |
| CLI | `click` or `typer` | Clean CLI ergonomics |
| Diff handling | `unidiff` / git plumbing (`git apply`, `git diff`) | Reuse git's own diff/patch machinery instead of inventing one |
| State/memory | SQLite (local file per project) | Simple, durable, no external dependency |
| Config | YAML (`.solvix.yml`) | Human-editable, standard for dev tooling |

**Target codebase language for MVP:** pick one to start (Python or JS/TS) — the indexer and test-runner integration are language-specific, so supporting one well beats two poorly.

### 7.2 Repository Structure

```
solvix/
├── cli.py                     # entry point: `solvix run "<task>"`
├── config.py                  # loads and validates .solvix.yml
├── indexer/
│   ├── chunker.py             # splits files into semantically meaningful chunks
│   ├── embedder.py            # calls embedding API, stores vectors
│   ├── symbol_index.py        # tree-sitter based function/class index
│   └── store.py                # vector store read/write (Chroma/FAISS wrapper)
├── context/
│   └── assembler.py           # given a task, decides what goes into the LLM context
├── reasoning/
│   ├── planner.py             # turns task + context into a step plan
│   ├── editor.py               # turns a plan step into a concrete diff proposal
│   └── llm_client.py           # thin wrapper around the local Ollama API (OpenAI-compatible, swappable)
├── execution/
│   ├── sandbox.py              # spins up/tears down the Docker sandbox
│   ├── patch_applier.py        # applies a diff via git apply, handles conflicts
│   └── test_runner.py          # runs project test suite/linter, parses results
├── memory/
│   └── task_state.py           # tracks attempts, errors, decisions per task (SQLite)
├── review/
│   └── pr_builder.py           # builds PR description, diff summary, reasoning trace
├── guardrails/
│   └── policy.py               # dangerous-op detection, path allow/deny list, confirmation prompts
└── tests/                      # tests for Solvix itself (not the target repo's tests)
```

### 7.3 Component Design

**Indexer** — chunks by function/class boundary (via `tree-sitter`), embeds each chunk with metadata (file path, symbol name, line range), and maintains a lightweight symbol map for exact lookups. Re-indexes only changed files on subsequent runs.

**Context Assembler** — given a task: (1) embeds the task description, (2) retrieves top-K similar chunks, (3) retrieves exact symbol matches when the task names a specific function/class/file, (4) pulls in one hop of directly related files (imports/callers), (5) includes project conventions and relevant tests, (6) ranks and truncates to fit the context budget.

**Reasoning Engine** — two-stage, not one-shot:
- *Planner* takes the task + context and returns an ordered list of steps (`{file, description of change}`), shown to the user for approval on risky/large tasks.
- *Editor* takes one step at a time plus current file content and returns a **unified diff only** (parsed programmatically), never a full-file rewrite — keeps changes localized and reviewable.

**Execution & Verification** — one Docker container per task, built from the repo's own environment where possible, network restricted to package installs only. The patch applier runs `git apply` on a fresh branch; if the patch doesn't apply cleanly, it's rejected and re-prompted (a common failure mode from stale context). The test runner detects and runs the project's existing test command, capturing stdout/stderr/exit code. On failure, error output plus the failing diff go back to the Editor for a revised attempt, capped at N retries (default 3).

**Memory/State** — a SQLite table per task run tracking attempt number, diff proposed, verification result, and error summary — used to avoid repeating failed approaches, build the final reasoning trace, and power a task-outcomes dashboard.

**Review/PR Builder** — compiles the diff, a plain-language plan summary, test results, and (if retries occurred) a short note on what was tried and fixed, then opens the PR via the GitHub API or outputs a local patch file.

**Guardrails** — a denylist of dangerous commands (`rm -rf`, `git push --force`, `DROP TABLE`, etc.) requiring explicit confirmation before any sandbox execution, plus a path allow/deny list enforced before a diff touching a denied path is even sent for application.

### 7.4 End-to-End Flow (Example)

1. `solvix run "Add rate limiting to the login endpoint"`
2. Config loaded; repo indexed (or incrementally updated).
3. Context assembler retrieves relevant files (`auth/login.py`, existing middleware, tests).
4. Planner returns a 2-step plan: (1) add rate-limit middleware, (2) wire it into the login route.
5. User approves plan (flagged because it touches `auth/`, a sensitive path).
6. Editor proposes diff for step 1 → sandbox applies → tests run → pass.
7. Editor proposes diff for step 2 → sandbox applies → tests run → fail (missing import).
8. Failure output fed back → Editor revises → tests run → pass.
9. PR builder compiles diff + trace + test results → opens PR.
10. Task state logged; dashboard updated.

### 7.5 Config Schema (`.solvix.yml`)

```yaml
language: python
test_command: pytest -q
lint_command: ruff check .
paths:
  deny:
    - "secrets/**"
    - "infra/**"
    - ".env*"
  sensitive:            # requires plan approval before editing
    - "auth/**"
    - "billing/**"
retries:
  max_attempts: 3
sandbox:
  base_image: auto      # use repo's Dockerfile if present
  network: install-only # allow package installs, block other egress
```

### 7.6 Testing Strategy (for Solvix itself)

- **Unit tests** for chunking, diff parsing, config validation, guardrail policy checks.
- **Golden-task benchmark** — a curated set (10–20) of real tasks against a sample repo with known-correct diffs, run on every change to Solvix to track resolution rate over time. This is the core quality metric tying back to the success metrics in Section 5.
- **Sandbox integration tests** — confirm containers are torn down, network is restricted, denylisted commands are actually blocked.

---

## 8. Non-Functional Requirements

- **Safety:** no destructive git operations (force-push, hard reset, delete branch) without explicit confirmation.
- **Reversibility:** every change is a reviewable diff/PR — nothing is applied directly to a protected branch.
- **Transparency:** the agent's reasoning and the exact commands it ran must be inspectable, not just the final diff.
- **Bounded retries:** a hard cap on retry loops (e.g. 3–5 attempts) to avoid runaway cost/time.
- **Cost visibility:** with a local model there's no per-token billing, but task duration and retry count should still be tracked and surfaced — cost now shows up as time/compute, not dollars, and a stuck retry loop can still waste real time on a laptop.
- **Isolation:** code execution happens in a sandboxed environment, not on the host machine directly.

---

## 9. Epics & User Stories

Each story follows: *As a [persona], I want [capability], so that [benefit].*

### Epic A: Repository Understanding

**A1.** As a developer, I want the agent to index my repository, so that it can retrieve relevant code without needing the entire codebase in context.
- *Acceptance:* Given a repo path, the system builds a searchable index (file + symbol + embedding based) within a reasonable time for repos up to X files.

**A2.** As a developer, I want the agent to find the right files for a given task automatically, so that I don't have to manually point it at the relevant code.
- *Acceptance:* Given a task description, the top-N retrieved files include the file(s) actually requiring changes, verified against a labeled test set.

**A3.** As a developer, I want the agent to understand project conventions (style, structure, existing patterns), so that its changes fit naturally into the codebase.
- *Acceptance:* Agent-authored code passes the project's existing linter without manual fixes in the common case.

### Epic B: Requirement Understanding

**B1.** As a developer, I want to give the agent a plain-language description of a feature or bug, so that I don't have to write a formal spec.
- *Acceptance:* Free-text input is accepted; agent produces a task plan before making changes.

**B2.** As a developer, I want the agent to ask clarifying questions when a requirement is ambiguous, so that it doesn't guess wrong and waste a cycle.
- *Acceptance:* If confidence in scope is low, agent surfaces a question instead of proceeding blindly.

**B3.** As a tech lead, I want the agent to break a larger requirement into a task plan I can review before it starts editing, so that I can catch scope issues early.
- *Acceptance:* Agent outputs an ordered plan before any code is modified, and waits for approval if the task is flagged as large/risky.

### Epic C: Code Modification

**C1.** As a developer, I want the agent to generate a code diff (not full-file rewrites), so that changes are minimal and reviewable.
- *Acceptance:* Output is a unified diff/patch, not a full file replacement, except when creating new files.

**C2.** As a developer, I want the agent to write or update tests alongside code changes, so that the change is verifiable and future-proof.
- *Acceptance:* For any behavior change, at least one corresponding test is added or updated.

**C3.** As a developer, I want the agent to run the existing test suite after making changes, so that regressions are caught immediately.
- *Acceptance:* Test suite runs automatically post-edit; results are captured.

**C4.** As a developer, I want the agent to retry and self-correct when tests fail, so that I don't have to manually debug its first attempt.
- *Acceptance:* On failure, error output is fed back to the reasoning engine, which proposes a revised diff, up to a configurable retry limit.

### Epic D: Human Oversight & Trust

**D1.** As a tech lead, I want every agent change delivered as a pull request, so that a human always reviews before merge.
- *Acceptance:* No commits land on protected branches without PR + approval flow.

**D2.** As a reviewer, I want to see the agent's reasoning trace alongside the diff, so that I understand why it made each change.
- *Acceptance:* PR description includes a summary of the plan, key decisions, and test results.

**D3.** As a developer, I want to reject or request changes to an agent's PR conversationally, so that I can course-correct without starting over.
- *Acceptance:* Feedback on a PR is fed back into the agent's context for a revised attempt.

**D4.** As an engineering manager, I want visibility into how many tasks the agent attempted, resolved, or failed, so that I can gauge its reliability over time.
- *Acceptance:* A dashboard/log shows task outcomes, retry counts, and time-to-resolution.

### Epic E: Safety & Guardrails

**E1.** As a developer, I want the agent to run in a sandboxed environment, so that it can't affect my machine or production systems directly.
- *Acceptance:* All code execution happens in an isolated container/VM with no access to host secrets or unrelated systems.

**E2.** As a developer, I want the agent to ask for explicit confirmation before destructive operations, so that mistakes are never silent.
- *Acceptance:* Any operation on a configurable "dangerous ops" list requires an explicit user confirmation step.

**E3.** As a developer, I want a hard cap on retries per task, so that a stuck task doesn't burn unbounded time or API cost.
- *Acceptance:* Configurable max retry count; task is marked "needs human help" once exceeded, rather than looping forever.

### Epic F: Developer Experience

**F1.** As a developer, I want to submit a task via CLI, so that I can integrate this into my existing terminal workflow.
- *Acceptance:* `solvix run "<task description>"` triggers the full loop against the current repo.

**F2.** As a developer, I want to see live progress (what file it's looking at, what it's trying), so that I'm not staring at a black box.
- *Acceptance:* Streaming status updates during retrieval, planning, editing, and testing phases.

**F3.** As a developer, I want to configure which parts of the repo the agent is allowed to touch, so that sensitive files are never modified.
- *Acceptance:* A config file (`.solvix.yml`) supports include/exclude paths, enforced before any write operation.

---

## 10. MVP Scope vs. Later Phases

**Phase 1 — MVP: prove the core loop works on a single repo**
- Single local/GitHub repo, one task at a time
- Basic embedding-based retrieval (no fine-tuned indexer yet)
- LLM plans + edits + runs tests, retries up to N times on failure
- CLI interface, human approves before merge
- Supports one language ecosystem to start (e.g. Python or JS)

**Phase 2 — Trust & UX**
- Web/PR-based interface (GitHub App style)
- Reasoning trace shown to the user (why this change, what was tried)
- Multi-file, multi-step tasks
- Config for project conventions (style guide, forbidden files/paths)

**Phase 3 — Scale & Autonomy**
- Multi-repo support
- Background/async task queue (submit a ticket, get a PR later)
- Learning from accepted/rejected PRs to improve future suggestions
- Support for multiple languages/frameworks

**Explicitly out of scope for MVP:** autonomous merging without human approval, production deploys, cross-repo refactors, non-code tasks (infra changes, data migrations).

---

## 11. Week-by-Week MVP Plan

| Week | Deliverable |
|---|---|
| 1 | Repo indexer (chunking + embeddings + symbol index) working standalone |
| 2 | Context assembler + basic planner (plan output only, no execution yet) |
| 3 | Editor producing diffs; patch applier + sandbox execution wired up |
| 4 | Test runner integration + retry loop |
| 5 | Guardrails (denylist, path policy) + config loading |
| 6 | PR builder + task state/memory + CLI polish |
| 7 | Golden-task benchmark built; run end-to-end, fix failure modes |
| 8 | Buffer for hardening, docs, and a first real dogfood task on your own repo |

---

## 12. Risks & Open Questions

- **Reliability ceiling:** what counts as "good enough" success rate before this is trustworthy for real use? Needs the labeled golden-task benchmark to answer.
- **Cost control:** retrieval + reasoning + retries can get expensive on large tasks — needs cost caps and visibility.
- **Ambiguous requirements:** plain-language input is inherently underspecified; the clarifying-question flow (B2) needs careful UX so it doesn't become annoying.
- **Scope creep in "complex":** keep "solve any complex problem" explicitly bounded per phase — start with well-defined, testable coding tasks before broadening.
- **Security:** running arbitrary generated code, even sandboxed, needs a real threat model (dependency installation, network access during test runs, etc.).
- **Tool-use vs. rigid pipeline:** the strict plan → diff → apply pipeline is safer and easier to debug at MVP; freer tool-use (direct file/shell access, like Claude Code's own agent loop) is more flexible but harder to constrain — worth reconsidering once the MVP benchmark is stable.

---

*This is a living document — scope, priorities, and acceptance criteria should be revisited as the MVP is built and tested against real tasks. The single most important next step is instrumenting the golden-task benchmark (Section 7.6) — that number should drive every subsequent architecture decision more than intuition will.*