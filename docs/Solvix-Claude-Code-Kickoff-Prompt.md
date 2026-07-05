# Solvix — Claude Code Kickoff Prompt (Sprint 1)

## Before you paste anything

1. Create a new local folder / repo for the project (e.g. `solvix/`).
2. Inside it, make a `docs/` folder and drop these two files in:
   - `Solvix-Master-Document.md`
   - `Solvix-Full-Story-Backlog.xlsx`
3. Open the folder in VS Code with Claude Code.
4. Paste the prompt below as your first message in a fresh Claude Code session.

Working this way (docs committed to the repo, not just pasted into chat) means Claude Code can re-read them anytime without you repeating yourself every session.

---

## First prompt (copy-paste this)

```
I'm starting a new project called Solvix — an AI coding agent that indexes a
repository, understands a plain-language requirement, proposes code changes,
verifies them with tests, and opens a PR for human review.

Read docs/Solvix-Master-Document.md first — it has the full vision, architecture,
tech stack, repo structure, and non-functional requirements. Follow it as the
spec for everything below, don't deviate from the tech stack or repo layout
without telling me why first.

For this session, I only want you to implement ONE story from the backlog
(docs/Solvix-Full-Story-Backlog.xlsx), Sprint 1:

Story ID: SLX-A1
Title: Index the repository
User story: As a developer, I want the agent to index my repository, so that
it can retrieve relevant code without needing the entire codebase in context.
Acceptance criteria: Given a repo path, the system builds a searchable index
(file + symbol + embedding based) within a reasonable time for repos up to a
few thousand files.

Scope for this session:
1. Set up the repo structure exactly as described in the Master Document
   section 7.2 (just the indexer/ module and its dependencies for now — don't
   scaffold every folder yet, we'll add the rest sprint by sprint).
2. Implement indexer/chunker.py — split source files into function/class-level
   chunks using tree-sitter.
3. Implement indexer/symbol_index.py — a lightweight symbol map (symbol name ->
   file:line).
4. Implement indexer/embedder.py and indexer/store.py — embedding generation
   and a local vector store (Chroma or FAISS, your call, tell me which and why).
5. Wire these into a single index_repo(repo_path) entry point that can be run
   and tested against a small sample repo.
6. Write unit tests covering chunking correctness and symbol lookup accuracy.

Before writing code:
- Confirm your understanding of the acceptance criteria back to me in 2-3
  sentences.
- Tell me which embedding approach you'll use (API-based vs local
  sentence-transformers) and why, given this is meant to run for a solo
  developer without requiring paid API keys by default.

After implementing:
- Run the tests and show me they pass.
- Tell me explicitly whether SLX-A1's acceptance criteria is met, and what (if
  anything) is still incomplete.
- Don't move on to Sprint 2 stories yet — stop and wait for me to review.
```

---

## Reusable template for every future sprint/story

Once Sprint 1 is reviewed and merged, use this shorter template for each new story — swap in the row from the backlog:

```
Continue the Solvix project (see docs/Solvix-Master-Document.md for the full
spec — don't deviate from established architecture/tech stack without asking).

Implement this story next:

Story ID: <paste ID, e.g. SLX-A2>
Title: <paste title>
User story: <paste user story>
Acceptance criteria: <paste acceptance criteria>

Before coding, confirm your understanding of the acceptance criteria back to
me in 2-3 sentences and flag anything ambiguous.

After implementing, run relevant tests, tell me explicitly whether the
acceptance criteria is met, and stop for my review before starting the next
story.
```

---

## Why this structure works

- **One story per session** keeps Claude Code's output reviewable — matches the "plan → diff → apply, human reviews" philosophy from your own Master Document (Epic D).
- **"Confirm before coding"** catches misunderstandings before any code is written, cheaper to fix at that stage than after.
- **"Stop after each story"** prevents silent scope creep across multiple stories in one uncontrolled pass — the same guardrail your own PRD calls for in the actual Solvix product (Epic E, retry/approval caps).
- Referencing the doc by **file path**, not pasted text, keeps every session consistent without you re-explaining the architecture each time.

---

## Quick-Fix Template (small bugs — NOT new stories)

The templates above are correctly heavyweight for a real feature. Using that same weight on a small, already-diagnosed bug wastes a lot of tokens for no benefit. Use this instead for anything that's a targeted fix, not new functionality:

```
Fix: <one-line description of the bug and root cause, if known>
Where: <file/function if you already know it, else "find the cause">

Do:
- Fix it
- Add exactly one test for it
- Run only the relevant test file, not the full suite — unless the fix
  touches a shared/core module (config.py, orchestrator.py), in which case
  run the full suite since those are used everywhere
- If this is a shared/core file, check other call sites for the same
  pattern (recall SLX-F3's additive-config fix — one bug like this is
  often not the only occurrence)

Report back with just:
- Pass/fail (don't paste full pytest output unless something failed)
- One line: acceptance met, yes/no, and why
```

**Before using this template, start a brand-new Claude Code session** (not mid-conversation from a long feature build) — the code and docs are real files on disk, so a fresh session loses nothing and avoids dragging along an entire feature's worth of prior context for an unrelated one-line fix.

**When to use which template:**

| Situation | Template |
|---|---|
| A new backlog story (anything in the spreadsheet) | Full story template (above) |
| A bug found via smoke-testing / real usage, root cause already known | Quick-Fix template |
| Something vague ("it's acting weird") | Investigate first (see below), don't jump straight to a fix template |

## Investigate-First Template (when you don't know the root cause yet)

Use this before asking for a fix when the cause isn't already clear — forcing a fix prompt too early leads to guessing, patch-over-symptom fixes, and wasted tokens redoing it properly later.

```
I saw this happen: <describe symptom/error, paste the actual error text>

Investigate the actual root cause first — don't propose or write a fix yet.
Tell me what you find, then stop and wait for me to decide next steps.
```

---

*Tip: as you complete stories, update the Status column in the backlog spreadsheet (To Do → In Progress → Done) so you always know where Sprint 1 stands before starting the next session.*
