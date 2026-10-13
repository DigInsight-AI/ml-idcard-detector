<!-- agent-context managed block: begin -->
## Agent Context

- Use English for all project-facing content, including code comments, documentation, commit messages, pull request text, issue text, and agent notes.
- This project uses the agent-context tool protocol v1. Use `agent-context tool capabilities.get` to discover available operations and their input, output, budget, and access rules.
- At the start of work that depends on project conventions, prior decisions, knowledge, or unfinished work, call `agent-context tool context.get` with the current task and caller session identity.
- Before ending meaningful development work, call `agent-context tool session.checkpoint` with the goal, progress, decisions, changed files, validations, blockers, and next steps.
- Treat `.agent-context-project.json` as stable project identity. Do not edit or remove it unless the user explicitly requests that change.
- agent-context never edits project files. Agents may only write shared state through operations marked `controlled-write`; memory lifecycle, knowledge import, and configuration changes require the user-facing CLI.
- Consume stdout as protocol JSON. If the tool is unavailable or returns an error, continue without inventing remembered context.
<!-- agent-context managed block: end -->
