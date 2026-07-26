# Project instructions

## Auto-approved command visibility

The following commands/tools are allowlisted in `.claude/settings.json` and run without a permission prompt:

- `curl -sL *`, `curl -sI *`, `curl -s -o /dev/null -w *` (read-only fetches/status checks)
- `mcp__Claude_Browser__preview_logs`, `preview_list`, `read_page`, `read_console_messages`, `read_network_requests`, `get_page_text` (read-only preview/browser inspection)

Whenever one of these runs, flag it visibly in the response with `***` (e.g. `*** auto-approved: curl -sL ... ***`) before or alongside the tool call, so it's clear the command ran without a manual approval step. Do not silently run them without this marker.
