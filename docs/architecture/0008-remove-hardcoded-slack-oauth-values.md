# 0008: Remove hardcoded Slack OAuth secret

**Status:** Resolved (commit `15d03e1`)

## Context

`scripts/runtime/get_slack_token.py` was inherited via this repo's fork
from GH05T3 with a real Slack app `CLIENT_ID`/`CLIENT_SECRET` pair
hardcoded in plaintext, source, tracked, and publicly visible on GitHub.
The same pair was live in this exact form in three places total: here,
and on both of GH05T3's own branches (`main` and
`claude/fix-multi-gpu-training-2WHKH`).

## Decision

Both values now read from `SLACK_CLIENT_ID`/`SLACK_CLIENT_SECRET` env
vars, with a clear `[ERROR]` message and `sys.exit(1)` instead of silently
running the OAuth flow with empty credentials. Fixed identically across
all three locations (this repo's commit `15d03e1`; GH05T3's `8fbe9b1` on
`main` and `d4be9d9` on `claude/fix-multi-gpu-training-2WHKH`) — see
GH05T3's own
[`docs/architecture/security-and-repo-hygiene-2026-07.md`](https://github.com/leerobber/GH05T3/blob/main/docs/architecture/security-and-repo-hygiene-2026-07.md)
for the full writeup, since the secret and the fix originated there.

## Consequences

Removing it from the current file does not undo its prior exposure in git
history in any of the three locations — the secret is still visible in
commit history on GitHub. The Slack app's client secret must be rotated at
api.slack.com regardless of this fix; that's an external action, not a
commit.
