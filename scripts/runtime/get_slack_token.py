"""
get_slack_token.py — one-shot OAuth flow to get a properly-scoped Slack bot token.

Run: python get_slack_token.py
Opens browser -> you click Allow -> token is saved to .env automatically.
"""
import http.server, urllib.parse, webbrowser, threading, json, sys
from pathlib import Path
import requests

CLIENT_ID     = "11126553666658.11119418273191"
CLIENT_SECRET = "09a430aff14a11cc1e284a26f9d67fc9"
REDIRECT_URI  = "http://localhost:3000/callback"
PORT          = 3000

BOT_SCOPES = ",".join([
    "channels:history", "channels:read", "channels:manage", "channels:join",
    "chat:write", "chat:write.public",
    "groups:history", "groups:read", "groups:write",
    "im:history", "im:read", "im:write",
    "mpim:history", "mpim:read", "mpim:write",
    "users:read", "users:read.email",
    "reactions:read", "files:read",
    "team:read",
])

AUTH_URL = (
    f"https://slack.com/oauth/v2/authorize"
    f"?client_id={CLIENT_ID}"
    f"&scope={BOT_SCOPES}"
    f"&redirect_uri={urllib.parse.quote(REDIRECT_URI, safe='')}"
)

token_result = {}

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_): pass  # silence request logs

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if "error" in params:
            err = params["error"][0]
            self._respond(f"<h2>Error: {err}</h2><p>Close this tab.</p>")
            token_result["error"] = err
            threading.Thread(target=self.server.shutdown).start()
            return

        code = params.get("code", [None])[0]
        if not code:
            self._respond("<h2>No code received.</h2>")
            return

        # Exchange code for token
        print("\n  [OAuth] Got code, exchanging for token...")
        r = requests.post("https://slack.com/api/oauth.v2.access", data={
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code":          code,
            "redirect_uri":  REDIRECT_URI,
        })
        data = r.json()

        if not data.get("ok"):
            err = data.get("error", "unknown")
            self._respond(f"<h2>Token exchange failed: {err}</h2>")
            token_result["error"] = err
            threading.Thread(target=self.server.shutdown).start()
            return

        bot_token = data.get("access_token", "")
        team      = data.get("team", {}).get("name", "")
        team_id   = data.get("team", {}).get("id", "")
        bot_user  = data.get("bot_user_id", "")

        token_result.update({
            "token":   bot_token,
            "team":    team,
            "team_id": team_id,
            "bot_user": bot_user,
        })

        self._respond(
            f"<h2>Success!</h2>"
            f"<p>Workspace: <b>{team}</b></p>"
            f"<p>Bot token saved to <code>.env</code></p>"
            f"<p>Close this tab and check the terminal.</p>"
        )
        threading.Thread(target=self.server.shutdown).start()

    def _respond(self, html):
        body = f"<html><body style='font-family:sans-serif;padding:40px'>{html}</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body.encode())


def update_env(token: str, team_id: str):
    env_path = Path(__file__).parent / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    updated = {}
    new_lines = []
    for line in lines:
        if line.startswith("SLACK_BOT_TOKEN="):
            new_lines.append(f"SLACK_BOT_TOKEN={token}")
            updated["SLACK_BOT_TOKEN"] = True
        elif line.startswith("SLACK_TEAM_ID="):
            new_lines.append(f"SLACK_TEAM_ID={team_id}")
            updated["SLACK_TEAM_ID"] = True
        else:
            new_lines.append(line)
    if "SLACK_BOT_TOKEN" not in updated:
        new_lines.insert(0, f"SLACK_BOT_TOKEN={token}")
    if "SLACK_TEAM_ID" not in updated:
        new_lines.insert(1, f"SLACK_TEAM_ID={team_id}")
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    # Add redirect URI to app first
    print("+---------------------------------------------------+")
    print("|  GH05T3 Slack OAuth Token Generator               |")
    print("+---------------------------------------------------+")
    print()
    print("STEP 1: Add redirect URI to your Slack app")
    print(f"  Go to: https://api.slack.com/apps/A0B3HCA815M/oauth")
    print(f"  Under 'Redirect URLs' click 'Add New Redirect URL'")
    print(f"  Paste: {REDIRECT_URI}")
    print(f"  Click 'Save URLs'")
    print()
    input("  Press ENTER when done to open the browser... ")

    print("\n  [OAuth] Starting local server on port 3000...")
    server = http.server.HTTPServer(("localhost", PORT), Handler)

    print(f"  [OAuth] Opening browser for authorization...")
    webbrowser.open(AUTH_URL)

    print(f"  [OAuth] Waiting for callback (authorize in browser)...")
    server.serve_forever()

    if "error" in token_result:
        print(f"\n  [ERROR] {token_result['error']}")
        sys.exit(1)

    token   = token_result["token"]
    team    = token_result["team"]
    team_id = token_result["team_id"]

    print(f"\n  [OK] Token received for workspace: {team}")
    print(f"  Token: {token[:24]}...")

    update_env(token, team_id)
    print(f"  [OK] Saved to .env")

    print()
    print("  Testing token...")
    r = requests.get("https://slack.com/api/conversations.list?limit=3",
                     headers={"Authorization": f"Bearer {token}"})
    data = r.json()
    if data.get("ok"):
        channels = [c["name"] for c in data.get("channels", [])]
        print(f"  [OK] Can see channels: {channels}")
    else:
        print(f"  [WARN] {data.get('error')} - may need to join channels first")

    print()
    print("  Setting up SovereignNation workspace channels...")
    import os
    os.environ["SLACK_BOT_TOKEN"] = token
    sys.path.insert(0, str(Path(__file__).parent))
    import slack_notify
    results = slack_notify.setup_workspace()
    print(f"\n  Done. {len(results)} channels ready.")
    print()
    print("  Posting test message to #general...")
    slack_notify.post("general", ":wave: GH05T3 bot is online. SovereignNation workspace configured.")
    print("  Check Slack — message should be in #general now.")
