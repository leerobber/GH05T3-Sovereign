# -*- coding: utf-8 -*-
"""
GH05T3 Frontend Server
Serves the React build with correct cache headers so the browser never
serves a stale index.html after a rebuild.
"""
import http.server
import os
import sys

PORT = 3210
# Resolve from repo root (this script lives in scripts/runtime/)
FRONTEND_BUILD = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend", "build")

class GH05T3Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        path = self.path.split("?")[0]
        # HTML entry points + SW must never be cached
        if (path in ("/", "/index.html", "/sw.js", "/manifest.json")
                or not path.split("/")[-1].count(".")):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
        elif "/static/" in path:
            # Hashed static assets can be cached forever
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        super().end_headers()

    def log_message(self, fmt, *args):
        # Quiet the per-request noise; keep errors
        if args and str(args[1]) not in ("200", "304"):
            super().log_message(fmt, *args)

if __name__ == "__main__":
    os.chdir(FRONTEND_BUILD)
    handler = GH05T3Handler
    with http.server.HTTPServer(("", PORT), handler) as httpd:
        print(f"[GH05T3] Frontend server running at http://localhost:{PORT}")
        sys.stdout.flush()
        httpd.serve_forever()
