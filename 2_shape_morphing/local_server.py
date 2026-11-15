"""Request checks for the local GUI servers (morph_gui.py, dim_gui.py).

The servers bind to 127.0.0.1, but any web page open in the same browser can still send requests
to http://localhost:<port>. Two attacks are blocked:
  * cross-site requests (CSRF): a "simple" POST (form / text/plain) from another site would run
    without a CORS preflight. POSTs must be Content-Type: application/json, which forces a
    preflight that these servers never approve, and an Origin header must be this server's.
  * DNS rebinding: a foreign host name resolved to 127.0.0.1 would let a page read the API.
    The Host header must be localhost / 127.0.0.1 with this server's port.
"""


def request_problem(handler):
    """None if the request may be served, else the reason to reject it (-> HTTP 403)."""
    port = handler.server.server_address[1]
    hosts = {f"localhost:{port}", f"127.0.0.1:{port}"}
    if handler.headers.get("Host", "") not in hosts:
        return "unexpected Host header (open the GUI as http://localhost:%d/)" % port
    origin = handler.headers.get("Origin")
    if origin is not None and origin not in {"http://" + h for h in hosts}:
        return "cross-origin request refused"
    if handler.command == "POST":
        ctype = handler.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return "POST requests must be Content-Type: application/json"
    return None
