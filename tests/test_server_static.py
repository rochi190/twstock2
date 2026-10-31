"""Static file serving (MIME, caching, 304, traversal / dotfile / symlink refusal) and robustness against malformed
requests, over a real in-process server."""
import email.utils
import os
import socket
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from server_fakes import ServerCase  # noqa: E402

import schema_check as sc  # noqa: E402


class StaticFiles(ServerCase):
    def test_root_serves_index_html(self):
        st, hdr, raw = self.get("/")
        self.assertEqual(st, 200)
        self.assertEqual(hdr["content-type"], "text/html; charset=utf-8")
        self.assertIn("首頁".encode("utf-8"), raw)
        self.assertEqual(self.get("/index.html")[2], raw)

    def test_mime_types(self):
        expect = {"/app.js": "application/javascript; charset=utf-8", "/style.css": "text/css; charset=utf-8",
                  "/data.json": "application/json; charset=utf-8", "/logo.svg": "image/svg+xml", "/pic.png": "image/png",
                  "/notes.txt": "text/plain; charset=utf-8", "/font.woff2": "font/woff2",
                  "/sub/page.html": "text/html; charset=utf-8"}
        for path, ctype in expect.items():
            st, hdr, raw = self.get(path)
            self.assertEqual(st, 200, path)
            self.assertEqual(hdr["content-type"], ctype, path)
            self.assertEqual(hdr["x-content-type-options"], "nosniff")

    def test_cache_control(self):
        for path in ("/", "/app.js", "/style.css"):
            self.assertEqual(self.get(path)[1]["cache-control"], "no-cache", path)
        self.assertIn("max-age", self.get("/pic.png")[1]["cache-control"])  # images may be cached

    def test_content_length_and_body_match(self):
        st, hdr, raw = self.get("/notes.txt")
        self.assertEqual(int(hdr["content-length"]), len(raw))
        self.assertEqual(raw.decode("utf-8"), "純文字")

    def test_etag_and_304(self):
        st, hdr, raw = self.get("/app.js")
        etag = hdr["etag"]
        self.assertTrue(etag)
        st2, hdr2, raw2 = self.get("/app.js", {"If-None-Match": etag})
        self.assertEqual((st2, raw2), (304, b""))
        self.assertEqual(hdr2["etag"], etag)
        self.assertEqual(self.get("/app.js", {"If-None-Match": '"nope", ' + etag})[0], 304)  # list of tags
        self.assertEqual(self.get("/app.js", {"If-None-Match": "*"})[0], 304)
        self.assertEqual(self.get("/app.js", {"If-None-Match": '"stale"'})[0], 200)
        # the file changes -> a new ETag and a full 200 again
        p = os.path.join(self.fe, "app.js")
        time.sleep(0.02)
        with open(p, "w") as fh:
            fh.write("console.log('changed, longer');")
        st3, hdr3, raw3 = self.get("/app.js", {"If-None-Match": etag})
        self.assertEqual(st3, 200)
        self.assertNotEqual(hdr3["etag"], etag)
        self.assertIn(b"changed", raw3)

    def test_if_modified_since(self):
        st, hdr, raw = self.get("/app.js")
        lm = hdr["last-modified"]
        self.assertEqual(self.get("/app.js", {"If-Modified-Since": lm})[0], 304)
        old = email.utils.formatdate(0, usegmt=True)
        self.assertEqual(self.get("/app.js", {"If-Modified-Since": old})[0], 200)
        self.assertEqual(self.get("/app.js", {"If-Modified-Since": "garbage"})[0], 200)
        # If-None-Match wins over If-Modified-Since
        self.assertEqual(self.get("/app.js", {"If-None-Match": '"x"', "If-Modified-Since": lm})[0], 200)

    def test_head_has_headers_but_no_body(self):
        st, hdr, raw = self.request("/app.js", "HEAD")
        self.assertEqual((st, raw), (200, b""))
        self.assertEqual(hdr["content-type"], "application/javascript; charset=utf-8")
        self.assertEqual(int(hdr["content-length"]), len(b"console.log('hi');"))

    def test_missing_file_and_directories(self):
        for path in ("/nope.js", "/sub/missing.html", "/emptydir", "/emptydir/", "/sub"):
            st, hdr, raw = self.get(path)
            self.assertEqual(st, 404, path)
            self.assertIn("text/html", hdr["content-type"])

    def test_path_traversal_is_refused(self):
        secret = b"outside the frontend directory"
        attempts = ["/../outside.txt", "/%2e%2e/outside.txt", "/%2E%2E/outside.txt", "/..%2foutside.txt", "/sub/../../outside.txt",
                    "/sub/%2e%2e/%2e%2e/outside.txt", "/....//outside.txt", "/..\\outside.txt", "/%5c..%5coutside.txt",
                    "//outside.txt", "/%2e%2e%2foutside.txt", "/../../../../../../etc/passwd", "/..%00/outside.txt",
                    "/%c0%ae%c0%ae/outside.txt", "/../backend/config.py"]
        for path in attempts:
            st, hdr, raw = self.get(path)
            self.assertNotEqual(st, 200, path)
            self.assertNotIn(secret, raw, path)
            self.assertNotIn(b"root:", raw, path)
            self.assertNotIn(b"DEFAULT_PORT", raw, path)
        # raw (un-normalised) request targets, as an attacker's client would send them
        for target in (b"/../outside.txt", b"/sub/../../outside.txt", b"/%2e%2e/outside.txt"):
            raw = self.raw_exchange(b"GET " + target + b" HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            self.assertNotIn(secret, raw, target)

    def test_dotfiles_are_refused(self):
        for path in ("/.secret", "/sub/.hidden", "/%2esecret", "/.git/config", "/sub/.", "/.%2e/outside.txt"):
            st, hdr, raw = self.get(path)
            self.assertIn(st, (403, 404), path)
            self.assertNotIn(b"top secret", raw, path)
            self.assertNotIn(b"hidden", raw if st == 200 else b"", path)
            self.assertNotEqual(st, 200, path)

    def test_symlink_pointing_outside_is_refused(self):
        if not os.path.islink(os.path.join(self.fe, "link.txt")):
            self.skipTest("no symlink support")
        st, hdr, raw = self.get("/link.txt")
        self.assertEqual(st, 404)
        self.assertNotIn(b"outside the frontend", raw)

    def test_nul_byte_and_weird_paths(self):
        for path in ("/app.js%00.png", "/%00", "/%20", "/%ff%fe", "/" + "a" * 3000):
            st, hdr, raw = self.get(path)
            self.assertIn(st, (400, 404, 414), path)
        self.assertEqual(self.get("/app.js?cachebust=123")[0], 200)  # query strings are ignored
        self.assertEqual(self.get("/app.js#frag")[0], 200)

    def test_api_json_is_never_confused_with_static(self):
        st, hdr, raw = self.get("/data.json")
        self.assertEqual(raw, b'{"a":1}')
        st, d, _ = self.json("/api/status")
        self.assertEqual(d["status"], "ready")


class MalformedRequests(ServerCase):
    def alive(self):
        st, d, _ = self.json("/api/status")
        self.assertEqual((st, d["status"]), (200, "ready"))

    def test_garbage_request_lines(self):
        for junk in (b"\x00\x01\x02\xff\xfe\r\n\r\n", b"GET\r\n\r\n", b"GET /\r\n\r\n", b"HELLO WORLD\r\n\r\n",
                     b"GET / HTTP/9.9\r\n\r\n", b"\r\n\r\n", b"GET /api/status HTTP/1.1\r\nBad Header\r\n\r\n",
                     b"GET /api/status HTTP/1.1\r\n" + b"X: " + b"a" * 70000 + b"\r\n\r\n", b"A" * 100000 + b"\r\n\r\n"):
            self.raw_exchange(junk, read_timeout=2.0)
            self.alive()

    def test_too_long_url(self):
        raw = self.raw_exchange(b"GET /api/search?q=" + b"a" * 70000 + b" HTTP/1.1\r\nHost: x\r\n\r\n", read_timeout=2.0)
        self.assertTrue(raw.startswith(b"HTTP/1.") and b" 414 " in raw.split(b"\r\n", 1)[0], raw[:80])
        self.alive()

    def test_invalid_percent_encoding_and_odd_queries(self):
        for q in ("/api/search?q=%", "/api/search?q=%zz", "/api/search?q=%E5%8F", "/api/search?q=%ff%fe%fd",
                  "/api/search?q=a&q=b&q=c", "/api/search?&&&==&", "/api/search?q[]=1",
                  "/api/history?symbol=%00", "/api/forecast?symbol=%E5%8F%B0%E7%A9%8D%E9%9B%BB%00",
                  "/api/search?" + "&".join(f"k{i}=1" for i in range(200))):
            st, hdr, raw = self.get(q)
            self.assertIn(st, (200, 400, 404), q)
            sc.strict_loads(raw)  # always valid JSON
        self.alive()

    def test_host_header_guard(self):
        st, hdr, raw = self.get("/api/status", {"Host": "evil.example.com"})
        self.assertEqual(st, 403)
        self.assertEqual(sc.check_error(sc.strict_loads(raw)), [])
        for host in ("localhost", "localhost:8000", "127.0.0.1:1234", "[::1]:80", "LOCALHOST"):
            self.assertEqual(self.get("/api/status", {"Host": host})[0], 200, host)
        self.assertEqual(self.get("/", {"Host": "attacker.test"})[0], 403)  # also for static files
        self.alive()

    def test_http10_without_host(self):
        raw = self.raw_exchange(b"GET /api/status HTTP/1.0\r\n\r\n")
        self.assertTrue(raw.startswith(b"HTTP/1.") and b" 200 " in raw.split(b"\r\n", 1)[0])

    def test_post_with_body_does_not_poison_the_connection(self):
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request("POST", "/api/status", body=b"GET /api/status HTTP/1.1\r\n\r\n" * 3)
            r = conn.getresponse()
            self.assertEqual(r.status, 405)
            r.read()
            self.assertEqual(r.getheader("Connection"), "close")
        finally:
            conn.close()
        self.alive()

    def test_client_disconnects_mid_request(self):
        for _ in range(20):
            s = socket.create_connection(("127.0.0.1", self.port))
            s.sendall(b"GET /api/forecast?symbol=2330 HTTP/1.1\r\nHost: x")
            s.close()
            s = socket.create_connection(("127.0.0.1", self.port))
            s.sendall(b"GET /api/forecast?symbol=2330 HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            s.close()  # hang up before the answer is written
        time.sleep(0.3)
        self.alive()

    def test_many_parallel_connections(self):
        import threading
        res = []

        def go():
            res.append(self.get("/api/status")[0])

        ts = [threading.Thread(target=go) for _ in range(60)]
        [t.start() for t in ts]
        [t.join(30) for t in ts]
        self.assertEqual(res, [200] * 60)

    def test_error_bodies_are_always_the_documented_shape(self):
        for path in ("/api/nope", "/api/history", "/api/forecast?symbol=!!", "/api/backtest?h=9", "/api/recommendations?h=9",
                     "/api/backtest/universe?top=0", "/api/search?limit=-1"):
            st, hdr, raw = self.get(path)
            d = sc.strict_loads(raw)
            self.assertGreaterEqual(st, 400, path)
            self.assertEqual(sc.check_error(d), [], path)
            self.assertEqual(hdr["content-type"], "application/json; charset=utf-8")


if __name__ == "__main__":
    unittest.main()
