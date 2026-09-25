"""Soak test for the SEMAI Console against the mock rig (docs/SPEC.md §6 Step 7.1).

Starts scripts/mock_rig.py (127.0.0.1:8081) and the console (127.0.0.1:8000,
--rig http://127.0.0.1:8081), runs for 5 minutes, and kills + restarts the mock
at the 2-minute mark. Checks:

  1. the console recovers within 10 s of the mock coming back
  2. /api/state never errors (HTTP 200 + valid JSON on every poll)
  3. nulls show as "—": the JSON keeps null (never 0/NaN) and the page maps null to "—"
  4. only one /stream connection is ever open on the mock
  5. the MOCK banner flag is true on every poll

Writes results/soak_test.txt. Exit code 0 when every check passes, 1 otherwise.

Usage: python scripts/soak_test.py [--duration 300] [--restart-at 120] [--quick]
(--quick = 40 s run with the restart at 15 s, for a smoke test only.)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import requests  # noqa: E402

PY = sys.executable
MOCK_URL = "http://127.0.0.1:8081"
CONSOLE_URL = "http://127.0.0.1:8000"
LOG_DIR = os.path.join(ROOT, "logs")
OUT = os.path.join(ROOT, "results", "soak_test.txt")
FLOAT_KEYS = ("vpd", "t_air", "rh", "t_leaf")


def start(cmd, log_name):
    os.makedirs(LOG_DIR, exist_ok=True)
    log = open(os.path.join(LOG_DIR, log_name), "a", encoding="utf-8")
    log.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} start {' '.join(cmd)}\n")
    log.flush()
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    p._semai_log = log  # type: ignore[attr-defined]
    return p


def stop(p):
    if p is None:
        return
    try:
        if p.poll() is None:
            p.kill()
            p.wait(timeout=10)
    except Exception as e:  # noqa: BLE001
        print("stop:", e, file=sys.stderr)
    try:
        p._semai_log.close()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


def wait_http(url, timeout_s=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return time.time() - t0
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.25)
    return None


def get_json(url, timeout=3.0):
    r = requests.get(url, timeout=timeout)
    return r.status_code, r.json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--restart-at", type=float, default=120.0)
    ap.add_argument("--quick", action="store_true", help="40 s smoke run, restart at 15 s")
    args = ap.parse_args()
    if args.quick:
        args.duration, args.restart_at = 40.0, 15.0

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say(f"SEMAI soak test  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"duration {args.duration:.0f} s, mock restart at {args.restart_at:.0f} s")
    say(f"mock {MOCK_URL}   console {CONSOLE_URL}")
    say("")

    checks = {}
    mock = console = None
    state_polls = 0
    state_errors = []
    mock_flag_false = 0
    connected_true = 0
    null_seen = 0
    zero_suspect = 0
    max_streams_open = 0
    stream_samples = 0
    stats_errors = 0
    restart_t = None
    recovered_after = None
    stream_back_after = None
    statuses_seen = set()

    try:
        mock = start([PY, os.path.join(ROOT, "scripts", "mock_rig.py"), "--port", "8081"], "soak_mock.log")
        t = wait_http(MOCK_URL + "/api/data")
        say(f"mock up after {t if t is not None else 'TIMEOUT'} s")
        if t is None:
            say("FAIL: mock rig did not start")
            checks["mock_started"] = False
            raise SystemExit(1)
        checks["mock_started"] = True

        console = start([PY, "-m", "semai.app", "--rig", MOCK_URL, "--port", "8000"], "soak_console.log")
        t = wait_http(CONSOLE_URL + "/api/state")
        say(f"console up after {t if t is not None else 'TIMEOUT'} s")
        if t is None:
            say("FAIL: console did not start")
            checks["console_started"] = False
            raise SystemExit(1)
        checks["console_started"] = True

        # static check: page maps null -> em dash
        try:
            html = requests.get(CONSOLE_URL + "/", timeout=3).text
            js_path = os.path.join(ROOT, "semai", "static", "app.js")
            js = open(js_path, encoding="utf-8").read() if os.path.exists(js_path) else ""
            has_dash = ("—" in js) or ("—" in html) or ("\\u2014" in js)
            checks["page_has_em_dash_for_null"] = has_dash
            say(f"page/JS contains the em dash for nulls: {has_dash}")
            checks["page_has_MOCK_DATA_string"] = "MOCK DATA" in html or "MOCK DATA" in js
        except Exception as e:  # noqa: BLE001
            checks["page_has_em_dash_for_null"] = False
            say(f"page fetch failed: {e}")

        t_start = time.time()
        next_stats = 0.0
        restarted = False
        last_state_ok = True
        while True:
            now = time.time() - t_start
            if now >= args.duration:
                break

            # --- restart the mock at the mark ---
            if not restarted and now >= args.restart_at:
                say(f"[{now:6.1f} s] killing the mock rig")
                stop(mock)
                time.sleep(3.0)
                mock = start([PY, os.path.join(ROOT, "scripts", "mock_rig.py"), "--port", "8081"], "soak_mock.log")
                tm = wait_http(MOCK_URL + "/api/data")
                restart_t = time.time()
                say(f"[{time.time()-t_start:6.1f} s] mock rig restarted (up after {tm} s); waiting for the console to recover")
                restarted = True
                last_state_ok = False

            # --- poll /api/state every second ---
            try:
                code, st = get_json(CONSOLE_URL + "/api/state")
                state_polls += 1
                if code != 200:
                    state_errors.append(f"{now:.1f}s HTTP {code}")
                else:
                    if not st.get("mock", False):
                        mock_flag_false += 1
                    if st.get("connected"):
                        connected_true += 1
                        if restart_t is not None and recovered_after is None and (time.time() - restart_t) >= 0:
                            # first connected=true after the restart, based on a fresh reading
                            age = st.get("data_age_s")
                            if age is not None and age < 4.0:
                                recovered_after = time.time() - restart_t
                                say(f"[{now:6.1f} s] console recovered {recovered_after:.1f} s after the mock restart")
                    d = st.get("data") or {}
                    for k in FLOAT_KEYS:
                        v = d.get(k)
                        if v is None:
                            null_seen += 1
                        elif isinstance(v, (int, float)) and v == 0:
                            zero_suspect += 1
                    if isinstance(d.get("status"), str):
                        statuses_seen.add(d["status"])
                    if st.get("data") is not None and st.get("data", {}).get("status", "").startswith("MOCK"):
                        pass
            except Exception as e:  # noqa: BLE001
                state_polls += 1
                state_errors.append(f"{now:.1f}s {type(e).__name__}: {e}")

            # --- mock stats every 5 s ---
            if now >= next_stats:
                next_stats = now + 5.0
                try:
                    code, ms = get_json(MOCK_URL + "/api/mock/stats")
                    if code == 200:
                        stream_samples += 1
                        so = int(ms.get("stream_connections_open", 0))
                        max_streams_open = max(max_streams_open, so)
                        if restart_t is not None and stream_back_after is None and so >= 1:
                            stream_back_after = time.time() - restart_t
                            say(f"[{now:6.1f} s] /stream reconnected {stream_back_after:.1f} s after the mock restart (open={so})")
                        if so > 1:
                            say(f"[{now:6.1f} s] WARNING: {so} /stream connections open")
                    else:
                        stats_errors += 1
                except Exception:  # noqa: BLE001
                    if restarted and restart_t is None:
                        pass
                    stats_errors += 1

            time.sleep(max(0.0, 1.0 - ((time.time() - t_start) - now)))

        # final stats sample
        try:
            code, ms = get_json(MOCK_URL + "/api/mock/stats")
            say(f"final mock stats: {json.dumps(ms)}")
            max_streams_open = max(max_streams_open, int(ms.get("stream_connections_open", 0)))
        except Exception as e:  # noqa: BLE001
            say(f"final mock stats failed: {e}")

        # --- evaluate ---
        checks["console_recovers_within_10s"] = recovered_after is not None and recovered_after <= 10.0
        checks["api_state_never_errors"] = len(state_errors) == 0
        checks["nulls_kept_as_null_not_zero"] = zero_suspect == 0
        checks["only_one_stream_connection"] = max_streams_open <= 1 and stream_samples > 0
        checks["mock_banner_flag_always_true"] = mock_flag_false == 0 and state_polls > 0

        say("")
        say("--- numbers ---")
        say(f"/api/state polls: {state_polls}, errors: {len(state_errors)}")
        say(f"polls with connected=true: {connected_true}")
        say(f"null float fields seen in /api/state: {null_seen} (mock emits ~1 in 20 readings with a null)")
        say(f"float fields equal to exactly 0 (suspect null->0): {zero_suspect}")
        say(f"mock /api/mock/stats samples: {stream_samples} (errors {stats_errors}); max stream_connections_open: {max_streams_open}")
        say(f"statuses seen: {sorted(statuses_seen)}")
        say(f"recovered after restart: {recovered_after if recovered_after is None else round(recovered_after, 1)} s (limit 10 s)")
        say(f"/stream back after restart: {stream_back_after if stream_back_after is None else round(stream_back_after, 1)} s")
        if state_errors:
            say("first errors: " + "; ".join(state_errors[:5]))
        say("")
        say("--- checks ---")
        for k, v in checks.items():
            say(f"{'PASS' if v else 'FAIL'}  {k}")
        ok = all(checks.values())
        say("")
        say("OVERALL: " + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    finally:
        stop(console)
        stop(mock)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
