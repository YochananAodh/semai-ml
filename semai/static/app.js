/* SEMAI Console page logic. Vanilla JS, no libraries, nothing loaded from the internet.
 * Cadence (SPEC section 6 step 6.5): /api/state once per second and /frame.jpg once per second. */
(function () {
  "use strict";
  var DASH = "—"; // em dash for null readings, never 0 or NaN
  var STR = {
    mock: "MOCK DATA",
    replay: "Replay of 2025 (model never saw this year) · outdoor Penang air",
    water: "uncalibrated",
    today: "today's Penang outlook (weather forecast)",
    wiltNA: "wilt model unavailable",
    fcNA: "forecast model unavailable",
    todayNA: "not fetched - run scripts/fetch_today.py while online"
  };
  var $ = function (id) { return document.getElementById(id); };

  // ------------------------------------------------------------ live state
  function fmt(v, dp) {
    if (v === null || v === undefined || typeof v !== "number" || !isFinite(v)) return DASH;
    return dp === 0 ? String(Math.round(v)) : v.toFixed(dp);
  }
  function chipClass(status) {
    if (!status) return "grey";
    var s = String(status);
    if (s.indexOf("MOCK_") === 0) s = s.slice(5);
    if (s === "OPTIMAL") return "green";
    if (s === "HIGH_VPD_MISTING") return "orange";
    if (s === "LOW_VPD_PURGE") return "blue";
    if (s.indexOf("ERR_") === 0) return "red";
    if (s === "WAITING_FOR_PICO" || s === "PICO_OFFLINE") return "grey";
    return "grey";
  }
  function setVerdict(w) {
    var el = $("verdict");
    el.className = "verdict";
    if (!w || !w.available) { el.textContent = STR.wiltNA; el.classList.add("na"); return; }
    if (!w.label) { el.textContent = "wilt: waiting for a camera frame"; el.classList.add("na"); return; }
    var p = (typeof w.p_wilted === "number") ? w.p_wilted.toFixed(2) : DASH;
    var ms = (typeof w.ms === "number") ? " · " + Math.round(w.ms) + " ms" : "";
    el.textContent = w.label + "  ·  p_wilted = " + p + ms;
    el.classList.add(String(w.label).toLowerCase());
  }
  var lastForecastAvail = null;
  function applyState(st) {
    $("dot").className = "dot " + (st.connected ? "on" : "off");
    $("rig-url").textContent = st.rig_url || "";
    $("updated").textContent = (st.data_age_s === null || st.data_age_s === undefined)
      ? "updated " + DASH + " s ago" : "updated " + Math.round(st.data_age_s) + " s ago";
    $("mock-banner").classList.toggle("hidden", !st.mock);
    $("server-time").textContent = st.server_time ? String(st.server_time).replace("T", " ").slice(0, 19) : "";
    var d = st.data || {};
    $("vpd").textContent = fmt(d.vpd, 2);
    $("t_air").textContent = fmt(d.t_air, 1);
    $("rh").textContent = fmt(d.rh, 0);
    $("t_leaf").textContent = fmt(d.t_leaf, 1);
    $("light").textContent = fmt(d.light, 0);
    $("water").textContent = fmt(d.water, 0);
    var chip = $("status");
    var status = (st.data && st.data.status) ? st.data.status : null;
    chip.textContent = status || DASH;
    chip.className = "chip " + chipClass(status);
    setVerdict(st.wilt);
    if (st.forecast_available !== lastForecastAvail) {
      lastForecastAvail = st.forecast_available;
      $("forecast-msg").classList.toggle("hidden", !!st.forecast_available);
      if (st.forecast_available) loadReplay();
      else drawChart();
    }
  }
  var stateInFlight = false;
  function pollState() {
    if (stateInFlight) return; // never faster than 1 Hz
    stateInFlight = true;
    fetch("/api/state", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(applyState)
      .catch(function () { $("dot").className = "dot off"; })
      .then(function () { stateInFlight = false; });
  }
  function refreshFrame() {
    $("frame").src = "/frame.jpg?ts=" + Date.now();
  }
  function tick() { pollState(); refreshFrame(); }
  setInterval(tick, 1000); // exactly once per second for both
  tick();

  // ------------------------------------------------------------ capture buttons
  function capture(label) {
    var msg = $("cap-msg");
    msg.textContent = "saving…";
    fetch("/api/capture?label=" + encodeURIComponent(label), { method: "POST" })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
      .then(function (res) {
        if (res.ok && res.j.ok) msg.textContent = "saved " + res.j.path + " · " + res.j.count + " " + label + " frame(s)";
        else msg.textContent = "capture failed: " + (res.j.error || "unknown");
      })
      .catch(function (e) { msg.textContent = "capture failed: " + e; });
  }
  $("cap-healthy").addEventListener("click", function () { capture("healthy"); });
  $("cap-wilted").addEventListener("click", function () { capture("wilted"); });

  // ------------------------------------------------------------ forecast replay
  var replay = { date: null, hours: null, idx: 0, playing: false, timer: null, available: false, error: null };
  var HOUR_MS = 500; // ~0.5 s per hour

  function loadReplay() {
    var date = $("replay-date").value || "2025-03-15";
    stopReplay();
    replay.date = date; replay.hours = null; replay.idx = 0; replay.error = null;
    $("hour-readout").textContent = "loading " + date + "…";
    fetch("/api/replay?date=" + encodeURIComponent(date), { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (j) {
        if (j.error) {
          // e.g. a typed year outside 2025: the model is fine, the date is not
          $("forecast-msg").textContent = j.error;
          $("forecast-msg").classList.remove("hidden");
          $("hour-readout").textContent = "hour " + DASH;
          drawChart();
          return;
        }
        if (!j.available) {
          replay.available = false;
          $("forecast-msg").textContent = STR.fcNA + (j.reason && j.reason !== STR.fcNA ? " (" + j.reason + ")" : "");
          $("forecast-msg").classList.remove("hidden");
          $("hour-readout").textContent = "hour " + DASH;
          drawChart();
          return;
        }
        replay.available = true;
        replay.hours = j.hours || [];
        replay.idx = 0;
        $("forecast-msg").classList.add("hidden");
        drawChart();
        updateReadout();
        startReplay();
      })
      .catch(function (e) {
        replay.available = false; replay.error = String(e);
        $("forecast-msg").textContent = STR.fcNA;
        $("forecast-msg").classList.remove("hidden");
        drawChart();
      });
  }
  function startReplay() {
    if (!replay.hours || !replay.hours.length) return;
    replay.playing = true;
    $("play").textContent = "Pause";
    if (replay.timer) clearInterval(replay.timer);
    replay.timer = setInterval(function () {
      replay.idx = (replay.idx + 1) % replay.hours.length;
      drawChart(); updateReadout();
    }, HOUR_MS);
  }
  function stopReplay() {
    replay.playing = false;
    $("play").textContent = "Play";
    if (replay.timer) { clearInterval(replay.timer); replay.timer = null; }
  }
  $("play").addEventListener("click", function () {
    if (replay.playing) stopReplay();
    else if (replay.hours && replay.hours.length) startReplay();
    else loadReplay();
  });
  $("replay-date").addEventListener("change", loadReplay);

  function updateReadout() {
    var h = replay.hours && replay.hours[replay.idx];
    if (!h) { $("hour-readout").textContent = "hour " + DASH; return; }
    var hh = (h.hour < 10 ? "0" : "") + h.hour + ":00";
    var txt = replay.date + " " + hh + " · VPD " + fmt(h.vpd, 2) + " kPa · P(mist in 2 h) " + fmt(h.prob, 2);
    txt += " · misting followed: " + (h.actual_dry === null || h.actual_dry === undefined ? DASH : (h.actual_dry ? "yes" : "no"));
    $("hour-readout").textContent = txt;
  }

  function drawChart() {
    var cv = $("chart"), ctx = cv.getContext("2d");
    var W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    var L = 46, R = 46, T = 30, B = 34;
    var pw = W - L - R, ph = H - T - B;
    var hours = replay.hours || [];
    ctx.font = "12px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    if (!replay.available || !hours.length) {
      ctx.fillStyle = "#667180"; ctx.textAlign = "center";
      ctx.fillText(replay.available ? "no data for " + replay.date : STR.fcNA, W / 2, H / 2);
      return;
    }
    var vmax = 2.0;
    for (var i = 0; i < hours.length; i++) if (typeof hours[i].vpd === "number" && hours[i].vpd > vmax) vmax = hours[i].vpd;
    vmax = Math.ceil(vmax * 2) / 2;
    var x = function (hr) { return L + (hr / 23) * pw; };
    var yV = function (v) { return T + ph - (v / vmax) * ph; };
    var yP = function (p) { return T + ph - p * ph; };
    var upto = replay.idx;

    // shading: hours where misting actually followed (actual_dry), revealed up to the current hour
    ctx.fillStyle = "rgba(214,59,47,0.14)";
    for (i = 0; i <= upto && i < hours.length; i++) {
      if (hours[i].actual_dry) ctx.fillRect(x(i) - pw / 46, T, pw / 23, ph);
    }
    // axes + grid
    ctx.strokeStyle = "#d9dee5"; ctx.lineWidth = 1;
    ctx.beginPath();
    for (var g = 0; g <= vmax + 1e-9; g += 0.5) { ctx.moveTo(L, yV(g)); ctx.lineTo(L + pw, yV(g)); }
    ctx.stroke();
    ctx.strokeStyle = "#1b2430";
    ctx.beginPath(); ctx.moveTo(L, T); ctx.lineTo(L, T + ph); ctx.lineTo(L + pw, T + ph); ctx.lineTo(L + pw, T); ctx.stroke();
    ctx.fillStyle = "#1b2430"; ctx.textAlign = "right";
    for (g = 0; g <= vmax + 1e-9; g += 0.5) ctx.fillText(g.toFixed(1), L - 4, yV(g));
    ctx.textAlign = "left";
    for (g = 0; g <= 1.0001; g += 0.25) ctx.fillText(g.toFixed(2), L + pw + 4, yP(g));
    ctx.textAlign = "center";
    for (i = 0; i < 24; i += 3) ctx.fillText(i + ":00", x(i), T + ph + 12);
    ctx.fillStyle = "#667180";
    ctx.fillText("hour of day (local)", L + pw / 2, T + ph + 26);
    ctx.save(); ctx.translate(12, T + ph / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("VPD (kPa)", 0, 0); ctx.restore();
    ctx.save(); ctx.translate(W - 10, T + ph / 2); ctx.rotate(Math.PI / 2); ctx.fillText("P(misting needed in next 2 h)", 0, 0); ctx.restore();

    // 1.2 kPa dashed threshold
    ctx.setLineDash([6, 4]); ctx.strokeStyle = "#d63b2f"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(L, yV(1.2)); ctx.lineTo(L + pw, yV(1.2)); ctx.stroke();
    ctx.setLineDash([]);

    // actual VPD line (revealed up to current hour)
    ctx.strokeStyle = "#1f6f5c"; ctx.lineWidth = 2.5;
    ctx.beginPath(); var started = false;
    for (i = 0; i <= upto && i < hours.length; i++) {
      var v = hours[i].vpd;
      if (typeof v !== "number") { started = false; continue; }
      if (!started) { ctx.moveTo(x(i), yV(v)); started = true; } else ctx.lineTo(x(i), yV(v));
    }
    ctx.stroke();
    // model probability line
    ctx.strokeStyle = "#7b3fd6"; ctx.lineWidth = 2;
    ctx.beginPath(); started = false;
    for (i = 0; i <= upto && i < hours.length; i++) {
      var p = hours[i].prob;
      if (typeof p !== "number") { started = false; continue; }
      if (!started) { ctx.moveTo(x(i), yP(p)); started = true; } else ctx.lineTo(x(i), yP(p));
    }
    ctx.stroke();
    // current hour marker
    var cur = hours[upto];
    if (cur) {
      ctx.strokeStyle = "#1b2430"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x(upto), T); ctx.lineTo(x(upto), T + ph); ctx.stroke(); ctx.setLineDash([]);
      if (typeof cur.vpd === "number") { ctx.fillStyle = "#1f6f5c"; ctx.beginPath(); ctx.arc(x(upto), yV(cur.vpd), 5, 0, 7); ctx.fill(); }
      if (typeof cur.prob === "number") { ctx.fillStyle = "#7b3fd6"; ctx.beginPath(); ctx.arc(x(upto), yP(cur.prob), 5, 0, 7); ctx.fill(); }
    }
    // legend
    var lx = L + 6, ly = T - 16;
    ctx.textAlign = "left"; ctx.font = "11px system-ui, sans-serif";
    function key(color, label, dash, fill) {
      if (fill) { ctx.fillStyle = color; ctx.fillRect(lx, ly - 5, 14, 10); }
      else { ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dash || []); ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 14, ly); ctx.stroke(); ctx.setLineDash([]); }
      ctx.fillStyle = "#1b2430"; ctx.fillText(label, lx + 18, ly);
      lx += 18 + ctx.measureText(label).width + 14;
    }
    key("#1f6f5c", "actual VPD (kPa)");
    key("#d63b2f", "1.2 kPa mist line", [4, 3]);
    key("#7b3fd6", "model P(mist in 2 h)");
    key("rgba(214,59,47,0.35)", "misting actually followed", null, true);
  }

  // ------------------------------------------------------------ today's outlook strip
  function loadToday() {
    fetch("/api/today", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(drawToday)
      .catch(function () { $("today-meta").textContent = STR.todayNA; drawTodayStrip([], false); });
  }
  function drawToday(j) {
    if (!j || !j.available || !j.hourly) { $("today-meta").textContent = STR.todayNA; drawTodayStrip([], false); return; }
    var now = Date.now();
    var all = j.hourly.slice();
    var from = all.filter(function (h) { return Date.parse(h.time) >= now - 30 * 60 * 1000; });
    var latest = false;
    var sel = from.slice(0, 24);
    if (sel.length < 24) { sel = all.slice(-24); latest = true; }
    var dry = sel.filter(function (h) { return h.dry; }).length;
    $("today-meta").textContent = "fetched " + (j.fetched_at_local || j.fetched_at_utc || "?") +
      " · next 24 h" + (latest ? " (latest available)" : "") + ": " + dry + " of " + sel.length +
      " hours dry (VPD > 1.2 kPa)";
    drawTodayStrip(sel, true);
  }
  function drawTodayStrip(sel, ok) {
    var cv = $("today-strip"), ctx = cv.getContext("2d");
    var W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    ctx.font = "10px system-ui, sans-serif"; ctx.textBaseline = "middle";
    if (!ok || !sel.length) { ctx.fillStyle = "#667180"; ctx.textAlign = "center"; ctx.fillText(STR.todayNA, W / 2, H / 2); return; }
    var L = 30, T = 6, B = 16, pw = W - L - 6, ph = H - T - B;
    var vmax = 1.5;
    sel.forEach(function (h) { if (typeof h.vpd === "number" && h.vpd > vmax) vmax = h.vpd; });
    var bw = pw / sel.length;
    ctx.strokeStyle = "#d9dee5"; ctx.setLineDash([3, 3]); ctx.beginPath();
    var y12 = T + ph - (1.2 / vmax) * ph; ctx.moveTo(L, y12); ctx.lineTo(L + pw, y12); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#667180"; ctx.textAlign = "right"; ctx.fillText("1.2", L - 3, y12); ctx.fillText("kPa", L - 3, T + 6);
    sel.forEach(function (h, i) {
      var v = (typeof h.vpd === "number") ? h.vpd : 0;
      var bh = (v / vmax) * ph;
      ctx.fillStyle = h.dry ? "#d63b2f" : "#2f7fd6";
      ctx.fillRect(L + i * bw + 1, T + ph - bh, Math.max(1, bw - 2), bh);
      if (i % 4 === 0) {
        ctx.fillStyle = "#1b2430"; ctx.textAlign = "center";
        ctx.fillText(String(h.time).slice(11, 16), L + i * bw + bw / 2, T + ph + 8);
      }
    });
  }
  loadToday();
  setInterval(loadToday, 5 * 60 * 1000);

  // initial replay load happens when /api/state first reports forecast_available
  drawChart();
})();
