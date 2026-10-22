/* Client-side player for the congestion-control dashboard.
 *
 * The episode is simulated in Python (real environment + real controllers); this script only
 * REPLAYS the recorded values in the browser, so Streamlit never has to redraw anything while
 * the animation runs (that redrawing is what made the charts blink). Pure SVG, no libraries.
 *
 * Data (one entry per controller): label, color, and per-step arrays
 *   capacity, rate, throughput (Mbps), latency_ms, queue_pct, loss_pct, plus rate_ymax, lat_ymax.
 */
(function (root) {
    "use strict";
    const NS = "http://www.w3.org/2000/svg";
    const C = {
        ink: "#1f2937", mute: "#6b7280", grid: "#e5e7eb", axis: "#9ca3af",
        blue: "#3b82f6", green: "#16a34a", red: "#c0262d", amber: "#e08a00"
    };
    const FONT = "-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif";

    const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const fx = (v, d) => Number(v).toFixed(d);
    const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

    function ticks(lo, hi, n) {
        const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / mag;
        const step = (r <= 1 ? 1 : r <= 2 ? 2 : r <= 5 ? 5 : 10) * mag, out = [];
        for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
        return out;
    }

    /* ------------------------------------------------------------------ line chart (SVG string) */
    function chartSVG(o) {
        const W = 520, H = 215, L = 50, R = o.right ? 52 : 14, T0 = 40, B = 32;
        const pw = W - L - R, ph = H - T0 - B;
        const X = (i) => L + (o.T > 0 ? (i * o.dt) / o.T : 0) * pw;
        const Y = (v, ax) => T0 + ph - ((clamp(v, ax.range[0], ax.range[1]) - ax.range[0]) / (ax.range[1] - ax.range[0])) * ph;
        let s = `<svg xmlns="${NS}" viewBox="0 0 ${W} ${H}" width="100%" font-family="${FONT}">`;
        s += `<text x="${L}" y="15" font-size="13" font-weight="600" fill="${C.ink}">${esc(o.title)}</text>`;
        let lx = L;                                                                       // legend
        o.series.forEach((se) => {
            if (se.faint) return;
            s += `<line x1="${lx}" y1="28" x2="${lx + 18}" y2="28" stroke="${se.color}" stroke-width="2"` +
                `${se.dash ? ' stroke-dasharray="5 3"' : ""}/>`;
            s += `<text x="${lx + 23}" y="32" font-size="10.5" fill="${C.mute}">${esc(se.name)}</text>`;
            lx += 23 + se.name.length * 5.6 + 14;
        });
        ticks(o.left.range[0], o.left.range[1], 4).forEach((v) => {                       // grid + left axis
            const y = Y(v, o.left);
            s += `<line x1="${L}" y1="${y}" x2="${W - R}" y2="${y}" stroke="${C.grid}"/>`;
            s += `<text x="${L - 6}" y="${y + 3.5}" font-size="10" text-anchor="end" fill="${C.mute}">${v}</text>`;
        });
        if (o.right) ticks(o.right.range[0], o.right.range[1], 4).forEach((v) => {
            s += `<text x="${W - R + 6}" y="${Y(v, o.right) + 3.5}" font-size="10" fill="${o.right.color}">${v}</text>`;
        });
        for (let t = 0; t <= o.T + 1e-9; t += 5) {                                         // x axis
            const x = L + (t / o.T) * pw;
            s += `<line x1="${x}" y1="${T0 + ph}" x2="${x}" y2="${T0 + ph + 4}" stroke="${C.axis}"/>`;
            s += `<text x="${x}" y="${T0 + ph + 16}" font-size="10" text-anchor="middle" fill="${C.mute}">${t}</text>`;
        }
        s += `<text x="${L + pw / 2}" y="${H - 3}" font-size="10.5" text-anchor="middle" fill="${C.mute}">time (s)</text>`;
        s += `<text transform="rotate(-90 11 ${T0 + ph / 2})" x="11" y="${T0 + ph / 2}" font-size="10.5" text-anchor="middle" fill="${C.mute}">${esc(o.left.label)}</text>`;
        if (o.right) s += `<text transform="rotate(90 ${W - 9} ${T0 + ph / 2})" x="${W - 9}" y="${T0 + ph / 2}" font-size="10.5" text-anchor="middle" fill="${o.right.color}">${esc(o.right.label)}</text>`;
        s += `<rect x="${L}" y="${T0}" width="${pw}" height="${ph}" fill="none" stroke="${C.axis}"/>`;
        let dot = "";
        o.series.forEach((se) => {                                                         // data
            const ax = se.axis === "right" ? o.right : o.left;
            const last = se.faint ? se.y.length - 1 : Math.min(o.k, se.y.length - 1);
            const pts = [];
            for (let i = 0; i <= last; i++) pts.push(fx(X(i), 1) + "," + fx(Y(se.y[i], ax), 1));
            if (se.fill && pts.length > 1)
                s += `<polygon points="${pts.join(" ")} ${fx(X(last), 1)},${fx(Y(ax.range[0], ax), 1)} ${fx(X(0), 1)},${fx(Y(ax.range[0], ax), 1)}" fill="${se.color}" fill-opacity="0.35"/>`;
            s += `<polyline points="${pts.join(" ")}" fill="none" stroke="${se.color}" stroke-width="${se.width || 1.6}"` +
                `${se.dash ? ' stroke-dasharray="5 3"' : ""}${se.faint ? ' stroke-opacity="0.22"' : ""} stroke-linejoin="round"/>`;
            if (se.dot) dot = `<circle cx="${fx(X(last), 1)}" cy="${fx(Y(se.y[last], ax), 1)}" r="3.6" fill="${se.color}"/>`;
        });
        return s + dot + "</svg>";
    }

    /* ------------------------------------------------------------------ network schematic (SVG string) */
    function schematic(d, m) {
        const sx = 80, sy = 43;                                   // 12 x 6 drawing units -> 960 x 258
        const X = (x) => x * sx, Y = (y) => (6 - y) * sy;
        const rect = (x0, x1, y0, y1, fill, stroke, sw) =>
            `<rect x="${X(x0)}" y="${Y(y1)}" width="${X(x1) - X(x0)}" height="${Y(y0) - Y(y1)}" fill="${fill}" ` +
            `stroke="${stroke || "none"}" stroke-width="${sw || 1}"/>`;
        const note = (x, y, t, size, color, bold) =>
            `<text x="${X(x)}" y="${Y(y) + 5}" font-size="${size || 17}" text-anchor="middle" fill="${color || C.ink}"` +
            `${bold ? ' font-weight="700"' : ""}>${esc(t)}</text>`;
        const mid = 3.0, hmax = 2.6, fill = clamp(d.queue_pct / 100, 0, 1);
        const hh = (v) => 0.15 + hmax * Math.min(v / m.rate_ymax, 1);
        const full = d.queue_pct > 99;
        const fcol = `rgb(${Math.round(255 * Math.min(1, 2 * fill))},${Math.round(200 * Math.min(1, 2 * (1 - fill)))},60)`;
        let s = `<svg xmlns="${NS}" viewBox="0 0 960 258" width="100%" font-family="${FONT}">`;
        s += rect(0.1, 1.9, 1.6, 4.4, "#dbeafe", "#555") + note(1.0, 2.95, "Sender", 18, C.ink, true);
        s += rect(1.9, 4.3, mid - hh(d.rate) / 2, mid + hh(d.rate) / 2, C.blue);
        s += note(3.1, 5.35, "offered " + fx(d.rate, 1) + " Mbps", 17, "#1d4ed8");
        s += rect(4.3, 6.7, 0.5, 5.0, "#f3f4f6", full ? C.red : "#555", full ? 3 : 1);
        s += rect(4.3, 6.7, 0.5, 0.5 + 4.5 * fill, fcol);
        s += note(5.5, 2.7, "buffer " + fx(d.queue_pct, 0) + "%", 19, d.queue_pct > 60 ? "#ffffff" : C.ink, true);
        if (d.loss_pct > 0) s += note(5.5, 5.4, "x dropping " + fx(d.loss_pct, 1) + "%", 16, C.red, true);
        s += rect(6.7, 9.3, mid - hh(d.capacity) / 2, mid + hh(d.capacity) / 2, "#d1d5db");
        s += rect(6.7, 9.3, mid - hh(d.throughput) / 2, mid + hh(d.throughput) / 2, C.green);
        s += note(8.0, 5.35, "link capacity " + fx(d.capacity, 1) + " Mbps", 17, "#374151");
        s += note(8.0, 0.55, "delivered " + fx(d.throughput, 1) + " Mbps", 17, "#15803d");
        s += rect(9.3, 11.9, 1.6, 4.4, "#dcfce7", "#555");
        s += note(10.6, 3.35, "Receiver", 18, C.ink, true) + note(10.6, 2.6, "latency " + fx(d.latency_ms, 0) + " ms", 16);
        return s + "</svg>";
    }

    function cardsHTML(d) {
        const items = [["Sending rate", fx(d.rate, 1) + " Mbps"], ["Link capacity", fx(d.capacity, 1) + " Mbps"],
        ["Buffer fill", fx(d.queue_pct, 0) + " %"], ["Latency", fx(d.latency_ms, 0) + " ms"],
        ["Throughput", fx(d.throughput, 1) + " Mbps"], ["Loss (this step)", fx(d.loss_pct, 1) + " %"]];
        return items.map((it) => `<div class="card"><div class="cl">${it[0]}</div><div class="cv">${it[1]}</div></div>`).join("");
    }

    /* ------------------------------------------------------------------ page */
    const CSS = `
  *{box-sizing:border-box} body{margin:0;font-family:${FONT};color:${C.ink};background:transparent}
  .bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:#fff;border:1px solid #d1d5db;border-radius:10px;padding:8px 12px;margin-bottom:10px}
  .bar button{border:1px solid #9ca3af;background:#f9fafb;border-radius:6px;padding:5px 12px;font-size:13px;cursor:pointer}
  .bar button:hover{background:#eef2ff} .bar input[type=range]{flex:1;min-width:140px} .bar .clock{font-size:13px;min-width:120px;color:${C.mute}}
  .bar select{font-size:13px;padding:3px}
  .panels{display:flex;gap:12px;align-items:flex-start} .panel{flex:1 1 0;min-width:0;background:#fff;border:1px solid #d1d5db;border-radius:10px;padding:10px}
  .single .panel{max-width:720px;margin:0 auto}
  .panel h3{margin:2px 0 8px;font-size:17px}
  .cards{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:6px} .single .cards{grid-template-columns:repeat(6,1fr)}
  .card{background:#f3f4f6;border-radius:8px;padding:5px 7px}.cl{font-size:10.5px;color:${C.mute}}.cv{font-size:15px;font-weight:600}
  .net,.ch{margin-top:2px}`;

    function init(el, data) {
        const n = data.n, dt = data.dt, T = n * dt;
        const style = document.createElement("style"); style.textContent = CSS; document.head.appendChild(style);
        el.innerHTML =
            `<div class="bar"><button id="play">Pause</button><button id="restart">Restart</button>` +
            `<input id="scrub" type="range" min="0" max="${n - 1}" value="0"><span class="clock" id="clock"></span>` +
            `<label>Speed <select id="speed"><option value="5">5 steps/s</option><option value="10">10 steps/s</option>` +
            `<option value="20" selected>20 (real time)</option><option value="40">40 steps/s</option><option value="80">80 steps/s</option></select></label></div>` +
            `<div class="panels ${data.panels.length === 1 ? "single" : ""}">` +
            data.panels.map((p, j) => `<div class="panel"><h3 style="color:${p.color}">${esc(p.label)}</h3><div class="cards" id="cards${j}"></div>` +
                `<div class="net" id="net${j}"></div><div class="ch" id="c1_${j}"></div><div class="ch" id="c2_${j}"></div><div class="ch" id="c3_${j}"></div></div>`).join("") + `</div>`;
        const $ = (id) => el.querySelector("#" + id);
        let k = clamp(data.start || 0, 0, n - 1), playing = !!data.autoplay, acc = 0, prev = null, dirty = true;

        function at(p, i) {
            return {
                rate: p.rate[i], capacity: p.capacity[i], throughput: p.throughput[i],
                latency_ms: p.latency_ms[i], queue_pct: p.queue_pct[i], loss_pct: p.loss_pct[i]
            };
        }
        function draw() {
            data.panels.forEach((p, j) => {
                $("cards" + j).innerHTML = cardsHTML(at(p, k));
                $("net" + j).innerHTML = schematic(at(p, k), p);
                const base = { T: T, dt: dt, k: k };
                $("c1_" + j).innerHTML = chartSVG(Object.assign({
                    title: "Sending rate vs capacity", left: { range: [0, p.rate_ymax], label: "Mbps" }, series: [
                        { y: p.capacity, color: "#000", faint: true, dash: true, name: "" },
                        { y: p.capacity, color: "#111", dash: true, name: "capacity" },
                        { y: p.rate, color: p.color, width: 2, name: "sending rate", dot: true }]
                }, base));
                $("c2_" + j).innerHTML = chartSVG(Object.assign({
                    title: "Buffer fill and latency", left: { range: [0, 100], label: "buffer fill (%)" },
                    right: { range: [0, Math.round(p.lat_ymax)], label: "latency (ms)", color: C.blue }, series: [
                        { y: p.queue_pct, color: C.amber, fill: true, name: "buffer fill (%)" },
                        { y: p.latency_ms, color: C.blue, axis: "right", name: "latency (ms)" }]
                }, base));
                const lossMax = Math.max(5, 1.2 * Math.max.apply(null, p.loss_pct.slice(0, k + 1)));
                $("c3_" + j).innerHTML = chartSVG(Object.assign({
                    title: "Throughput and loss", left: { range: [0, p.rate_ymax], label: "Mbps" },
                    right: { range: [0, Math.ceil(lossMax)], label: "loss (%)", color: C.red }, series: [
                        { y: p.throughput, color: C.green, name: "throughput (Mbps)" },
                        { y: p.loss_pct, color: C.red, axis: "right", name: "loss (%)" }]
                }, base));
            });
            $("scrub").value = k;
            $("clock").textContent = "t = " + fx(k * dt, 2) + " s / " + fx(T, 1) + " s";
            $("play").textContent = playing ? "Pause" : (k >= n - 1 ? "Replay" : "Play");
        }
        function frame(ts) {
            if (playing) {
                if (prev !== null) acc += ((ts - prev) / 1000) * Number($("speed").value);
                const adv = Math.floor(acc); acc -= adv;
                if (adv > 0) { k = Math.min(k + adv, n - 1); dirty = true; }
                if (k >= n - 1) { playing = false; dirty = true; }
            }
            prev = ts;
            if (dirty) { dirty = false; draw(); }
            requestAnimationFrame(frame);
        }
        $("play").onclick = () => { if (!playing && k >= n - 1) k = 0; playing = !playing; acc = 0; dirty = true; };
        $("restart").onclick = () => { k = 0; playing = true; acc = 0; dirty = true; };
        $("scrub").oninput = (e) => { k = Number(e.target.value); playing = false; dirty = true; };
        draw();
        requestAnimationFrame(frame);
    }

    const api = { chartSVG, schematic, cardsHTML, init };
    if (typeof module !== "undefined" && module.exports) module.exports = api; else root.Player = api;
})(typeof window !== "undefined" ? window : this);