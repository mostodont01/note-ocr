// Movable, resizable selection box with a small toolbar. Injected on click.
(() => {
  if (window.__notocrOverlay) { window.__notocrOverlay.remove(); window.__notocrOverlay = null; return; }
  const root = document.createElement("div");
  window.__notocrOverlay = root;
  root.style.cssText = "all:initial;position:fixed;inset:0;z-index:2147483647;pointer-events:none;font:13px system-ui,sans-serif";
  const box = document.createElement("div");
  const w = Math.min(700, innerWidth - 80), h = Math.min(260, innerHeight - 120);
  Object.assign(box.style, {
    position: "fixed", left: (innerWidth - w) / 2 + "px", top: (innerHeight - h) / 2 + "px",
    width: w + "px", height: h + "px", border: "2px dashed #72a7ff", background: "rgba(114,167,255,0.08)",
    boxShadow: "0 0 0 9999px rgba(0,0,0,0.25)", pointerEvents: "auto", cursor: "move", boxSizing: "border-box",
  });
  const handle = document.createElement("div");
  Object.assign(handle.style, {
    position: "absolute", right: "-7px", bottom: "-7px", width: "14px", height: "14px",
    background: "#72a7ff", borderRadius: "3px", cursor: "nwse-resize",
  });
  box.appendChild(handle);
  const bar = document.createElement("div");
  Object.assign(bar.style, {
    position: "absolute", left: "0", top: "-40px", display: "flex", gap: "6px", alignItems: "center",
    background: "#202329", color: "#e3e7ed", padding: "5px 8px", borderRadius: "6px", cursor: "default",
    whiteSpace: "nowrap",
  });
  const status = document.createElement("span");
  status.style.color = "#929ba8";
  const btn = (label, fn, primary) => {
    const b = document.createElement("button");
    b.textContent = label;
    Object.assign(b.style, {
      all: "initial", font: "13px system-ui,sans-serif", padding: "4px 10px", borderRadius: "4px",
      cursor: "pointer", background: primary ? "#72a7ff" : "#282c33", color: primary ? "#101722" : "#e3e7ed",
    });
    b.addEventListener("mousedown", (e) => e.stopPropagation());
    b.addEventListener("click", (e) => { e.stopPropagation(); fn(); });
    bar.appendChild(b);
    return b;
  };
  const go = async (mode) => {
    const r = box.getBoundingClientRect();
    root.style.visibility = "hidden"; // not in the screenshot
    await new Promise((res) => requestAnimationFrame(() => requestAnimationFrame(res)));
    const reply = await browser.runtime.sendMessage({
      type: "capture", mode, dpr: devicePixelRatio,
      rect: { x: r.left + 2, y: r.top + 2, w: r.width - 4, h: r.height - 4 },
    });
    root.style.visibility = "visible";
    if (!reply || !reply.ok) { status.textContent = (reply && reply.error) || "failed"; status.style.color = "#f28b82"; return; }
    status.style.color = "#6fcf97";
    status.textContent = reply.summary || "sent to note-OCR";
    if (mode === "open") setTimeout(close, 1200);
  };
  const close = () => { root.remove(); window.__notocrOverlay = null; };
  btn("Open in note-OCR", () => go("open"), true);
  btn("Quick save", () => go("quick"));
  btn("✕", close);
  bar.appendChild(status);
  box.appendChild(bar);
  root.appendChild(box);
  document.documentElement.appendChild(root);

  let drag = null;
  box.addEventListener("mousedown", (e) => {
    const r = box.getBoundingClientRect();
    drag = { mode: e.target === handle ? "resize" : "move", x: e.clientX, y: e.clientY, l: r.left, t: r.top, w: r.width, h: r.height };
    e.preventDefault();
  });
  addEventListener("mousemove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (drag.mode === "move") { box.style.left = drag.l + dx + "px"; box.style.top = drag.t + dy + "px"; }
    else { box.style.width = Math.max(80, drag.w + dx) + "px"; box.style.height = Math.max(40, drag.h + dy) + "px"; }
  }, true);
  addEventListener("mouseup", () => { drag = null; }, true);
  addEventListener("keydown", (e) => { if (e.key === "Escape") close(); if (e.key === "Enter") go("open"); }, { once: false });
})();
