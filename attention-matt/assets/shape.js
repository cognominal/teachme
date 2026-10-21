/* shape.js — the shared shape-arithmetic component for this workspace.
 *
 * Two jobs:
 *   Shape.product(a, b)      — does A·B exist, and with what shape?
 *   Shape.explorer(host, …)  — a live drill: type two shapes, watch the rule fire.
 *
 * The rule implemented is the batched one from the start — A's LAST axis against
 * B's SECOND-TO-LAST — because that is what PyTorch implements and what lesson 03
 * needs once a head axis appears. For plain 2-D matrices it collapses to
 * "inner numbers must match".
 *
 * Classic script, no modules, no fetch: works offline from file://.
 *
 * Usage:
 *   Shape.explorer(document.getElementById("drill"), {
 *     presets: [{ label: "Q · Kᵀ", note: "every query against every key", a: "4x8", b: "8x4" }]
 *   });
 */
window.Shape = (function () {
  "use strict";

  function esc(s) {
    return String(s).replace(/[&<>]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c];
    });
  }

  /* "3x4" / "3×4" / "3 x 4" / "2,4,8" -> [3,4] */
  function parse(text) {
    var parts = String(text).trim().split(/[x×*,\s]+/).filter(function (s) { return s.length; });
    if (parts.length < 2) throw new Error('write at least two dimensions, like "3x4"');
    return parts.map(function (p) {
      if (!/^\d+$/.test(p) || +p < 1) throw new Error('"' + p + '" is not a dimension — use whole numbers, like "3x4"');
      return +p;
    });
  }

  function compact(dims) { return dims.join("×"); }
  function fmt(dims) { return dims.join(" × "); }
  function of(matrix) {
    var dims = [matrix.length];
    var cur = matrix[0];
    while (Array.isArray(cur)) { dims.push(cur.length); cur = cur[0]; }
    return dims;
  }

  /* product(a, b): a and b are arrays of dimensions. */
  function product(a, b) {
    var innerA = a[a.length - 1];
    var innerB = b[b.length - 2];
    var ok = a.length >= 2 && b.length >= 2 && innerA === innerB;
    return {
      ok: ok,
      innerA: innerA,
      innerB: innerB,
      shape: ok ? a.slice(0, -1).concat([b[b.length - 1]]) : null
    };
  }

  var MAX_CELLS = 144;

  function gridHtml(dims) {
    var rows = dims[dims.length - 2], cols = dims[dims.length - 1];
    if (rows * cols > MAX_CELLS) {
      var w = Math.max(24, Math.min(96, cols * 4));
      var h = Math.max(18, Math.min(96, rows * 4));
      return '<div class="shape-blob" style="width:' + w + "px;height:" + h + 'px"></div>';
    }
    var cells = "";
    for (var i = 0; i < rows * cols; i++) cells += "<i></i>";
    return '<div class="shape-grid" style="grid-template-columns:repeat(' + cols + ",11px)\">" + cells + "</div>";
  }

  function boxHtml(name, dims, markRow, markCol) {
    var lead = dims.length > 2 ? dims.slice(0, -2).join("×") + " of " : "";
    var rows = dims[dims.length - 2], cols = dims[dims.length - 1];
    return '<div class="shape-box">' +
      '<div class="shape-cols">' + lead + '<span class="' + (markCol ? "inner" : "") + '">' + cols + "</span> cols</div>" +
      '<div class="shape-body">' +
        '<div class="shape-rows"><span class="' + (markRow ? "inner" : "") + '">' + rows + "</span></div>" +
        gridHtml(dims) +
      "</div>" +
      '<div class="shape-name">' + esc(name) + " · " + compact(dims) + "</div>" +
    "</div>";
  }

  function explorer(host, spec) {
    spec = spec || {};
    host.classList.add("shapes");

    var form = document.createElement("div");
    form.className = "shape-form";
    form.innerHTML = "<label>A <input id=\"shape-a\" size=\"7\" value=\"" + (spec.a || "3x4") +
      "\"></label><span class=\"op\">·</span><label>B <input id=\"shape-b\" size=\"7\" value=\"" +
      (spec.b || "4x2") + "\"></label>";
    host.appendChild(form);

    var presetBox = document.createElement("div");
    presetBox.className = "shape-presets";
    host.appendChild(presetBox);

    var verdict = document.createElement("div");
    verdict.className = "shape-verdict";
    host.appendChild(verdict);

    var diagram = document.createElement("div");
    diagram.className = "shape-diagram";
    host.appendChild(diagram);

    var inA = form.querySelector("#shape-a");
    var inB = form.querySelector("#shape-b");
    var buttons = [];

    function markPressed(active) {
      buttons.forEach(function (b) {
        if (b.preset === active) b.setAttribute("aria-pressed", "true");
        else b.removeAttribute("aria-pressed");
      });
    }

    function render() {
      var a, b;
      try { a = parse(inA.value); } catch (e) { a = null; }
      try { b = parse(inB.value); } catch (e) { b = null; }

      if (!a || !b) {
        verdict.className = "shape-verdict bad";
        verdict.innerHTML = "<strong>Cannot read that shape.</strong> " + esc((a ? "" : "A: ") + "write dimensions as numbers joined by <code>x</code>, like <code>3x4</code>.");
        diagram.innerHTML = "";
        return;
      }

      var p = product(a, b);
      var innerNote = "A's last axis <span class=\"inner\">" + p.innerA + "</span>, B's second-to-last <span class=\"inner\">" +
        p.innerB + "</span>";

      if (p.ok) {
        verdict.className = "shape-verdict ok";
        verdict.innerHTML = "<strong>Product exists: " + compact(p.shape) + ".</strong> " + innerNote +
          " — they match, so every row of A meets a column of B. The outer numbers survive: " +
          compact(a) + " → " + compact(p.shape) + ".";
      } else {
        verdict.className = "shape-verdict bad";
        verdict.innerHTML = "<strong>No product.</strong> " + innerNote +
          " — they differ, so some dot product would have to pair different-length vectors. " +
          "Try the other order, or transpose one of them.";
      }

      diagram.innerHTML =
        boxHtml("A", a, false, true) +
        '<span class="shape-op">·</span>' +
        boxHtml("B", b, true, false) +
        '<span class="shape-op">=</span>' +
        (p.ok ? boxHtml("A · B", p.shape, false, false)
              : "<div class=\"shape-box\"><div class=\"shape-name\">no shape — the product does not exist</div></div>");
    }

    (spec.presets || []).forEach(function (p) {
      var btn = document.createElement("button");
      btn.className = "btn btn--ghost";
      btn.textContent = p.label;
      btn.title = p.note || "";
      btn.preset = p;
      btn.addEventListener("click", function () {
        inA.value = p.a; inB.value = p.b;
        markPressed(p);
        render();
      });
      presetBox.appendChild(btn);
      buttons.push(btn);
    });

    inA.addEventListener("input", function () { markPressed(null); render(); });
    inB.addEventListener("input", function () { markPressed(null); render(); });
    render();
    return host;
  }

  return { parse: parse, fmt: fmt, compact: compact, of: of, product: product, explorer: explorer, escapeHtml: esc };
})();
