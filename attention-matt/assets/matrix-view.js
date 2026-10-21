/* matrix-view.js — the shared matrix renderer for this workspace.
 *
 * One idea: a matrix on screen should look like the matrix in the code and in the
 * paper — row i is the i-th token, column j is the j-th key. Shading shows
 * magnitude, the number in the cell shows the value, so a grid can be read both
 * ways at once.
 *
 * Reused by:
 *   lesson 00 — raw Q·Kᵀ scores
 *   lesson 01 — attention weights
 *   lesson 02 — masked (lower-triangular) weights, via opts.cellClass
 *
 * Classic script, no modules, no fetch: works offline from file://.
 *
 * Usage:
 *   MatrixView.render(host, W, { rowLabels: tokens, colLabels: tokens, digits: 2,
 *                                intensityMax: 1, title: "weights" });
 */
window.MatrixView = (function () {
  "use strict";

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  /* heatmap(m, opts) -> HTML string.
   * opts:
   *   rowLabels   array of row labels (default: 0,1,2…) — the "asking" side
   *   colLabels   array of column labels (default: 0,1,2…) — the "answering" side
   *   digits      decimals shown (default 2)
   *   intensityMax  value that maps to full shading (default: largest magnitude)
   *   cellText    function (value, i, j) -> string shown in the cell
   *   cellClass   function (value, i, j) -> extra class name, or null
   *   corner      label for the empty top-left cell
   */
  function heatmap(m, opts) {
    opts = opts || {};
    var digits = opts.digits == null ? 2 : opts.digits;
    var cells = [];
    var max = 0;
    for (var i = 0; i < m.length; i++) {
      for (var j = 0; j < m[i].length; j++) {
        var v = m[i][j];
        if (isFinite(v)) max = Math.max(max, Math.abs(v));
        cells.push(v);
      }
    }
    var denom = opts.intensityMax != null ? opts.intensityMax : (max || 1);

    function label(list, k, kind) {
      if (list && list[k] != null) return esc(list[k]);
      return String(k);
    }

    function show(v, i, j) {
      if (opts.cellText) return opts.cellText(v, i, j);
      if (!isFinite(v)) return v > 0 ? "∞" : "−∞";
      return (Math.abs(v) < 5e-13 ? 0 : v).toFixed(digits);
    }

    var html = '<table class="heatmap"><thead><tr><th>' +
      (opts.corner != null ? esc(opts.corner) : "") + "</th>";
    var cols = m.length ? m[0].length : 0;
    for (var c = 0; c < cols; c++) html += '<th class="col">' + label(opts.colLabels, c, "col") + "</th>";
    html += "</tr></thead><tbody>";
    for (var r = 0; r < m.length; r++) {
      html += '<tr><td class="head">' + label(opts.rowLabels, r, "row") + "</td>";
      for (var k = 0; k < m[r].length; k++) {
        var val = m[r][k];
        var extra = opts.cellClass ? opts.cellClass(val, r, k) : null;
        var w = Math.max(0, Math.min(100, (isFinite(val) ? Math.abs(val) : denom) / denom * 100));
        html += '<td class="' + (extra || "") + '" style="--w:' + w.toFixed(1) + '" title="' +
          label(opts.rowLabels, r, "row") + " → " + label(opts.colLabels, k, "col") + ": " + show(val, r, k) +
          '">' + show(val, r, k) + "</td>";
      }
      html += "</tr>";
    }
    return html + "</tbody></table>";
  }

  function render(host, m, opts) {
    host.innerHTML = heatmap(m, opts);
    return host;
  }

  return { heatmap: heatmap, render: render, escapeHtml: esc };
})();
