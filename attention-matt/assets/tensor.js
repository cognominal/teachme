/* tensor.js — the shared mini-tensor component for this workspace.
 *
 * Matrices are arrays of rows: [[1,2],[3,4]] is a 2×2 matrix.
 * That matches the (T, d) shape prose used in "Attention Is All You Need",
 * so a shape you read in the paper is a shape you can see in the console.
 *
 * Pure ES5, no dependencies, works offline, works in Node/Bun for testing.
 */
(function (root) {
  "use strict";

  /* ---------- shape helpers ---------- */

  function shape(m) {
    if (!Array.isArray(m) || m.length === 0) {
      throw new Error("not a matrix: expected a non-empty array of rows");
    }
    if (!Array.isArray(m[0])) {
      throw new Error("not a matrix: rows must be arrays (a plain vector is 1-D)");
    }
    var cols = m[0].length;
    for (var i = 1; i < m.length; i++) {
      if (!Array.isArray(m[i])) throw new Error("ragged matrix: row " + i + " is not an array");
      if (m[i].length !== cols) {
        throw new Error("ragged matrix: row " + i + " is width " + m[i].length + ", row 0 is width " + cols);
      }
    }
    return [m.length, cols];
  }

  function describe(m) {
    try { var s = shape(m); return "(" + s[0] + "x" + s[1] + ")"; } catch (e) { return "(?)"; }
  }

  function zeros(rows, cols) {
    var out = [];
    for (var i = 0; i < rows; i++) out.push(new Array(cols).fill(0));
    return out;
  }

  /* ---------- linear algebra ---------- */

  /* matmul(a, b): (m×n) · (n×p) -> (m×p). */
  function matmul(a, b) {
    var sa = shape(a), sb = shape(b);
    if (sa[1] !== sb[0]) {
      throw new Error(
        "matmul: " + describe(a) + " . " + describe(b) +
        " — inner dimensions must match (left has " + sa[1] + " columns, right has " + sb[0] + " rows)"
      );
    }
    var out = [];
    for (var i = 0; i < sa[0]; i++) {
      var row = new Array(sb[1]);
      for (var j = 0; j < sb[1]; j++) {
        var sum = 0;
        for (var k = 0; k < sa[1]; k++) sum += a[i][k] * b[k][j];
        row[j] = sum;
      }
      out.push(row);
    }
    return out;
  }

  /* transpose(a): (m×n) -> (n×m). */
  function transpose(a) {
    var s = shape(a), out = [];
    for (var j = 0; j < s[1]; j++) {
      var row = new Array(s[0]);
      for (var i = 0; i < s[0]; i++) row[i] = a[i][j];
      out.push(row);
    }
    return out;
  }

  /* scale(a, k): multiply every entry by k. */
  function scale(a, k) {
    return a.map(function (row) { return row.map(function (x) { return x * k; }); });
  }

  function maxOf(row) {
    var m = row[0];
    for (var i = 1; i < row.length; i++) if (row[i] > m) m = row[i];
    return m;
  }

  /* softmaxRows(a): normalise each ROW into a probability distribution.
   * exp() is shifted by the row max first so large scores cannot overflow. */
  function softmaxRows(a) {
    shape(a);
    return a.map(function (row) {
      var m = maxOf(row);
      var exps = row.map(function (x) { return Math.exp(x - m); });
      var total = exps.reduce(function (s, x) { return s + x; }, 0);
      return exps.map(function (x) { return x / total; });
    });
  }

  /* softmaxCols(a): the same thing down columns. Wrong axis for attention —
   * it exists so the lesson can diagnose that mistake precisely. */
  function softmaxCols(a) {
    return transpose(softmaxRows(transpose(a)));
  }

  function rowSums(a) {
    return a.map(function (row) { return row.reduce(function (s, x) { return s + x; }, 0); });
  }

  /* ---------- deterministic random numbers (same numbers every reload) ---------- */

  function mulberry32(seed) {
    var t = seed >>> 0;
    return function () {
      t = (t + 0x6D2B79F5) >>> 0;
      var r = Math.imul(t ^ (t >>> 15), 1 | t);
      r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* randn(rows, cols, seed): reproducible normal-ish noise, used for test inputs. */
  function randn(rows, cols, seed) {
    var rnd = mulberry32(seed == null ? 1 : seed);
    var out = [];
    for (var i = 0; i < rows; i++) {
      var row = new Array(cols);
      for (var j = 0; j < cols; j++) {
        var u = Math.max(rnd(), 1e-12), v = rnd();
        row[j] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      }
      out.push(row);
    }
    return out;
  }

  /* ---------- comparison ---------- */

  function maxAbsDiff(a, b) {
    var sa = shape(a), sb = shape(b);
    if (sa[0] !== sb[0] || sa[1] !== sb[1]) return Infinity;
    var worst = 0;
    for (var i = 0; i < sa[0]; i++) {
      for (var j = 0; j < sa[1]; j++) {
        var d = Math.abs(a[i][j] - b[i][j]);
        if (d > worst) worst = d;
      }
    }
    return worst;
  }

  function isClose(a, b, eps) {
    return maxAbsDiff(a, b) <= (eps == null ? 1e-9 : eps);
  }

  /* fmt: pretty-print a matrix with fixed decimals, for console inspection. */
  function fmt(a, digits) {
    var d = digits == null ? 3 : digits;
    if (!Array.isArray(a)) return String(a);
    if (!Array.isArray(a[0])) return "[" + a.map(function (x) { return x.toFixed(d); }).join(", ") + "]";
    var w = 0;
    var strs = a.map(function (row) {
      return row.map(function (x) {
        var s = (Math.abs(x) < 5e-13 ? 0 : x).toFixed(d);
        if (s.length > w) w = s.length;
        return s;
      });
    });
    return strs.map(function (row) {
      return "[ " + row.map(function (s) { return ("     " + s).slice(-w); }).join("  ") + " ]";
    }).join("\n");
  }

  /* ---------- reference implementations (the target the learner matches) ---------- */

  /* refWeights(Q, K) -> (T×T): how much each query pulls from each key. */
  function refWeights(Q, K) {
    var dk = shape(Q)[1];
    return softmaxRows(scale(matmul(Q, transpose(K)), 1 / Math.sqrt(dk)));
  }

  /* refAttention(Q, K, V) -> (T×dv): each row is a weighted average of V's rows. */
  function refAttention(Q, K, V) {
    return matmul(refWeights(Q, K), V);
  }

  var T = {
    shape: shape, describe: describe, zeros: zeros,
    matmul: matmul, transpose: transpose, scale: scale,
    softmaxRows: softmaxRows, softmaxCols: softmaxCols, rowSums: rowSums,
    mulberry32: mulberry32, randn: randn,
    maxAbsDiff: maxAbsDiff, isClose: isClose, fmt: fmt,
    refWeights: refWeights, refAttention: refAttention
  };

  root.T = T;
  if (typeof module !== "undefined" && module.exports) module.exports = T;
})(typeof globalThis !== "undefined" ? globalThis : this);
