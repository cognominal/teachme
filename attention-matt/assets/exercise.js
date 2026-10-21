/* exercise.js — the shared code-exercise component.
 *
 * A lesson declares parts. Each part is the BODY of a function: the names in
 * `scope` plus the names in `params` are in scope, and the learner ends with a
 * `return`. The component compiles the body, runs the lesson's tests against
 * it, and on failure names the specific wrong answer by comparing the
 * learner's output against variant references.
 *
 * Classic script, no modules, no fetch: works offline from file://.
 *
 * Usage:
 *   Exercise.mount({
 *     host: document.getElementById("ex1"),
 *     lessonId: "0001",
 *     prelude: { T: window.T, matmul: window.T.matmul },
 *     parts: [{
 *       id: "weights",
 *       title: "Part A — the weight matrix",
 *       prompt: "…html…",
 *       scope:    ["T", "matmul", "transpose", "scale", "softmaxRows"],
 *       params:   ["Q", "K"],
 *       stub:     "// …\nreturn null;",
 *       solution: "// …\nreturn softmaxRows(scaled);",
 *       tests:    [{ name: "…", run: function (student) { return { ok: true, detail: "…" }; } }],
 *       diagnose: [{ when: function (student) { return true; }, hint: "…html…" }]
 *     }]
 *   });
 * Tests call `student(...)` with their own inputs.
 * A later part can pull an earlier part's function in with
 * `sources: { attentionWeights: "weights" }` — resolved live, at run time.
 */
window.Exercise = (function () {
  "use strict";

  function h(tag, cls, html) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (html != null) node.innerHTML = html;
    return node;
  }

  function store(key, value) {
    try {
      if (value === undefined) return window.localStorage.getItem(key);
      window.localStorage.setItem(key, value);
    } catch (e) { /* file:// or private mode: the lesson still works, minus persistence */ }
    return null;
  }

  /* Line numbers come from the message when V8 provides them, and from the
     stack otherwise: runtime ReferenceError/TypeError messages carry no
     position, but their frames do. */
  function renumber(message) {
    return String(message).replace(/(?:anonymous|<anonymous>):(\d+):(\d+)/g, function (_m, line, col) {
      var n = parseInt(line, 10) - 2;
      return "your line " + (n < 1 ? 1 : n) + ", col " + col;
    });
  }

  function lineFromStack(err) {
    var stack = String((err && err.stack) || "");
    var m = stack.match(/(?:anonymous|<anonymous>):(\d+):(\d+)/);
    if (!m) return "";
    var n = parseInt(m[1], 10) - 2;
    return n >= 1 ? " (your line " + n + ")" : "";
  }

  function describeError(err) {
    var msg = renumber((err && err.message) || String(err));
    return /\d+:\d+/.test(msg) ? msg : msg + lineFromStack(err);
  }

  function attempt(fn) {
    try { return { value: fn() }; } catch (e) { return { error: e }; }
  }

  function mount(spec) {
    var host = spec.host;
    host.classList.add("exercise");
    var head = h("div", "ex-head");
    head.appendChild(h("span", null, spec.heading || "Exercise"));
    var statusEl = h("span", "status", "");
    head.appendChild(statusEl);
    host.appendChild(head);
    if (spec.note) host.appendChild(h("div", "helpers", spec.note));

    var registry = {};
    var boxes = [];
    (spec.parts || []).forEach(function (part) {
      var box = mountPart(host, spec, part, registry);
      registry[part.id] = box;
      boxes.push(box);
    });

    function refreshStatus() {
      var all = boxes.every(function (b) { return b.isPassed(); });
      statusEl.textContent = all && boxes.length ? "all parts pass ✓" : "";
    }
    boxes.forEach(function (b) { b.onPass = refreshStatus; });
    host.statusEl = statusEl;
    host.parts = boxes;
    return boxes;
  }

  function mountPart(host, spec, part, registry) {
    var prelude = spec.prelude || {};
    var sources = part.sources || {};
    var scope = part.scope || [];
    var params = part.params || [];
    var paramList = scope.concat(params);
    var storageKey = "teach:" + (spec.lessonId || "lesson") + ":" + part.id;

    function resolveScope() {
      return scope.map(function (name) {
        if (Object.prototype.hasOwnProperty.call(prelude, name)) return prelude[name];
        if (Object.prototype.hasOwnProperty.call(sources, name)) {
          var upstream = registry[sources[name]];
          if (!upstream) throw new Error("lesson bug: no part named " + sources[name]);
          try {
            return upstream.fn();
          } catch (e) {
            var broken = function () {
              throw new Error("part \"" + sources[name] + "\" did not compile (" + e.message.split("\n")[0] + ")");
            };
            return broken;
          }
        }
        throw new Error("lesson bug: nothing provides \"" + name + "\"");
      });
    }

    /* fn(): compile the learner's body into a callable, with the scope values
       bound as leading arguments. Tests then call it with the params only. */
    function fn() {
      var raw = new Function(paramList.join(","), ta.value);
      var bound = resolveScope();
      return function () {
        return raw.apply(null, bound.concat(Array.prototype.slice.call(arguments)));
      };
    }

    var box = h("div", "part");
    box.partId = part.id;
    box.id = "part-" + part.id;
    box.appendChild(h("h3", null, part.title));
    if (part.prompt) box.appendChild(h("div", null, part.prompt));

    var shown = scope.concat(params.map(function (p) { return p + " <em>(test input)</em>"; }));
    box.appendChild(h("div", "helpers", "In scope: " +
      shown.map(function (n) { return "<code>" + n + "</code>"; }).join(" ") +
      " — plain nested JS arrays, no libraries to import."));

    var ta = document.createElement("textarea");
    ta.className = "editor";
    ta.spellcheck = false;
    ta.setAttribute("autocapitalize", "off");
    ta.setAttribute("autocorrect", "off");
    ta.rows = Math.min(20, Math.max(6, part.stub.split("\n").length + 2));
    ta.value = store(storageKey) || part.stub;
    ta.addEventListener("input", function () {
      store(storageKey, ta.value);
      ta.classList.toggle("dirty", ta.value !== part.stub);
    });
    ta.addEventListener("keydown", function (e) {
      if ((e.key === "Enter" || e.key === "r" || e.key === "R") && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        run();
      } else if (e.key === "Tab") {
        e.preventDefault();
        var s = ta.selectionStart, epos = ta.selectionEnd;
        ta.value = ta.value.slice(0, s) + "  " + ta.value.slice(epos);
        ta.selectionStart = ta.selectionEnd = s + 2;
      }
    });
    var wrap = h("div", "editor-wrap");
    wrap.appendChild(ta);
    box.appendChild(wrap);

    var controls = h("div", "controls");
    var runBtn = h("button", "btn btn--primary", "Run tests");
    var resetBtn = h("button", "btn btn--ghost", "Reset to stub");
    var solutionBtn = h("button", "btn btn--ghost", "Show a solution");
    controls.appendChild(runBtn);
    controls.appendChild(resetBtn);
    controls.appendChild(solutionBtn);
    controls.appendChild(h("span", "hint", "⌘/Ctrl + Enter"));
    box.appendChild(controls);

    var feedback = h("div", "feedback");
    box.appendChild(feedback);
    host.appendChild(box);

    var passed = false;
    var attempts = 0;

    function run() {
      attempts++;
      feedback.innerHTML = "";
      feedback.className = "feedback";

      try {
        fn(); /* compile check only: syntax errors surface here */
      } catch (e) {
        feedback.className = "feedback fail";
        feedback.appendChild(h("div", "verdict", "Syntax error — nothing ran. Check brackets, quotes and commas."));
        feedback.appendChild(h("div", "console-out", escapeHtml(describeError(e))));
        return;
      }

      var results = [];
      var thrown = [];
      (part.tests || []).forEach(function (t) {
        var student = attempt(fn);
        if (student.error) {
          var compile = describeError(student.error);
          results.push({ name: t.name, ok: false, detail: compile });
          thrown.push(compile);
          return;
        }
        var r = attempt(function () { return t.run(student.value); });
        if (r.error) {
          var msg = describeError(r.error);
          results.push({ name: t.name, ok: false, detail: msg });
          thrown.push(msg);
        } else {
          results.push({ name: t.name, ok: !!(r.value && r.value.ok), detail: (r.value && r.value.detail) || "" });
        }
      });

      var failed = results.filter(function (r) { return !r.ok; });
      var allOk = results.length > 0 && failed.length === 0;
      feedback.className = "feedback " + (allOk ? "pass" : "fail");
      feedback.appendChild(h("div", "verdict", allOk
        ? "✓ All checks pass. " + (attempts === 1 ? "First try." : attempts + " attempts.")
        : "Not yet — " + failed.length + " of " + results.length + " checks failed."));

      var list = h("ul", "tests");
      results.forEach(function (r) {
        var li = h("li");
        li.appendChild(h("span", r.ok ? "mark-ok" : "mark-bad", r.ok ? "✓" : "✗"));
        li.appendChild(document.createTextNode(" " + r.name));
        if (!r.ok && r.detail) li.appendChild(h("span", "detail", " — " + r.detail));
        list.appendChild(li);
      });
      feedback.appendChild(list);

      if (!allOk) {
        var hints = [];
        if (thrown.some(function (m) { return /is not defined/.test(m); })) {
          hints.push("An unknown name: check the spelling against the <em>In scope</em> list above. " +
            "Every helper name must match exactly, and the test inputs are only available under their listed names.");
        }
        (part.diagnose || []).forEach(function (d) {
          if (hints.length) return;
          var hit = attempt(function () {
            var student = fn();
            return d.when(student);
          });
          if (!hit.error && hit.value) hints.push(d.hint);
        });
        if (hints.length) feedback.appendChild(h("div", "diagnosis", "<strong>Most likely cause</strong>" + hints[0]));
      }

      passed = allOk;
      box.setAttribute("data-passed", allOk ? "yes" : "no");
      if (box.onPass) box.onPass();
    }

    runBtn.addEventListener("click", run);
    resetBtn.addEventListener("click", function () {
      ta.value = part.stub;
      store(storageKey, part.stub);
      ta.classList.remove("dirty");
      feedback.innerHTML = "";
      box.removeAttribute("data-passed");
      passed = false;
      if (box.onPass) box.onPass();
    });
    solutionBtn.addEventListener("click", function () {
      if (attempts === 0 && !passed) {
        feedback.className = "feedback fail";
        feedback.innerHTML = "";
        feedback.appendChild(h("div", "verdict", "Attempt it first — the effort is what makes it stick. Then compare."));
        return;
      }
      ta.value = part.solution;
      store(storageKey, part.solution);
      ta.classList.add("dirty");
    });

    box.editor = ta;
    box.fn = fn;
    box.isPassed = function () { return passed; };
    box.grade = run;
    return box;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c];
    });
  }

  return { mount: mount, escapeHtml: escapeHtml, renumber: renumber };
})();
