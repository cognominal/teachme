/* pyexercise.js — the shared "run my Python" component.
 *
 * The lesson does not contain your implementation any more: it lives in a real
 * file under exercises/, edited in a real editor, graded by the local service
 * (`./bin/grader`, tools/grader/serve.py) which runs actual torch. This widget
 * fetches that report and draws it — the numbers on the page come from your
 * code, not from a copy of your code.
 *
 * Usage:
 *   PyExercise.mount(document.getElementById("exercise"), {
 *     lesson: "0001",
 *     file: "exercises/0001_attention.py",
 *     page: "lessons/0001-attention-as-a-weighted-average.html",
 *     onResult: function (report) { drawMatrices(report.figures); }
 *   });
 *
 * Offline (opened with file:// and no service running) it degrades to the
 * terminal instruction rather than pretending to check anything.
 */
window.PyExercise = (function () {
  "use strict";

  var DEFAULT_PORT = 8757;

  function h(tag, cls, html) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (html != null) node.innerHTML = html;
    return node;
  }

  function endpointFor(spec) {
    if (spec.endpoint) return spec.endpoint;
    if (location.protocol === "http:" || location.protocol === "https:") return "";
    return "http://127.0.0.1:" + DEFAULT_PORT;
  }

  function offlineBox(spec) {
    return '<div class="verdict">The local grader is not running, so this page cannot check your code.</div>' +
      "<p>In a terminal, in this workspace:</p>" +
      '<div class="console-out">./bin/grader</div>' +
      "<p>then reload <code>" + spec.page + "</code> from " +
      "<code>http://127.0.0.1:" + DEFAULT_PORT + "/" + spec.page + "</code></p>" +
      "<p>The terminal loop does not need this service at all:</p>" +
      '<div class="console-out">./bin/grade 0001</div>';
  }

  function render(report, spec, host, feedback, status) {
    feedback.className = "feedback";
    feedback.innerHTML = "";
    status.textContent = "";

    if (report.error) {
      var err = report.error;
      feedback.className = "feedback fail";
      if (err.kind === "missing") {
        feedback.appendChild(h("div", "verdict", "✗ Nothing to grade yet."));
        feedback.appendChild(h("div", null, "The checks look for <code>" + report.source + "</code>."));
      } else if (err.kind === "syntax") {
        feedback.appendChild(h("div", "verdict", "✗ Python could not compile your file."));
        feedback.appendChild(h("div", null, PyExercise.escapeHtml(err.message)));
        if (err.text) feedback.appendChild(h("div", "console-out", err.line + " → " + PyExercise.escapeHtml(err.text)));
      } else {
        feedback.appendChild(h("div", "verdict", "✗ Your file raised while loading."));
        feedback.appendChild(h("div", "console-out", PyExercise.escapeHtml(err.message)));
      }
      return;
    }

    var allPassed = report.parts.length > 0 && report.parts.every(function (p) { return p.passed; });
    if (allPassed) status.textContent = "all parts pass ✓";

    feedback.className = "feedback " + (allPassed ? "pass" : "fail");
    if (allPassed) {
      feedback.appendChild(h("div", "verdict", "✓ All checks pass — your code, running real torch."));
    } else {
      var failing = report.parts.filter(function (p) { return !p.passed; }).length;
      feedback.appendChild(h("div", "verdict", "Not yet — " + failing + " of " + report.parts.length + " parts failing."));
    }

    report.parts.forEach(function (part) {
      var box = h("div", "pyex-part");
      box.appendChild(h("h4", null, PyExercise.escapeHtml(part.title) +
        (part.passed ? ' <span class="mark-ok">✓</span>' : ' <span class="mark-bad">✗</span>') +
        (part.passed ? "" : " " + PyExercise.escapeHtml(part.verdict.replace(/^✗ /, "")))));
      if (part.raised) box.appendChild(h("div", "console-out", PyExercise.escapeHtml(part.raised)));
      var list = h("ul", "tests");
      part.tests.forEach(function (test) {
        var li = h("li");
        li.appendChild(h("span", test.ok ? "mark-ok" : "mark-bad", test.ok ? "✓" : "✗"));
        li.appendChild(document.createTextNode(" " + test.name));
        if (!test.ok && test.detail) li.appendChild(h("span", "detail", " — " + test.detail));
        list.appendChild(li);
      });
      box.appendChild(list);
      if (!part.passed && part.diagnosis) {
        box.appendChild(h("div", "diagnosis", "<strong>Most likely cause</strong>" + part.diagnosis));
      }
      feedback.appendChild(box);
    });

    var stamp = new Date().toLocaleTimeString();
    feedback.appendChild(h("div", "pyex-stamp", "checked " + stamp + " · " + report.source));
  }

  function mount(host, spec) {
    spec = spec || {};
    var endpoint = endpointFor(spec);
    host.classList.add("pyex");
    host.innerHTML = "";

    var head = h("div", "pyex-head");
    head.appendChild(h("span", null, spec.heading || "Your implementation"));
    var status = h("span", "status", "");
    head.appendChild(status);
    host.appendChild(head);

    host.appendChild(h("div", "helpers",
      (spec.intro || "") +
      "<p>Edit <code>" + spec.file + "</code> in your editor, save, then run the checks. " +
      "The same checks run in the terminal: <code>./bin/grade " + spec.lesson + "</code></p>"));

    var controls = h("div", "controls");
    var runBtn = h("button", "btn btn--primary", "Run the checks on my file");
    var fileBtn = h("button", "btn btn--ghost", "Show me a solution (after you have tried)");
    controls.appendChild(runBtn);
    controls.appendChild(fileBtn);
    host.appendChild(controls);

    var solutionBox = h("div", "console-out", "");
    solutionBox.style.display = "none";
    host.appendChild(solutionBox);

    var feedback = h("div", "feedback", "");
    host.appendChild(feedback);

    var running = false;

    function grade() {
      if (running) return;
      running = true;
      runBtn.disabled = true;
      status.textContent = "running…";
      var controller = typeof AbortController !== "undefined" ? new AbortController() : null;
      var timer = setTimeout(function () { if (controller) controller.abort(); }, 60000);

      fetch(endpoint + "/grade?lesson=" + encodeURIComponent(spec.lesson), { signal: controller ? controller.signal : undefined })
        .then(function (response) {
          if (!response.ok) throw new Error("HTTP " + response.status);
          return response.json();
        })
        .then(function (report) {
          clearTimeout(timer);
          render(report, spec, host, feedback, status);
          if (spec.onResult) {
            try { spec.onResult(report); } catch (e) { /* the page's problem, not the grader's */ }
          }
        })
        .catch(function (error) {
          clearTimeout(timer);
          feedback.className = "feedback fail";
          feedback.innerHTML = offlineBox(spec);
          status.textContent = "";
        })
        .then(function () {
          clearTimeout(timer);
          running = false;
          runBtn.disabled = false;
        });
    }

    runBtn.addEventListener("click", grade);
    fileBtn.addEventListener("click", function () {
      if (solutionBox.style.display === "none") {
        solutionBox.textContent =
          "./bin/grade " + spec.lesson + " --show-solution\n\n" +
          "(that prints the reference implementation; it is also at exercises/solutions/" +
          spec.lesson + "_attention.py)";
        solutionBox.style.display = "block";
      } else {
        solutionBox.style.display = "none";
      }
    });

    grade(); /* first look: show the verdict for whatever is in the file right now */
    return { grade: grade, host: host };
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c];
    });
  }

  return { mount: mount, escapeHtml: escapeHtml };
})();
