/* quiz.js — the shared retrieval-practice component.
 *
 * Two item types:
 *   { q, options: [{ text, correct }], explain }        multiple choice
 *   { q, type: "recall", accept: [/regex/i, "literal"], explain, placeholder }
 *
 * Multiple-choice options in this workspace are written to be the same
 * length, so formatting never leaks the answer.
 * Correct choices are only revealed after the learner commits.
 *
 * Classic script, no modules, no fetch.
 */
window.Quiz = (function () {
  "use strict";

  var LETTERS = ["A", "B", "C", "D", "E"];

  function h(tag, cls, html) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  }

  function mount(host, spec) {
    var root = h("div", "quiz");
    if (spec.title) root.appendChild(h("h3", null, spec.title));
    var scoreEl = h("div", "quiz-score", "");
    var states = spec.items.map(function () { return { answered: false, correct: false }; });

    spec.items.forEach(function (item, i) {
      var qBox = h("div", "q");
      qBox.appendChild(h("div", "prompt", '<span class="num">Question ' + (i + 1) + "</span>" + item.q));
      var explain = h("div", "explain", "");

      if (item.type === "recall") {
        var input = document.createElement("input");
        input.className = "recall-input";
        input.type = "text";
        input.placeholder = item.placeholder || "type it from memory";
        input.setAttribute("autocomplete", "off");
        input.spellcheck = false;
        var check = h("button", "btn btn--primary", "Check");
        check.style.marginTop = "0.5rem";
        function grade() {
          if (states[i].answered) return;
          var v = input.value.trim();
          if (!v) return;
          var ok = (item.accept || []).some(function (pat) {
            return pat instanceof RegExp ? pat.test(v) : v.toLowerCase() === String(pat).toLowerCase();
          });
          states[i].answered = true;
          states[i].correct = ok;
          input.style.borderColor = ok ? "var(--ok)" : "var(--bad)";
          input.disabled = true;
          check.disabled = true;
          explain.innerHTML = (ok ? "<strong>Yes.</strong> " : "<strong>Not quite.</strong> ") + (item.explain || "");
          updateScore();
        }
        check.addEventListener("click", grade);
        input.addEventListener("keydown", function (e) { if (e.key === "Enter") grade(); });
        qBox.appendChild(input);
        qBox.appendChild(check);
        qBox.appendChild(explain);
        root.appendChild(qBox);
        return;
      }

      var opts = h("div", "options");
      item.options.forEach(function (opt, j) {
        var btn = h("button", "option");
        btn.type = "button";
        btn.appendChild(h("span", "key", LETTERS[j]));
        btn.appendChild(h("span", null, opt.text));
        btn.addEventListener("click", function () {
          if (states[i].answered) return;
          states[i].answered = true;
          states[i].correct = !!opt.correct;
          Array.prototype.forEach.call(opts.children, function (child, k) {
            child.disabled = true;
            if (item.options[k].correct) child.classList.add("correct");
            else if (k === j) child.classList.add("wrong");
          });
          explain.innerHTML =
            (opt.correct ? "<strong>Correct.</strong> " : "<strong>Not this one.</strong> ") + (item.explain || "");
          updateScore();
        });
        opts.appendChild(btn);
      });
      qBox.appendChild(opts);
      qBox.appendChild(explain);
      root.appendChild(qBox);
    });

    function updateScore() {
      var done = states.filter(function (s) { return s.answered; }).length;
      var right = states.filter(function (s) { return s.correct; }).length;
      scoreEl.textContent = done + " / " + spec.items.length + " answered · " + right + " correct";
      if (done === spec.items.length) {
        scoreEl.innerHTML += right === spec.items.length
          ? " — clean sweep. Come back tomorrow and do it without looking."
          : " — re-read the explanation on the misses; they mark exactly where the model is thin.";
      }
    }

    root.appendChild(scoreEl);
    if (spec.reset !== false) {
      var reset = h("button", "btn btn--ghost", "Start over");
      reset.style.marginTop = "1rem";
      reset.addEventListener("click", function () {
        states.forEach(function (s) { s.answered = false; s.correct = false; });
        /* Rebuild by re-mounting into the same host. */
        host.innerHTML = "";
        mount(host, spec);
      });
      root.appendChild(reset);
    }
    host.appendChild(root);
    return root;
  }

  return { mount: mount };
})();
