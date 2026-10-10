// Story site behaviour: embed charts, glossary popovers, active nav link.
(function () {
  "use strict";

  // Charts: specs are inlined as <script type="application/json" id="spec-KEY">.
  function embed(el) {
    if (el.dataset.done || !el.offsetParent) return; // skip hidden (closed <details>)
    var node = document.getElementById("spec-" + el.dataset.spec);
    if (!node || typeof vegaEmbed === "undefined") return;
    el.dataset.done = "1";
    vegaEmbed(el, JSON.parse(node.textContent), { actions: false, renderer: "svg" })
      .catch(function (err) { el.textContent = "Chart failed to load: " + err; });
  }
  function embedAll() { document.querySelectorAll(".chart[data-spec]").forEach(embed); }
  // wait for web fonts so Vega measures axis labels with the real font
  (document.fonts ? document.fonts.ready : Promise.resolve()).then(embedAll);
  // charts inside <details> need a visible width, so embed them on open
  document.querySelectorAll("details").forEach(function (d) {
    d.addEventListener("toggle", function () { if (d.open) embedAll(); });
  });

  // Glossary popovers: <dfn class="term" data-term="key"> -> <dt data-key="key"> + <dd>.
  var tip = document.getElementById("tip");
  function show(term) {
    var dt = document.querySelector('dl.glossary dt[data-key="' + term.dataset.term + '"]');
    if (!dt) return;
    tip.innerHTML = "";
    var b = document.createElement("b");
    b.textContent = dt.textContent;
    tip.appendChild(b);
    tip.appendChild(document.createTextNode(dt.nextElementSibling.textContent));
    tip.style.display = "block";
    var r = term.getBoundingClientRect();
    var left = Math.min(window.scrollX + r.left, window.scrollX + document.documentElement.clientWidth - tip.offsetWidth - 16);
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = window.scrollY + r.bottom + 8 + "px";
  }
  function hide() { tip.style.display = "none"; }
  document.querySelectorAll("dfn.term").forEach(function (t) {
    t.addEventListener("mouseenter", function () { show(t); });
    t.addEventListener("focus", function () { show(t); });
    t.addEventListener("mouseleave", hide);
    t.addEventListener("blur", hide);
  });

  // Highlight the nav link of the section in view.
  var links = {};
  document.querySelectorAll(".nav-links a[href^='#']").forEach(function (a) {
    links[a.getAttribute("href").slice(1)] = a;
  });
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting || !links[e.target.id]) return;
        Object.values(links).forEach(function (a) { a.classList.remove("active"); });
        links[e.target.id].classList.add("active");
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    document.querySelectorAll("section[id]").forEach(function (s) { io.observe(s); });
  }
})();
