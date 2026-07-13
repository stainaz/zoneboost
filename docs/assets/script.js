// zoneboost docs site — shared behavior (mobile nav, docs sidebar, copy buttons, docs search)

(function () {
  "use strict";

  function initMainNavToggle() {
    var toggle = document.querySelector(".nav-toggle");
    var nav = document.querySelector(".main-nav");
    if (!toggle || !nav) return;
    toggle.addEventListener("click", function () {
      var open = nav.classList.toggle("open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  function initSidebarToggle() {
    var toggle = document.querySelector(".sidebar-toggle");
    var sidebar = document.querySelector(".docs-sidebar");
    if (!toggle || !sidebar) return;
    toggle.addEventListener("click", function () {
      var open = sidebar.classList.toggle("open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  function flashCopied(btn, label) {
    var original = btn.textContent;
    btn.textContent = label || "Copied";
    btn.disabled = true;
    setTimeout(function () {
      btn.textContent = original;
      btn.disabled = false;
    }, 1400);
  }

  function copyText(text, btn) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () {
        flashCopied(btn);
      });
    } else {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand("copy"); } catch (e) { /* no-op */ }
      document.body.removeChild(ta);
      flashCopied(btn);
    }
  }

  function initInlineCopyButtons() {
    document.querySelectorAll("[data-copy]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        copyText(btn.getAttribute("data-copy"), btn);
      });
    });
  }

  function initCodeBlockCopyButtons() {
    document.querySelectorAll("pre > code").forEach(function (codeEl) {
      var pre = codeEl.parentElement;
      if (pre.querySelector(".copy-btn")) return;
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "copy-btn";
      btn.textContent = "Copy";
      btn.addEventListener("click", function () {
        copyText(codeEl.textContent, btn);
      });
      pre.appendChild(btn);
    });
  }

  // ---- docs search ----
  // Small hand-maintained index: one entry per page/section heading.
  // "crumb" is the parent doc page title, shown under each result.
  var SEARCH_INDEX = [
    { title: "Getting Started", crumb: "Docs", url: "getting-started.html" },
    { title: "Installation", crumb: "Getting Started", url: "getting-started.html#installation" },
    { title: "Quickstart", crumb: "Getting Started", url: "getting-started.html#quickstart" },
    { title: "Regression", crumb: "Getting Started › Quickstart", url: "getting-started.html#regression" },
    { title: "Classification", crumb: "Getting Started › Quickstart", url: "getting-started.html#classification" },
    { title: "Requirements", crumb: "Getting Started", url: "getting-started.html#requirements" },
    { title: "Development", crumb: "Getting Started", url: "getting-started.html#development" },

    { title: "How It Works", crumb: "Docs", url: "how-it-works.html" },
    { title: "The Weak Learner", crumb: "How It Works", url: "how-it-works.html#weak-learner" },
    { title: "Main Effects", crumb: "How It Works › Weak Learner", url: "how-it-works.html#main-effects" },
    { title: "Interactions", crumb: "How It Works › Weak Learner", url: "how-it-works.html#interactions" },
    { title: "Empirical Bayes Shrinkage (overview)", crumb: "How It Works › Weak Learner", url: "how-it-works.html#density-confidence" },
    { title: "Continuous vs. Categorical Zones", crumb: "How It Works", url: "how-it-works.html#continuous-vs-categorical" },
    { title: "Missing Values", crumb: "How It Works", url: "how-it-works.html#missing-values" },
    { title: "From Regression to Classification", crumb: "How It Works", url: "how-it-works.html#classification" },
    { title: "Adaptive Interaction Order", crumb: "How It Works", url: "how-it-works.html#adaptive-interaction-order" },
    { title: "Cross-Fitted Cell Means", crumb: "How It Works", url: "how-it-works.html#cross-fitted-cell-means" },
    { title: "Empirical Bayes Shrinkage", crumb: "How It Works", url: "how-it-works.html#empirical-bayes-shrinkage" },
    { title: "Lasso Stacking", crumb: "How It Works", url: "how-it-works.html#lasso-stacking" },
    { title: "Soft Zone Boundaries", crumb: "How It Works", url: "how-it-works.html#soft-zone-boundaries" },
    { title: "Cyclic Backfitting", crumb: "How It Works", url: "how-it-works.html#cyclic-backfitting" },
    { title: "Monotonic Constraints", crumb: "How It Works", url: "how-it-works.html#monotonic-constraints" },
    { title: "Pair Screening", crumb: "How It Works", url: "how-it-works.html#pair-screening" },
    { title: "Hierarchical Zones (Grouped Data)", crumb: "How It Works", url: "how-it-works.html#hierarchical-zones" },
    { title: "Native Multinomial Boosting", crumb: "How It Works", url: "how-it-works.html#native-multinomial-boosting" },
    { title: "Prediction Intervals (Regressor)", crumb: "How It Works", url: "how-it-works.html#prediction-intervals" },
    { title: "Probability Calibration (Classifier)", crumb: "How It Works", url: "how-it-works.html#probability-calibration" },
    { title: "Honest Data Splits (Calibration & Final Refit)", crumb: "How It Works", url: "how-it-works.html#honest-data-splits" },
    { title: "Adaptive Boundary Continuity", crumb: "How It Works", url: "how-it-works.html#adaptive-boundary-continuity" },
    { title: "Quantile Regression", crumb: "How It Works", url: "how-it-works.html#quantile-regression" },
    { title: "Actuarial Losses (Poisson, Gamma, Tweedie)", crumb: "How It Works", url: "how-it-works.html#actuarial-losses" },
    { title: "Conformalized Quantile Regression (CQR)", crumb: "How It Works", url: "how-it-works.html#conformalized-quantile-regression" },
    { title: "Global Shape Constraints", crumb: "How It Works", url: "how-it-works.html#global-shape-constraints" },
    { title: "How It Compares", crumb: "How It Works", url: "how-it-works.html#how-it-compares" },

    { title: "API Reference", crumb: "Docs", url: "api-reference.html" },
    { title: "Parameters", crumb: "API Reference", url: "api-reference.html#parameters" },
    { title: "ZoneBoostRegressor Fitted Attributes", crumb: "API Reference", url: "api-reference.html#regressor-attributes" },
    { title: "ZoneBoostClassifier Fitted Attributes", crumb: "API Reference", url: "api-reference.html#classifier-attributes" },
    { title: "ConformalizedQuantileRegressor Parameters", crumb: "API Reference", url: "api-reference.html#cqr-parameters" },
    { title: "BootstrapStability Parameters", crumb: "API Reference", url: "api-reference.html#bootstrap-parameters" },
    { title: "compare_models Signature", crumb: "API Reference", url: "api-reference.html#compare-models-signature" },
    { title: "evidence_card Signature", crumb: "API Reference", url: "api-reference.html#evidence-card-signature" },
    { title: "Scope & Compatibility", crumb: "API Reference", url: "api-reference.html#scope" },

    { title: "Explaining Predictions", crumb: "Docs", url: "explaining-predictions.html" },
    { title: "explain(X)", crumb: "Explaining Predictions", url: "explaining-predictions.html#explain" },
    { title: "feature_importance(X)", crumb: "Explaining Predictions", url: "explaining-predictions.html#feature-importance" },
    { title: "Classification Semantics", crumb: "Explaining Predictions", url: "explaining-predictions.html#classification-semantics" },
    { title: "Explanation Reliability", crumb: "Explaining Predictions", url: "explaining-predictions.html#explanation-reliability" },
    { title: "Bootstrap Stability", crumb: "Explaining Predictions", url: "explaining-predictions.html#bootstrap-stability" },
    { title: "Evidence Report", crumb: "Explaining Predictions", url: "explaining-predictions.html#evidence-report" },
    { title: "Audited Human Editing", crumb: "Explaining Predictions", url: "explaining-predictions.html#audited-human-editing" },
    { title: "Zone-Native Counterfactuals", crumb: "Explaining Predictions", url: "explaining-predictions.html#zone-native-counterfactuals" },
    { title: "Time-Based Drift Comparison", crumb: "Explaining Predictions", url: "explaining-predictions.html#time-based-drift-comparison" },
    { title: "Model Evidence Cards", crumb: "Explaining Predictions", url: "explaining-predictions.html#model-evidence-cards" },
    { title: "How This Differs from SHAP/LIME", crumb: "Explaining Predictions", url: "explaining-predictions.html#vs-shap-lime" },

    { title: "Benchmarks", crumb: "Docs", url: "benchmarks.html" },
    { title: "Methodology", crumb: "Benchmarks", url: "benchmarks.html#methodology" },
    { title: "Regression: California Housing", crumb: "Benchmarks", url: "benchmarks.html#regression" },
    { title: "Reproducing", crumb: "Benchmarks", url: "benchmarks.html#reproduce" }
  ];

  function initDocsSearch() {
    var wrap = document.querySelector(".docs-search");
    if (!wrap) return;
    var input = wrap.querySelector("input");
    var results = wrap.querySelector(".search-results");
    if (!input || !results) return;

    function render(items) {
      if (!items.length) {
        results.innerHTML = '<div class="no-results">No matches</div>';
        results.classList.add("open");
        return;
      }
      results.innerHTML = items.map(function (item) {
        return '<a href="' + item.url + '">' +
          '<div class="result-title"></div>' +
          '<div class="result-crumb"></div>' +
          '</a>';
      }).join("");
      // set text content safely (avoid injecting query text as HTML)
      var links = results.querySelectorAll("a");
      items.forEach(function (item, i) {
        links[i].querySelector(".result-title").textContent = item.title;
        links[i].querySelector(".result-crumb").textContent = item.crumb;
      });
      results.classList.add("open");
    }

    function search(query) {
      var q = query.trim().toLowerCase();
      if (!q) {
        results.classList.remove("open");
        results.innerHTML = "";
        return;
      }
      var matches = SEARCH_INDEX.filter(function (item) {
        return item.title.toLowerCase().indexOf(q) !== -1 ||
          item.crumb.toLowerCase().indexOf(q) !== -1;
      }).slice(0, 8);
      render(matches);
    }

    input.addEventListener("input", function () {
      search(input.value);
    });
    input.addEventListener("keydown", function (e) {
      if (e.key === "Escape") {
        input.value = "";
        results.classList.remove("open");
      }
    });
    document.addEventListener("click", function (e) {
      if (!wrap.contains(e.target)) {
        results.classList.remove("open");
      }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    initMainNavToggle();
    initSidebarToggle();
    initInlineCopyButtons();
    initCodeBlockCopyButtons();
    initDocsSearch();
  });
})();
