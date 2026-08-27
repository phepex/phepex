// Documentation version switcher for the furo theme.
//
// The container (.phepex-version-switcher) is server-rendered with the `hidden`
// attribute so it occupies no space until this script reveals it. Reveal happens
// only after switcher.json (URL and current version in data- attributes) is
// fetched and yields at least one valid entry, so a JS-disabled page, a failed
// fetch (e.g. a page opened over file://), or a slow connection shows no empty
// box rather than a placeholder.
//
// switcher.json is an array of objects; recognised fields:
//   version : machine id matched against data-current-version (required)
//   name    : label shown in the dropdown (falls back to version)
//   url      : absolute URL navigated to on selection (required)
// Unknown fields are ignored. Order in the file is preserved in the dropdown.

(function () {
  "use strict";

  function reveal(container) {
    // Remove `hidden` (display:none -> block at opacity 0), then flip the class
    // on the next frame so the CSS opacity transition runs. Only opacity is
    // animated; no height/size change, so no reflow beyond the element appearing.
    container.hidden = false;
    window.requestAnimationFrame(function () {
      container.classList.add("is-visible");
    });
  }

  function build(container) {
    var select = container.querySelector(".phepex-version-switcher__select");
    if (!select) {
      return;
    }
    var url = container.getAttribute("data-switcher-url");
    var current = container.getAttribute("data-current-version");
    if (!url) {
      return;
    }

    // Default HTTP caching: switcher.json changes only on deploy, so a cached
    // copy (revalidated per the server's Cache-Control) avoids one request per
    // in-docs navigation. A stale version list until the cache expires is
    // acceptable; the entries it points at do not move.
    fetch(url)
      .then(function (resp) {
        if (!resp.ok) {
          throw new Error("switcher.json HTTP " + resp.status);
        }
        return resp.json();
      })
      .then(function (entries) {
        if (!Array.isArray(entries)) {
          throw new Error("switcher.json is not an array");
        }
        var frag = document.createDocumentFragment();
        var count = 0;
        entries.forEach(function (entry) {
          if (!entry || !entry.url || !entry.version) {
            return;
          }
          var opt = document.createElement("option");
          opt.value = entry.url;
          opt.textContent = entry.name || entry.version;
          if (entry.version === current) {
            opt.selected = true;
          }
          frag.appendChild(opt);
          count += 1;
        });
        if (count === 0) {
          // Nothing to switch to; leave the container hidden.
          return;
        }
        select.appendChild(frag);
        select.addEventListener("change", function () {
          if (select.value) {
            window.location.href = select.value;
          }
        });
        reveal(container);
      })
      .catch(function (err) {
        // Leave the container hidden; do not disrupt the page.
        console.warn("phepex version switcher:", err);
      });
  }

  function init() {
    var containers = document.querySelectorAll(".phepex-version-switcher");
    for (var i = 0; i < containers.length; i++) {
      build(containers[i]);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
