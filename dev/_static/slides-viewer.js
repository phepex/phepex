// In-page viewer for the Typst slide deck rendered on docs/slides.md.
//
// Each slide is one SVG file swapped into a single <img>, rather than eight SVGs inlined in
// the page: Typst names every glyph <symbol id="..."> and those ids recur across slides, so
// inlining would put duplicate ids in one document (see docs/conf.py). The trade-off is that
// SVG text is glyph outlines, hence the authored outline list the page carries as the
// accessible and searchable text alternative; alt text is taken from it here.
//
// Loaded on every page, so it must do nothing when there is no deck.
function phepexInitDeck() {
  "use strict";

  const deck = document.querySelector(".phepex-deck");
  if (!deck) return;

  const stage = deck.querySelector(".phepex-deck-stage");
  const bar = deck.querySelector(".phepex-deck-bar");
  const position = deck.querySelector(".phepex-deck-position");
  const note = deck.querySelector(".phepex-deck-unavailable");
  const base = deck.dataset.base || "";

  // One entry per slide, so the outline that supplies the alt text also fixes the slide
  // count -- the two cannot drift apart here. docs/conf.py checks the list against the
  // exported deck at build time.
  const titles = Array.from(document.querySelectorAll(".phepex-deck-outline li"), (li) =>
    (li.textContent || "").trim().replace(/\s+/g, " "),
  );
  const count = titles.length;

  function unavailable() {
    if (stage) stage.hidden = true;
    if (bar) bar.hidden = true;
    if (note) note.hidden = false;
  }

  // A docs build without Typst exports no SVGs, and nothing in the page says so: the markup
  // is static and MyST does not substitute into it. Take the first image failing to load as
  // the signal, which also covers a partial export.
  if (count < 1 || !stage) return unavailable();
  stage.addEventListener("error", unavailable);

  const src = (i) => `${base}${i + 1}.svg`;
  let current = -1;

  function show(index, focus) {
    const i = ((index % count) + count) % count;
    if (i === current) return;
    current = i;
    stage.src = src(i);
    const title = titles[i];
    stage.alt = title
      ? `Slide ${i + 1} of ${count}: ${title}`
      : `Slide ${i + 1} of ${count}`;
    if (position) position.textContent = `${i + 1} / ${count}`;
    // replaceState, not pushState: paging through the deck must not fill the back button.
    history.replaceState(null, "", `#slide-${i + 1}`);
    if (focus) stage.focus();
    // Warm the next slide so a click or arrow key paints without a fetch.
    if (count > 1) new Image().src = src((i + 1) % count);
  }

  stage.addEventListener("click", () => show(current + 1, false));

  stage.addEventListener("keydown", (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    const step = { ArrowRight: 1, ArrowLeft: -1, PageDown: 1, PageUp: -1 }[event.key];
    if (step !== undefined) show(current + step, false);
    else if (event.key === "Home") show(0, false);
    else if (event.key === "End") show(count - 1, false);
    else return; // leave every other key to the page (furo binds "/" to search)
    event.preventDefault();
  });

  deck.querySelector(".phepex-deck-prev")?.addEventListener("click", () => {
    show(current - 1, true);
  });
  deck.querySelector(".phepex-deck-next")?.addEventListener("click", () => {
    show(current + 1, true);
  });
  deck.querySelector(".phepex-deck-fullscreen")?.addEventListener("click", () => {
    if (document.fullscreenElement) document.exitFullscreen?.();
    else stage.requestFullscreen?.();
  });

  // Deep link: #slide-3 opens on that slide, and the hash stays shareable.
  const fromHash = () => {
    const match = /^#slide-(\d+)$/.exec(location.hash);
    return match ? parseInt(match[1], 10) - 1 : 0;
  };
  window.addEventListener("hashchange", () => show(fromHash(), false));

  if (bar) bar.hidden = false;
  deck.dataset.ready = "1";
  show(fromHash(), false);
}

// furo emits html_js_files at the end of <body>, so the deck is normally parsed by now; the
// guard matches version-switcher.js and keeps this correct if that ever changes.
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", phepexInitDeck);
} else {
  phepexInitDeck();
}
