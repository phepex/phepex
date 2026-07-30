// phepex-intro.typ
// Minimalist Typst slide deck introducing phepex
// Compile with:
//   typst compile phepex-intro.typ phepex-intro.pdf

#set document(
  title: "phepex intro",
  author: "Felix Werner (MPIK)",
)

#set page(
  width: 16in,
  height: 9in,
  margin: (
    left: 0.85in,
    right: 0.85in,
    top: 0.60in,
    bottom: 0.45in,
  ),
  fill: white,
)

#let ink = rgb("#120b28")
#let muted = rgb("#615c70")
#let faint = rgb("#ded7f4")
#let card-bg = rgb("#f8f7fd")
#let accent = rgb("#9c86de")

#set text(
  font: "Fira Sans",
  size: 20pt,
  fill: ink,
)

#show raw: set text(font: "Fira Mono", weight: "medium")
#show raw.where(block: false): set text(size: 1.1em)

#set par(
  leading: 0.68em,
)

// Explicit markers: Typst's level-2 default is U+2023 (‣), which Fira Sans lacks, so it
// would fall back to a serif face for that glyph alone on the nested-list slides.
#set list(
  spacing: 0.6em,
  marker: ([•], [–]),
)

#set enum(
  spacing: 0.36em,
)

#let repo = "github.com/phepex/phepex"
#let footer-text = "F. Werner (MPIK) · " + repo

#let footer(txt) = [
  #v(1fr)
  #line(length: 100%, stroke: 0.7pt + faint)
  #v(0.08in)
  #text(size: 11pt, fill: muted)[#txt]
]

#let slide(title, subtitle, body) = [
  #block(width: 100%, height: 100%)[
    #v(0.05in)
    #text(size: 40pt, weight: "bold")[#title]

    #if subtitle != none [
      #v(0.08in)
      #text(size: 18pt, fill: muted)[#subtitle]
    ]

    #v(0.38in)
    #text(size: 21pt)[#body]

    #footer(footer-text)
  ]
]

#let title-slide() = [
  #block(width: 100%, height: 100%)[
    #v(2.15in)
    //#text(size: 78pt, weight: "bold")[phepex]
    #image("phepex-logo-light.svg", height: 150pt)
    #v(0.16in)
    #text(size: 30pt, fill: ink)[
      Fast DSP and pulse extraction for PMT/SiPM waveforms
    ]
    #v(0.42in)
    #text(size: 21pt, fill: accent)[
      C++ kernels · Python bindings · Extracted from CTAO libdvr
    ]
    #footer(footer-text)
  ]
]

#let card(body, title: none, center-title: none) = rect(
  width: 100%,
  radius: 8pt,
  inset: 17pt,
  fill: card-bg,
  stroke: 0.6pt + faint,
)[
  #if title != none [
    #text(size: 17pt, weight: "bold", fill: accent)[#title]
    #v(0.12in)
  ]
  #if center-title != none [
    #align(center, [
      #text(size: 17pt, weight: "bold", fill: accent)[#center-title]
    ])
    #v(0.12in)

  ]
  #text(size: 17pt)[#body]
]

#let large-card(body, title: none) = rect(
  width: 100%,
  radius: 8pt,
  inset: 17pt,
  fill: card-bg,
  stroke: 0.6pt + faint,
)[
  #if title != none [
    #text(size: 23pt, weight: "bold", fill: accent)[#title]
    #v(0.12in)
  ]
  #text(size: 23pt)[#body]
]

#let small-card(title, body) = rect(
  width: 100%,
  radius: 8pt,
  inset: 14pt,
  fill: card-bg,
  stroke: 0.6pt + faint,
)[
  #text(size: 15pt, weight: "bold", fill: accent)[#title]
  #v(0.08in)
  #text(size: 14pt, fill: muted)[#body]
]

#title-slide()

#pagebreak()

#slide(
  "Why phepex?",
  "A small package for a very common waveform-processing problem",
  [
    - CTA-style events contain blocks of many waveforms.
    - For many workflows, the expensive part is DSP and pulse extraction.
    - `libdvr` already contained optimized extraction code.
    - `phepex` factors that functionality out into a focused library.
    - The result: easier reuse from both C++ and Python.

    #v(0.28in)

    #rect(
      width: 100%,
      radius: 8pt,
      inset: 18pt,
      fill: card-bg,
      stroke: 0.6pt + rgb("#d9e7f5"),
    )[
      #text(size: 18pt, fill: accent, weight: "bold")[
        For ctapipe users:
      ]
      #text(size: 18pt)[
        phepex focuses on low-level waveform and pulse extraction, rather than full event processing.
      ]
    ]
  ],
)

#pagebreak()

#slide(
  "Relationship to libdvr",
  "phepex is the reusable DSP and pulse-extraction core",
  [
    #grid(
      columns: (1.2fr, 1.0fr),
      gutter: 0.42in,
      [
        #card(
          title: "libdvr",
          [
            - CTAO data volume reduction library.
            - Contains event-level DVR logic.
            - Originally included DSP and pulse-extraction functionality.
            - Optimized for production-style reduction workflows.
          ],
        )
      ],
      [
        #card(
          title: "phepex",
          [
            - Splits out the DSP and pulse-extraction pieces.
            - Keeps them independent and easier to test.
            - Adds Python bindings.
            - v0.2 brings significant speed-ups.
          ],
        )
      ],
    )

    #v(0.38in)

    #text(size: 23pt, fill: accent, weight: "bold")[
      phepex is now the extraction engine of libdvr.
    ]
  ],
)

#pagebreak()

#slide(
  "Relationship to ctapipe",
  "Designed to fit the mental model ctapipe users already have",
  [
    #grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      gutter: 0.18in,
      [
        #small-card("Waveforms in", "channel × pixel × sample arrays (uint16 or float)")
      ],
      [
        #small-card("Signal processing", "upsampling, deconvolution, filtering")
      ],
      [
        #small-card("Features out", "neighbor summation, pulse time & integral")
      ],
      [
        #small-card("Downstream use", "cleaning, reconstruction, monitoring, ...")
      ],
    )

    #v(0.55in)

    #grid(
      columns: (1fr, 1fr),
      gutter: 0.42in,
      [
        #card(
          title: "ctapipe provides",
          [
            - Event sources.
            - Calibration machinery.
            - Containers and pipeline structure.
            - Higher-level analysis components.
          ],
        )
      ],
      [
        #card(
          title: "phepex focuses on",
          [
            - Fast low-level waveform operations.
            - Pulse extraction algorithms.
            - C++ performance with Python access.
            - Reusable kernels that can be embedded elsewhere.
          ],
        )
      ],
    )
  ],
)

#pagebreak()

#slide(
  "What is inside?",
  "A focused set of DSP and extraction building blocks",
  [
    - C++ implementation of waveform-processing kernels.
      - `preprocess_waveform(s)` – upsampling, deconvolution, smoothing.
    - Pulse extraction routines derived from the libdvr work.
      - `neighbor_peak_indices`, `adaptive_centroid`.
    - Python bindings for interactive, analysis, and pipeline use.
      - Plus a `FastFlashCamExtractor` drop-in for ctapipe.
    - Minimal dependency surface; straightforward to embed.
    - Compact library boundary: no full event model required.
    - Useful for benchmarking extraction choices outside a full pipeline.

    #v(0.42in)

    #large-card(
      title: "Package goal",
      [ Make the reusable part of waveform processing fast, and easy to call, test and compare. ],
    )
  ],
)

#pagebreak()

#slide(
  "v0.2: speed-ups",
  "Beyond factoring out and wrapping the libdvr kernels",
  [
    - Faster waveform processing and pulse extraction for waveform blocks.
      - SIMD-friendly tiled processing + CPU pipeline saturation.
    - Zero-copy Python interfaces for both uint16 (R1/DL0) and float32 (ctapipe) data.
    - Lower barrier to benchmarking extraction variants.

    #v(0.35in)

    #grid(
      columns: (0.95fr, 1fr, 0.95fr),
      gutter: 0.22in,
      [
        #card(
          center-title: "FlashCam upsampling/deconvolution",
          [
            #align(center, [
              #table(
                columns: (auto, auto),
                align: (right, right),
                stroke: none,
                inset: (x: 6pt, y: 3pt),

                [scipy/numpy], [980 µs/op],
                [phepex (Python)], [134 µs/op],
                [fc-utils (C)], [147 µs/op],
                [phepex (C++)], [96 µs/op],
              )

              #text("+630% (Python) / +53% (C++)", size: 20pt, weight: "bold", fill: accent)

              `phepex/benchmarks/cpp/microbench.cpp`
            ])
          ],
        )
      ],
      [
        #card(
          center-title: "ctapipe throughput",
          [
            #align(center, [
              #table(
                columns: (auto, auto),
                align: (right, right),
                stroke: none,
                inset: (x: 6pt, y: 3pt),

                [ctapipe `FlashCamExtractor`], [370 ev/s],
                [`FastFlashCamExtractor`], [1235 ev/s],
                [ ], [],
                [ ], [],
              )

              #text("+233%", size: 20pt, weight: "bold", fill: accent)

              `phepex/benchmarks/benchmark-fast-extractor.py`
            ])
          ],
        )
      ],
      [
        #card(
          center-title: "libdvr throughput (incl. smoothing)",
          [
            #align(center, [
              #table(
                columns: (auto, auto),
                align: (right, right),
                stroke: none,
                inset: (x: 6pt, y: 3pt),

                [before refactor], [1338 ev/s/thread],
                [with phepex 0.2], [2175 ev/s/thread],
                [ ], [],
                [ ], [],
              )

            #text("+62%", size: 20pt, weight: "bold", fill: accent)

            `libdvr/cmd/dvr-benchmark/main.cpp`
            ])
          ],
        )
      ],
    )
  ],
)

#pagebreak()

#slide(
  "Outlook",
  "Testing/integration + plans",
  [
    - Verify `FastFlashCamExtractor` with end-to-end time/charge resolution curves.
    - Integrate 'new' FlashCam saturation recovery & time extraction algorithms.
    - Create PR into ctapipe.
      - All cameras will benefit from the drop-in routines.
    - Tuning for DL0 (zero-suppressed) data model.
    - Support for sample-level per-channel time offsets (LSTCam).
  ],
)

#pagebreak()

#slide(
  "Takeaway",
  "phepex makes the PMT/SiPM signal extraction core reusable",
  [
    - Extracted from libdvr’s DSP and pulse-extraction functionality.
    - Packaged as a focused C++ library.
    - Exposed to Python for ctapipe-style analysis workflows.
    - v0.2 adds important speed-ups.
    - Useful wherever waveform extraction should be fast, testable, and independent.

    #v(1fr)

    #large-card(
      title: "Resources",
      [
        - Repository: #link("https://github.com/phepex/phepex")[github.com.phepex/phepex]
        - Documentation: #link("https://phepex.github.io/phepex")[phepex.github.io/phepex]
      ],
    )
  ],
)
