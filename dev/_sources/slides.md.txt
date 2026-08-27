# Introduction slides

An 8-slide deck introducing phepex for readers new to it: where it sits relative to libdvr
and ctapipe, which kernels it exposes, the v0.2 speed-ups with their measured figures, and
what is planned next.

<div class="phepex-deck" data-base="_static/slides/slide-">
  <img class="phepex-deck-stage" src="_static/slides/slide-1.svg" tabindex="0"
       alt="Slide 1 of 8: title slide" />
  <div class="phepex-deck-bar" hidden>
    <button class="phepex-deck-prev" type="button" aria-label="Previous slide">&larr;</button>
    <span class="phepex-deck-position" role="status" aria-live="polite"></span>
    <button class="phepex-deck-next" type="button" aria-label="Next slide">&rarr;</button>
    <button class="phepex-deck-fullscreen" type="button" aria-label="Present fullscreen">&#x26F6;</button>
  </div>
  <p class="phepex-deck-unavailable" hidden>
    The slides were not rendered for this build (Typst unavailable); see
    <code>docs/slides/README.md</code>.
  </p>
</div>

{{ slides_download }}

<details class="phepex-deck-outline-details">
<summary>Slide outline</summary>
<ol class="phepex-deck-outline">
  <li>phepex — fast DSP and pulse extraction for PMT/SiPM waveforms; C++ kernels, Python bindings, extracted from CTAO libdvr.</li>
  <li>Why phepex? Events carry blocks of many waveforms and DSP plus pulse extraction is often the expensive part; phepex factors the extraction code out of libdvr into a library reusable from C++ and Python.</li>
  <li>Relationship to libdvr: libdvr keeps the event-level data-volume-reduction logic, phepex takes the DSP and pulse-extraction kernels, adds Python bindings, and is now libdvr's extraction engine.</li>
  <li>Relationship to ctapipe: waveforms in (channel × pixel × sample, uint16 or float), signal processing (upsampling, deconvolution, filtering), features out (neighbor summation, pulse time & integral). ctapipe supplies event sources,
      calibration, containers and analysis; phepex the fast low-level waveform operations.</li>
  <li>What is inside: <code>preprocess_waveform(s)</code> for upsampling, deconvolution and smoothing; <code>neighbor_peak_indices</code> and <code>adaptive_centroid</code> for extraction; Python bindings including a <code>FastFlashCamExtractor</code> drop-in for ctapipe.</li>
  <li>v0.2 speed-ups: FlashCam upsampling/deconvolution 980 µs/op (scipy/numpy) → 134 (phepex Python); 147 µs/op (fc-utils in C) → 96 (phepex C++); ctapipe throughput 370 → 1235 events/s; libdvr throughput 1338 → 2175 events/s/thread.</li>
  <li>Plans: end-to-end verification with time and charge resolution curves, new-style FlashCam saturation recovery and time extraction, a PR into ctapipe, support for DL0 zero-suppressed data and per-channel sample-level time offsets for LSTCam.</li>
  <li>Takeaway: the PMT/SiPM extraction core, packaged as a focused C++ library and exposed to Python, with v0.2 speed-ups.</li>
</ol>
</details>
