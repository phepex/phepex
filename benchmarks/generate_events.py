# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Max-Planck-Institut für Kernphysik
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.

"""Generate synthetic gamma-like FlashCam-MST events for benchmarking.

Physics (shower charge image + arrival-time gradient) is produced with ctapipe's
validated toy models; the perf-critical waveform synthesis (pulse convolution + NSB)
is delegated to the C++ ``phepex.generate_waveforms``.
"""

from __future__ import annotations

import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from ctapipe.image.toymodel import SkewedGaussian, obtain_time_image
from ctapipe.instrument import (
    CameraDescription,
    CameraGeometry,
    CameraReadout,
    FromNameWarning,
    OpticsDescription,
    SubarrayDescription,
    TelescopeDescription,
)
from ctapipe.instrument.optics import ReflectorShape, SizeType

from phepex import generate_waveforms


def build_flashcam_mst_subarray(n_samples: int = 22) -> SubarrayDescription:
    """One-telescope FlashCam-MST subarray (1764 px). Optics are nominal MST values
    (unused by the extractor, which only needs the readout + geometry)."""
    # Bundled FlashCam reference data is exactly what we want for this synthetic
    # benchmark, so silence ctapipe's advisory that .from_name may differ from real data.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FromNameWarning)
        geometry = CameraGeometry.from_name("FlashCam")
        readout = CameraReadout.from_name("FlashCam")
    readout.n_samples = n_samples
    camera = CameraDescription(name="FlashCam", geometry=geometry, readout=readout)
    optics = OpticsDescription(
        name="MST",
        size_type=SizeType.MST,
        n_mirrors=1,
        equivalent_focal_length=16.0 * u.m,
        effective_focal_length=16.445 * u.m,
        mirror_area=88.0 * u.m**2,
        n_mirror_tiles=90,
        reflector_shape=ReflectorShape.DAVIES_COTTON,
    )
    telescope = TelescopeDescription(name="MST_FlashCam", optics=optics, camera=camera)
    return SubarrayDescription(
        "FlashCam-MST-benchmark",
        tel_positions={1: [0, 0, 0] * u.m},
        tel_descriptions={1: telescope},
        reference_location=EarthLocation.from_geocentric(0, 0, 0, unit=u.m),
    )


def _draw_shower_images(subarray, n_events, rng):
    """Draw ``n_events`` unique gamma-like (charge, peak_time) images.

    Returns two (n_events, n_pix) float64 arrays: photo-electron charge per pixel and
    the pulse peak time in ns.  Shower parameters (position, size, orientation,
    skewness, intensity, time gradient) are randomised per event so no two are alike.
    """
    geom = subarray.tel[1].camera.geometry
    readout = subarray.tel[1].camera.readout
    n_pix = geom.n_pixels
    n_samples = readout.n_samples
    sample_width = (1 / readout.sampling_rate).to_value(u.ns)
    unit = geom.pix_x.unit
    radius = float(np.hypot(geom.pix_x.to_value(unit), geom.pix_y.to_value(unit)).max())
    window_mid = n_samples // 2 * sample_width  # centre of the readout window (ns)

    charge = np.empty((n_events, n_pix), dtype=np.float64)
    time_ns = np.empty((n_events, n_pix), dtype=np.float64)

    # Vectorised parameter draws (reproducible via rng).
    r = radius * 0.7 * np.sqrt(rng.uniform(0, 1, n_events))
    phi = rng.uniform(0, 2 * np.pi, n_events)
    cx = r * np.cos(phi)
    cy = r * np.sin(phi)
    length = rng.uniform(0.03, 0.20, n_events)
    width = length * rng.uniform(0.25, 0.6, n_events)
    psi = rng.uniform(0, 2 * np.pi, n_events)
    skew = rng.uniform(0.1, 0.6, n_events)
    intensity = 10 ** rng.uniform(2.0, 3.7, n_events)  # ~100 .. 5000 p.e.
    time_grad = rng.uniform(-5.0, 5.0, n_events)  # ns / m
    time_int = window_mid + rng.uniform(-4.0, 4.0, n_events)

    for i in range(n_events):
        model = SkewedGaussian(
            x=u.Quantity(cx[i], unit),
            y=u.Quantity(cy[i], unit),
            length=u.Quantity(length[i], unit),
            width=u.Quantity(width[i], unit),
            psi=u.Quantity(psi[i], u.rad),
            skewness=skew[i],
        )
        image, _, _ = model.generate_image(
            geom, intensity=intensity[i], nsb_level_pe=0, rng=rng
        )
        charge[i] = image
        time_ns[i] = obtain_time_image(
            geom.pix_x,
            geom.pix_y,
            u.Quantity(cx[i], unit),
            u.Quantity(cy[i], unit),
            u.Quantity(psi[i], u.rad),
            u.Quantity(time_grad[i], u.ns / unit),
            u.Quantity(time_int[i], u.ns),
        )
    return charge, time_ns


def generate_events(subarray, n_events=10000, nsb_rate_ghz=0.2, seed=0):
    """Generate ``n_events`` gamma-like waveforms held in memory.

    Returns
    -------
    waveforms : float32 ndarray, shape (n_events, 1, n_pixels, n_samples)
        Per-event waveforms (single gain channel), signal + NSB.
    charge, time_ns : float64 ndarrays, shape (n_events, n_pixels)
        The true injected photo-electron charge and peak time (for accuracy checks).
    """
    rng = np.random.default_rng(seed)
    readout = subarray.tel[1].camera.readout
    charge, time_ns = _draw_shower_images(subarray, n_events, rng)

    waveforms = generate_waveforms(
        charge,
        time_ns,
        reference_pulse=np.ascontiguousarray(
            readout.reference_pulse_shape[0], dtype=np.float64
        ),
        ref_sample_width_ns=readout.reference_pulse_sample_width.to_value(u.ns),
        sample_width_ns=(1 / readout.sampling_rate).to_value(u.ns),
        n_samples=readout.n_samples,
        upsampling=10,
        nsb_rate_ghz=nsb_rate_ghz,
        seed=seed + 1,
    )
    # Add the single gain-channel axis expected by ImageExtractor (no copy).
    waveforms = waveforms[:, np.newaxis, :, :]
    return waveforms, charge, time_ns


if __name__ == "__main__":
    import time

    sub = build_flashcam_mst_subarray()
    for n in (100,):
        t0 = time.perf_counter()
        wf, q, t = generate_events(sub, n_events=n, seed=1)
        dt = time.perf_counter() - t0
        print(
            f"{n} events: {dt:.2f}s ({dt / n * 1000:.1f} ms/event) "
            f"-> 10000 ~ {dt / n * 10000:.0f}s"
        )
        print(f"  waveforms {wf.shape} {wf.dtype} {wf.nbytes / 1e6:.1f} MB")
        print(
            f"  mean lit pixels/event: {(q > 0.5).sum(1).mean():.0f}, "
            f"median total charge: {np.median(q.sum(1)):.0f} pe"
        )
