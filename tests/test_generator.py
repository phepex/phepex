# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Max-Planck-Institut für Kernphysik
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.

"""Validate the C++ waveform generator.

Design goal: physical correctness (not bit-exact reproduction of ctapipe's WaveformModel).
For deposits inside the readout window the two agree to float precision -- that interior
agreement is a useful regression anchor (``test_signal_bit_exact``) -- but at the window
edges the generator is deliberately more physical than WaveformModel: the deposit time is
floor-snapped to the upsampled grid, and a pulse centred just outside the window still
contributes its in-window tail rather than being dropped (see
``test_early_deposit_physical``).
The NSB component must be a stationary Poisson process with the requested mean rate.
"""

import warnings

import numpy as np
import pytest

from phepex import generate_waveforms

pytest.importorskip("ctapipe")
# astropy ships as a ctapipe dependency, so import it only after the guard above.
import astropy.units as u
from ctapipe.image.toymodel import WaveformModel
from ctapipe.instrument import CameraReadout, FromNameWarning

N_SAMPLES = 22
UPSAMPLING = 10


@pytest.fixture(scope="module")
def readout():
    # Bundled FlashCam reference data is intended here; silence the .from_name advisory.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FromNameWarning)
        return CameraReadout.from_name("FlashCam")


def _cpp_args(readout):
    return {
        "reference_pulse": np.ascontiguousarray(
            readout.reference_pulse_shape[0], dtype=np.float64
        ),
        "ref_sample_width_ns": readout.reference_pulse_sample_width.to_value(u.ns),
        "sample_width_ns": (1 / readout.sampling_rate).to_value(u.ns),
        "n_samples": N_SAMPLES,
        "upsampling": UPSAMPLING,
    }


def test_signal_bit_exact(readout):
    """Interior agreement: for mid-window deposits the generator == WaveformModel to
    machine precision. (At the window edges the two intentionally diverge; see
    ``test_early_deposit_physical``.)"""
    wm = WaveformModel.from_camera_readout(readout)
    n_pix = readout.reference_pulse_shape.shape[0]  # placeholder; real n_pix below
    n_pix = 1764
    rng = np.random.default_rng(123)
    charge = rng.uniform(0, 500, n_pix)
    sample_width = (1 / readout.sampling_rate).to_value(u.ns)
    mid = N_SAMPLES // 2 * sample_width
    time_ns = rng.uniform(mid - 8, mid + 8, n_pix)  # kept away from window edges

    ref = wm.get_waveform(charge.copy(), time_ns.copy(), N_SAMPLES)[0]  # (n_pix, n_samp)
    got = generate_waveforms(
        charge[None].copy(),
        time_ns[None].copy(),
        nsb_rate_ghz=0.0,
        seed=0,
        **_cpp_args(readout),
    )[0]

    assert got.shape == (n_pix, N_SAMPLES)
    assert np.max(np.abs(got - ref)) < 1e-5  # float32 output precision


def test_early_deposit_physical(readout):
    """Physical correctness at the window-start edge, where WaveformModel is not.

    (a) Shift-equivariance: shifting a deposit earlier by an integer number of readout
        samples shifts the output by exactly that many samples over the overlap region --
        including when the shift carries the pulse centre to a negative time.
        WaveformModel breaks this by dropping any deposit whose sample index is < 0.
    (b) A pulse centred before the window still contributes its in-window rising tail
        (non-zero output), whereas WaveformModel zeroes it out entirely.
    """
    wm = WaveformModel.from_camera_readout(readout)
    sample_width = (1 / readout.sampling_rate).to_value(u.ns)

    def gen(t):
        return generate_waveforms(
            np.array([[300.0]]),
            np.array([[t]]),
            nsb_rate_ghz=0.0,
            seed=0,
            **_cpp_args(readout),
        )[0, 0]

    # (a) shift a mid-window deposit earlier by m readout samples, past t = 0
    t_in = (N_SAMPLES // 2) * sample_width
    m = N_SAMPLES // 2 + 3
    t_early = t_in - m * sample_width
    assert t_early < 0  # the shifted pulse centre is before the window
    wf_in, wf_early = gen(t_in), gen(t_early)
    # exact integer-sample shift over the overlap region
    assert np.array_equal(wf_early[: N_SAMPLES - m], wf_in[m:])

    # (b) the early pulse still renders an in-window tail; WaveformModel drops it
    wm_early = wm.get_waveform(np.array([300.0]), np.array([t_early]), N_SAMPLES)[0, 0]
    assert wf_early.sum() > 1.0
    assert wm_early.sum() == 0.0


def test_nsb_rate_and_stationarity(readout):
    """NSB integral per pixel matches rate * window; process is stationary in time."""
    n_pix = 1764
    sample_width = (1 / readout.sampling_rate).to_value(u.ns)
    nsb_rate = 0.2  # GHz == 200 MHz
    charge = np.zeros((1, n_pix))
    time_ns = np.zeros((1, n_pix))
    wf = generate_waveforms(
        charge, time_ns, nsb_rate_ghz=nsb_rate, seed=7, **_cpp_args(readout)
    )[0]

    expected = nsb_rate * N_SAMPLES * sample_width  # p.e. per pixel
    integral = wf.sum(axis=1)
    assert np.isclose(integral.mean(), expected, rtol=0.03)
    # stationary: mean per-sample amplitude flat across the trace (edges within 15%)
    per_sample = wf.mean(axis=0)
    assert per_sample.std() / per_sample.mean() < 0.15


def test_nsb_kernel_trim(readout):
    """Guard the NSB-only kernel-tail trim (see src/generate.cpp): NSB deposits use
    only the kernel's significant taps (>=99.99% of its integral) for speed, which
    must (a) NOT touch the signal path and (b) preserve the NSB rate to tolerance."""
    wm = WaveformModel.from_camera_readout(readout)
    n_pix = 1764
    sample_width = (1 / readout.sampling_rate).to_value(u.ns)

    # (a) trim is NSB-only: signal-only output is still bit-exact vs WaveformModel.
    rng = np.random.default_rng(5)
    charge = rng.uniform(0, 500, n_pix)
    mid = N_SAMPLES // 2 * sample_width
    time_ns = rng.uniform(mid - 8, mid + 8, n_pix)
    ref = wm.get_waveform(charge.copy(), time_ns.copy(), N_SAMPLES)[0]
    got = generate_waveforms(
        charge[None].copy(),
        time_ns[None].copy(),
        nsb_rate_ghz=0.0,
        seed=0,
        **_cpp_args(readout),
    )[0]
    assert np.max(np.abs(got - ref)) < 1e-5

    # (b) trim quality: NSB rate preserved to <1% (the dropped tail is <0.01% of
    # each pulse; a grossly-too-aggressive trim would shift the windowed rate).
    nsb_rate = 0.2
    wf = generate_waveforms(
        np.zeros((300, n_pix)),
        np.zeros((300, n_pix)),
        nsb_rate_ghz=nsb_rate,
        seed=11,
        **_cpp_args(readout),
    )
    expected = nsb_rate * N_SAMPLES * sample_width
    assert np.isclose(wf.reshape(-1, N_SAMPLES).sum(1).mean(), expected, rtol=0.01)


def test_reproducible_seed(readout):
    n_pix = 100
    rng = np.random.default_rng(1)
    charge = rng.uniform(50, 500, (4, n_pix))
    time_ns = rng.uniform(30, 55, (4, n_pix))
    a = generate_waveforms(
        charge, time_ns, nsb_rate_ghz=0.2, seed=42, **_cpp_args(readout)
    )
    b = generate_waveforms(
        charge, time_ns, nsb_rate_ghz=0.2, seed=42, **_cpp_args(readout)
    )
    assert np.array_equal(a, b)


if __name__ == "__main__":
    ro = CameraReadout.from_name("FlashCam")
    test_signal_bit_exact(ro)
    print("signal interior agreement: OK")
    test_early_deposit_physical(ro)
    print("early-deposit physical correctness: OK")
    test_nsb_rate_and_stationarity(ro)
    print("nsb rate/stationarity: OK")
    test_nsb_kernel_trim(ro)
    print("nsb kernel-trim guard: OK")
    test_reproducible_seed(ro)
    print("reproducible seed: OK")
    print("all generator validations passed")
