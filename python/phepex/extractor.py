# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 Max-Planck-Institut für Kernphysik
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.

"""ctapipe-integrated FlashCam extractor built on the phepex C++ kernels.

``FastFlashCamExtractor`` is ctapipe's ``FlashCamExtractor`` with the pole-zero
deconvolution, neighbour-sum clipping and neighbour peak search, window integration and
leading-edge timing all done in C++ (via ``phepex``). Parameter estimation
(``_get_deconvolution_parameters``) and the traitlet config come from the ctapipe base
class; the clip + peak search run over the valid (non-edge) deconvolution samples.

This is the only phepex module that imports ctapipe.
"""

from __future__ import annotations

import numpy as np
from ctapipe.containers import DL1CameraContainer
from ctapipe.image.extractor import FlashCamExtractor

from . import _core
from .kernels import (
    adaptive_centroid,
    deconvolve,
    deconvolve_valid_range,
    extract_around_peak,
    neighbor_peak_indices,
)

__all__ = ["FastFlashCamExtractor"]


class FastFlashCamExtractor(FlashCamExtractor):
    """FlashCamExtractor with the numeric pipeline done in C++ (phepex kernels)."""

    def __call__(
        self, waveforms, tel_id, selected_gain_channel, broken_pixels
    ) -> DL1CameraContainer:
        upsampling = self.upsampling.tel[tel_id]
        integration_window_width = self.window_width.tel[tel_id]
        integration_window_shift = self.window_shift.tel[tel_id]
        neighbour_sum_clipping = self.neighbour_sum_clipping.tel[tel_id]
        leading_edge_timing = self.leading_edge_timing.tel[tel_id]
        leading_edge_rel_descend_limit = self.leading_edge_rel_descend_limit.tel[tel_id]

        pole_zeros, gains, shifts, pz2ds = self._get_deconvolution_parameters(tel_id)
        pz, gain, shift, pz2d = pole_zeros[0], gains[0], shifts[0], pz2ds[0]

        neighbors = self.subarray.tel[tel_id].camera.geometry.neighbor_matrix_sparse
        local_weight = self.local_weight.tel[tel_id]
        clip_off = neighbour_sum_clipping == 0.0 or np.isinf(neighbour_sum_clipping)

        # C++: deconvolution, then clip + peak search over the valid deconvolution
        # samples.
        #
        # Note: unlike ctapipe's FlashCamExtractor, which clips and argmaxes the full
        # trace, the clip and neighbour peak search here run only over [lo, hi) -- the
        # non-edge deconvolution samples. The 2*(upsampling-1) samples at each end (and
        # the extra 3*upsampling-2 at the start when pole_zero != 0) are contaminated by
        # the upsample+filtfilt boxcar, so they are excluded on purpose. This is a
        # deliberate divergence from FlashCamExtractor for peaks in those edge regions.
        t_waveforms = deconvolve(waveforms, 0.0, upsampling, pz)
        lo, hi = deconvolve_valid_range(upsampling, waveforms.shape[-1], pz)
        nn = (
            t_waveforms
            if clip_off
            else _core.pos_soft_clip(t_waveforms, float(neighbour_sum_clipping), lo, hi)
        )
        peak_index = neighbor_peak_indices(
            nn, neighbors, local_weight, broken_pixels, lo, hi
        )

        charge, peak_time = extract_around_peak(
            t_waveforms,
            peak_index,
            integration_window_width,
            integration_window_shift,
            self.sampling_rate_ghz[tel_id] * upsampling,
        )

        if leading_edge_timing:
            d_waveforms = deconvolve(waveforms, 0.0, upsampling, 1)
            peak_index = np.round(peak_index - pz2d).astype(int)
            n_samples = d_waveforms.shape[-1]
            np.clip(peak_index, 0, n_samples - 1, out=peak_index)
            peak_time = adaptive_centroid(
                d_waveforms, peak_index, leading_edge_rel_descend_limit
            )
            peak_time /= self.sampling_rate_ghz[tel_id] * upsampling

        if gain != 0:
            charge /= gain
        if shift != 0:
            peak_time -= shift

        if selected_gain_channel is not None:
            charge = charge[0]
            peak_time = peak_time[0]

        return DL1CameraContainer(image=charge, peak_time=peak_time, is_valid=True)
