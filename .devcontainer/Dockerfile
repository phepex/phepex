# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If
# a copy of the MPL was not distributed with this file, You can obtain one at
# https://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0

FROM library/ubuntu

RUN apt -y update && \
    apt -y upgrade && \
    apt -y install python3-pip curl git build-essential cmake ninja-build clang-format clangd && \
    pip install --break-system-packages \
        ctapipe \
        nanobind "scikit-build-core>=0.10" pytest ruff && \
    apt clean
