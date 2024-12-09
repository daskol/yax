# Copyright 2024 Daniel Bershatsky
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04 AS base

LABEL org.opencontainers.image.source="https://github.com/daskol/yax"

RUN --mount=type=cache,target=/var/cache/apt \
    apt update -y && \
    apt install -y --no-install-recommends \
        python3.12 python-is-python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --break-system-packages -U flax 'jax[cuda12]<0.4.36' numpy

FROM base AS dev

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --break-system-packages -U 'pytest>=8.2' isort ruff && \
    pip install --break-system-packages -U \
        datasets sentencepiece tokenizers transformers ytsaurus-client

FROM base AS wheel

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --break-system-packages -U build

WORKDIR /usr/src/yax

ADD . .

RUN python -m build -nw && \
    rm -rf build yax.egg-info

FROM base

COPY --from=wheel /usr/src/yax/dist/yax-0.0.0-py3-none-any.whl .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --break-system-packages -U yax-0.0.0-py3-none-any.whl
