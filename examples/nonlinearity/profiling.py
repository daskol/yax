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

import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil, floor, log
from typing import Iterator

import jax
import jax.profiler

from profile_pb2 import Label, Profile

SI_UNITS = ('', 'ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi', 'Yi', 'Ri', 'Qi')


def calc_memory_usage(profile: Profile):
    ix = find_space(profile)
    # TODO(@daskol): We also can add filtering by source name.
    memory_usage = 0
    for i, sample in enumerate(profile.sample):
        if any(is_buffer(profile, x) for x in sample.label):
            memory_usage += sample.value[ix]
    return memory_usage


def find_space(profile: Profile) -> int:
    for i, ent in enumerate(profile.sample_type):
        if profile.string_table[ent.type] == 'space':
            return i
    raise RuntimeError('Malformed profile structure.')


def format_space(value: float) -> str:
    """Format a disk or memory space value in human-readable way.

    >>> format_space(2048)
    '2kiB'
    """
    if value == 0:
        return '0B'
    digits = log(value) / log(1024)
    order = int(floor(digits))
    value /= 1024**order
    unit = SI_UNITS[order]
    return f'{ceil(value)}{unit}B'


def is_buffer(profile: Profile, label: Label) -> bool:
    return all([
        profile.string_table[label.key] == 'kind',
        profile.string_table[label.str] == 'buffer',
    ])


@dataclass(repr=False)
class Span:
    """An  memory usage at the beginning and the ending of a context."""

    bakend: str

    begin: int

    end: int

    @property
    def change(self) -> int:
        """A memory usage as change between terminal and initial values."""
        return self.end - self.begin

    def __repr__(self) -> str:
        return format_space(self.change)

    def __float__(self) -> float:
        return float(self.change)

    def __int__(self) -> int:
        return self.change


def get_memory_usage(backend: str | None = None) -> int:
    gzipped = jax.profiler.device_memory_profile(backend)
    raw = gzip.decompress(gzipped)
    profile = Profile().FromString(raw)
    return calc_memory_usage(profile)


@contextmanager
def memory_usage(backend: str | None = None) -> Iterator[Span]:
    if backend is None:
        backend = jax.default_backend()
    before = get_memory_usage(backend)
    span = Span(backend, before, before)
    yield span
    span.end = get_memory_usage(backend)
