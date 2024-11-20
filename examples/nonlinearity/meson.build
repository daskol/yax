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

project(
    'fewbit',
    'cpp',
    version: '0.0.0',
    default_options: ['cpp_std=c++20'],
    meson_version: '>=1.3.2',
)

nanobind = dependency('nanobind', static: true)
py = import('python').find_installation()
py.extension_module(
    '_fewbit',
    sources: ['fewbit.cc'],
    dependencies: [nanobind],
    install: true,
)

protoc = find_program('protoc', required: false)
if protoc.found()
    custom_target(
        'proto-py',
        build_always_stale: false,
        build_by_default: true,
        command: [
            protoc.full_path(),
            '-I@CURRENT_SOURCE_DIR@',
            '--python_out=@CURRENT_SOURCE_DIR@',
            '@PLAINNAME@',
        ],
        input: ['profile.proto'],
        output: ['profile_pb2.py'],
    )
endif
