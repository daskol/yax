project('fewbit', 'cpp', version: '0.0.0')

add_project_arguments('-std=c++20', language: 'cpp')

nanobind = dependency('nanobind', static: true)
py = import('python').find_installation()
py.extension_module(
  '_fewbit',
  sources: ['fewbit.cc'],
  dependencies: [nanobind],
  install: true,
)
