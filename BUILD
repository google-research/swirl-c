load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//learning/deepmind/public/tools/python_interpreter:build_defs.bzl", "python_interpreter_binary")
load("//tools/build_defs/license:license.bzl", "license")

package(
    default_applicable_licenses = ["//third_party/py/swirl_c:license"],
    default_visibility = [":internal"],
)

license(
    name = "license",
    package_name = "swirl_c",
)

licenses(["notice"])

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//third_party/py/swirl_c/...",
    ],
)

# swirl_lm public API
# This is a single py_library rule which centralizes all files/deps.
#
# Use this command to generate the list of dependencies:
#
#   blaze query 'kind("py_.*library", //third_party/py/swirl_c/... - //third_party/py/swirl_c:swirl_c)' | sed -e 's/\(.*\)/        "\1",/'
pytype_strict_library(
    name = "swirl_c",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/swirl_c/boundary",
        "//third_party/py/swirl_c/boundary:bc_types",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/common:utils",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/numerics:gradient",
        "//third_party/py/swirl_c/physics:constant",
        "//third_party/py/swirl_c/physics:fluid",
        "//third_party/py/swirl_c/physics/thermodynamics:generic",
    ],
)

python_interpreter_binary(
    name = "swirl_c_python_interpreter",
    exec_properties = {
        "mem": "20g",  # This is to avoid forge OOMs while generating .par.
    },
    deps = [
        ":swirl_c",
        "//learning/deepmind/public/tools/ml_python:core_deps",
    ],
)

py_strict_test(
    name = "swirl_c_test",
    srcs = ["swirl_c_test.py"],
    deps = [
        "//third_party/py/swirl_c/boundary",
        "//third_party/py/swirl_c/boundary:bc_types",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/common:utils",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/numerics:gradient",
        "//third_party/py/swirl_c/physics:constant",
        "//third_party/py/swirl_c/physics:fluid",
        "//third_party/py/swirl_c/physics/thermodynamics:generic",
    ],
)
