load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(
    default_applicable_licenses = ["//third_party/py/swirl_c:license"],
    default_visibility = [
        "//research/simulation/tensorflow/fluid:sim_research_fluid",
        "//third_party/py/swirl_c:__subpackages__",
    ],
)

licenses(["notice"])

pytype_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        ":types",
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "testing_utils",
    srcs = ["testing_utils.py"],
    deps = [
        ":types",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "types",
    srcs = ["types.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":testing_utils",
        ":types",
        ":utils",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "testing_utils_test",
    srcs = ["testing_utils_test.py"],
    deps = [
        ":testing_utils",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "types_test",
    srcs = ["types_test.py"],
    deps = [
        ":types",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)
