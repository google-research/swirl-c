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
    name = "simple",
    srcs = ["simple.py"],
    deps = [
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "simple_test",
    srcs = ["simple_test.py"],
    deps = [
        ":simple",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:testing_utils",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/physics:constant",
        "//third_party/py/swirl_c/physics:physics_models",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)
