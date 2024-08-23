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
    name = "generic",
    srcs = ["generic.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/common:utils",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/physics:constant",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "generic_test",
    srcs = ["generic_test.py"],
    tags = ["requires-dragonfish"],
    deps = [
        ":generic",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:testing_utils",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/physics:constant",
        "//third_party/py/tensorflow",
    ],
)
