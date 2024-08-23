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
    name = "boundary",
    srcs = ["boundary.py"],
    deps = [
        ":bc_types",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/common:utils",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "boundary_test",
    timeout = "long",
    srcs = ["boundary_test.py"],
    shard_count = 6,
    tags = [
        "optonly",
        "requires-dragonfish",
    ],
    deps = [
        ":bc_types",
        ":boundary",
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

pytype_strict_library(
    name = "bc_types",
    srcs = ["bc_types.py"],
    deps = ["//third_party/py/tensorflow"],
)
