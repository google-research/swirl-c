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
    name = "buoyant_bubble",
    srcs = ["buoyant_bubble.py"],
    tags = ["optonly"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/boundary:bc_types",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:initializer",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/numerics:kernel_op_types",
        "//third_party/py/swirl_c/physics:constant",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "buoyant_bubble_test",
    timeout = "long",
    srcs = ["buoyant_bubble_test.py"],
    args = ["--xla_jf_conv_full_precision=true"],
    tags = [
        "optonly",
        "requires-dragonfish:4",
    ],
    deps = [
        ":buoyant_bubble",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//research/simulation/tensorflow/fluid/framework/post_processing:data_processing",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:driver",
        "//third_party/py/swirl_c/physics:constant",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)
