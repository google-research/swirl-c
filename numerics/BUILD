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
    name = "gradient",
    srcs = ["gradient.py"],
    deps = [
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "kernel_op_types",
    srcs = ["kernel_op_types.py"],
)

pytype_strict_library(
    name = "interpolation",
    srcs = ["interpolation.py"],
    deps = [
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "gradient_test",
    srcs = ["gradient_test.py"],
    deps = [
        ":gradient",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:testing_utils",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "interpolation_test",
    srcs = ["interpolation_test.py"],
    deps = [
        ":interpolation",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:testing_utils",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)
