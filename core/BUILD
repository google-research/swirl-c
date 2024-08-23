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
    name = "parameter",
    srcs = ["parameter.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/boundary:bc_types",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/numerics:kernel_op_types",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/yaml",
    ],
)

pytype_strict_library(
    name = "simulation",
    srcs = ["simulation.py"],
    deps = [
        ":parameter",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/equations:compressible_navier_stokes",
        "//third_party/py/swirl_c/numerics/time_integration",
        "//third_party/py/swirl_c/physics:physics_models",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "initializer",
    srcs = ["initializer.py"],
    deps = [
        ":parameter",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "driver",
    srcs = ["driver.py"],
    deps = [
        ":initializer",
        ":parameter",
        ":simulation",
        "//third_party/py/absl/logging",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_lm/base:driver",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "simulation_test",
    srcs = ["simulation_test.py"],
    deps = [
        ":parameter",
        ":simulation",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:testing_utils",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/numerics/time_integration",
        "//third_party/py/swirl_c/physics:physics_models",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "initializer_test",
    srcs = ["initializer_test.py"],
    deps = [
        ":initializer",
        ":parameter",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:testing_utils",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)
