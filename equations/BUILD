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
    name = "compressible_navier_stokes",
    srcs = ["compressible_navier_stokes.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/boundary",
        "//third_party/py/swirl_c/boundary:bc_types",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/common:utils",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/numerics:gradient",
        "//third_party/py/swirl_c/numerics:interpolation",
        "//third_party/py/swirl_c/numerics/riemann_solver",
        "//third_party/py/swirl_c/numerics/time_integration:rhs_type",
        "//third_party/py/swirl_c/physics:diffusion",
        "//third_party/py/swirl_c/physics:physics_models",
        "//third_party/py/swirl_lm/numerics:interpolation",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "compressible_navier_stokes_test",
    timeout = "long",
    srcs = ["compressible_navier_stokes_test.py"],
    shard_count = 20,
    tags = [
        "optonly",
    ],
    deps = [
        ":compressible_navier_stokes",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/boundary:bc_types",
        "//third_party/py/swirl_c/common:testing_utils",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/numerics:kernel_op_types",
        "//third_party/py/swirl_c/physics:constant",
        "//third_party/py/swirl_c/physics:physics_models",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)
