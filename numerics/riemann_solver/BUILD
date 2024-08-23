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
    name = "riemann_solver",
    srcs = ["riemann_solver.py"],
    deps = [
        ":hll",
        ":hllc",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
    ],
)

pytype_strict_library(
    name = "hll",
    srcs = ["hll.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/physics:physics_models",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "hllc",
    srcs = ["hllc.py"],
    deps = [
        ":hll",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/physics:physics_models",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "hll_test",
    srcs = ["hll_test.py"],
    deps = [
        ":hll",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:testing_utils",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/physics:constant",
        "//third_party/py/swirl_c/physics:physics_models",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "riemann_solver_test",
    srcs = ["riemann_solver_test.py"],
    deps = [
        ":hll",
        ":riemann_solver",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/physics:physics_models",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "hllc_test",
    srcs = ["hllc_test.py"],
    deps = [
        ":hllc",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:testing_utils",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/physics:constant",
        "//third_party/py/swirl_c/physics:physics_models",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)
