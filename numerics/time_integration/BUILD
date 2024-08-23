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
    name = "rhs_type",
    srcs = ["rhs_type.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/physics:physics_models",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "rk2",
    srcs = ["rk2.py"],
    deps = [
        ":rhs_type",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/physics:physics_models",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "rk3",
    srcs = ["rk3.py"],
    deps = [
        ":rhs_type",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/physics:physics_models",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "time_integration",
    srcs = ["time_integration.py"],
    deps = [
        ":rhs_type",
        ":rk2",
        ":rk3",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/physics:physics_models",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "rk2_test",
    srcs = ["rk2_test.py"],
    deps = [
        ":rk2",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:testing_utils",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/physics:physics_models",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "rk3_test",
    srcs = ["rk3_test.py"],
    deps = [
        ":rk3",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:testing_utils",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/physics:physics_models",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "time_integration_test",
    srcs = ["time_integration_test.py"],
    deps = [
        ":rhs_type",
        ":time_integration",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:testing_utils",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/physics:physics_models",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)
