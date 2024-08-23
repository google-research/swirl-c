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
    name = "fluid",
    srcs = ["fluid.py"],
    deps = [
        ":constant",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/boundary",
        "//third_party/py/swirl_c/boundary:bc_types",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/common:utils",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/numerics:gradient",
        "//third_party/py/swirl_c/numerics:interpolation",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "constant",
    srcs = ["constant.py"],
)

pytype_strict_library(
    name = "physics_models",
    srcs = ["physics_models.py"],
    deps = [
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/physics/additional_models:gravity",
        "//third_party/py/swirl_c/physics/thermodynamics:generic",
        "//third_party/py/swirl_c/physics/transport:simple",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "diffusion",
    srcs = ["diffusion.py"],
    deps = [
        ":fluid",
        ":physics_models",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/numerics:gradient",
        "//third_party/py/swirl_c/numerics:interpolation",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "diffusion_test",
    srcs = ["diffusion_test.py"],
    deps = [
        ":constant",
        ":diffusion",
        ":physics_models",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/numerics:kernel_op_types",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "fluid_test",
    srcs = ["fluid_test.py"],
    tags = ["requires-dragonfish"],
    deps = [
        ":constant",
        ":fluid",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:testing_utils",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/swirl_c/numerics:kernel_op_types",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "physics_models_test",
    srcs = ["physics_models_test.py"],
    deps = [
        ":physics_models",
        "//third_party/py/absl/logging",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_c/common:testing_utils",
        "//third_party/py/swirl_c/common:types",
        "//third_party/py/swirl_c/core:parameter",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)
