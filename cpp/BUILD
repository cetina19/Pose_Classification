load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_binary(
    name = "main",
    srcs = ["main.cpp"],
    deps = [
        "@protobuf_archive//:protobuf",
        "@opencv//:opencv",
        "@mediapipe//:mediapipe",
        "@zlib//:zlib",
    ],
)

cc_library(
    name = "opencv",
    hdrs = glob(["external/opencv/include/**"]),
    includes = ["external/opencv/include"],
    linkopts = ["-lopencv_core", "-lopencv_imgproc", "-lopencv_highgui", "-lopencv_imgcodecs"],
)

cc_library(
    name = "mediapipe",
    hdrs = glob(["external/mediapipe/framework/**"]),
    includes = ["external/mediapipe/framework"],
    srcs = glob(["external/mediapipe/framework/**/*.cc"]),
    deps = [
        "@protobuf_archive//:protobuf",
    ],
)

cc_library(
    name = "protobuf",
    hdrs = glob(["external/protobuf_archive/src/**"]),
    includes = ["external/protobuf_archive/src"],
    srcs = glob(["external/protobuf_archive/src/**/*.cc"]),
)

cc_library(
    name = "zlib",
    srcs = glob(["*.c"]),
    hdrs = glob(["*.h"]),
    includes = ["."],
)