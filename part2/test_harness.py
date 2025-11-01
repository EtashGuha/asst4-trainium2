import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal
from neuronxcc.nki import benchmark

from conv2d import fused_conv2d_maxpool as conv2d

from conv2d_numpy import conv2d_cpu_torch
import logging
import argparse
import io
import sys

import subprocess

logging.disable(logging.OFF)


def save_trace(profile_name, neff_file_name):
    """Run the profiler and save the NEFF and NTFF files with the specified name."""
    subprocess.run(
        [
            "neuron-profile",
            "capture",
            "-n",
            neff_file_name,
            "-s",
            profile_name + ".ntff",
        ],
        check=True,
    )

    subprocess.run(["mv", neff_file_name, profile_name + ".neff"], check=True)

    print(
        f"\n\nNEFF / NTFF files generated with names: {profile_name + '.neff'}, {profile_name + '.ntff'}"
    )


def test_correctness_conv2d_kernel(
    kernel,
    use_cpu_impl=False,
    use_larger_images=False,
    use_bias=False,
    use_maxpool=False,
):
    kernel = baremetal(kernel)
    ref_impl = conv2d_cpu_torch

    input_channels_list = [128, 256]
    output_channels_list = [128, 256]
    kernel_size_list = [3]
    batch_size_list = [4]
    image_dims_list = [(32, 16)]
    pool_size = 2 if use_maxpool else 1

    if use_larger_images:
        input_channels_list = [256]
        output_channels_list = [256]
        image_dims_list = [(224, 224)]

    for input_channels in input_channels_list:
        for output_channels in output_channels_list:
            for kernel_size in kernel_size_list:
                for batch_size in batch_size_list:
                    for image_dims in image_dims_list:
                        X = np.random.rand(
                            batch_size, input_channels, image_dims[0], image_dims[1]
                        ).astype(np.float32)
                        W = np.random.rand(
                            output_channels, input_channels, kernel_size, kernel_size
                        ).astype(np.float32)
                        bias = (
                            np.zeros(output_channels).astype(np.float32)
                            if not use_bias
                            else np.random.rand(output_channels).astype(np.float32)
                        )

                        args = [X, W, bias]
                        kwargs = {"pool_size": pool_size}

                        out = kernel(*args, **kwargs)
                        out_ref = ref_impl(*args, **kwargs)

                        if not np.allclose(out, out_ref):
                            print(
                                f"Output mismatch for input_channels: {input_channels}, \
                        output_channels: {output_channels}, kernel_size: {kernel_size}, batch_size: {batch_size},\
                         image_dims: {image_dims}, use_bias: {use_bias}, use_maxpool: {use_maxpool}"
                            )

                            return False

    return True


def test_performance_conv2d_kernel(
    kernel,
    dtype=np.float32,
    batch_size=1,
    in_channels=256,
    out_channels=256,
    image_height=224,
    image_width=224,
    kernel_height=3,
    kernel_width=3,
    pool_size=1,
):
    # a performance requirement map (dtype, image_height) ->
    # [relaxed performance threshold, optimized performance threshold]
    performance_requirements_by_dtype_size = {
        (np.float32, 224): [3640, 4626],
        (np.float16, 224): [1570, 1018],
        (np.float32, 32): [112, 112],
        (np.float16, 32): [86, 86],
    }

    X = np.random.rand(batch_size, in_channels, image_height, image_width).astype(dtype)
    W = np.random.rand(out_channels, in_channels, kernel_height, kernel_width).astype(
        dtype
    )
    bias = np.random.rand(out_channels).astype(dtype)

    args = [X, W, bias]
    kwargs = {"pool_size": pool_size}

    dtype_str = "float32" if dtype == np.float32 else "float16"

    bench_func = nki.benchmark(
        warmup=5,
        iters=20,
        save_neff_name=f"file_pool_{pool_size}_{dtype_str}_{image_height}.neff",
        additional_compile_opt="--disable-dge",
    )(kernel)
    text_trap = io.StringIO()
    sys.stdout = text_trap
    bench_func(*args, **kwargs)
    sys.stdout = sys.__stdout__
    p99_us_student = bench_func.benchmark_result.nc_latency.get_latency_percentile(99)
    print(f"\n\nExecution Time for student implementation: {p99_us_student} μs")

    if (
        p99_us_student
        > performance_requirements_by_dtype_size[(dtype, image_height)][0]
    ):
        print(
            f"Performance requirement not met: need to be under {performance_requirements_by_dtype_size[(dtype, image_height)][0]} μs"
        )
        return False, False
    elif (
        p99_us_student
        > performance_requirements_by_dtype_size[(dtype, image_height)][1]
    ):
        print(
            f"Performance requirement partially met: better to be under {performance_requirements_by_dtype_size[(dtype, image_height)][1]} μs"
        )
        return True, False
    else:
        return True, True


def get_performance_score(test_result, total_score):
    relaxed_result, optimized_result = test_result
    if optimized_result:
        print("Performance test passed 😍")
        return total_score
    elif relaxed_result:
        print("Can you make it faster? 🧐")
        return (
            total_score * 0.95
        )  # students get most of the score with meeting the relaxed time constraint
    else:
        print("Performance test failed 😢")
        return 0  # got 0 for performance otherwise


# write a function g which when passed a function f, returns a new function that when called with some *args and **kwargs, calls
# nki.simulate_kernel(f, *args, **kwargs) and returns the result
def simulate_kernel_wrapper(kernel):
    def temp_func(*args, **kwargs):
        return nki.simulate_kernel(kernel, *args, **kwargs)

    return temp_func


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_maxpool", action="store_true", help="Use smaller images for testing"
    )
    parser.add_argument(
        "--profile", type=str, default=None, help="File to save the neff file"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use nki.simulate_kernel to run student implementation",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for random number generation"
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.simulate:
        conv2d = simulate_kernel_wrapper(conv2d)

    correctness_score = 0.0
    performance_score = 0.0
    ec = 0.0
    # running correctness tests
    print(
        "Running correctness test for conv2d kernel with smaller images...",
        end="",
        flush=True,
    )
    test_result = test_correctness_conv2d_kernel(conv2d, use_larger_images=False)
    if test_result:
        correctness_score += 2.5
        print("Passed 😎")
    else:
        print("Failed 😢")

    print(
        "Running correctness test for conv2d kernel with larger images...",
        end="",
        flush=True,
    )
    test_result = test_correctness_conv2d_kernel(conv2d, use_larger_images=True)
    if test_result:
        correctness_score += 2.5
        print("Passed 😇")
    else:
        print("Failed 😢")

    print(
        "Running correctness test for conv2d kernel with larger images + bias...",
        end="",
        flush=True,
    )
    test_result = test_correctness_conv2d_kernel(
        conv2d, use_bias=True, use_larger_images=True
    )
    if test_result:
        correctness_score += 2.5
        print("Passed 😍")
    else:
        print("Failed 😢")

    if args.test_maxpool:
        print(
            "Running correctness test for conv2d kernel with larger images + bias + maxpool...",
            end="",
            flush=True,
        )
        test_result = test_correctness_conv2d_kernel(
            conv2d, use_bias=True, use_maxpool=True, use_larger_images=True
        )
        if test_result:
            correctness_score += 2.5
            print("Passed 😍")
        else:
            print("Failed 😢")

    print("Comparing performance with reference kernel (no maxpool, float32)...")
    test_result = test_performance_conv2d_kernel(conv2d, pool_size=1, dtype=np.float32)
    performance_score += get_performance_score(test_result, 17.5)

    if args.profile is not None:
        save_trace(args.profile + "_float32", "file_pool_1_float32_224.neff")

    print("Comparing performance with reference kernel (no maxpool, float16)...")
    test_result = test_performance_conv2d_kernel(conv2d, pool_size=1, dtype=np.float16)
    performance_score += get_performance_score(test_result, 17.5)

    if args.profile is not None:
        save_trace(args.profile + "_float16", "file_pool_1_float16_224.neff")

    print(
        "Comparing performance with reference kernel (no maxpool, float32, smaller image)... [EC]"
    )
    test_result = test_performance_conv2d_kernel(
        conv2d, pool_size=1, dtype=np.float32, image_height=32, image_width=16
    )
    ec += get_performance_score(test_result, 1.25)

    if args.profile is not None:
        save_trace(args.profile + "_float32_smaller", "file_pool_1_float32_32.neff")

    print(
        "Comparing performance with reference kernel (no maxpool, float16, smaller image)... [EC]"
    )
    test_result = test_performance_conv2d_kernel(
        conv2d, pool_size=1, dtype=np.float16, image_height=32, image_width=16
    )
    ec += get_performance_score(test_result, 1.25)

    if args.profile is not None:
        save_trace(args.profile + "_float16_smaller", "file_pool_1_float16_32.neff")

    if args.test_maxpool:
        print("Comparing performance with reference kernel (with maxpool, float32)...")
        test_result = test_performance_conv2d_kernel(
            conv2d, pool_size=2, dtype=np.float32
        )
        performance_score += get_performance_score(test_result, 7.5)

        if args.profile is not None:
            save_trace(args.profile + "_pool_float32", "file_pool_2_float32_224.neff")

        print("Comparing performance with reference kernel (with maxpool, float16)...")
        test_result = test_performance_conv2d_kernel(
            conv2d, pool_size=2, dtype=np.float16
        )
        performance_score += get_performance_score(test_result, 7.5)

        if args.profile is not None:
            save_trace(args.profile + "_pool_float16", "file_pool_2_float16_224.neff")

        print(
            "Comparing performance with reference kernel (with maxpool, float32, smaller image)... [EC]"
        )
        test_result = test_performance_conv2d_kernel(
            conv2d, pool_size=2, dtype=np.float32, image_height=32, image_width=16
        )
        ec += get_performance_score(test_result, 1.25)

        if args.profile is not None:
            save_trace(
                args.profile + "_pool_float32_smaller", "file_pool_2_float32_32.neff"
            )

        print(
            "Comparing performance with reference kernel (with maxpool, float16, smaller image)... [EC]"
        )
        test_result = test_performance_conv2d_kernel(
            conv2d, pool_size=2, dtype=np.float16, image_height=32, image_width=16
        )
        ec += get_performance_score(test_result, 1.25)

        if args.profile is not None:
            save_trace(
                args.profile + "_pool_float16_smaller", "file_pool_2_float16_32.neff"
            )

    print(
        "Your final score is: ",
        "" if args.test_maxpool else "(without maxpool) ",
        correctness_score + performance_score + ec,
        "\tTotal obtainable: ",
        65 if args.test_maxpool else 45,
    )
    print(
        "Correctness: ",
        correctness_score,
        "\tTotal obtainable: ",
        10 if args.test_maxpool else 7.5,
    )
    print(
        "Performance: ",
        performance_score,
        "\tTotal obtainable: ",
        50 if args.test_maxpool else 35,
    )
    print("Extra Credit:", ec, "\tTotal obtainable: ", 5 if args.test_maxpool else 2.5)
