# By: FISA
# BERDNYK Mariia

import argparse
import time
import math
import numpy as np
from PIL import Image
from numba import cuda

DEFAULT_THREADS_PER_BLOCK = (8, 8)

LOW = 51
HIGH = 102
HIGHEST_COLOR_PIXEL = 255
CLAMPING_VALUE = 175
WEAK_PIXEL = 127

kernel = np.array([[1, 4, 6, 4, 1],
                   [4, 16, 24, 16, 4],
                   [6, 24, 36, 24, 6],
                   [4, 16, 24, 16, 4],
                   [1, 4, 6, 4, 1]]) / 256

def compute_threads_and_blocks(imagetab, threadsperblock=DEFAULT_THREADS_PER_BLOCK):
    width, height = imagetab.shape[:2]
    blockspergrid_x = math.ceil(width / threadsperblock[0])
    blockspergrid_y = math.ceil(height / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    return threadsperblock,blockspergrid
    
def load_image(inputFile):
    im = Image.open(inputFile)
    imagetab = np.array(im)
    return imagetab

#TD corrections by professor
@cuda.jit
def RGBToBWKernel(source, destination, offset):
    width, height = source.shape[:2]
    x, y = cuda.grid(2)
    if x < width and y < height:
        r_x = (x + offset) % width
        r_y = (y + offset) % height
        destination[r_x, r_y] = np.uint8(0.3 * source[r_x, r_y, 0] + 0.59 * source[r_x, r_y, 1] + 0.11 * source[r_x, r_y, 2])

def bw_kernel(imagetab, threadsperblockInt = DEFAULT_THREADS_PER_BLOCK):
    s_image = cuda.to_device(imagetab)
    d_image = cuda.device_array((imagetab.shape[0], imagetab.shape[1], 3), dtype = np.uint8)
    threadsperblock, blockspergrid = compute_threads_and_blocks(imagetab, threadsperblockInt)

    for off in range(33, 1, -1):
        RGBToBWKernel[blockspergrid, threadsperblock](s_image, d_image,off) 
        cuda.synchronize()
                    
    output = d_image.copy_to_host()
    cuda.synchronize()
    return output

@cuda.jit
def gaussian_blur_kernel(source, destination, filter_kernel):
    x, y = cuda.grid(2)
    if x < source.shape[0] and y < source.shape[1]:
        for c in range(source.shape[2]) if source.ndim == 3 else [0]:  # Loop over channels if 3D, else consider only one channel
            for c in range(source.shape[2]): # Deal with 3D dimmentional image as well as 2D
                output_pixel = 0.0
                for i in range(filter_kernel.shape[0]):
                    for j in range(filter_kernel.shape[1]):
                        if 0 <= x + i - 2 < source.shape[0] and 0 <= y + j - 2 < source.shape[1]:
                            output_pixel += source[x + i - 2, y + j - 2, c] * filter_kernel[i, j]
                        else:
                            # When a neighboring pixel is missing we will use the value of the current pixel as substitute.
                            output_pixel += source[x, y, c] * filter_kernel[i, j]

                destination[x, y, c] = np.float32(output_pixel)

def gaussian_blur(imagetab, threadsperblockInt = DEFAULT_THREADS_PER_BLOCK):
    s_image = cuda.to_device(imagetab)
    f_image = cuda.to_device(kernel)
    d_image = cuda.device_array(imagetab.shape, dtype = np.uint8) # To be careful with the shape of the image
    threadsperblock,blockspergrid = compute_threads_and_blocks(imagetab, threadsperblockInt)
    gaussian_blur_kernel[blockspergrid, threadsperblock](s_image, d_image, f_image)
    output = d_image.copy_to_host()

    cuda.synchronize()
    return output

@cuda.jit
def sobel_filter_kernel(source, destination):
    x, y = cuda.grid(2)
    if x < source.shape[0] - 2 and y < source.shape[1] - 2:
        for c in range(source.shape[2]) if source.ndim == 3 else [0]:  # Iterate over color channels if 3D, else consider only one channel
            # Horizontal Sobel Kernel
            gx = (source[x, y, c] - source[x + 2, y, c] +
                2 * source[x, y + 1, c] - 2 * source[x + 2, y + 1, c] +
                source[x, y + 2, c] - source[x + 2, y + 2, c])

            # Vertical Sobel Kernel
            gy = (source[x, y, c] - source[x, y + 2, c] +
                2 * source[x + 1, y, c] - 2 * source[x + 1, y + 2, c] +
                source[x + 2, y, c] - source[x + 2, y + 2, c])
            
            # Convolve the image with these two kernels separately and then combine the results to obtain the gradient magnitude.
            mag = math.sqrt(gx**2 + gy**2)

            # Clamp the values to HIGHEST_COLOR_PIXEL as in the paper
            if mag > HIGHEST_COLOR_PIXEL:
                mag = CLAMPING_VALUE
            if mag < 0:
                mag = 0
            
            destination[x + 1, y + 1, c] = mag  # Assign gradient magnitude to destination

def sobel_filter(imagetab, threadsperblockInt = DEFAULT_THREADS_PER_BLOCK):
    s_image = cuda.to_device(imagetab)
    d_image = cuda.device_array_like(s_image)

    threadsperblock, blockspergrid = compute_threads_and_blocks(imagetab, threadsperblockInt)

    sobel_filter_kernel[blockspergrid, threadsperblock](s_image, d_image)
    output = d_image.copy_to_host()

    cuda.synchronize()
    return output

@cuda.jit
def threshold_kernel(source, destination):
    x, y = cuda.grid(2)
    if x < source.shape[0] and y < source.shape[1]:
        if source.ndim == 2:  # 2D image
            if source[x, y] >= HIGH:
                destination[x, y] = HIGHEST_COLOR_PIXEL
            elif source[x, y] < LOW:
                destination[x, y] = 0
            else:
                destination[x, y] = WEAK_PIXEL # weak pixels are brought to 127
        elif source.ndim == 3:  # 3D image (RGB)
            for c in range(source.shape[2]):  # Iterate over color channels
                if source[x, y, c] >= HIGH:
                    destination[x, y, c] = HIGHEST_COLOR_PIXEL
                elif source[x, y, c] < LOW:
                    destination[x, y, c] = 0
                else: 
                    destination[x, y, c] = WEAK_PIXEL # weak pixels are brought to 127

def threshold(imagetab, threadsperblockInt = DEFAULT_THREADS_PER_BLOCK):
    s_image = cuda.to_device(imagetab)
    d_image = cuda.device_array_like(s_image)

    threadsperblock, blockspergrid = compute_threads_and_blocks(imagetab, threadsperblockInt)
    threshold_kernel[blockspergrid, threadsperblock](s_image, d_image)
    output = d_image.copy_to_host()

    cuda.synchronize()

    return output

@cuda.jit
def hysterisis_kernel(source, destination):
    x, y = cuda.grid(2)
    if x < source.shape[0] and y < source.shape[1]:
        if source.ndim == 2:
            if source[x, y] == WEAK_PIXEL:
                # transforming weak pixels into strong ones, if and only if at least one of the pixels around the one being processed is a strong one
                if (source[x - 1, y - 1] == HIGHEST_COLOR_PIXEL or
                        source[x - 1, y] == HIGHEST_COLOR_PIXEL or
                        source[x - 1, y + 1] == HIGHEST_COLOR_PIXEL or
                        source[x, y - 1] == HIGHEST_COLOR_PIXEL or
                        source[x, y + 1] == HIGHEST_COLOR_PIXEL or
                        source[x + 1, y - 1] == HIGHEST_COLOR_PIXEL or
                        source[x + 1, y] == HIGHEST_COLOR_PIXEL or
                        source[x + 1, y + 1] == HIGHEST_COLOR_PIXEL):
                    destination[x, y] = HIGHEST_COLOR_PIXEL
                else:
                    destination[x, y] = 0
            else:
                destination[x, y] = destination[x, y]
        elif source.ndim == 3:
            for c in range(source.shape[2]):
                if source[x, y, c] == WEAK_PIXEL:
                    if (source[x - 1, y - 1, c] == HIGHEST_COLOR_PIXEL or
                            source[x - 1, y, c] == HIGHEST_COLOR_PIXEL or
                            source[x - 1, y + 1, c] == HIGHEST_COLOR_PIXEL or
                            source[x, y - 1, c] == HIGHEST_COLOR_PIXEL or
                            source[x, y + 1, c] == HIGHEST_COLOR_PIXEL or
                            source[x + 1, y - 1, c] == HIGHEST_COLOR_PIXEL or
                            source[x + 1, y, c] == HIGHEST_COLOR_PIXEL or
                            source[x + 1, y + 1, c] == HIGHEST_COLOR_PIXEL):
                        destination[x, y, c] = HIGHEST_COLOR_PIXEL
                    else:
                        destination[x, y, c] = 0
                else:
                    destination[x, y, c] = destination[x, y, c]

def hysterisis(imagetab, threadsperblockInt = DEFAULT_THREADS_PER_BLOCK):
    s_image = cuda.to_device(imagetab)
    d_image = cuda.device_array_like(s_image)
    threadsperblock, blockspergrid = compute_threads_and_blocks(imagetab, threadsperblockInt)
    hysterisis_kernel[blockspergrid, threadsperblock](s_image, d_image)
    output = d_image.copy_to_host()

    cuda.synchronize()

    return output

def parse_args():
    parser = argparse.ArgumentParser(description="Image processing using CUDA")
    parser.add_argument("inputImage", help="Input image file path")
    parser.add_argument("outputImage", help="Output image file path")
    parser.add_argument("--tb", type=int, help="Size of a thread block for all operations")
    parser.add_argument("--bw", action="store_true", help="Perform only the bw_kernel")
    parser.add_argument("--gauss", action="store_true", help="Perform the bw_kernel and the gauss_kernel")
    parser.add_argument("--sobel", action="store_true", help="Perform all kernels up to sobel_kernel")
    parser.add_argument("--threshold", action="store_true", help="Perform all kernels up to threshold_kernel")
    return parser.parse_args()

def main():
    args = parse_args()

    input_image_path = args.inputImage
    output_image_path = args.outputImage

    start_time = time.time()

    image_source = load_image(input_image_path)

    if args.tb:
        threads_per_block = (args.tb, args.tb)
    else:
        threads_per_block = DEFAULT_THREADS_PER_BLOCK

    if args.bw:
        image_output = bw_kernel(image_source, threads_per_block)
    elif args.gauss:
        image_output = gaussian_blur(bw_kernel(image_source, threads_per_block), threads_per_block)
    elif args.sobel:
        image_output = sobel_filter(gaussian_blur(bw_kernel(image_source, threads_per_block), threads_per_block), threads_per_block)
        image_output = Image.fromarray(image_output)
        image_output.save(output_image_path)
    elif args.threshold:
        image_output = threshold(sobel_filter(gaussian_blur(bw_kernel(image_source, threads_per_block), threads_per_block), threads_per_block), threads_per_block)
    else:
        image_output = hysterisis(threshold(sobel_filter(gaussian_blur(bw_kernel(image_source, threads_per_block), threads_per_block), threads_per_block), threads_per_block), threads_per_block)

    if not args.sobel:
        m = Image.fromarray(image_output)
        m.save(output_image_path)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

if __name__ == "__main__":
    main()