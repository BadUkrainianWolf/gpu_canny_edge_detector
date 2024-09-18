import numpy as np
from numba import cuda
from threads_and_blocks_calculator import compute_threads_and_blocks
from math import sqrt

@cuda.jit
def sobel_filter_kernel(source, destination):
    x, y = cuda.grid(2)
    if x < source.shape[0] - 2 and y < source.shape[1] - 2:
        for c in range(source.shape[2]):  # Iterate over color channels
            gx = (source[x, y, c] - source[x + 2, y, c] +
                  2 * source[x, y + 1, c] - 2 * source[x + 2, y + 1, c] +
                  source[x, y + 2, c] - source[x + 2, y + 2, c])

            gy = (source[x, y, c] - source[x, y + 2, c] +
                  2 * source[x + 1, y, c] - 2 * source[x + 1, y + 2, c] +
                  source[x + 2, y, c] - source[x + 2, y + 2, c])

            destination[x + 1, y + 1, c] = sqrt(gx**2 + gy**2)
            if destination[x + 1, y + 1, c] > 255:
                destination[x + 1, y + 1, c] = 175
            if destination[x + 1, y + 1, c] < 0:
                destination[x + 1, y + 1, c] = 0



def sobel_filter(imagetab, threadsperblockInt=(8, 8)):
    s_image = cuda.to_device(imagetab)
    d_image = cuda.device_array_like(s_image)
    threadsperblock, blockspergrid = compute_threads_and_blocks(imagetab, threadsperblockInt)
    sobel_filter_kernel[blockspergrid, threadsperblock](s_image, d_image)
    output = d_image.copy_to_host()

    cuda.synchronize()
    return output

# if len(sys.argv) < 3:
#    print("Usage: ", sys.argv[0], " <inputFile> <outputFile>")
#    sys.exit(-1)

# inputFile = sys.argv[1]
# outputFile = sys.argv[2]

# image = load_image(inputFile)

# output_image = sobel_filter(image)

# m = Image.fromarray(output_image)
# m.save(outputFile)