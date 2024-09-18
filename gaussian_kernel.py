from numba import cuda
import numpy as np
from threads_and_blocks_calculator import compute_threads_and_blocks

kernel = np.array([[1, 4, 6, 4, 1],
                   [4, 16, 24, 16, 4],
                   [6, 24, 36, 24, 6],
                   [4, 16, 24, 16, 4],
                   [1, 4, 6, 4, 1]]) / 256

@cuda.jit
def gaussian_blur_kernel(source, destination, filter_kernel):
    x, y = cuda.grid(2)
    if x < source.shape[0] and y < source.shape[1]:
        # Do the gaussian blur with count of corners
        for c in range(source.ndim):
            output_pixel = 0.0
            for i in range(filter_kernel.shape[0]):
                for j in range(filter_kernel.shape[1]):
                    if 0 <= x + i - 2 < source.shape[0] and 0 <= y + j - 2 < source.shape[1]:
                        output_pixel += source[x + i - 2, y + j - 2, c] * filter_kernel[i, j]
                    else:
                        output_pixel += source[x, y, c] * filter_kernel[i, j]

            destination[x, y, c] = np.float32(output_pixel)

def gaussian_blur(imagetab, threadsperblockInt=(8, 8)):
    s_image = cuda.to_device(imagetab)
    f_image = cuda.to_device(kernel)
    d_image = cuda.device_array((imagetab.shape[0],imagetab.shape[1],3),dtype = np.uint8)
    threadsperblock,blockspergrid = compute_threads_and_blocks(imagetab, threadsperblockInt)
    gaussian_blur_kernel[blockspergrid, threadsperblock](s_image, d_image, f_image)
    output = d_image.copy_to_host()

    cuda.synchronize()
    return output


# if len(sys.argv) < 3:
#    print("Usage: ", sys.argv[0], " <inputFile> <outputFile>")
#    sys.exit(-1)


# inputFile = sys.argv[1]
# outputFile = sys.argv[2]

# imagetab = load_image(inputFile)

# output = gaussian_blur(imagetab)

# m = Image.fromarray(output)
# m.save(outputFile)