from numba import cuda

from sobel_kernel import sobel_filter
from threads_and_blocks_calculator import compute_threads_and_blocks, load_image
from threshold_kernel import threshold

@cuda.jit
def hysterisis_kernel(source, destination):
    x, y = cuda.grid(2)
    if x < source.shape[0] and y < source.shape[1]:
        if source.ndim == 2:
            if source[x, y] == 127:
                if (source[x - 1, y - 1] == 255 or
                        source[x - 1, y] == 255 or
                        source[x - 1, y + 1] == 255 or
                        source[x, y - 1] == 255 or
                        source[x, y + 1] == 255 or
                        source[x + 1, y - 1] == 255 or
                        source[x + 1, y] == 255 or
                        source[x + 1, y + 1] == 255):
                    destination[x, y] = 255
                else:
                    destination[x, y] = 0
            else:
                destination[x, y] = destination[x, y]
        elif source.ndim == 3:
            for c in range(source.shape[2]):
                if source[x, y, c] == 127:
                    if (source[x - 1, y - 1, c] == 255 or
                            source[x - 1, y, c] == 255 or
                            source[x - 1, y + 1, c] == 255 or
                            source[x, y - 1, c] == 255 or
                            source[x, y + 1, c] == 255 or
                            source[x + 1, y - 1, c] == 255 or
                            source[x + 1, y, c] == 255 or
                            source[x + 1, y + 1, c] == 255):
                        destination[x, y, c] = 255
                    else:
                        destination[x, y, c] = 0
                else:
                    destination[x, y, c] = destination[x, y, c]


def hysterisis(imagetab, threadsperblockInt=(8, 8)):
    s_image = cuda.to_device(imagetab)
    d_image = cuda.device_array_like(s_image)
    threadsperblock, blockspergrid = compute_threads_and_blocks(imagetab, threadsperblockInt)
    hysterisis_kernel[blockspergrid, threadsperblock](s_image, d_image)
    output = d_image.copy_to_host()

    cuda.synchronize()

    return output


# if len(sys.argv) < 3:
#    print("Usage: ", sys.argv[0], " <inputFile> <outputFile>")
#    sys.exit(-1)

# inputFile = sys.argv[1]
# outputFile = sys.argv[2]

# image_source = load_image(inputFile)

# image_sobel = sobel_filter(image_source)

# image_output_thresh = threshold(image_sobel)

# output = hysterisis(image_output_thresh)

# m = Image.fromarray(output)
# m.save(outputFile)