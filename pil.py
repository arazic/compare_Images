from PIL import Image, ImageStat
from PIL import ImageChops

def _compute_manhattan_distance(diff_image):
    """
            Computes a percentage of similarity of the difference image given.

            :param PIL.Image diff_image:
                An image in RGB mode computed from ImageChops.difference
            """
    import numpy

    number_of_pixels = diff_image.size[0] * diff_image.size[1]
    return (
        # To obtain a number in 0.0 -> 100.0
            100.0
            * (
                # Compute the sum of differences
                    numpy.sum(diff_image)
                    /
                    # Divide by the number of channel differences RGB * Pixels
                    float(3 * number_of_pixels)
            )
            # Normalize between 0.0 -> 1.0
            / 255.0
    )


if __name__ == '__main__':
    # Calculate difference as a ratio.
    # stat = ImageStat.Stat(diff)
    # diff_ratio = sum(stat.mean) / (len(stat.mean) * 255)
    #
    # print(diff_ratio * 100)

    im1 = Image.open(r'C:\Users\user\PycharmProjects\compare_Images\JJ\cookie0.jpg')
    im2 = Image.open(r'C:\Users\user\PycharmProjects\compare_Images\JJ\pistol4.jpg')

    diff = ImageChops.difference(im1, im2)
    if diff.getbbox():
        print("images are different")
        print(_compute_manhattan_distance(diff))
    else:
        print("images are the same")


    # print(diff)
    # diff.save(open('difference.jpg','wb'))
