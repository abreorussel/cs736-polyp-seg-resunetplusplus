import numpy as np
import cv2
from scipy.ndimage.interpolation import rotate

def save_image(image, mask, path, binary=True):
    image = np.array(image)
    if binary == True:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path[0], image)
    cv2.imwrite(path[1], mask)

def concat_images(images, rows, cols):
    _, h, w, _ = images.shape
    images = images.reshape((rows, cols, h, w, 3))
    images = images.transpose(0, 2, 1, 3, 4)
    images = images.reshape((rows * h, cols * w, 3))
    return images

def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size

def subtract(image):
    image = image / 255
    return image

def resize(image, size):
    size = check_size(size)
    image = cv2.resize(image, size)
    return image

def center_crop(image, mask, crop_size, size):
    h, w, _ = image.shape
    crop_size = check_size(crop_size)
    top = (h - crop_size[0]) // 2
    left = (w - crop_size[1]) // 2
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    image = image[top:bottom, left:right, :]
    mask = mask[top:bottom, left:right, :]

    image = resize(image, size)
    mask = resize(mask, size)

    return image, mask

def random_crop(image, mask, crop_size, size):
    crop_size = check_size(crop_size)
    h, w, _ = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    image = image[top:bottom, left:right, :]
    mask = mask[top:bottom, left:right, :]

    image = resize(image, size)
    mask = resize(mask, size)

    return image, mask

def horizontal_flip(image, mask, size):
    image = image[:, ::-1, :]
    mask = mask[:, ::-1, :]

    image = resize(image, size)
    mask = resize(mask, size)

    return image, mask

def vertical_flip(image, mask, size):
    image = image[::-1, :, :]
    mask = mask[::-1, :, :]

    image = resize(image, size)
    mask = resize(mask, size)

    return image, mask

def scale_augmentation(image, mask, scale_range, crop_size, size):
    scale_size = np.random.randint(*scale_range)
    image = cv2.resize(image, (scale_size, scale_size))
    mask = cv2.resize(mask, (scale_size, scale_size))
    image, mask = random_crop(image, mask, crop_size, size)
    return image, mask

def random_rotation(image, mask, size, angle_range=(0, 90)):
    h1, w1, _ = image.shape
    h2, w2, _ = mask.shape

    angle = np.random.randint(*angle_range)
    image = rotate(image, angle)
    image = resize(image, (h1, w1))

    mask = rotate(mask, angle)
    mask = resize(mask, (h2, w2))

    image = resize(image, size)
    mask = resize(mask, size)

    return image, mask

def cutout(image_origin, mask_origin, mask_size, mask_value='mean'):
    image = np.copy(image_origin)
    mask = np.copy(mask_origin)

    if mask_value == 'mean':
        mask_value = image.mean()
    elif mask_value == 'random':
        mask_value = np.random.randint(0, 256)

    h, w, _ = image.shape
    top = np.random.randint(0 - mask_size // 2, h - mask_size)
    left = np.random.randint(0 - mask_size // 2, w - mask_size)
    bottom = top + mask_size
    right = left + mask_size
    if top < 0:
        top = 0
    if left < 0:
        left = 0

    image[top:bottom, left:right, :].fill(mask_value)
    mask[top:bottom, left:right, :].fill(0)

    image = resize(image, size)
    mask = resize(mask, size)

    return image, mask

def brightness_augment(img, mask, factor=0.5):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #convert to hsv
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform()) #scale channel V uniformly
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
    rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)

    image = resize(rgb, size)
    mask = resize(mask, size)

    return image, mask

def rgb_to_grayscale(img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = [img, img, img]
    img = np.transpose(img, (1, 2, 0))

    image = resize(img, size)
    mask = resize(mask, size)
    return image, mask