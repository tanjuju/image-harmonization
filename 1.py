import numpy as np
from PIL import Image
import torchvision.transforms.functional as tf
import torchvision.transforms as transforms
from util import util

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)

def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True

path='a0006.jpg'
path1='a0007.jpg'
image = Image.open(path).convert('RGB')
image= tf.resize(image, [256, 256])

transform_list = []
transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=Image.BICUBIC)))
transform_list += [transforms.ToTensor()]
transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform=transforms.Compose(transform_list)
image1=transform(image)#归一化到-1 - 1之间
image1=image1.unsqueeze(0).cuda()
image1=util.tensor2im(image1)

image=np.array(image,dtype=np.float32)
print(image)
print(image1)
image1_1 = Image.fromarray(image1)
image1_1.save(path1, quality=100)

image2=Image.open(path1).convert('RGB')
# image2= tf.resize(image2, [256, 256])
image2=np.array(image2,dtype=np.uint8)
print(image2)
print(image==image1)
print(image1==image2)
print(image==image2)