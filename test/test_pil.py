
__doc__ = """针对常用的图像操作库使用中的⚠️事项进行测试，并总结如下：

1、PIL/OpenCV读取的图片结构和输入pytorch的结构差一个transpose，numpy rollaxis和transpose都可以实现
2、OpenCV读取图片的结果也是numpy array的结构
3、PIL Image size读取的是 (width, height)
   numpy/opencv shape为(height, width, channel)
4、PIL Image的操作中输入size为(width, height),
   opencv操作中输入的size为(width, height)
"""
import numpy as np 
import torch
import cv2
from PIL import Image

if __name__ == "__main__":
    img_path = "images/图片RGB值表.png"
    print("========================== PIL ===========================")
    ##PIL.Image
    pil_img = Image.open(img_path).convert('RGB')
    print(type(pil_img), pil_img.size)
    pil_img_resized = pil_img.resize(size=(500, 600), resample=Image.Resampling.BILINEAR)
    pil_img_resized.save(img_path + ".resized.png")

    print("========================= Numpy ==========================")
    np_img = np.array(pil_img, dtype=np.uint8)
    print(type(np_img), np_img.shape)
    # following operation is equal
    np_img_trans = np.transpose(np_img, (2,0,1))
    np_img = np.rollaxis(np_img, 2)
    torch.testing.assert_close(np_img_trans, np_img, rtol=1e-6, atol=1e-6)
    print("Transpose the array")
    print(type(np_img), np_img.shape)  
    print(type(np_img_trans), np_img_trans.shape)  


    ## ToTensor
    tensor_img = torch.from_numpy(np_img).to(dtype=torch.float32)
    print(type(tensor_img), tensor_img.shape, tensor_img.size())

    print("========================= OpenCV ==========================")
    opencv_img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)
    print(type(opencv_img), opencv_img.shape)
    opencv_color = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    print(type(opencv_color), opencv_color.shape)
    opencv_color_resized = cv2.resize(opencv_color, dsize=(500, 600), interpolation=cv2.INTER_AREA)
    cv2.imwrite(img_path+".cv2resize.png", opencv_color_resized)
