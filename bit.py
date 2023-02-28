import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("lena.png", flags=0)  # flags=0 读取为灰度图像
height, width = img.shape[:2]  # 图片的高度和宽度
# imgRec = np.zeros((height, width), dtype=np.uint8)  # 创建零数组

plt.figure(figsize=(10, 8))
for l in range(9, 0, -1):
    plt.subplot(3, 3, (9-l)+1, xticks=[], yticks=[])
    if l == 9:
        plt.imshow(img, cmap='gray'), plt.title('Original')
    else:
        imgBit = np.empty((height, width), dtype=np.uint8)  # 创建空数组
        for w in range(width):
            for h in range(height):
                x = np.binary_repr(img[w,h], width = 8)  # 以字符串形式返回输入数字的二进制表示形式
                x = x[::-1]
                a = x[l-1]
                imgBit[w,h] = int(a)  #第i位二进制的值
        plt.imshow(imgBit, cmap='gray')
        plt.title(f"{bin((l-1))}")
plt.show()
