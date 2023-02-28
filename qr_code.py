# encoding:UTF-8 
import cv2                                  # opencv库，用来显示二维码
import numpy as np                          # 科学计算库，用来创建图像
import random                               # 随机数库，用来生成随机数

image = np.zeros((512, 512), np.uint8)      # 生成一个512x512的单通道图像
image.fill(255)                             # 图片全白
for x in range(image.shape[0]):             # 遍历图像
    for y in range(image.shape[1]):
        num = random.randint(0,1)           # 从0，1之间任取一个数
        if num == 0:                        # 数字是0，对应的颜色是0，即黑色
            color = 0
        else:
            num = 255                       # 数字是1，对应的颜色是255，即白色
        image[x,y] = num                    # 将颜色赋值给图像
cv2.imshow("qr_code", image)                # 显示图像
cv2.waitKey()
cv2.destroyAllWindows()