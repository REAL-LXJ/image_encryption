# encoding:UTF-8 

from unittest import result
import qrcode                                                           # 二维码库  
import numpy as np                                                      # 科学计算库
from pyzbar import pyzbar                                               # 解码库
from PIL import Image                                                   # 图像文件库
import cv2                                                              # 计算机视觉库
import matplotlib.pyplot as plt                                         # 画图库                                   
import random                                                           # 随机数库
import time
import math


#*******************************  函数类  *******************************# 
#******** 1. 二维码密匙类 ********#
class QrCodeKey:

    def __init__(self, key_size, box_size):
        self.key_size = key_size
        self.box_size = box_size
    '''
    description: 生成密钥
    param {*} key_size：密钥长度
    return {*} key_array：生成的密钥
    '''
    def _generate_key(self):
        key_array = np.random.randint(0,2,self.key_size)                          # 生成一个0，1组成的512x512的数组                                                  # 打印数组
        return key_array

    '''
    description: 将密钥插入二维码中
    param {*} box_size：二维码大小
    param {*} key_array：生成的密钥
    param {*} imgpath：生成二维码的路径
    return {*} imgpath：生成二维码的路径
    '''
    def _insert_key_qrcode(self, key_array, imgpath):
        qr = qrcode.QRCode(                                                  # 创建一个二维码
            version = None,                                                  # 该参数表示二维码的大小
            error_correction = qrcode.constants.ERROR_CORRECT_Q,             # 二维码的纠错范围
            box_size = self.box_size,                                        # 每个点（方块）中的像素个数，！！！更改图像尺寸
            border = 4                                                       # 二维码距图像外围边框距离，默认为4
        )

        array_str = str(key_array)                                           # 将数组转为字符串
        qr.add_data(array_str)                                               # 将字符串传入二维码中
        qr.make(fit = True)                                                  # 自适应二维码大小

        img = qr.make_image(fill_color = "black", back_color = "white")      # 生成二维码图像，颜色为黑色，背景色为白色
        #print(img.size)                                                     # 打印二维码尺寸大小
        img.show()                                                           # 显示图像
        img.save("qrcode.png")                                               # 保存二维码      
        return imgpath

    '''
    description: 对二维码进行解码
    param {*} imgpath：生成二维码的路径
    return {*} barcodeData：解码出的密钥
    '''
    def _decode_qrcode(self, imgpath):
        qrcode_image = Image.open(imgpath)                                   # 打开二维码图片
        barcodes =  pyzbar.decode(qrcode_image)                              # 解码
        for barcode in barcodes:
            barcodeData = barcode.data.decode("utf-8")
        return barcodeData

#******** 2. 图像随机块类 ********#
class ImageRandomBlock:

    '''
    description: 将图像分割成m行n列
    param {*} img：待分割的图像
    param {*} m：行数
    param {*} n：列数
    return {*}
    '''
    def _devide_image_block(self, image, m, n):
        height, width = image.shape[0], image.shape[1]                                               # 获取原始图像的高宽
        grid_height = int(height*1.0/(m-1)+0.5)                                                      # 每个网格的高
        grid_width = int(width*1.0/(n-1)+0.5)                                                        # 每个网格的宽

        # 满足整除关系时的高宽
        suit_height = grid_height*(m-1)
        suit_width = grid_width*(n-1)

        # 图像缩放到合适的尺寸
        image_resize = cv2.resize(image, (suit_width, suit_height), cv2.INTER_LINEAR)
        # 生成网格采样点矩阵
        gx, gy = np.meshgrid(np.linspace(0, suit_width, n), np.linspace(0, suit_height, m))
        gx = gx.astype(np.int_)                                                                      
        gy = gy.astype(np.int_)

        # 创建一个图像张量，前两维表示分块后图像的位置(第m行，第n列)，后三维表示每个分块后的图像信息
        divide_image = np.zeros([m-1, n-1, grid_height, grid_width], np.uint8)

        # 图像分块
        for i in range(m-1):
            for j in range(n-1):      
                divide_image[i,j] = image_resize[gy[i][j]:gy[i+1][j+1], gx[i][j]:gx[i+1][j+1]]
        
        return divide_image

    '''
    description: 显示原始图像分块后的图像
    param {*} divide_image
    return {*}
    '''
    def _display_blocks(self, divide_image):
        m, n = divide_image.shape[0], divide_image.shape[1]                                          # 获取分块图像的行列
        for i in range(m):
            for j in range(n):
                value = i*n+j+1
                #print("value = \n", value)
                plt.subplot(m,n,value)
                plt.imshow(divide_image[i,j], cmap='gray')
                plt.axis('off')
                plt.title(str(value), fontsize = 8, verticalalignment='center', horizontalalignment='center')

    '''
    description: 还原原始图像
    param {*} divide_image
    return {*}
    '''
    def _restore_image(self, divide_image):
        m, n, grid_width, grid_height = [divide_image.shape[0],divide_image.shape[1],               #每行，每列的图像块数
                                        divide_image.shape[2],divide_image.shape[3]]                #每个图像块的尺寸
        
        restore_image = np.zeros([m*grid_height, n*grid_width], np.uint8)
        restore_image[0:grid_height,0:]
        for i in range(m):
            for j in range(n):
                restore_image[i*grid_height:(i+1)*grid_height, j*grid_width:(j+1)*grid_width] = divide_image[i,j]
        return restore_image

    '''
    description: 随机打乱图像
    param {*} divide_image 分块后的图像
    return {*} restore_image 打乱后的分块图像
    '''
    def _sort_image_blocks(self, divide_image):
        '''
        description: 显示随机打乱分块图像
        param {*} divide_image
        param {*} dict：图像键值对，key-value，key为0-15数字，value为分块的图像
        return {*}
        '''
        def _display_scramble_blocks(divide_image, dict):
            m, n = divide_image.shape[0], divide_image.shape[1]
            for i in range(m):
                for j in range(n):
                    value = i*n+j+1
                    plt.subplot(m,n,value)
                    plt.imshow(dict[value], cmap='gray')
                    plt.axis('off')
                    plt.title(str(value), fontsize = 8, verticalalignment='center', horizontalalignment='center')
            #plt.show()
        '''
        description: 显示随机打乱分块图像
        param {*} divide_image
        param {*} dict：图像键值对，key-value，key为0-15数字，value为分块的图像
        return {*}
        '''
        def _restore_scramble_image(divide_image, dict):
            m, n, grid_width, grid_height = [divide_image.shape[0],divide_image.shape[1],               
                                        divide_image.shape[2],divide_image.shape[3]]   
            restore_image = np.zeros([m*grid_height, n*grid_width], np.uint8)
            restore_image[0:grid_height,0:]  
            for i in range(m):
                for j in range(n):
                    value = i*n+j+1
                    restore_image[i*grid_height:(i+1)*grid_height, j*grid_width:(j+1)*grid_width] = dict[value]
            return restore_image          
        block_list = list(range(1, 17))
        #print("原始图像块列表 = ", block_list)                  # 打印原始图像块列表
        random.shuffle(block_list)
        #print("置乱图像块列表 = ", block_list)                  # 打印置乱后的图像块列表
        dict = {block_list[0]:divide_image[0][0],block_list[1]:divide_image[0][1],block_list[2]:divide_image[0][2],block_list[3]:divide_image[0][3],\
                block_list[4]:divide_image[1][0],block_list[5]:divide_image[1][1],block_list[6]:divide_image[1][2],block_list[7]:divide_image[1][3],\
                block_list[8]:divide_image[2][0],block_list[9]:divide_image[2][1],block_list[10]:divide_image[2][2],block_list[11]:divide_image[2][3],\
                block_list[12]:divide_image[3][0],block_list[13]:divide_image[3][1],block_list[14]:divide_image[3][2],block_list[15]:divide_image[3][3]}
        #_display_scramble_blocks(divide_image, dict)          # 显示分块后的图像
        restore_image = _restore_scramble_image(divide_image, dict)
        return restore_image

#******** 3. 图像加解密类-Arnold+logistic ********#
class ImageNoldLog:
    
    '''
    description: 猫脸置乱算法
    param {*} image 
    return {*} pic 
    '''
    def _image_arnold(self, image, Scramble_arnold_a, Scramble_arnold_b):
        row, col = image.shape
        pic = np.zeros((row,col), np.uint8)
        a = Scramble_arnold_a # 1
        b = Scramble_arnold_b # 1
        for i in range(row):
            for j in range(col):
                x = (i+b*j)%row
                y = (a*i+(a*b+1)*j)%col
                pic[x,y] = image[i,j]
        return pic 

    '''
    description: 猫脸置乱算法还原
    param {*} image 
    return {*} pic 
    '''
    def _image_dearnold(self, image, Scramble_arnold_a, Scramble_arnold_b):
        row, col = image.shape
        pic = np.zeros((row, col), np.uint8)
        a = Scramble_arnold_a
        b = Scramble_arnold_b
        for i in range(row):
            for j in range(col):
                x = ((a * b + 1) * i - b * j) % row
                y = (-a * i + j) % col
                pic[x, y] = image[i, j]
        return pic

    def _image_logistic(self, image, key):
        # 图像的宽高
        [w,h] = image.shape
        # 混沌系统初始条件
        x1 = key[0]
        x2 = key[1]
        x3 = key[2]
        # 分岔参数u
        u1 = key[3]
        u2 = key[4]
        u3 = key[5]
        # 加密次数
        n = key[6]
        # 一个临时数组，用于返回加密后的图像，可以不影响原始图像
        img_tmp = np.zeros((w,h))
        # 对原始图像的每个像素都处理n次
        for k in range(n):
            for i in range(w):
                for j in range(h):
                    # 计算混沌序列值
                    x1 = u1*x1*(1-x1)
                    x2 = u2*x2*(1-x2)
                    x3 = u3*x3*(1-x3)
                    # 混沌值位于[0,1]区间内，所以可以看做是一个系数，乘以最大灰度值并转成整数用于异或运算即可
                    r1 = int(x1*255)
                    r2 = int(x2*255)
                    r3 = int(x3*255)
                    img_tmp[i][j] = (((r1+r2)^r3)+image[i][j])%256
            # 下一轮加密重新初始化混沌系统
            x1 = key[0]
            x2 = key[1]
            x3 = key[2]
        return img_tmp

    '''
    description: 混沌扩散算法
    param {*} image 
    return {*} pic 

    def _image_logistic(self, image, x, u):
        row, col = image.shape
        sum = row * col
        x = u * x * (1-x)
        array = np.zeros(sum + 1000)                         
        array[0] = x    # ? test
        for i in range(sum + 999):
            array[i + 1] = u * array[i] * (1 - array[i])
        array = array[1000 : len(array)]
        array = np.array(array * 255, dtype='uint8')
        code = np.reshape(array, (row, col))
        xor = image ^ code
        pic = xor
        return pic  
    '''

    '''
    description: 混沌扩散算法还原
    param {*} image 
    return {*} pic 
    '''
    def _image_delogistic(self, image, x, u):
        '''
        row, col = image.shape
        x = u * x * (1-x)
        array = np.zeros(row * col)
        array[1] = x
        for i in range(1, row * col - 1):
            array[i + 1] = u * array[i] * (1 - array[i])
        array = np.array(array * 255, dtype='uint8')
        code = np.reshape(array, (row, col))
        xor = image ^ code
        pic = xor
        return pic   
        '''
        row, col = image.shape
        sum = row * col
        x = u * x * (1-x)
        array = np.zeros(sum + 1000)                         
        array[0] = x    # ? test
        for i in range(sum + 999):
            array[i + 1] = u * array[i] * (1 - array[i])
        array = array[1000 : len(array)]
        array = np.array(array * 255, dtype='uint8')
        code = np.reshape(array, (row, col))
        xor = image ^ code
        pic = xor
        return pic  




#******** 4. 待开发类 ********#

class MakeIdea:
    '''
    description: 图像平移
    param {*} image
    param {*} m
    param {*} n
    return {*} image 
    '''
    def _image_circle_shift(self, image, m, n):
        h,w = image.shape
        image = np.vstack((image[(h-m):,:],image[:(h-m),:]))
        image = np.hstack((image[:,(w-n):],image[:,:(w-n)]))
        return image

    '''
        description: 匹配二维码和图像
        param {*} image
        param {*} key_array
        return {*}
    '''
    def _match_image_and_key(self, image, key_array):
        sum0 = 0
        sum1 = 0
        if image.shape == key_array.shape:
            rows, cols = image.shape
        #print(type(image))
        #cv2.imshow("image", image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #print(key_array)
        #print(np.sum(key_array == 0))
        #print(np.sum(key_array == 1))
        for i in range(rows):
            for j in range(cols):
                if key_array[i,j] == 0:# 上左移
                    sum0 += 1
                    #image[i,j] = image[i+1,j+1]
                    #image[i,j] = 0
                elif key_array[i,j] == 1:# 下右移
                    #image[i,j] = image[i-1,j-1]
                    #image[i,j] = 255
                    sum1 += 1
        print(sum0)
        print(sum1)
        #print(sum0+sum1)
        return image

#******** 5. 算法评估类 ********#
class Evaluate:

    # 直方图
    def _evaluate_histogram(self, orig_img, pro_img):

        fig1 = plt.figure('原图灰度直方图')
        plt.hist(orig_img.ravel(), 256)
        plt.title('orig_histogram')
        #plt.show()

        fig2 = plt.figure('置乱图灰度直方图')
        #equ = cv2.equalizeHist(scr_img1)
        plt.hist(pro_img.ravel(), 256)
        plt.title('pro_histogram')
        plt.show()

    # 像素相关性
    def _evaluate_pixel_correlation(self, orig_img, pro_img):
        def _calculate_correlation(channel, N):
            h, w = channel.shape
            row = np.random.randint(0, h-1, N)    # 随机产生pixels个[0,w-1)范围内的整数序列
            col = np.random.randint(0, w-1, N)
            x = []
            h_y = []
            v_y = []
            d_y = []
            for i in range(N):
                # 选择当前一个像素
                x.append(channel[row[i]][col[i]])
                # 水平相邻像素是它的右侧也就是同行下一列的像素
                h_y.append(channel[row[i]][col[i]+1])
                # 垂直相邻像素是它的下方也就是同列下一行的像素
                v_y.append(channel[row[i]+1][col[i]])
                # 对角线相邻像素是它的右下即下一行下一列的那个像素
                d_y.append(channel[row[i]+1][col[i]+1])
            # 三个方向的合到一起
            x = x*3
            y = h_y+v_y+d_y

            # 计算E(x)
            ex = 0
            for i in range(N):
                ex += channel[row[i]][col[i]]
            ex = ex / N
            # 计算D(x)
            dx = 0
            for i in range(N):
                dx += (channel[row[i]][col[i]]-ex)**2
            dx /= N

            # 水平相邻像素h_y
            # 计算E(y)
            h_ey = 0
            for i in range(N):
                h_ey += channel[row[i]][col[i]+1]
            h_ey /= N
            # 计算D(y)
            h_dy = 0
            for i in range(N):
                h_dy += (channel[row[i]][col[i]+1]-h_ey)**2
            h_dy /= N
            # 计算协方差
            h_cov = 0
            for i in range(N):
                h_cov += (channel[row[i]][col[i]]-ex)*(channel[row[i]][col[i]+1]-h_ey)
            h_cov /= N
            h_Rxy = h_cov/(np.sqrt(dx)*np.sqrt(h_dy))

            # 垂直相邻像素v_y
            # 计算E(y)
            v_ey = 0
            for i in range(N):
                v_ey += channel[row[i]+1][col[i]]
            v_ey /= N
            # 计算D(y)
            v_dy = 0
            for i in range(N):
                v_dy += (channel[row[i]+1][col[i]]-v_ey)**2
            v_dy /= N
            # 计算协方差
            v_cov = 0
            for i in range(N):
                v_cov += (channel[row[i]][col[i]]-ex)*(channel[row[i]+1][col[i]]-v_ey)
            v_cov /= N
            v_Rxy = v_cov/(np.sqrt(dx)*np.sqrt(v_dy))

            # 对角线相邻像素d_y
            # 计算E(y)
            d_ey = 0
            for i in range(N):
                d_ey += channel[row[i]+1][col[i]+1]
            d_ey /= N
            # 计算D(y)
            d_dy = 0
            for i in range(N):
                d_dy += (channel[row[i]+1][col[i]+1]-d_ey)**2
            d_dy /= N
            # 计算协方差
            d_cov = 0
            for i in range(N):
                d_cov += (channel[row[i]][col[i]]-ex)*(channel[row[i]+1][col[i]+1]-d_ey)
            d_cov /= N
            d_Rxy = d_cov/(np.sqrt(dx)*np.sqrt(d_dy))

            return h_Rxy, v_Rxy, d_Rxy, x, y

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
        plt.tick_params(labelsize = 10)
        plt.subplot(221)
        plt.imshow(orig_img, cmap='gray')
        plt.title('原始图像', fontsize = 10)
        plt.subplot(222)
        ori_Rxy = _calculate_correlation(orig_img, N = 3000)
        plt.scatter(ori_Rxy[3], ori_Rxy[4], s=1, c='red')
        plt.title('原始图像像素相关性', fontsize = 10)
        plt.subplot(223)
        plt.imshow(pro_img, cmap='gray')
        plt.title('密文图像', fontsize = 10)
        pro_Rxy = _calculate_correlation(pro_img, N = 3000)
        plt.subplot(224)
        plt.scatter(pro_Rxy[3],pro_Rxy[4],s=1, c='blue')
        plt.title('密文图像像素相关性', fontsize = 10)
        
        print("*****该图像各方向的相关系数为*****")
        print(' 图像各方向 \t   Horizontal \t Vertical \t Diagonal')
        print(' 原始图像像素相关性 \t{:.4f}    {:.4f}    {:.4f}'.format(ori_Rxy[0],ori_Rxy[1],ori_Rxy[2]))
        print(' 密文图像像素相关性 \t{:.4f}    {:.4f}    {:.4f}'.format(pro_Rxy[0],pro_Rxy[1],pro_Rxy[2]))
        plt.show()

    # 信息熵
    def _evaluate_entropy(self, pro_img):
        w, h = pro_img.shape
        gray, num = np.unique(pro_img, return_counts=True)
        gray_entropy = 0

        for i in range(len(gray)):
            p1 = num[i]/(w*h)
            gray_entropy -= p1*(math.log(p1,2))
        return gray_entropy

    # 加密质量
    def _evaluate_EQ(self, orig_img, pro_img):
        w,h = orig_img.shape
        H1 = orig_img
        H2 = pro_img
        IH1 = {}
        IH2 = {}
        EQ = 0
        for i in range(512):
            IH1[i] = 0
            IH2[i] = 0

        for i in range(w):
            for j in range(h):
                IH1[H1[i][j]] += 1
                IH2[H2[i][j]] += 1

        for i in range(512):
            EQ += abs(IH1[i] - IH2[i])
        EQ /= 512
        return EQ

    def _evaluate_NPCR(self, img1, img2):
        w,h = img1.shape
        ar, num = np.unique((img1!=img2),return_counts=True)
        npcr = (num[0] if ar[0]==True else num[1])/(w*h)
        return npcr

    def _evaluate_UACI(self, img1, img2):
        w,h = img1.shape
        img1 = img1.astype(np.int16)
        img2 = img2.astype(np.int16)
        sum = np.sum(abs(img1-img2))
        uaci = sum/255/(w*h)
        return uaci




#*******************************  函数类  *******************************# 

#******************************* 功能函数 *******************************#        
# 1.生成密钥二维码
def generate_qrcode_key():
    key_size = (512,512)                                                 # 密钥长度
    box_size = 10                                                        # 可更改二维码尺寸，1-(97x97), 10-(970x970)
    imgpath = "qrcode.png"                                               # 二维码位置

    qrkey = QrCodeKey(key_size, box_size)
    # 1.1 生成密钥
    key_array = qrkey._generate_key()                                    # 生成密钥，256位0、1组成的密钥
    #print(key_array.shape)                                              # 打印密钥
    # 1.2 将密钥插入二维码
    #imgpath = qrkey._insert_key_qrcode(key_array, imgpath)               # 密钥插入二维码
    # 1.3 解码二维码得到密钥
    #barcodeData = qrkey._decode_qrcode(imgpath)                          # 解码二维码得到密钥
    #print("secret_key = \n",barcodeData)                                 # 打印密钥
    return key_array

# 2.图像随机分块
def image_random_blocks(gray_image):
    
    imgranbk = ImageRandomBlock()

    #* 2.1 图像分块
    m = 4
    n = 4
    divide_image = imgranbk._devide_image_block(gray_image, m+1, n+1)
    #print(divide_image.shape)
    #fig1 = plt.figure('图像分块')
    #imgranbk._display_blocks(divide_image)
    #plt.show()

    #* 2.2 随机排序
    random_image = imgranbk._sort_image_blocks(divide_image)
    #print(random_image.shape)
    #fig2 = plt.figure('随机图像还原')
    #plt.imshow(random_image, cmap='gray')
    #plt.axis('off')
    #plt.title('random_image')
    #plt.show()

    '''
    fig3 = plt.figure('分块图像的还原')
    restore_image = imgranbk._restore_image(divide_image)            #图像缩放法分块还原
    plt.imshow(restore_image, cmap='gray')
    plt.axis('off')
    plt.title('restore_image')
    plt.show()
    '''

    return random_image

# 3.图像加密
def image_encryption(random_image, \
                    Scramble_arnold_a, Scramble_arnold_b, Scramble_arnold_times, \
                    Spread_logistic_x1, Spread_logistic_x2, Spread_logistic_x3,\
                    Spread_logistic_u1, Spread_logistic_u2, Spread_logistic_u3,\
                    Spread_logistic_times):
    
    imgencry = ImageNoldLog()                                                 
    
    #******* 图像加密是对随机排序后的图片进行加密 ******#
    #! 猫脸置乱
    a = Scramble_arnold_a
    b = Scramble_arnold_b
    scramble_times = Scramble_arnold_times
    for _ in range(scramble_times):
        arnold_image = imgencry._image_arnold(random_image, a, b)
    #! logistic扩散
    x1 = Spread_logistic_x1   # 0 < x < 1
    x2 = Spread_logistic_x2
    x3 = Spread_logistic_x3
    u1 = Spread_logistic_u1   # 3.5699456...<u<=4
    u2 = Spread_logistic_u2
    u3 = Spread_logistic_u3
    n = Spread_logistic_times
    key = [x1,x2,x3,u1,u2,u3,n] 
    logistic_image = imgencry._image_logistic(arnold_image, key)
    '''
    for _ in range(spread_times):
        logistic_image = imgencry._image_logistic(arnold_image, x, u)
    '''

    return logistic_image

# 4.图像解密
def image_decryption(encryption_image, \
                    Scramble_arnold_a, Scramble_arnold_b, Scramble_arnold_times, \
                    Spread_logistic_x, Spread_logistic_u, Spread_logistic_times):
    #print("image decryption starting *******")
    # 先扩散，后置乱
    imgdecry = ImageNoldLog()
    #! logistic逆扩散
    x = Spread_logistic_x # 0 < x < 1
    u = Spread_logistic_u   # 3.5699456...<u<=4
    spread_times = Spread_logistic_times
    for _ in range(spread_times):
        delogistic_image = imgdecry._image_delogistic(encryption_image, x, u)
    a = Scramble_arnold_a
    b = Scramble_arnold_b
    scramble_times = Scramble_arnold_times
    for _ in range(scramble_times):
        dearnold_image = imgdecry._image_dearnold(delogistic_image, a, b)
    return dearnold_image

# 5.图像评估    
def image_evaluate(orig_img, scr_img1, scr_img2):
    imgel = Evaluate()
    #1. 直方图
    imgel._evaluate_histogram(orig_img, scr_img1)
    #2. 像素相关性
    imgel._evaluate_pixel_correlation(orig_img, scr_img1)
    #3. 图像信息熵
    gray_entropy = imgel._evaluate_entropy(scr_img1)
    print('图像信息熵:{:.4}'.format(gray_entropy))
    #4. 加密质量
    EQ = imgel._evaluate_EQ(orig_img, scr_img1)
    print('加密质量:{:.0f}'.format(EQ))
    #5. NPCR+UACI
    npcr = imgel._evaluate_NPCR(scr_img1, scr_img2)
    print('NPCR  :{:.4%}'.format(npcr))
    uaci = imgel._evaluate_UACI(scr_img1, scr_img2)
    print('UACI  :{:.4%}'.format(uaci))
#******************************* 功能函数 *******************************#


if __name__ == "__main__":

    print('开始处理图像！')
    #******** 1.生成密钥二维码 ********#
    key_array = generate_qrcode_key()

    img = cv2.imread("lena.png")                            # 读入原始BGR图像
    #print("orig_image_shape = ",img.shape)                 # 打印原始图像信息
    gray_img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        # BGR转灰度图
    gray_img2 = gray_img1.copy()
    gray_img2[511, 511] = 255 - gray_img2[511, 511]
    x = np.binary_repr(gray_img2[511, 511], width=8) 
    print(gray_img2)
    #print(gray_img2[255, 255])

    #******** 2.图像随机分块   ********#
    random_image1 = image_random_blocks(gray_img1)
    random_image2 = image_random_blocks(gray_img2)

    #******** 3.图像加密      ********#
     #******** 需要调节的参数   ********#
      #******** 3.1 图像置乱--猫脸算法 ********#
    Scramble_arnold_a = 1                                   # 猫脸算法参数1 a(像素位置操作)         a∈[1,+∞],a∈+N
    Scramble_arnold_b = 1                                   # 猫脸算法参数2 b(像素位置操作)         b∈[1,+∞],b∈+N
    Scramble_arnold_times = 3                               # 猫脸算法参数3 times(轮数)            times∈[30,700],times∈+N
      #******** 3.2 图像扩散--logistic算法 ********#
    key1 = [0.343,0.432,0.63,3.769,3.82,3.85,5] 
    Spread_logistic_x1 = key1[0]                             # 混沌算法参数1 x1(logistic扩散初值)     x∈(0,1)          
    Spread_logistic_x2 = key1[1]                             # 混沌算法参数2 x2(logistic扩散初值)     x∈(0,1)    
    Spread_logistic_x3 = key1[2]                             # 混沌算法参数3 x3(logistic扩散初值)     x∈(0,1)    
    Spread_logistic_u1 = key1[3]                             # 混沌算法参数4 u1(logistic扩散参数)     u∈(3.5699456,4)     
    Spread_logistic_u2 = key1[4]                             # 混沌算法参数5 u2(logistic扩散参数)     u∈(3.5699456,4)
    Spread_logistic_u3 = key1[5]                             # 混沌算法参数6 u3(logistic扩散参数)     u∈(3.5699456,4)
    Spread_logistic_times = key1[6]                          # 混沌算法参数3 times(轮数)              times∈[500,2000],times∈+N   
    entire_times = 3                                         # 整轮次数                                                         
    for i in range(entire_times):
        encryption_image1 = image_encryption(random_image1,\
                                    Scramble_arnold_a, Scramble_arnold_b, Scramble_arnold_times, \
                                    Spread_logistic_x1, Spread_logistic_x2, Spread_logistic_x3,\
                                    Spread_logistic_u1, Spread_logistic_u2, Spread_logistic_u3,\
                                    Spread_logistic_times)
        encryption_image2 = image_encryption(random_image2,\
                                    Scramble_arnold_a, Scramble_arnold_b, Scramble_arnold_times, \
                                    Spread_logistic_x1, Spread_logistic_x2, Spread_logistic_x3,\
                                    Spread_logistic_u1, Spread_logistic_u2, Spread_logistic_u3,\
                                    Spread_logistic_times)
    
    cv2.imwrite('pro_image.png',encryption_image1, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    
    #******** 4. 图像解密    ********#
    '''
    for i in range(entire_times):
        decryption_image = image_decryption(encryption_image, \
                                            Scramble_arnold_a, Scramble_arnold_b, Scramble_arnold_times, \
                                            Spread_logistic_x, Spread_logistic_u, Spread_logistic_times)
    '''
    #******** 5. 评价图像     ********#
    image_evaluate(gray_img1, encryption_image1, encryption_image2)

    #******** 6. 显示图像     ********#

    '''
    for i in range(11):
        time.sleep(0.5)
        print('\r当前进度：{0}{1}%'.format('▉▉▉▉'*i,(i*10)), end='')
    print('处理完成！')
    '''
    
    #matrix = gray_img - decryption_image
    #print(np.all(matrix == 0))
    #img1 = np.hstack((gray_img, random_image))
    #img2 = np.hstack((img1, encryption_image))
    #img3 = np.hstack((img2, decryption_image))
    #result = np.hstack((img3, gray_img))
    #result = cv2.resize(result, None, fx = 0.5, fy = 0.5)
    #cv2.imshow("orignal_image", gray_img)                       # 原始图像--灰度图
    #cv2.imshow("random_image", random_image)                    # 随机分块后的图像--明文图像
    #cv2.imshow("encryption_image", encryption_image)            # 加密后的密文图像
    #cv2.imshow("decryption_image", decryption_image)            # 解密后的明文图像--随机分块后的图像   
    #cv2.imshow("restore_image", gray_img)                       # 还原后的图像
    #cv2.imshow("result", result)

    #cv2.imshow("encryption_image", encryption_image)            # 加密后的密文图像
    #cv2.imshow("decryption_image", decryption_image)            # 解密后的明文图像
    
    '''
    print("图像依次为：\n")
    print("1.原始灰度图像\n")
    print("2.打乱后的图像\n")
    print("3.加密后的图像\n")
    print("4.解密后的图像\n")
    print("5.还原后的图像\n")
    '''
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    


    
