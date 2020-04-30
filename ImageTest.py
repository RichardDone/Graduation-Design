import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageTk,ImageFilter,ImageEnhance
import math
from skimage import measure,color,morphology,img_as_ubyte,img_as_float,transform
import pandas as pd

def coverup(location_all,leaf1,leaf2):
    leaf_test = np.zeros(im3.shape[:], dtype=np.uint8)
    for t in location_all[leaf1]:
        leaf_test[t[0], t[1]] = 255
    for t in location_all[leaf2]:
        leaf_test[t[0], t[1]] = 255

    x = []
    y = []
    for i in range(leaf_test.shape[1]):
        for j in range(leaf_test.shape[0]):
            if (leaf_test[j, i] > 0):
                x.append(i)
                y.append(j)

    f1 = np.polyfit(x, y, 5)
    p1 = np.poly1d(f1)
    yvals = p1(x)

    plt.imshow(opencv2skimage(leaf_test))
    plt.plot(x, yvals, 'r')
    plt.show()

def opencv2skimage(any_opencv_image):
    sk_image = img_as_float(any_opencv_image)
    return sk_image

def skimage2opencv(any_skimage_image):
    op_image = img_as_ubyte(any_skimage_image)
    return op_image

def angle(x1,y1,x2,y2):
    if y1==y2:
        len=9999
    else:
        len = (math.fabs(x1-x2)/math.fabs(y1-y2))
    return len

def resize(w_box, h_box, pil_image):  # 参数是：要适应的窗口宽、高、Image.open后的图片
    w, h = pil_image.size  # 获取图像的原始大小
    f1 = 1.0 * w_box / w
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)

def showImage(img):
    img = Image.fromarray(img)
    img_show = img.copy()
    resize(800,600,img_show)
    img_show.show()

# 读取图片
filepath = 'images/'
filename = '20180711'
filetype = '.png'
img_png = Image.open(filepath+filename+filetype)

# 将PIL格式转换为np矩阵
img = np.asarray(img_png)
# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 二值化
ret, binary = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
# 读取轮廓
img1, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print(type(contours))
# print(type(contours[0]))
# print(len(contours))

left_x=0  # 左上角横坐标
left_y=0  # 左上角纵坐标
wid=0     #矩形宽度
hei=0     #矩形高度

for i in range(0, len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    if (w*h)>30000 and (w*h)<5000000 and h<2000:
            # print(x,y,w,h)
            left_x=x
            left_y=y
            wid = w
            hei = h
            cv2.rectangle(img, (x-60, y-50), (x + w+80, y + h), (0, 0, 255), 2)

# 显示带轮廓的原图像
# img_png = Image.fromarray(img)
# resize(800,600,img_png)
# img_png.show()

# 裁剪图像，显示的就是
img_cut = img_png.crop((left_x-60, left_y-50, left_x + wid+80, left_y + hei))

# img_show = img_cut.copy()
# resize(800,600,img_show)
# img_show.show()

# 图像灰度化
img_cut = img_cut.convert('L')    #调用Image中的函数
# 图像增强
img_cut = ImageEnhance.Brightness(img_cut).enhance(1.0)  # 亮度增强 1.0为原始图像
img_cut = ImageEnhance.Color(img_cut).enhance(800.0)  # 色度增强 1.0为原始图像
img_cut = ImageEnhance.Contrast(img_cut).enhance(15.5)  # 对比度增强
# img_cut = ImageEnhance.Sharpness(img_cut).enhance(100)  # 锐化增强
# img_show = img_cut.copy()
# resize(800,600,img_show)
# img_show.show()
# 二值化反转
img_cut = np.asarray(img_cut)
ret,img_cut = cv2.threshold(img_cut, 175, 255, cv2.THRESH_BINARY_INV)
# showImage(img_cut)

# 骨架提取
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)) #卷积核，定义一个3x3的十字形结构元素

skel = np.zeros(img_cut.shape, np.uint8) #骨架图，初始化是原图
erode = np.zeros(img_cut.shape, np.uint8)
temp = np.zeros(img_cut.shape, np.uint8)

skel_num=0

while True:
    # 图像腐蚀
    erode = cv2.erode(img_cut, element)
    #图像膨胀
    temp = cv2.dilate(erode, element)

    # 消失的像素是skeleton的一部分
    temp = cv2.subtract(img_cut, temp)   #相减，img_cut-temp即消失的部分，也就是轮廓的一部分
    skel = cv2.bitwise_or(skel, temp) #或运算，将删除的部分添加到骨架图
    img_cut = erode.copy()
    if cv2.countNonZero(img_cut) == 0:
        break
    skel_num = skel_num+1
img_cut = skel

# 去噪点（去掉面积较小的连通域）
_, labels, stats, centroids = cv2.connectedComponentsWithStats(img_cut)
i = 0
for istat in stats:
    if istat[4] < 2:
        if istat[3] > istat[4]:
            r = istat[3]
        else:
            r = istat[4]
        cv2.rectangle(img_cut, tuple(istat[0:2]), tuple(istat[0:2] + istat[2:4]), 0, thickness=-1)  # 26
    i = i + 1
# showImage(img_cut)

# 霍夫直线变化
# PIL换成np数组
im = np.asarray(img_cut)
# 统计概率霍夫线变换
# 边缘检测
edges = cv2.Canny(im, 50, 150, apertureSize=3)  # apertureSize参数默认其实就是3
# showImage(edges)

# HoughLinesP参数1是二值图；
# 参数2是举例r的精度，值越大考虑越多的线；
# 参数3是θ的精度，值越小考虑线越多；
# 参数4是累加数阈值，值越小考虑的线越多 minLineLength是直线最短长度，MaxLineGap直线内是两点最大间隔
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=7, maxLineGap=50)
min_angle = 2000000
maxX = 0
minX = 0
# 两个drawing用来画检测出的直线
drawing = np.zeros(im.shape[:], dtype=np.uint8)
drawing2 = np.zeros(im.shape[:], dtype=np.uint8)
for line in lines:
    x1, y1, x2, y2 = line[0]
    if min_angle>=angle(x1,y1,x2,y2) and math.fabs(y1-y2)>100:
        min_angle = angle(x1,y1,x2,y2)
        maxX = max(x1,x2)
        minX = max(x1,x2)
        # print(x1,y1,x2,y2)
    #     cv2.line(drawing, (x1, y1), (x2, y2), (255, 255, 255), 2)
    # else:
    #     cv2.line(drawing2, (x1, y1), (x2, y2), (255, 255, 255), 2)
# showImage(drawing)
# showImage(drawing2)

# 显示茎秆
# im2 = im.copy()
# im2_height = im2.shape[0]
# im2_width = im2.shape[1]
#
# for i in range(im2_height):
#     for j in range(im2_width):
#         if(j<=maxX+20 and j>=minX-35):
#             pass
#         else:
#             im2[i,j]=0
# showImage(im2)

#显示叶片
im3 = im.copy()
im3_height = im3.shape[0]
im3_width = im3.shape[1]

for i in range(im3_height):
    for j in range(im3_width):
        if(j<=maxX+20 and j>=minX-20):
            im3[i,j]=0
        else:
            pass
# showImage(im3)


# 标记提取叶片
# 骨架提取导致叶片变的断断续续
# 所以先进行膨胀，让细碎的连通域组成一个较大的连通域
img_extract = im3.copy()
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)) #卷积核，定义一个5x5的十字形结构元素
i=5
while i>0:
    img_extract = cv2.dilate(img_extract,element)
    i=i-1

# 标记各个连通域
labels = measure.label(img_extract, connectivity=2)

# 定义数组存储分割出来的叶片骨架
img_leaf = []

# 定义数组存储质心坐标
centroids_leaf = []

# 定义数组存储外接矩形坐标
bbox = []

# 定义数组存储连通域内部坐标值
location_all =[]

# 通过label标记，遍历各个连通域
for region in measure.regionprops(labels):
    # 连通域外接矩形的坐标
    minr, minc, maxr, maxc = region.bbox
    # 连通域内部像素坐标值,列表嵌套，一组坐标为单位存储
    location = region.coords
    # 定义和原图同样大小的临时图片，用来画出树叶
    leaf_test = np.zeros(im3.shape[:], dtype=np.uint8)

    # 判断长宽，去除较小的连通域
    if (maxc-minc)<50 or (maxr-minr)<80:
    # if (im3.shape[0]-minr)<100 or (maxc-minc)<20:
        pass
    else:
        # 存储质心坐标
        centroids_leaf.append(region.centroid)
        # 存储矩形坐标
        bbox.append(region.bbox)
        # 存储内部像素坐标值
        location_all.append(location)
        cv2.rectangle(img_extract,(minc, minr), (maxc,maxr),(255,255,255),2)
        for t in location:
            leaf_test[t[0],t[1]]=255
        leaf = leaf_test[minr:maxr,minc:maxc]
        img_leaf.append(leaf)

# for i in range(0,len(bbox)):
#     if(bbox[i][2]-bbox[i][0])<80 or (bbox[i][3]-bbox[i][1])<50:
#         if i > 0:
#             distance1 = (centroids_leaf[i][0]-centroids_leaf[i-1][0])**2 +(centroids_leaf[i][1]-centroids_leaf[i-1][1])**2
#             distance2 = (centroids_leaf[i][0]-centroids_leaf[i+1][0])**2 +(centroids_leaf[i][1]-centroids_leaf[i+1][1])**2
#             if distance1>distance2:
#                 coverup(location_all,i,i+1)
                # minr = min(bbox[i][0],bbox[i+1][0])
                # minc = min(bbox[i][1],bbox[i+1][1])
                # maxr = max(bbox[i][2],bbox[i+1][2])
                # maxc = max(bbox[i][3],bbox[i+1][3])
                # cv2.rectangle(img_extract,(minc, minr), (maxc,maxr),(255,255,255),2)
                # leaf_test = np.zeros(im3.shape[:], dtype=np.uint8)
                # for t in location_all[i]:
                #     leaf_test[t[0],t[1]]=255
                # for t in location_all[i+1]:
                #     leaf_test[t[0],t[1]]=255
                # leaf = leaf_test[minr:maxr,minc:maxc]
                # img_leaf.append(leaf)
                # i = i+1
            # else:
            #     coverup(location_all,i,i-1)
                # minr = min(bbox[i][0], bbox[i - 1][0])
                # minc = min(bbox[i][1], bbox[i - 1][1])
                # maxr = max(bbox[i][2], bbox[i - 1][2])
                # maxc = max(bbox[i][3], bbox[i - 1][3])

                # cv2.rectangle(img_extract, (bbox[i - 1][1], bbox[i - 1][0]), (bbox[i - 1][3], bbox[i - 1][2]),
                #               (0, 0, 0), 2)
                # cv2.rectangle(img_extract, (minc, minr), (maxc, maxr), (255, 255, 255), 2)
                #
                # leaf_test = np.zeros(im3.shape[:], dtype=np.uint8)
                # for t in location_all[i]:
                #     leaf_test[t[0], t[1]] = 255
                # for t in location_all[i - 1]:
                #     leaf_test[t[0], t[1]] = 255
                # leaf = leaf_test[minr:maxr, minc:maxc]
                # # img_leaf.pop()
                # img_leaf.append(leaf)
                # i = i+1
        # elif i ==0:
        #     coverup(location_all,i,i+1)
            # minr = min(bbox[i][0], bbox[i + 1][0])
            # minc = min(bbox[i][1], bbox[i + 1][1])
            # maxr = max(bbox[i][2], bbox[i + 1][2])
            # maxc = max(bbox[i][3], bbox[i + 1][3])
            # cv2.rectangle(img_extract, (minc, minr), (maxc, maxr), (255, 255, 255), 2)
            # leaf_test = np.zeros(im3.shape[:], dtype=np.uint8)
            # for t in location_all[i]:
            #     leaf_test[t[0], t[1]] = 255
            # for t in location_all[i + 1]:
            #     leaf_test[t[0], t[1]] = 255
            # leaf = leaf_test[minr:maxr, minc:maxc]
            # img_leaf.append(leaf)
            # i=i+1
    # else:
    #     if(i<len(bbox)-1) and ((bbox[i+1][2]-bbox[i+1][0])<80 or (bbox[i+1][3]-bbox[i+1][1])<50):
    #         continue
    #     elif (i>0) and ((bbox[i-1][2]-bbox[i-1][0])<80 or (bbox[i-1][3]-bbox[i-1][1])<50):
    #         continue
    #     else:
    #         minr = bbox[i][0]
    #         minc = bbox[i][1]
    #         maxr = bbox[i][2]
    #         maxc = bbox[i][3]
    #         leaf_test = np.zeros(im3.shape[:], dtype=np.uint8)
    #         cv2.rectangle(img_extract,(minc, minr), (maxc,maxr),(255,255,255),2)
    #         for t in location_all[i]:
    #             leaf_test[t[0],t[1]]=255
    #         leaf = leaf_test[minr:maxr,minc:maxc]
    #         img_leaf.append(leaf)

showImage(img_extract)

print("叶片的数量为：",len(img_leaf))

# # 存储叶片基部点坐标
#
# array_bottompoint = []
#
# for i in range(len(img_leaf)):
#     leaf = img_leaf[i]
#     k = 3
#     element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 卷积核，定义一个3x3的十字形结构元素
#     while k > 0:
#         leaf = cv2.erode(leaf, element)
#         k -= 1
#
#     flag = 0
#     leaf_height, leaf_width = leaf.shape
#
#     if centroids_leaf[i][1]>maxX:
#         for y in range(leaf_height-1,-1,-1):
#             for x in range(leaf_width):
#                 if(leaf[y,x]==255):
#                     array_bottompoint.append([i+1,y+bbox[i][0],x+bbox[i][1]])
#                     flag=1
#                     print("第",i+1,"个叶片的基部点坐标：","Height:",array_bottompoint[i][1],"Width:",array_bottompoint[i][2])
#                     break
#             if(flag):
#                 break
#         # plt.imshow(opencv2skimage(leaf))
#         # plt.plot(array_bottompoint[i][2]-bbox[i][1],array_bottompoint[i][1]-bbox[i][0],'o',color='r')
#         # plt.show()
#     else:
#         for y in range(leaf_height - 1, -1,-1):
#             for x in range(leaf_width-1,-1,-1):
#                 if (leaf[y, x] == 255):
#                     array_bottompoint.append([i + 1, y + bbox[i][0], x + bbox[i][1]])
#                     flag = 1
#                     print("第", i + 1, "个叶片的基部点坐标：", "Height:", array_bottompoint[i][1], "Width:",
#                           array_bottompoint[i][2])
#                     break
#             if (flag):
#                 break
#         # plt.imshow(opencv2skimage(leaf))
#         # plt.plot(array_bottompoint[i][2] - bbox[i][1], array_bottompoint[i][1] - bbox[i][0], 'o', color='r')
#         # plt.show()
#
# data_toppoint = pd.DataFrame(array_bottompoint)
# data_toppoint.to_csv(filename + '_bottompoint.csv', header=['blade','height/像素点', 'width/像素点'], index=False, encoding="gbk")



# # 存储叶片最高点坐标
# array_toppoint = []
#
# for i in range(len(img_leaf)):
#     leaf = img_leaf[i]
#     k = 3
#     element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 卷积核，定义一个3x3的十字形结构元素
#     while k > 0:
#         leaf = cv2.erode(leaf, element)
#         k -= 1
#
#     flag = 0
#     leaf_height, leaf_width = leaf.shape
#     for y in range(leaf_height):
#         for x in range(leaf_width):
#             if (leaf[y, x] == 255):
#                 array_toppoint.append([i+1,y+bbox[i][0],x+bbox[i][1]])
#                 flag =1
#                 print("第", i + 1, "个叶片的最高点坐标为：", "Height:", array_toppoint[i][1], "Width:",array_toppoint[i][2])
#                 break
#         if(flag):
#             break
#         # plt.imshow(opencv2skimage(leaf))
#         # plt.plot(array_toppoint[i][1],array_toppoint[i][0],'o',color='r')
#         # plt.show()
#
# data_toppoint = pd.DataFrame(array_toppoint)
# data_toppoint.to_csv(filename + '_toppoint.csv', header=['blade','height/像素点', 'width/像素点'], index=False, encoding="gbk")

# # 存储叶片角度
# array_angel = []
#
# for i in range(len(img_leaf)):
#     leaf = img_leaf[i]
#
#     k= 3
#     element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 卷积核，定义一个3x3的十字形结构元素
#     while k>0:
#         leaf = cv2.erode(leaf, element)
#         k-=1
#
#     leaf_height, leaf_width = leaf.shape
#     leaf_y = int((2 / 3) * leaf_height)
#     leaf_x = int((2 / 3) * leaf_width)
#
#     for y in range(leaf_height):
#         if (y == leaf_y):
#             for x in range(leaf_width):
#                 if (leaf[y, x] == 255):
#                     leaf_x = x
#                     break
#         if (y > leaf_y):
#             break
#     # print(leaf_y, leaf_x, leaf_height, leaf_width)
#     plt.imshow(opencv2skimage(leaf))
#     if centroids_leaf[i][1] > maxX:
#         leaf_angle = math.atan(math.fabs(leaf_height - leaf_y) / math.fabs(leaf_x))
#         plt.plot([0, leaf_x], [leaf_height, leaf_y],"r")
#     else:
#         leaf_angle = math.atan(math.fabs(leaf_height - leaf_y) / math.fabs(leaf_width - leaf_x))
#         plt.plot([leaf_width, leaf_x], [leaf_height, leaf_y],"r")
#
#     array_angel.append([i+1,leaf_angle * 180 / math.pi])
#     print("第",i+1,"个叶片的角度为：{:.4f}".format(leaf_angle * 180 / math.pi))
#
#     angle_num = str(i + 1)
#     angle_savepath = 'angle_pic/'
#     plt.savefig(angle_savepath + filename + '_angle_blade' + angle_num + filetype)
#     plt.clf()
#     # plt.show()
#
# data_angel = pd.DataFrame(array_angel)
# data_angel.to_csv(filename+'_angle.csv',header=['blade','angle/°'],index=False,encoding="gbk",float_format="%.4f")
#
# # 存储叶片长度
# array_length = []
#
# for leaf_i in range(len(img_leaf)):
#     leaf1 = img_leaf[leaf_i]
#     k= 3
#     element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 卷积核，定义一个3x3的十字形结构元素
#     while k>0:
#         leaf1 = cv2.erode(leaf1, element)
#         k-=1
#     x=[]
#     y=[]
#     for i in range(leaf1.shape[1]):
#         for j in range(leaf1.shape[0]):
#             if(leaf1[j,i]>0):
#                 x.append(i)
#                 y.append(j)
#
#     f1 = np.polyfit(x, y, 5)
#     p1 = np.poly1d(f1)
#     yvals = p1(x)
#
#     # 计算每个叶片的长度，即曲线长度
#     area_list = []  # 存储每一微小步长的曲线长度
#     for i in range(1,len(x)):
#         # 计算每一微小步长的曲线长度，dx = x_{i}-x{i-1}，索引从1开始
#         dl_i = np.sqrt( (x[i]-x[i-1])**2 + (yvals[i]-yvals[i-1])**2 )
#         # 将计算结果存储起来
#         area_list.append(dl_i)
#     area = sum(area_list)  # 求和计算曲线在t:[0,2*pi]的长度
#     area = area/24*0.635   # 24个像素=0.635厘米
#
#     array_length.append([leaf_i+1,area])
#     print("第",leaf_i+1,"个叶片：{:.4f}厘米".format(area))
#
#     plt.imshow(opencv2skimage(leaf1))
#     plt.plot(x, yvals, 'r')
#
#     length_num = str(leaf_i + 1)
#     length_savepath = 'length_pic/'
#     plt.savefig(length_savepath + filename+'_length_blade' + length_num + filetype)
#     plt.clf()
#     # plt.show()
#
# data_length = pd.DataFrame(array_length)
# data_length.to_csv(filename+'_length.csv',header=['blade','length/cm'],index=False,encoding='gbk',float_format="%.4f")
#
# # 求植株高度
# img_plant = im.copy()
# img_plant_height = img_plant.shape[0]
# img_plant_width = img_plant.shape[1]
#
# location_top = img_plant_height
# location_bottom = 0
# location_level = 0
#
# for i in range(img_plant_height):
#     for j in range(img_plant_width):
#         if img_plant[i,j]==255:
#             location_top=i
#             location_level=j
#             break
#     if location_top==i:
#         break
#
# for i in range(img_plant_height-1,-1,-1):
#     for j in range(img_plant_width):
#         if img_plant[i,j]==255:
#             location_bottom = i
#             break
#     if location_bottom==i:
#         break
#
# plant_height = location_bottom - location_top
# plant_height = plant_height/24*0.635
# print("高度为：{:.4f}厘米".format(plant_height))
#
# plt.imshow(opencv2skimage(img_plant))
# plt.plot([location_level,location_level],[location_bottom,location_top],"r")
# plt.savefig(filename+'_height'+filetype)
# plt.clf()
# # plt.show()
#
#
#
#
