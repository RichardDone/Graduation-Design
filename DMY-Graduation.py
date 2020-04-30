# -*- coding: utf-8 -*-
"""
Author: Richard_Done
Last edited: April 8,2020
"""
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import math
from skimage import measure,color,morphology,img_as_ubyte,img_as_float,transform

from tkinter import filedialog,messagebox
from tkinter import *
from PIL import Image,ImageTk,ImageFilter,ImageEnhance
import win32clipboard as clip
import win32con     #pip install pywin32
from io import BytesIO

global img_all  # 全局存储所有窗口展示过的图片
img_all = []
global img_png  # 定义全局变量图像的

Myheight= 600    #高
Mywidth= 800     #宽

def opencv2pil(any_opencv_image):
    pil_img = Image.fromarray(any_opencv_image)
    return pil_img

def pil2opencv(any_pil_image):
    op_image = np.asarray(any_pil_image)
    return op_image

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

#图片自适应画布大小
def resize( w_box, h_box, pil_image): #参数是：要适应的窗口宽、高、Image.open后的图片
      w, h = pil_image.size #获取图像的原始大小
      f1 = 1.0*w_box/w
      f2 = 1.0*h_box/h
      factor = min([f1, f2])
      width = int(w*factor)
      height = int(h*factor)
      return pil_image.resize((width, height), Image.ANTIALIAS)

def showImage(img):
    img_show = img.copy()
    img_show = resize(window.winfo_width(), window.winfo_height(),img_show)
    canvas.delete(ALL)
    photo = ImageTk.PhotoImage(img_show)
    canvas.create_image(int(window.winfo_width() / 2), 0, anchor='n', image=photo)  # 图片锚定点（n图片顶端的中间点位置）放在画布（400,0）坐标处
    canvas.image = photo

#基础功能模块

#打开图片
def openfiles():
    global img_png
    global filename
    global filetype

    file_path = filedialog.askopenfilename(title='选择文件',filetype=[('all files', '.*'),('JPG','jpg'),('PNG','png')])
    if file_path!='':
        print("打开路径",file_path)
        file = file_path.split('/')
        filename,filetype = file[len(file)-1].split('.')
        print("文件名："+filename,"文件类型："+filetype)
        img_png = Image.open(file_path)
        img_all.append(img_png)
        showImage(img_png)

#保存图片，格式为PNG
def savefiles():
    global img_png
    file_path = filedialog.asksaveasfilename(title='保存文件', filetypes=[("PNG", ".png")])

    if file_path!='':
        print("保存路径",file_path)
        img_png.save(str(file_path) + '.png', 'PNG')

def ImgCopy():
    global img_png
    output = BytesIO()
    img_png.convert("RGB").save(output,"BMP")
    data=output.getvalue()[14:]
    clip.OpenClipboard() #打开剪贴板
    clip.EmptyClipboard()  #先清空剪贴板
    clip.SetClipboardData(win32con.CF_DIB, data)  #将图片放入剪贴板
    clip.CloseClipboard()

def Imgpast():
    global img_png
    img_tep=img_png.copy()
    output = BytesIO()
    img_png.convert("RGB").save(output, "BMP")
    data = output.getvalue()[14:]
    clip.OpenClipboard()  # 打开剪贴板
    clip.EmptyClipboard()  # 先清空剪贴板
    clip.SetClipboardData(win32con.CF_DIB, data)  # 将图片放入剪贴板
    clip.CloseClipboard()

    canvas.delete(ALL)

#撤销函数
def ImgUndo():
    global img_png
    if(len(img_all)>1):
        img_all.pop()
        img_png = img_all[len(img_all)-1]
    elif(len(img_all)==1):
        img_png = img_all[0]
    elif(len(img_all)==0):
        pass
    showImage(img_png)

#关于，程序信息
def Imghelp():
    top=Toplevel()
    top.geometry("400x200")
    top.title('关于')
    longtext="""
    Software Information
    Author: Richard_Done
    Last edited: April,8,2020
    """
    Textlabel=Label(top,anchor='s',text=longtext,font=('微软雅黑',15),width=40,height=30)
    Textlabel.pack()


#图像处理功能实现模块

#裁剪图像，将植株提取出来
def ImgCut():
    global img_png

    path = filename + '_plant'
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)

    # 将PIL格式转换为np矩阵
    img = np.asarray(img_png)
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, binary = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
    # 读取轮廓
    img1, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    left_x = 0  # 左上角横坐标
    left_y = 0  # 左上角纵坐标
    wid = 0  # 矩形宽度
    hei = 0  # 矩形高度

    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if (w * h) > 30000 and (w * h) < 5000000 and h < 2000:
            # print(x,y,w,h)
            left_x = x
            left_y = y
            wid = w
            hei = h
            cv2.rectangle(img, (x - 60, y - 50), (x + w + 80, y + h), (0, 0, 255), 2)

    # 裁剪图像，显示的就是
    img_cut = img_png.crop((left_x - 60, left_y - 50, left_x + wid + 80, left_y + hei))
    img_all.append(img_cut)
    showImage(img_cut)
    img_cut.save(filename+'_plant/'+filename+"_1裁剪."+filetype)

#图像灰度化+图像增强+二值化
def ImgGrayEhanceINV():
    global img_png
    global img_aera

    img_png = img_all[len(img_all)-1]

    # 图像灰度化
    img_png = img_png.convert('L')    #调用Image中的函数

    # 图像增强
    img_png = ImageEnhance.Brightness(img_png).enhance(1.0)  # 亮度增强 1.0为原始图像
    img_png = ImageEnhance.Color(img_png).enhance(800.0)  # 色度增强 1.0为原始图像
    img_png = ImageEnhance.Contrast(img_png).enhance(15.5)  # 对比度增强

    # 二值化反转
    img_png = np.asarray(img_png)
    ret, img_png = cv2.threshold(img_png, 175, 255, cv2.THRESH_BINARY_INV)
    img_aera = img_png.copy()

    # opencv转化为PIL
    img_png = opencv2pil(img_png)
    img_all.append(img_png)
    showImage(img_png)
    img_png.save(filename+'_plant/'+ filename+"_2预处理."+filetype)

# 骨架提取+去噪
def ImgSkeleton_RemoveNoise():
    global img_png
    global img_skel

    im = np.asarray(img_png) #将PIL转换为np数组格式
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)) #卷积核，定义一个3x3的十字形结构元素

    skel = np.zeros(im.shape, np.uint8) #骨架图，初始化是原图
    erode = np.zeros(im.shape, np.uint8)
    temp = np.zeros(im.shape, np.uint8)

    skel_num=0
    while True:
        # 图像腐蚀
        erode = cv2.erode(im, element)
        #图像膨胀
        temp = cv2.dilate(erode, element)

        # 消失的像素是skeleton的一部分
        temp = cv2.subtract(im, temp)   #相减，im-temp即消失的部分，也就是轮廓的一部分
        # cv2.imshow('skeleton part %d' % (i,), temp)
        skel = cv2.bitwise_or(skel, temp) #或运算，将删除的部分添加到骨架图
        im = erode.copy()
        if cv2.countNonZero(im) == 0:
            break
        skel_num += 1
    img_png = skel

    # 去噪点（去掉面积较小的连通域）
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(img_png)
    i = 0
    for istat in stats:
        if istat[4] < 2:
            if istat[3] > istat[4]:
                r = istat[3]
            else:
                r = istat[4]
            cv2.rectangle(img_png, tuple(istat[0:2]), tuple(istat[0:2] + istat[2:4]), 0, thickness=-1)  # 26
        i = i + 1
    img_skel = img_png
    img_png = opencv2pil(img_png)
    img_all.append(img_png)
    showImage(img_png)
    img_png.save(filename+'_plant/'+ filename + "_3骨架提取." + filetype)

# 提取叶片信息
def LeafExtract():
    global img_png

    # PIL换成np数组
    im = np.asarray(img_png)
    # 统计概率霍夫线变换
    # 边缘检测
    edges = cv2.Canny(im, 50, 150, apertureSize=3)  # apertureSize参数默认其实就是3

    # HoughLinesP参数1是二值图；
    # 参数2是举例r的精度，值越大考虑越多的线；
    # 参数3是θ的精度，值越小考虑线越多；
    # 参数4是累加数阈值，值越小考虑的线越多 minLineLength是直线最短长度，MaxLineGap直线内是两点最大间隔
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=7, maxLineGap=50)
    min_angle = 2000000
    maxX = 0
    minX = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if min_angle >= angle(x1, y1, x2, y2) and math.fabs(y1 - y2) > 100:
            min_angle = angle(x1, y1, x2, y2)
            maxX = max(x1, x2)
            minX = max(x1, x2)

    # 显示叶片
    im3 = im.copy()
    im3_height = im3.shape[0]
    im3_width = im3.shape[1]

    for i in range(im3_height):
        for j in range(im3_width):
            if (j <= maxX + 20 and j >= minX - 20):
                im3[i, j] = 0
            else:
                pass
    img_png = opencv2pil(im3)
    img_all.append(img_png)
    showImage(img_png)

    img_png.save(filename+'_plant/'+ filename + "_4叶片提取." + filetype)

    # 标记叶片
    img_extract = im3.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 卷积核，定义一个3x3的十字形结构元素
    i = 5
    while i > 0:
        img_extract = cv2.dilate(img_extract, element)
        i = i - 1

    # 标记各个连通域
    labels = measure.label(img_extract, connectivity=2)

    # 定义数组存储分割出来的叶片骨架
    img_leaf = []

    # 定义数组存储质心坐标
    centroids_leaf = []

    # 定义数组存储外接矩形坐标
    bbox = []

    # 通过label标记，遍历各个连通域
    for region in measure.regionprops(labels):
        # 连通域外接矩形的坐标
        minr, minc, maxr, maxc = region.bbox
        # 连通域内部像素坐标值,列表嵌套，一组坐标为单位存储
        location = region.coords
        # 定义和原图同样大小的临时图片，用来画出树叶
        leaf_test = np.zeros(im3.shape[:], dtype=np.uint8)

        # 判断长宽，去除较小的连通域
        if (maxc - minc) < 50 or (maxr - minr) < 80:
            pass
        else:
            # 存储质心坐标
            centroids_leaf.append(region.centroid)

            # 存储矩形坐标
            bbox.append(region.bbox)

            cv2.rectangle(img_extract, (minc, minr), (maxc, maxr), (255, 255, 255), 2)
            for t in location:
                leaf_test[t[0], t[1]] = 255
            leaf = leaf_test[minr:maxr, minc:maxc]
            img_leaf.append(leaf)

    img_png = opencv2pil(img_extract)
    img_all.append(img_png)
    showImage(img_png)
    img_png.save(filename+'_plant/'+ filename + "_5叶片标记." + filetype)

    # print("叶片的数量为：", len(img_leaf))
    messagebox.showinfo('叶片数量', '叶片数量为：{}'.format(len(img_leaf)))

    # 创建存储数据文件夹
    data_path = filename+ '_plant/data_csv'
    isExists = os.path.exists(data_path)

    if not isExists:
        os.makedirs(data_path)


    # 存储叶片基部点坐标
    array_bottompoint = []

    for i in range(len(img_leaf)):
        leaf = img_leaf[i]
        k = 3
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 卷积核，定义一个3x3的十字形结构元素
        while k > 0:
            leaf = cv2.erode(leaf, element)
            k -= 1

        flag = 0
        leaf_height, leaf_width = leaf.shape

        if centroids_leaf[i][1] > maxX:
            for y in range(leaf_height - 1, -1, -1):
                for x in range(leaf_width):
                    if (leaf[y, x] == 255):
                        array_bottompoint.append([i + 1, y + bbox[i][0], x + bbox[i][1]])
                        flag = 1
                        # print("第", i + 1, "个叶片的基部点坐标：", "Height:", array_bottompoint[i][1], "Width:",
                        #       array_bottompoint[i][2])
                        break
                if (flag):
                    break
        else:
            for y in range(leaf_height - 1, -1, -1):
                for x in range(leaf_width - 1, -1, -1):
                    if (leaf[y, x] == 255):
                        array_bottompoint.append([i + 1, y + bbox[i][0], x + bbox[i][1]])
                        flag = 1
                        # print("第", i + 1, "个叶片的基部点坐标：", "Height:", array_bottompoint[i][1], "Width:",
                        #       array_bottompoint[i][2])
                        break
                if (flag):
                    break
    data_toppoint = pd.DataFrame(array_bottompoint)
    data_toppoint.to_csv(filename+ '_plant/data_csv/'+ filename + '_bottompoint.csv', header=['blade', 'height/pixel', 'width/pixel'], index=False,
                         encoding="gbk")

    # 存储每个叶片的最高点坐标
    array_toppoint = []

    for i in range(len(img_leaf)):
        leaf = img_leaf[i]
        k = 3
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 卷积核，定义一个3x3的十字形结构元素
        while k > 0:
            leaf = cv2.erode(leaf, element)
            k -= 1

        flag = 0
        leaf_height, leaf_width = leaf.shape
        for y in range(leaf_height):
            for x in range(leaf_width):
                if (leaf[y, x] == 255):
                    array_toppoint.append([i+1,y+bbox[i][0],x+bbox[i][1]])
                    flag =1
                    # print("第", i + 1, "个叶片的最高点坐标为：", "Height:", y + bbox[i][1], "Width:", x + bbox[i][2])
                    break
            if(flag):
                break

    data_toppoint = pd.DataFrame(array_toppoint)
    data_toppoint.to_csv(filename+ '_plant/data_csv/' +filename + '_toppoint.csv', header=['blade','height/pixel', 'width/pixel'], index=False, encoding="gbk")

    # 存储叶片角度
    path = filename+ '_plant/angle_pic'
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)


    array_angel = []

    for i in range(len(img_leaf)):
        leaf = img_leaf[i]

        k = 3
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 卷积核，定义一个3x3的十字形结构元素
        while k > 0:
            leaf = cv2.erode(leaf, element)
            k -= 1

        leaf_height, leaf_width = leaf.shape
        leaf_y = int((2 / 3) * leaf_height)
        leaf_x = int((2 / 3) * leaf_width)

        for y in range(leaf_height):
            if (y == leaf_y):
                for x in range(leaf_width):
                    if (leaf[y, x] == 255):
                        leaf_x = x
                        break
            if (y > leaf_y):
                break
        # print(leaf_y, leaf_x, leaf_height, leaf_width)
        plt.imshow(opencv2skimage(leaf))
        if centroids_leaf[i][1] > maxX:
            leaf_angle = math.atan(math.fabs(leaf_height - leaf_y) / math.fabs(leaf_x))
            plt.plot([0, leaf_x], [leaf_height, leaf_y], "r")
        else:
            leaf_angle = math.atan(math.fabs(leaf_height - leaf_y) / math.fabs(leaf_width - leaf_x))
            plt.plot([leaf_width, leaf_x], [leaf_height, leaf_y], "r")

        array_angel.append([i + 1, leaf_angle * 180 / math.pi])
        # print("第", i + 1, "个叶片的角度为：{:.4f}".format(leaf_angle * 180 / math.pi))

        angle_num = str(i + 1)
        angle_savepath = filename + '_plant/angle_pic/'
        plt.savefig(angle_savepath + filename + '_angle_blade' + angle_num + "." + filetype)
        plt.clf()

    data_angel = pd.DataFrame(array_angel)
    data_angel.to_csv(filename + '_plant/data_csv/'+ filename + '_angle.csv', header=['blade', 'angle/°'], index=False, encoding="gbk",float_format="%.4f")

    # 存储叶片长度
    path = filename + '_plant/length_pic'
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)

    array_length = []

    for leaf_i in range(len(img_leaf)):
        leaf1 = img_leaf[leaf_i]
        k = 3
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 卷积核，定义一个3x3的十字形结构元素
        while k > 0:
            leaf1 = cv2.erode(leaf1, element)
            k -= 1
        x = []
        y = []
        for i in range(leaf1.shape[1]):
            for j in range(leaf1.shape[0]):
                if (leaf1[j, i] > 0):
                    x.append(i)
                    y.append(j)

        f1 = np.polyfit(x, y, 5)
        p1 = np.poly1d(f1)
        yvals = p1(x)

        # 计算每个叶片的长度，即曲线长度
        area_list = []  # 存储每一微小步长的曲线长度
        for i in range(1, len(x)):
            # 计算每一微小步长的曲线长度，dx = x_{i}-x{i-1}，索引从1开始
            dl_i = np.sqrt((x[i] - x[i - 1]) ** 2 + (yvals[i] - yvals[i - 1]) ** 2)
            # 将计算结果存储起来
            area_list.append(dl_i)
        area = sum(area_list)  # 求和计算曲线在t:[0,2*pi]的长度
        area = area / 24 * 0.635  # 24个像素=0.635厘米

        array_length.append([leaf_i + 1, area])
        # print("第", leaf_i + 1, "个叶片：{:.4f}厘米".format(area))

        plt.imshow(opencv2skimage(leaf1))
        plt.plot(x, yvals, 'r')

        length_num = str(leaf_i + 1)
        length_savepath = filename+'_plant/length_pic/'
        plt.savefig(length_savepath + filename + '_length_blade' + length_num + "."+ filetype)
        plt.clf()

    data_length = pd.DataFrame(array_length)
    data_length.to_csv(filename + '_plant/data_csv/'+filename + '_length.csv', header=['blade', 'length/cm'], index=False, encoding='gbk',float_format="%.4f")

#植株高度和面积
def Plant_Height_Aera():
    # 求植株高度
    img_plant = img_skel.copy()
    img_plant_height = img_plant.shape[0]
    img_plant_width = img_plant.shape[1]

    location_top = img_plant_height
    location_bottom = 0
    location_level = 0

    for i in range(img_plant_height):
        for j in range(img_plant_width):
            if img_plant[i, j] == 255:
                location_top = i
                location_level = j
                break
        if location_top == i:
            break

    for i in range(img_plant_height - 1, -1, -1):
        for j in range(img_plant_width):
            if img_plant[i, j] == 255:
                location_bottom = i
                break
        if location_bottom == i:
            break

    plant_height = location_bottom - location_top
    plant_height = plant_height / 24 * 0.635
    # print("高度为：{:.4f}厘米".format(plant_height))
    messagebox.showinfo("植株高度", "植株高度为：{:.4f}厘米".format(plant_height))

    plt.imshow(opencv2skimage(img_plant))
    plt.plot([location_level, location_level], [location_bottom, location_top], "r")
    plt.savefig(filename+'_plant/'+filename + '_height')
    plt.clf()

    # 求植株投影面积
    img_height, img_width = img_aera.shape
    aera = 0
    for x in range(img_height):
        for y in range(img_width):
            if (img_aera[x, y] == 255):
                aera += 1
    # print("面积为：{}像素点".format(aera)
    messagebox.showinfo("投影面积", "投影面积为：{}像素点".format(aera))


    plant_info = []
    plant_info.append([filename,plant_height,aera])
    data_plant = pd.DataFrame(plant_info)
    data_plant.to_csv(filename + '_plant/data_csv/' + filename + '_height_aera.csv', header=['plant', 'height/cm','aera/pixel'], index=False, encoding='gbk',float_format="%.4f")

def Run():
    ImgCut()
    ImgGrayEhanceINV()
    ImgSkeleton_RemoveNoise()
    LeafExtract()
    Plant_Height_Aera()


#第1步，建立窗口window
window=Tk()

#第2步，给窗口可视化取名字
window.title('MIP')
window.update()

#第3步，设定窗口大小（宽*高）
window.geometry('%dx%d'%(Mywidth,Myheight))   #这里的乘是小x

#第4步，创建一个菜单栏，可以理解为一个容器，放在窗口的上方
menubar = Menu(window)

#第5步，设计下拉菜单和主菜单
# content=[['打开','保存'],['复制','剪切','撤销'],['图像灰度化','灰度直方图','大小相同图像相加','对数变换','LoG算子图像锐化','Ostu图像分割'],['帮助']]
# Main=['文件', '编辑', '功能', '关于']

#第6步，创建一个文件菜单（默认不下拉，包含新建，打开，保存，另为存）
filemenu=Menu(menubar,tearoff=0)
filemenu.add_command(label='打开', command=openfiles)
filemenu.add_command(label='保存', command=savefiles)
filemenu.add_separator()    # 添加一条分隔线
filemenu.add_command(label='退出', command=window.quit) # 用tkinter里面自带的quit()函数
menubar.add_cascade(label='文件',menu=filemenu)

#第7步，创建一个编辑菜单
editmenu=Menu(menubar,tearoff=0)
editmenu.add_command(label='复制', command=ImgCopy)
editmenu.add_command(label='剪切', command=Imgpast)
editmenu.add_command(label='撤销', command=ImgUndo)
menubar.add_cascade(label='编辑',menu=editmenu)

#第8步，创建一个功能菜单
funmenu=Menu(menubar,tearoff=0)
funmenu.add_command(label='图像裁剪', command=ImgCut)
funmenu.add_command(label='图像预处理', command=ImgGrayEhanceINV)
funmenu.add_command(label='骨架提取', command=ImgSkeleton_RemoveNoise)
funmenu.add_command(label='叶片提取', command=LeafExtract)
funmenu.add_command(label='植株高度和面积', command=Plant_Height_Aera)
funmenu.add_command(label='一键运行', command=Run)
menubar.add_cascade(label='功能',menu=funmenu)

#第9步，创建一个帮助菜单
helpmenu=Menu(menubar,tearoff=0)
helpmenu.add_command(label='关于', command=Imghelp)
menubar.add_cascade(label='帮助',menu=helpmenu)

#第10步，创建一个画布
canvas=Canvas(window, bg='gray')
canvas.pack(fill='both',expand='yes')

# 将window的menu属性设置为M
window['menu'] = menubar
window.mainloop()
