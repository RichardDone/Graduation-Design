# -*- coding: utf-8 -*-
"""
Author: DMY
Last edited: June 13,2019
"""
import PIL
import tkinter as tk
import matplotlib.pyplot as plt
import cv2
from skimage import morphology,draw
import numpy as np
from tkinter import filedialog
from tkinter import *
from PIL import Image,ImageTk,ImageFilter,ImageEnhance
import win32clipboard as clip
import win32con     #pip install pywin32
from io import BytesIO


global img_tep
global img_png           # 定义全局变量图像的
Myheight= 600    #高
Mywidth= 800     #宽


#图片自适应画布大小
def resize( w_box, h_box, pil_image): #参数是：要适应的窗口宽、高、Image.open后的图片
      w, h = pil_image.size #获取图像的原始大小
      f1 = 1.0*w_box/w
      f2 = 1.0*h_box/h
      factor = min([f1, f2])
      width = int(w*factor)
      height = int(h*factor)
      return pil_image.resize((width, height), Image.ANTIALIAS)


#基础功能模块

#打开图片
def openfiles():
    global img_png
    global Imgwidth #图片原本宽度
    global Imgheight  #图片原本高度
    file_path = filedialog.askopenfilename(title='选择文件',filetype=[('all files', '.*'),('JPG','.jpg'),('PNG','png')])
    if file_path!='':
        print("打开路径",file_path)
        img_png = Image.open(file_path)
        canvas.delete(ALL)
        Imgwidth,Imgheight = img_png.size
        img_png=resize(window.winfo_width() ,window.winfo_height() ,img_png)
        photo=ImageTk.PhotoImage(img_png)
        canvas.create_image(int(window.winfo_width() /2),0,anchor='n',image=photo)   # 图片锚定点（n图片顶端的中间点位置）放在画布（400,0）坐标处
        canvas.image=photo

        global img_tep
        img_tep=img_png.copy()
        # imglabel=Label(window,image=photo)
        # imglabel.image=photo
        # imglabel.pack()
        # plt.imshow(img_png)
        # plt.show()

#保存图片，格式为PNG
def savefiles():
    global img_png
    file_path = filedialog.asksaveasfilename(title='保存文件', filetypes=[("PNG", ".png")])
    img_png=img_png.resize((Imgwidth,Imgheight))
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
    global img_tep
    img_tep=img_png.copy()
    output = BytesIO()
    img_png.convert("RGB").save(output, "BMP")
    data = output.getvalue()[14:]
    clip.OpenClipboard()  # 打开剪贴板
    clip.EmptyClipboard()  # 先清空剪贴板
    clip.SetClipboardData(win32con.CF_DIB, data)  # 将图片放入剪贴板
    clip.CloseClipboard()

    canvas.delete(ALL)

#撤销函数，只能撤销一步
def ImgUndo():
    global img_tep
    global img_png
    img_png=img_tep.copy()
    canvas.delete(ALL)
    photo = ImageTk.PhotoImage(img_tep)
    canvas.create_image(int(window.winfo_width() / 2), 0, anchor='n', image=photo)
    canvas.image = photo

#关于，程序信息
def Imghelp():
    top=Toplevel()
    top.geometry("400x200")
    top.title('关于')
    longtext="""
    Software Information
    DIP1.0
    Author: DMY
    Last edited: June 13,2019
    """
    Textlabel=Label(top,anchor='s',text=longtext,font=('微软雅黑',15),width=40,height=30)
    Textlabel.pack()


#图像处理功能实现模块

#图像灰度化+图像增强
def ImgGrayEhance():
    global img_png
    global img_tep
    img_tep=img_png.copy()

    img = np.asarray(img_png)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

    img1, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(type(contours))
    print(type(contours[0]))
    print(len(contours))
    for i in range(71, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # x, y, w, h = cv2.boundingRect(contours[52])
    # print(x,y,x+w,y+h)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

    # cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    cv2.imshow("img", img)

    # 图像灰度化
    img_png = img_png.convert('L')    #调用Image中的函数

    # 图像增强
    img_png = ImageEnhance.Brightness(img_png).enhance(1.0)  # 亮度增强 1.0为原始图像
    img_png = ImageEnhance.Color(img_png).enhance(1.0)  # 色度增强 1.0为原始图像
    img_png = ImageEnhance.Contrast(img_png).enhance(1.5)  # 对比度增强
    img_png = ImageEnhance.Contrast(img_png).enhance(1.5)  # 锐化增强

    canvas.delete(ALL)
    photo = ImageTk.PhotoImage(img_png)
    canvas.create_image(int(window.winfo_width() /2),0,anchor='n',image=photo)
    canvas.image=photo

# 图像二值化反转
def ImgBinary_INV():
    global img_png
    global img_tep
    img_tep = img_png.copy()

    # img_png = Image.fromarray()

    canvas.delete(ALL)
    photo = ImageTk.PhotoImage(img_png)
    canvas.create_image(int(window.winfo_width() / 2), 0, anchor='n', image=photo)
    canvas.image = photo

# 骨架提取
def ImgSkeleton():
    global img_png
    global img_tep
    img_tep = img_png.copy()

    im = np.asarray(img_png) #将PIL转换为np数组格式
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)) #卷积核，定义一个3x3的十字形结构元素

    skel = np.zeros(im.shape, np.uint8) #骨架图，初始化是原图
    erode = np.zeros(im.shape, np.uint8)
    temp = np.zeros(im.shape, np.uint8)

    i = 0
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
        i += 1

    # 将opencv格式转换回PIL
    img_png = Image.fromarray(skel)

    canvas.delete(ALL)
    photo = ImageTk.PhotoImage(img_png)
    canvas.create_image(int(window.winfo_width() / 2), 0, anchor='n', image=photo)
    canvas.image = photo

# 去噪点
def ImgNoiseRemoval():
    global img_png
    global img_tep
    img_tep = img_png.copy()

    im = np.asarray(img_png) #将PIL转换为np数组格式
    # 根据连通域面积去除噪点
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(im)
    i = 0
    for istat in stats:
        if istat[4] < 1.2:
            if istat[3] > istat[4]:
                r = istat[3]
            else:
                r = istat[4]
            cv2.rectangle(im, tuple(istat[0:2]), tuple(istat[0:2] + istat[2:4]), 0, thickness=-1)  # 26
        i = i + 1
    # 将opencv格式转换回PIL
    img_png = Image.fromarray(im)

    canvas.delete(ALL)
    photo = ImageTk.PhotoImage(img_png)
    canvas.create_image(int(window.winfo_width() / 2), 0, anchor='n', image=photo)
    canvas.image = photo

#霍夫直线检测
def ImgHough_line():
    global img_png
    global img_tep
    img_tep = img_png.copy()

    im = np.asarray(img_png) #将PIL转换为np数组格式

    edges = cv2.Canny(im, 150, 150, apertureSize=3)  # apertureSize参数默认其实就是3
    cv2.imshow("edges", edges)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)
    for line in lines:
        rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
        a = np.cos(theta)  # theta是弧度
        b = np.sin(theta)
        x0 = a * rho  # 代表x = r * cos（theta）
        y0 = b * rho  # 代表y = r * sin（theta）
        x1 = int(x0 + 1000 * (-b))  # 计算直线起点横坐标
        y1 = int(y0 + 1000 * a)  # 计算起始起点纵坐标
        x2 = int(x0 - 1000 * (-b))  # 计算直线终点横坐标
        y2 = int(y0 - 1000 * a)  # 计算直线终点纵坐标    注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
        cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 点的坐标必须是元组，不能是列表。
    cv2.imshow("image-lines", im)

    # 将opencv格式转换回PIL
    img_png = Image.fromarray(im)

    canvas.delete(ALL)
    photo = ImageTk.PhotoImage(img_png)
    canvas.create_image(int(window.winfo_width() / 2), 0, anchor='n', image=photo)
    canvas.image = photo

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
funmenu.add_command(label='图像灰度化和增强', command=ImgGrayEhance)
funmenu.add_command(label='图像二值化反转', command=ImgBinary_INV)
funmenu.add_command(label='骨架提取', command=ImgSkeleton)
funmenu.add_command(label='去除噪点', command=ImgNoiseRemoval)
funmenu.add_command(label='霍夫变换', command=ImgHough_line)
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
