# -*- coding: utf-8 -*-
"""
Author: DMY
Last edited: June 13,2019
"""
import PIL
import tkinter as tk
import matplotlib.pyplot as plt
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

#卷积函数
def conv(image, weight):
    image1=np.array(image)
    height, width = image1.shape[0],image1.shape[1]
    h, w = weight.shape
    # 经滑动卷积操作后得到的新的图像的尺寸
    new_h = height -h + 1
    new_w = width -w + 1
    new_image = np.zeros((height, width),dtype=np.float)
    # 进行卷积操作,实则是对应的窗口覆盖下的矩阵对应元素值相乘,卷积操作
    for i in range(new_h):
        for j in range(new_w):
            new_image[i+1, j+1] = abs(np.sum(image1[i:i+h, j:j+w] * weight))
    # 去掉矩阵乘法后的小于0的和大于255的原值,重置为0和255
    new_image = new_image.clip(0, 255)
    new_image = np.rint(new_image).astype('uint8')
    return new_image


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
#图像灰度化
def ImgGray():
    global img_png
    global img_tep
    img_tep=img_png.copy()
    img_png = img_png.convert('L')    #调用Image中的函数

    canvas.delete(ALL)
    photo = ImageTk.PhotoImage(img_png)
    canvas.create_image(int(window.winfo_width() /2),0,anchor='n',image=photo)
    canvas.image=photo

#图像增强
def ImgGrayEnhance():
    global img_png
    global img_tep
    img_tep = img_png.copy()
    img_png=img_png.convert('L')    #先对图片灰度化

    img_png = ImageEnhance.Brightness(img_png).enhance(1.0) #亮度增强 1.0为原始图像
    img_png = ImageEnhance.Color(img_png).enhance(1.0) #色度增强 1.0为原始图像
    img_png = ImageEnhance.Contrast(img_png).enhance(1.5) #对比度增强
    img_png = ImageEnhance.Contrast(img_png).enhance(1.5) #锐化增强


    canvas.delete(ALL)
    photo = ImageTk.PhotoImage(img_png)
    canvas.create_image(int(window.winfo_width()/ 2), 0, anchor='n', image=photo)
    canvas.image = photo

#大小相同的图片相加
def ImgAdd():
    global img_png
    global img_tep
    img_tep = img_png.copy()
    img1=img_png
    file_path = filedialog.askopenfilename(title='选择文件', filetype=[('all files', '.*'), ('JPG', '.jpg')])
    img2= Image.open(file_path)
    img2 = resize(Mywidth, Myheight, img2)

    if img1.size==img2.size and img1.mode==img2.mode:
        img_png=Image.blend(img1,img2,0.5)
        canvas.delete(ALL)
        photo = ImageTk.PhotoImage(img_png)
        canvas.create_image(int(window.winfo_width()  / 2), 0, anchor='n', image=photo)
        canvas.image = photo
    else:
        top=Toplevel()
        top.geometry("300x200")
        top.title('错误提示')
        imglabel=Label(top,anchor='center',text='图片属性不一致',font=('微软雅黑',15),width=20,height=30)
        imglabel.pack()


#对数变化
def Imglog():
    global img_png
    global img_tep
    img_tep = img_png.copy()
    img_png=np.array(img_png)
    c = 255 / np.log(256)
    row, column = img_png.shape[0], img_png.shape[1]
    for i in range(row):
        for j in range(column):
            img_png[i][j]=c*np.log(1.0+img_png[i][j])

    img_png=Image.fromarray(np.uint8(img_png))
    canvas.delete(ALL)
    photo = ImageTk.PhotoImage(img_png)
    canvas.create_image(int(Mywidth / 2), 0, anchor='n', image=photo)
    canvas.image = photo

#Log图像锐化
def Logsharpen():
    global img_png
    global img_tep
    img_tep = img_png.copy()
    #img_png=img_png.filter(ImageFilter.GaussianBlur(radius=1))  #高斯平滑


    #高斯滤波器
    kernel_3x3 = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])
    kernel_3x3 = kernel_3x3 / kernel_3x3.sum()  #加权平均

    # 高通滤波器与图片进行卷积
    img_png=conv(img_png,kernel_3x3)

    #拉普拉斯算子变换
    img_png=np.array(img_png)
    r, c = img_png.shape[0],img_png.shape[1]
    new_image = np.zeros((r, c))
    L_sunzi = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    #L_sunnzi = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    for i in range(r-2):
        for j in range(c-2):
            new_image[i+1, j+1] = abs(np.sum(img_png[i:i + 3, j:j + 3] * L_sunzi))

    # 将图片由np数组格式转换为Image格式
    img_png=Image.fromarray(np.uint8(new_image))
    #展示到画布中
    canvas.delete(ALL)
    photo = ImageTk.PhotoImage(img_png)
    canvas.create_image(int(Mywidth / 2), 0, anchor='n', image=photo)
    canvas.image = photo

#Ostu阈值分割
def ImgOstu():
    global img_png
    global img_tep
    img_tep = img_png.copy()

    img_png=img_png.convert('L')
    img_png = np.array(img_png)
    Imgh,Imgw = img_png.shape[0],img_png.shape[1]
    MN = Imgh*Imgw
    avgValue = 0.0
    nHistogram = [0] * 256  #灰度直方图
    fHistogram = [0] * 256  #归一化直方图
    #求灰度直方图
    for i in range(Imgh):
        for j in range(Imgw):
            pixel=int(img_png[i][j])
            nHistogram[pixel]=nHistogram[pixel]+1
    #灰度直方图归一化，并且求取整幅图像平均灰度average
    for i in range(256):
        fHistogram[i]=nHistogram[i]/float(MN)
        avgValue=avgValue+fHistogram[i]*i

    threshold = 0      #阈值
    maxVariance = 0.0
    w = 0.0
    u = 0.0
    for i in range(256):
        w = w + fHistogram[i]     #假设当前灰度i为阈值, 0~i 灰度的像素(假设像素值在此范围的像素叫做前景像素) 所占整幅图像的比例
        u = u + fHistogram[i]*i    #灰度i之前的像素(0~i)的平均灰度值： 前景像素的平均灰度值

        t = avgValue*w-u
        if w!=0 and w!=1:
            variance = t * t / (w * (1 - w))
            if variance > maxVariance:
                maxVariance = variance
                threshold = i   #Ostu阈值

    img_png = Image.fromarray(np.uint8(img_png))
    img_png = img_png.point(lambda p: p > threshold and 255)

    # 展示到画布中
    canvas.delete(ALL)
    photo = ImageTk.PhotoImage(img_png)
    canvas.create_image(int(Mywidth / 2), 0, anchor='n', image=photo)
    canvas.image = photo
# 1、计算直方图并归一化histogram
# 2、计算图像灰度均值avgValue.
# 3、计算直方图的零阶w[i]和一级矩u[i]
# 4、计算并找到最大的类间方差（between-class variance）
# variance[i]=(avgValue*w[i]-u[i])*(avgValue*w[i]-u[i])/(w[i]*(1-w[i]))
# 对应此最大方差的灰度值即为要找的阈值
# 5、用找到的阈值二值化图像




#第1步，建立窗口window
window=Tk()

#第2步，给窗口可视化取名字
window.title('DIP')
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
funmenu.add_command(label='图像灰度化', command=ImgGray)
funmenu.add_command(label='图像增强', command=ImgGrayEnhance)
funmenu.add_command(label='大小相同图像相加', command=ImgAdd)
funmenu.add_command(label='对数变换', command=Imglog)
funmenu.add_command(label='LoG算子图像锐化', command=Logsharpen)
funmenu.add_command(label='Ostu图像分割', command=ImgOstu)
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
