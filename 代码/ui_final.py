import tkinter as tk
from tkinter import ttk, messagebox, font
from PIL import Image, ImageTk
import cv2
import socket
import pickle
import struct

# 导入口罩检测代码
from pytorch_infer import inference


# 初始化套接字连接为None
client_socket = None
payload_size = struct.calcsize("Q")

def open_camera():
    global client_socket
    host_ip = '192.168.137.136'  # RKNN开发板的IP地址
    port = 9999              # 与服务器端相同的端口号

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((host_ip, port))
    except Exception as e:
        messagebox.showerror("Error", f"Could not connect to remote device: {e}")
        return

    btn_start.config(state='disabled')
    btn_stop.config(state='normal')
    camera_loop()

def close_camera():
    global client_socket
    if client_socket:
        client_socket.close()
    client_socket = None
    panel.config(image='')
    btn_start.config(state='normal')
    btn_stop.config(state='disabled')

def update_image(frame):
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    panel.imgtk = imgtk
    panel.config(image=imgtk)

def receive_frame():
    global client_socket
    data = b""
    while len(data) < payload_size:
        packet = client_socket.recv(4*1024)
        if not packet: return None
        data += packet
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    while len(data) < msg_size:
        data += client_socket.recv(4*1024)
    frame_data = data[:msg_size]
    return pickle.loads(frame_data)

def camera_loop():
    frame = receive_frame()
    if frame is None:
        close_camera()
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    update_image(frame)
    if detection_results:
        info_text.set("检测到人脸")
    else:
        info_text.set("未检测到人脸")

    window.after(10, camera_loop)

# 创建主窗口
window = tk.Tk()
window.title("口罩检测系统")

# 设置窗口初始大小
window.geometry("800x600")

# 加载并显示背景图像
bg_image = Image.open("./background.jpg")  # 替换为你的图像文件
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = ttk.Label(window, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# 添加标题
title_font = font.Font(family='Arial', size=20, weight='bold')  # 定义加粗字体样式
chinese_title = ttk.Label(window, text="口罩检测系统", font=title_font, background='#f0f0f0')
chinese_title.pack(side=tk.TOP, pady=10)
english_title = ttk.Label(window, text="Mask Detection System", font=title_font, background='#f0f0f0')
english_title.pack(side=tk.TOP)

# 创建框架用于放置视频流和信息标签
stream_frame = ttk.Frame(window)
stream_frame.place(relx=0.5, rely=0.3, anchor='center')

# 创建视频流的标签
panel = ttk.Label(stream_frame)
panel.pack(padx=10, pady=10)

info_text = tk.StringVar()
info_label = ttk.Label(stream_frame, textvariable=info_text, foreground='blue', font=("Arial", 16))
info_label.pack()

# 创建一个新框架用于放置按钮
button_frame = ttk.Frame(window)
button_frame.pack(side=tk.BOTTOM, pady=20)

# 使用ttk.Button代替tk.Button，并设置样式
style = ttk.Style()
style.configure('TButton', font=('Arial', 12), borderwidth='4')
style.map('TButton', foreground=[('active', '!disabled', 'green')], background=[('active', 'black')])

btn_start = ttk.Button(button_frame, text="开启摄像头", command=open_camera)
btn_start.pack(side=tk.LEFT, padx=10)

btn_stop = ttk.Button(button_frame, text="停止摄像头", state='disabled', command=close_camera)
btn_stop.pack(side=tk.LEFT, padx=10)

btn_exit = ttk.Button(button_frame, text="退出程序", command=window.destroy)
btn_exit.pack(side=tk.LEFT, padx=10)

window.mainloop()
