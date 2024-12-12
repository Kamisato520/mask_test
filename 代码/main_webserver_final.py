from flask import Flask, render_template, Response, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import socket
import cv2
import numpy as np
import struct
import pickle
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
# 导入口罩检测代码
from pytorch_infer_forweb import inference

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 设置一个秘钥，用于会话安全

# Flask-Login 配置
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# 用户模型
class User(UserMixin):
    def __init__(self, id, password_hash):
        self.id = id
        self.password_hash = password_hash

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# 模拟的用户信息，实际应用中应该从数据库中获取
users = {'1': User(id='1', password_hash=generate_password_hash('1'))}

# 加载用户的回调函数
@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# 创建 socket 对象
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    # 连接到提供图像的服务器
    sock.connect(('192.168.137.136', 9999))  # 使用您提供的服务器地址和端口
except ConnectionRefusedError:
    print("无法连接到图像服务器。请检查服务器是否正在运行并监听该端口。")
    exit(1)

def gen_frames():
    data = b""
    payload_size = struct.calcsize("Q")
    while True:
        # 接收帧长度
        while len(data) < payload_size:
            packet = sock.recv(4 * 1024)  # 4K
            if not packet: break
            data += packet
        if not data: break

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # 接收完整的帧数据
        while len(data) < msg_size:
            data += sock.recv(4 * 1024)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        # 解序列化帧数据

        frame = pickle.loads(frame_data)
        output_info, processed_frame = inference(frame, draw_result=True, show_result=False, target_shape=(360, 360))
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users.get(username)
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('video_stream'))  # 修改重定向到 video_stream
        else:
            flash('Invalid username or password')
    return render_template('login_v2.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # 简单的验证逻辑，实际应用中需要更复杂的验证和密码加密
        if username not in users:
            new_user = User(id=username, password_hash='')
            new_user.set_password(password)
            users[username] = new_user
            return redirect(url_for('login'))
        else:
            flash('Username already exists')
    return render_template('register.html')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_stream')
@login_required
def video_stream():
    return render_template('video_stream.html')

@app.route('/introduction')
def introduction():
    return render_template('introduction.html')

@app.route('/suggestions', methods=['GET', 'POST'])
def suggestions():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        suggestion = request.form['suggestion']

        # 将建议保存到文本文件
        with open('suggestions.txt', 'a', encoding='utf-8') as file:
            file.write(f'姓名: {name}, 邮箱: {email}, 建议: {suggestion}\n')

        # 重定向到主页或感谢页面
        return redirect(url_for('thank_you'))
    return render_template('suggestions.html')

@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



