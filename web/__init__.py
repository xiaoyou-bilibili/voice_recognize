from email.mime import base
from flask import Flask, request, Response, render_template
from flask_cors import CORS
import json

from sympy import im
from core import recognition, audio_db_path, register
import os
import base64

# 初始化flaskAPP
app = Flask(__name__)
# r'/*' 是通配符，让本服务器所有的URL 都允许跨域请求
# 允许跨域请求
CORS(app, resources=r'/*')


# 返回JSON字符串
def return_json(data):
    return Response(json.dumps(data, ensure_ascii=False), mimetype='application/json')


# 声纹识别
@app.route('/voice_recognize/recognize', methods=['POST'])
def recognize_image():
    # 直接把音频信息保存为文件
    file = request.files['file']
    file.save("./web/static/row.wav")
    # 返回json类型字符串
    return return_json(recognition('./web/static/row.wav'))


# 添加声纹信息
@app.route('/voice_recognize/add', methods=['POST'])
def add_face():
    # 获取所有的参数
    data = request.form
    row = data["data"]
    name = data["name"]
    row = row.replace("data:audio/wav;base64,","")
    # 保存音频文件到数据库中
    save_path = os.path.join(audio_db_path, "%s.wav" % name)
    # 保存文件
    with open(save_path,"wb") as f:
        f.write(base64.b64decode(row))
    # 更新特征信息
    register(save_path, name)
    # 直接返回空    
    return return_json({})


# 主页显示HTML
@app.route('/', methods=['GET'])
def index():
    return render_template('content.html')
