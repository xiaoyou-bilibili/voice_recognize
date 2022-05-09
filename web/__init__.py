from flask import Flask, request, Response, render_template
from flask_cors import CORS
import json
from core import recognition, audio_db_path
import os

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
    file = request.files['file']
    file.save("./web/static/row2.wav")
    print(data)
    # 读取文件并保存
    # name = data["name"]
    # img = base64_to_cv(data["img"])
    # img = cv2.resize(img, (250, 250))
    # # 获取图片的特征信息
    # feature = get_feature(img)
    # # 保存图片的特征
    # store_feature("0", name, feature, "")
    # save_path = os.path.join(audio_db_path, user_name + os.path.basename(path)[-4:])
    # 返回生成的图片和种子
    return return_json({})


# 主页显示HTML
@app.route('/', methods=['GET'])
def index():
    return render_template('content.html')
