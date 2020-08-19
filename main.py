# coding:utf-8
import os
from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import keras
from keras_vggface import utils
# メモリープロファイル用コード
from memory_profiler import profile
#
import gc

classes = ["Bagwell", "Burrows", "Mahone", "Scofield", "Tancredi"]
classes_color = ["red", "green", "blue", "fuchsia", "yellow"]
num_classes = len(classes)
image_size = 200

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 色設定
def check_color(color):
  if color == "red":
    return (255, 0, 0)
  elif color == "green":
    return (0, 128, 0)
  elif color == "blue":
    return (0, 0, 255)
  elif color == "fuchsia":
    return (255, 0, 255)
  elif color == "yellow":
    return (255, 255, 0)
  else:
    print("non defined color. check_color_err.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS #変数filenameの.より後ろの文字列がALLOWED_EXTENSIONSのどれかに該当するかどうかつまり有効な拡張子かを判定している

model = load_model('./PrisonBreak_Character_Classification_VGGFaceVGG16.h5')#学習済みモデルをロードする
graph = tf.get_default_graph()

@app.route('/', methods=['GET', 'POST'])
@profile
def upload_file():
    global graph
    with graph.as_default():
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('ファイルがありません')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('ファイルがありません')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(UPLOAD_FOLDER, filename))
                filepath = os.path.join(UPLOAD_FOLDER, filename)

                #受け取った画像を読み込み、np形式に変換
                img = image.load_img(filepath, color_mode='rgb')
                img = image.img_to_array(img)
                img = np.array(img, dtype=np.uint8)

                # 画像から顔部分を抽出(認識ミスで顔以外も多少含まれる)
                # 検出しやすいようグレースケールに変換する
                img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_g = np.array(img_g, dtype=np.uint8)

                # カスケード型分類器を使用して画像ファイルから顔部分を検出する
                # カスケード型分類器に使用する分類器のデータ（xmlファイル）を読み込み
                HAAR_FILE = "./haarcascades/haarcascade_frontalface_alt.xml"
                cascade = cv2.CascadeClassifier(HAAR_FILE)
                face = cascade.detectMultiScale(img_g, scaleFactor=1.2, minNeighbors=2, minSize=(20, 20))
                #メモリ解放
                del img_g
                gc.collect()

                # openCVで画像を扱うのでBGR形式に変換する
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # 顔の座標を表示する
                face_cut_data = []

                i = 1
                for x,y,w,h in face:
                    # 顔として囲った部分を分類機にかけて人物特定する
                    face_cut = img[y:y+h, x:x+w]
                    face_cut = cv2.resize(face_cut, (image_size, image_size))

                    data = np.array([face_cut])

                    result = model.predict(data)[0]
                    predicted = result.argmax()
                    del result
                    gc.collect()
                    # 顔部分の出力ファイル名を設定
                    face_cut_image = os.path.join(UPLOAD_FOLDER, 'cut_face_' + str(i) + '.jpg')
                    i += 1

                    # 顔部分に対応したラベルを付けて保存する
                    face_cut_data.append([face_cut_image, classes[predicted]])

                    # 顔部分の出力
                    cv2.imwrite(face_cut_image, face_cut)

                    # 顔をに囲いを付加する
                    color = classes_color[predicted]
                    cv2.rectangle(img, (x,y), (x+w,y+h), check_color(color), 2)

                answer_filepath = os.path.join(UPLOAD_FOLDER, 'face_rectangle.jpg')
                cv2.imwrite(answer_filepath, img)

                return render_template("result.html",answer="", answer_image=answer_filepath, face_cut_data=face_cut_data )

        return render_template("index.html",answer="")
@app.route('/uploads/<filename>')
# uploadsディレクトリのファイルを表示するために必要
def output_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)
    #app.run()