# OpenCVのインポート
import cv2
import glob
import os
import numpy as np
import face_recognition


# 画像のあるディレクトリから全画像を読み込む
def fileRead(input_path):
    data = []
    get_files = glob.glob(f'{input_path}/*.jpg') + glob.glob(f'{input_path}/*.jpeg') + glob.glob(f'{input_path}/*.png')
    for file in get_files:
        data.append(file)
    return data


# スクリーンショット取得時に日本語名ファイルで保存される。日本語名ファイルはcv2.imreadでは読めないので別途関数を用意する
def imread(image_picture):
    img_array = np.fromfile(image_picture, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


if __name__ == '__main__':
    # 画像の入力ディレクトリ
    input_path = "Prison_Break"

    # 画像の出力ディレクトリ
    output_path = "C:/Users/Ryota Arinobu/Desktop/get_image/Prison_Break_face"

    # 画像ファイルの読み込み
    image_data = fileRead(input_path)

    print(f'読み込んだ画像の枚数:{len(image_data)}')

    num = 1
    # 画像を一枚ずつ読み込む
    for image_picture in image_data:
        image = face_recognition.load_image_file(image_picture)
        face_locations = face_recognition.face_locations(image)
        print(image_picture)
        bgr = imread(image_picture)
        # print(bgr)
        for face_location in face_locations:
            top, right, bottom, left = face_location

            # 顔に囲いとラベルを付加する
            face_cut = bgr[top:bottom, left:right]

            # きりとった画像を出力する
            path = os.path.join(output_path, f'face_cut_{num}.jpg')
            cv2.imwrite(path, face_cut)
            num += 1
