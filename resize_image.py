import cv2
import os
from cut_image import fileRead

# 顔画像フォルダ名
face_dir = "Prison_Break_face"

# 顔画像フォルダを開く
faces = fileRead(face_dir)
print(f'読み込んだ画像の枚数:{len(faces)}')

num = 1
for face in faces:
    f = cv2.imread(face, 1)
    resize_face = cv2.resize(f, (200, 200))

    # ディレクトリ名指定
    dirname = "resize"

    # ディレクトリがなかったら作る
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    # ファイル名
    path = "face_resize" + str(num) + ".png"
    file_name = os.path.join(dirname, path)
    num += 1
    # print(file_name)

    # 保存
    cv2.imwrite(file_name, resize_face)
