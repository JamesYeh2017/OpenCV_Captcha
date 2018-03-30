
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.externals import joblib # use pickle to 序列化 Python 物件
from sklearn.preprocessing import StandardScaler
import requests
import numpy
import cv2
from datetime import datetime
import time


def create_prediction_data():
    def crawler_captcha():
        with open('./captcha/captcha.jpg', 'wb') as f:
            res = requests.get('http://gcis.nat.gov.tw/pub/kaptcha.jpg')  # network/image
            f.write(res.content)  # 寫入

    def find_contours():
        crawler_captcha()
        pil_image = Image.open('./captcha/captcha.jpg').convert('RGB')
        open_cv_image = numpy.array(pil_image)  # image => number
        # print(open_cv_image)
        plt.imshow(open_cv_image)  # use plt to read number and show img
        # plt.show()

        imgray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)  # convert color 轉換顏色
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)  # (原圖, 閾值, 最大值), 大於閥值就是最大, 反之最小
        image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 找出輪廓,            階層式輪廓, 終點座標

        cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key=lambda x: x[1])

        ary = []
        for (c, _) in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            # print((x, y, w, h))
            if w >= 15 and h == 24:
                ary.append((x, y, w, h))
        return open_cv_image, ary

    def save_captcha_data(open_cv_image, ary):
        # fig = plt.figure()
        ct = int(time.mktime(datetime.now().timetuple()))
        for img in os.listdir('./captcha/prediction/'):
            os.remove('./captcha/prediction/{}'.format(img))
        for index, (x, y, w, h) in enumerate(ary):
            roi = open_cv_image[y:y+h, x:x+w]
            thresh = roi.copy()
            # a = fig.add_subplot(1, len(ary), index + 1)
            fig = plt.figure()
            plt.imshow(thresh)
            # plt.show()
            plt.savefig('./captcha/prediction/{}.jpg'.format(ct+index), dpi=100)  # 測試集資料

    result = find_contours()
    while len(result[1]) < 6:
        print("lenCaptcha :", len(result[1]))
        result = find_contours()
    print("lenCaptcha :", len(result[1]))
    save_captcha_data(open_cv_image=result[0], ary=result[1])


def load_model():
    mlp = joblib.load('./captcha/captcha.pkl')
    return mlp

# def showPredictionData():
#     fig = plt.figure(figsize=(20, 20))
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
#
#     for idx, img in enumerate(os.listdir('./captcha/prediction/')):
#         pil_image = Image.open('captcha/prediction/{}'.format(img)).convert('1')
#         ax = fig.add_subplot(10, 12, idx+1, xticks=[], yticks=[])
#         ax.imshow(pil_image, cmap=plt.cm.binary, interpolation='nearest')  # 先查看測試集資料
        # pil_image.show()


def resize_and_standard():
    data = []
    basewidth = 50
    fig = plt.figure(figsize=(20, 20))
    cnt = 0
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for idx, img in enumerate(os.listdir('captcha/prediction/')):
        pil_image = Image.open('captcha/prediction/{}'.format(img)).convert('1')

        ax = fig.add_subplot(10, 12, idx+1, xticks=[], yticks=[])
        ax.imshow(pil_image, cmap=plt.cm.binary, interpolation='nearest')  # 先查看測試集資料
        pil_image.show()

        wpercent = (basewidth / float(pil_image.size[0]))
        hsize = int((float(pil_image.size[1]) * float(wpercent)))
        img = pil_image.resize((basewidth, hsize), Image.ANTIALIAS)  # 轉換成相對應大小
        data.append([pixel for pixel in iter(img.getdata())])
    scaler = StandardScaler()
    scaler.fit(data)
    data_scaled = scaler.transform(data)  # 標準化
    return data_scaled


def main():
    create_prediction_data()
    # showPredictionData()
    mlp = load_model()
    data_scaled = resize_and_standard()
    captcha = mlp.predict(data_scaled)
    print(captcha)

if __name__ == '__main__':
    main()
    print("done")

