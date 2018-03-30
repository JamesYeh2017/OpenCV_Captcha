import requests
from PIL import Image
import numpy
from matplotlib import pyplot as plt
import cv2
from datetime import datetime
import time
import os


def crawler_captcha():
    with open('./captcha/captcha.jpg', 'wb') as f:
        res = requests.get('http://gcis.nat.gov.tw/pub/kaptcha.jpg')  # network/image
        f.write(res.content)  # 寫入

    # image = Image.open('./captcha/captcha.jpg')
    # image.show()


def find_contours():
    crawler_captcha()
    pil_image = Image.open('./captcha/captcha.jpg').convert('RGB')
    open_cv_image = numpy.array(pil_image)  # image => number
    # print(open_cv_image)
    plt.imshow(open_cv_image)  # use plt to read number and show img
    # plt.show()

    imgray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)  # convert color 轉換顏色
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)  # (原圖, 閾值, 最大值), 大於閥值就是最大, 反之最小
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找出輪廓(物件,階層式輪廓,終點座標)

    cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key=lambda x: x[1])

    ary = []
    for (c, _) in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        # print((x, y, w, h))
        if w >= 15 and h == 24:
            ary.append((x, y, w, h))

    len_captcha = len(ary)
    return len_captcha, open_cv_image, ary


def save_captcha_data(captcha, open_cv_image, ary):
    # fig = plt.figure()
    ct = int(time.mktime(datetime.now().timetuple()))
    for index, (x, y, w, h) in enumerate(ary):
        roi = open_cv_image[y:y+h, x:x+w]
        thresh = roi.copy()
        # a = fig.add_subplot(1, len(ary), index + 1)
        fig = plt.figure()
        plt.imshow(thresh)
        # plt.show()
        if captcha:
            plt.savefig('./captcha/{}/{}_{}.jpg'.format(captcha[index], ct, index), dpi=100) # 訓練集資料
        else:
            plt.savefig('./captcha/prediction/{}.jpg'.format(ct+index), dpi=100) # 測試集資料


def main():
    result = find_contours()
    while int(result[0]) < 6:
        print("lenCaptcha :", result[0])
        result = find_contours()
    print("lenCaptcha :", result[0])
    open_cv_image = result[1]
    ary = result[2]

    captcha = str(input("驗證碼 : "))
    if len(captcha) < 6:
        captcha = str(input("驗證碼 : "))
    else:
        captcha = captcha if str(input("確認 : ")) else captcha

    save_captcha_data(captcha, open_cv_image, ary)

if __name__ == '__main__':
    for i in range(1):
        print(i)
        main()
        print("done")
