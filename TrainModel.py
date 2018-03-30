import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier # 多層神經元分類器
import numpy
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler  # 標準化
from sklearn.externals import joblib


def resize():
    digits = []
    labels = []
    basewidth = 50  # 圖形縮小
    fig = plt.figure(figsize=(20, 20))  # open a picture
    cnt = 0
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)  # 切一個小圖
    for i in range(0, 10):
        for img in os.listdir('./captcha/{}/'.format(i)):
            pil_image = Image.open('./captcha/{}/{}'.format(i, img)).convert('1')  # conver('1')轉成黑白

            wpercent = (basewidth / float(pil_image.size[0]))
            hsize = int((float(pil_image.size[1]) * float(wpercent)))   # 高
            img = pil_image.resize((basewidth, hsize), Image.ANTIALIAS)  # 縮小可以減短訓練時間
            # img.show()                 # 寬, 高

            ax = fig.add_subplot(30, 12, cnt + 1, xticks=[], yticks=[])
            ax.imshow(img, cmap=plt.cm.binary, interpolation='nearest')
            ax.text(0, 7, str(i), color="red", fontsize=20)
            cnt = cnt + 1

            digits.append([pixel for pixel in iter(img.getdata())])  # 特徵  50*33
            labels.append(i)  # 目標答案
    return digits, labels


def train_model(digits, labels, max_iter):
    digit_ary = numpy.array(digits)
    print("digit shape", digit_ary.shape)
    scaler = StandardScaler()
    scaler.fit(digit_ary)
    x_scaled = scaler.transform(digit_ary)
    mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), activation='logistic', max_iter=max_iter)  # 3層神經層,iter次數越高準度越高
    mlp.fit(x_scaled, labels)
    return mlp, x_scaled


def model_accuracy(labels, mlp, x_scaled):
    predicted = mlp.predict(x_scaled)
    target = numpy.array(labels)
    predicted_list = list(predicted == target)
    mistake = filter(lambda x: x is not True, predicted_list)
    accuracy = (len(predicted_list)-len(list(mistake)))/len(predicted_list)*100
    print("accuracy :", accuracy, "%")
    return accuracy


def dump_model(mlp):
    joblib.dump(mlp, './captcha/captcha.pkl')


def main():
    result = resize()
    results = train_model(digits=result[0], labels=result[1], max_iter=30000)
    accuracy = model_accuracy(labels=result[1], mlp=results[0], x_scaled=results[1])
    if accuracy > 80:
        dump_model(mlp=results[0])

if __name__ == '__main__':
    main()
    print("done")

