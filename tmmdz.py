import cv2
import dlib
import numpy as np
import ipywidgets as widgets
from IPython.display import display


# Функция для наложения бровей на изображение
def add_eyebrows(image, x, y, w, h, brow_img):
    brow_img = cv2.resize(brow_img, (w, h))
    if brow_img.shape[2] == 4:
        alpha = brow_img[:, :, 3] / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])
        brow = brow_img[:, :, :3]
    else:
        brow = brow_img
        alpha = None  # add this line to initialize alpha

    roi = image[y:y + h, x:x + w]
    if alpha is not None:  # add this check to use alpha only if it has been initialized
        bg = cv2.multiply(1.0 - alpha, roi, dtype=cv2.CV_8UC3)
        fg = cv2.multiply(alpha, brow, dtype=cv2.CV_8UC3)
        dst = cv2.add(bg, fg)
        image[y:y + h, x:x + w] = dst
    else:
        image[y:y + h, x:x + w] = brow


# Загрузка каскада Хаара для обнаружения лиц
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Загрузим детектор лица
detector = dlib.get_frontal_face_detector()

# Загрузим предиктор, для поиска точек на лице
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Загрузка изображения с бровями
eyebrows_image = cv2.imread('Eyebrows_1.png')

# Подключим камеру
cap = cv2.VideoCapture(0)

# Создание окна для трекбара
cv2.namedWindow('BGR')

# Размер трекбара
cv2.resizeWindow('BGR', 640, 240)

# Трекбары будут работать с цветами
def empty(a):
    pass

cv2.createTrackbar('Blue', 'BGR', 0, 255, empty)
cv2.createTrackbar('Green', 'BGR', 0, 255, empty)
cv2.createTrackbar('Red', 'BGR', 0, 255, empty)


def createBox(img, points, scale=5, masked=False, cropped=True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [points], (255, 255, 255))
        img = cv2.bitwise_and(img, mask)

    if cropped:
        bbox = cv2.boundingRect(points)
        x, y, w, h = bbox
        imgCrop = img[y:y + h, x:x + w]
        imgCrop = cv2.resize(imgCrop, (0, 0), None, scale, scale)
        return imgCrop
    else:
        return mask


global bi
bi = 'Eyebrows_1.png'

while True:
    _, img = cap.read()

    # Функция детектора работает только в градации серого, переведем фото
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    # Используем детектор
    faces = detector(gray)

    for face in faces:
        x1 = face.left()  # левая точка
        y1 = face.top()  # верхняя точка
        x2 = face.right()  # правая точка
        y2 = face.bottom()  # нижняя точка

        # Поиск ориентиров
        landmarks = predictor(image=gray, box=face)

        myPoints = []
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            myPoints.append([x, y])

        myPoints = np.array(myPoints)
        imgLeftEyeBrow = createBox(img, myPoints[17:22])
        imgRightEyeBrow = createBox(img, myPoints[22:27])

        maskLeftEyebrow = createBox(img, myPoints[17:22], 8, masked=True, cropped=False)
        maskRightEyebrow = createBox(img, myPoints[22:27], 8, masked=True, cropped=False)
        imgColorLeftEyebrow = np.zeros_like(maskLeftEyebrow)
        imgColorRightEyebrow = np.zeros_like(maskRightEyebrow)
        b = cv2.getTrackbarPos("Blue", "BGR")
        g = cv2.getTrackbarPos("Green", "BGR")
        r = cv2.getTrackbarPos("Red", "BGR")

        imgColorLeftEyebrow[:] = b, g, r
        imgColorLeftEyebrow = cv2.bitwise_and(maskLeftEyebrow, imgColorLeftEyebrow)
        imgColorLeftEyebrow = cv2.GaussianBlur(imgColorLeftEyebrow, (7, 7), 10)

        imgColorRightEyebrow[:] = b, g, r
        imgColorRightEyebrow = cv2.bitwise_and(maskRightEyebrow, imgColorRightEyebrow)
        imgColorRightEyebrow = cv2.GaussianBlur(imgColorRightEyebrow, (7, 7), 10)

        result = cv2.addWeighted(img, 1, imgColorLeftEyebrow, 0.4, 0)
        result = cv2.addWeighted(result, 1, imgColorRightEyebrow, 0.4, 0)

        # Конвертация кадра в оттенки серого
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Обнаружение лиц на кадре
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Наложение выбранных усов на обнаруженные лица
        for (x, y, w, h) in faces:
            add_eyebrows(result, x, y, w, h // 2, cv2.imread(bi, cv2.IMREAD_UNCHANGED))

        # Отображение кадра
        cv2.imshow('BGR', result)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            print('Изменили на первый вариант')
            bi = 'Eyebrows_1.png'

        if cv2.waitKey(1) & 0xFF == ord('b'):
            print('Изменили на второй вариант')
            bi = 'Eyebrows_2.png'

        if cv2.waitKey(1) & 0xFF == ord('c'):
            print('Изменили на третий вариант')
            bi = 'Eyebrows_3.png'

        # Ожидание нажатия клавиши 'q' для выхода из цикла
        if cv2.waitKey(1) & 0xFF == ord('d'):
            break

    # Для выхода необходимо нажать Esc
    if cv2.waitKey(delay=1) == 27:
        break

# Завершаем съемку
cap.release()

cv2.destroyAllWindows()