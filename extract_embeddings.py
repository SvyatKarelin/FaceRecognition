from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# класс предобработки картинок для обучения модели
# выделяет на картинке человеческое лицо и вычисляет для него векторную гиперсферу в 128-мерном пространстве
# вычисленная гиперсфера будет использоваться для обучения модели распознавателя лиц
class embedders_extractor:
	def __init__(self, detector, embedder):

		# собираем пути к картинкам для обучения модели
		print("[INFO] обрабатываем картинки...")
		imagePaths = list(paths.list_images("dataset"))

		# заведем списки для подсчитанных векторов лиц и
		# соответствующих им имен людей
		knownEmbeddings = []
		knownNames = []

		# общий счетчик лиц
		total = 0

		# обрабатываем все найденные пути к картинкам
		for (i, imagePath) in enumerate(imagePaths):
			# достаем имя человека из пути к картинке
			print("[INFO] обрабатываем картинку {}/{}".format(i + 1,
				len(imagePaths)))
			name = imagePath.split(os.path.sep)[-2]

			# загружаем картинку, приводим к размеру 600х600
			image = cv2.imread(imagePath)
			image = imutils.resize(image, width=600)
			# берем высоту и ширину картинки
			(h, w) = image.shape[:2]

			# создаем blob (binary large object) из картинки для отправки детектору на обработку
			imageBlob = cv2.dnn.blobFromImage(
				cv2.resize(image, (300, 300)), 1.0, (300, 300),
				(104.0, 177.0, 123.0), swapRB=False, crop=False)

			# находим на картинке лицо
			detector.setInput(imageBlob)
			detections = detector.forward()

			# если хотя бы одно лицо найдено
			if len(detections) > 0:
				# работаем только с одним лицом на картинке
				# поэтому берем лицо, у которого рассчитанная вероятность наибольшая
				i = np.argmax(detections[0, 0, :, 2])
				confidence = detections[0, 0, i, 2]

				if confidence > 0.5:
					# вычисляем координаты прямоугольника лица
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					# и вырезаем его из картинки
					face = image[startY:endY, startX:endX]
					(fH, fW) = face.shape[:2]

					# размер картинки должен быть не меньше 20х20 для обучения модели
					if fW < 20 or fH < 20:
						continue

					# создаем из лица blob и отправляем его в модель
					# для получения 128-мерной гиперсферы с просчитанными векторами
					faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
						(96, 96), (0, 0, 0), swapRB=True, crop=False)
					embedder.setInput(faceBlob)
					vec = embedder.forward()

					# сохраняем вектора и имя человека в списках
					knownNames.append(name)
					knownEmbeddings.append(vec.flatten())
					total += 1

		# сохраняем подсчитанные вектора и имя челокека на диск
		print("[INFO] сохраняем {} рассчитанных массивов данных...".format(total))
		data = {"embeddings": knownEmbeddings, "names": knownNames}
		f = open("output/embeddings.pickle", "wb")
		f.write(pickle.dumps(data))
		f.close()