import numpy as np
import imutils
import pickle
import cv2
import os
from extract_embeddings import embedders_extractor
from train_model import modelTrain

# класс распознавателя лиц
class FaceRecognition:
	def __init__(self, confidence):
		self.confidence = confidence

		# используем встроенный в OpenCV детектор на базе глубокого обучения
		# который работает на основе SSD (Single Shot Detector) с использованием нейросети ResNet
		print("[INFO] загружаем детектор человеческих лиц...")
		# загружаем протомодель
		protoPath = os.path.sep.join(["face_detection_model/", "deploy.prototxt"])
		# загружаем веса в формате caffe
		modelPath = os.path.sep.join(["face_detection_model/", "res10_300x300_ssd_iter_140000.caffemodel"])
		self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

		print("[INFO] загружаем модель нейросети...")
		self.embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

		# предобработка картинок для обучения нейросети
		embedders_extractor(self.detector, self.embedder)
		# обучаем нейросеть
		modelTrain()

		# загружаем обученную модель и метки классов (лиц)
		self.recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
		self.le = pickle.loads(open("output/le.pickle", "rb").read())

	def recognize(self, image):
		# загружаем тестовую картинку и ресайхим ее к размеру 600х600
		image = cv2.imread(image)
		image = imutils.resize(image, width=600)
		(h, w) = image.shape[:2]

		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(image, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		# отдаем картинку определителю лиц
		self.detector.setInput(imageBlob)
		detections = self.detector.forward()

		# обрабатываем все найденные прямоугольники с лицами
		for i in range(0, detections.shape[2]):
			# берем текущую вероятность того, что это лицо
			confidence = detections[0, 0, i, 2]

			# отбрасываем лица с вероятностью меньше пороговой (0.5 по умолчанию)
			if confidence > self.confidence:
				# вырезаем прямоугольник с лицом из картинки
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				face = image[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# лица размером меньше 20х20 не рассматриваем
				if fW < 20 or fH < 20:
					continue

				# создаем из лица blob и отправляем его в модель
				# для получения 128-мерной гиперсферы с просчитанными векторами
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
					(0, 0, 0), swapRB=True, crop=False)
				self.embedder.setInput(faceBlob)
				vec = self.embedder.forward()

				# выполняем классификацию для предсказания класса текущей картинки
				preds = self.recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = self.le.classes_[j]

				# рисуем на исходной картинке прямоугольник с вычисленной вероятностью принадлежности
				# к предсказанному классу
				text = "{}: {:.2f}%".format(name, proba * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(image, (startX, startY), (endX, endY),
					(0, 0, 255), 2)
				cv2.putText(image, text, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

		# выводим результат на экран
		cv2.imshow("Image", image)
		cv2.waitKey(0)