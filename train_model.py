from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# класс для обучения модели распознавателя лиц
class modelTrain():
    def __init__(self):

        # загружаем обсчитанные вектора для лиц
        print("[INFO] загружаем вектора...")
        data = pickle.loads(open("output/embeddings.pickle", "rb").read())

        # загружаем метки классов (имена людей на картинках)
        print("[INFO] загружаем метки класов...")
        le = LabelEncoder()
        labels = le.fit_transform(data["names"])

        # обучаем модель на полученных ранее 128-мерных векторах для каждого лица
        print("[INFO] обучаем модель...")
        # используем метод опорных векторов (Linear Vector Support Machine) библиотеки scikit-learn
        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(data["embeddings"], labels)

        # сохраняем обученную модель на диск
        f = open("output/recognizer.pickle", "wb")
        f.write(pickle.dumps(recognizer))
        f.close()

        # сохраняем метки классов (имена людей)
        f = open("output/le.pickle", "wb")
        f.write(pickle.dumps(le))
        f.close()