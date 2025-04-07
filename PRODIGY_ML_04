import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


data = []
labels = []
main_folder_path = r"E:\ProdigyMaterial\Task4-HandGesture\images"
for label, subfolder in enumerate(os.listdir(main_folder_path), start=1):
    subfolders_path = os.path.join(main_folder_path, subfolder)
    if os.path.isdir(subfolders_path):
        for filename in os.listdir(subfolders_path):
            img_path = os.path.join(subfolders_path, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
            image = cv2.resize(image, (64, 64))  
            data.append(image.flatten())  
            labels.append(label)

datanew = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(datanew, labels, test_size=0.2, random_state=24)

knn = KNeighborsClassifier(n_neighbors=2)  
knn.fit(X_train, y_train)

y_predict_train = knn.predict(X_train)
y_predict_test = knn.predict(X_test)

training_accuracy = accuracy_score(y_train, y_predict_train)
testing_accuracy = accuracy_score(y_test, y_predict_test)
print("Training accuracy:", training_accuracy)

conf_matrix_train = confusion_matrix(y_train, y_predict_train)
print("Confusion Matrix (Training):")
print(conf_matrix_train)

plt.figure(figsize=(10, 6))
plt.imshow(conf_matrix_train, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Training)")
plt.colorbar()
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()

#plotting TPR FNR

tp_train = np.diag(conf_matrix_train)
fn_train = np.sum(conf_matrix_train, axis=1) - tp_train
tpr_for_train = tp_train / np.sum(conf_matrix_train, axis=1)
fnr_for_train = fn_train / np.sum(conf_matrix_train, axis=1)

plt.figure(figsize=(10, 9))
plt.subplot(2, 1, 1)
classes = np.arange(1, len(tpr_for_train) + 1)
bars_width = 0.2
plt.bar(classes - bars_width/2, tpr_for_train, bars_width, label='True positive rate (TPR)', color='b')
plt.bar(classes + bars_width/2, fnr_for_train, bars_width, label='False negative rate (FNR)', color='r')
plt.xlabel('Classes')
plt.ylabel('Rate')
plt.title('TPR and FNR for 8 classes (Training)')
plt.xticks(classes)
plt.legend()
plt.tight_layout()
