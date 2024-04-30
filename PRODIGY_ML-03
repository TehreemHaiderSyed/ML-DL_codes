import os
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cats_folder = r'C:\Users\Dell\Downloads\ProdigyMaterial\Task3-SVM\Cats&DogsImgclassif\train\cats'
dogs_folder = r'C:\Users\Dell\Downloads\ProdigyMaterial\Task3-SVM\Cats&DogsImgclassif\train\dogs'

def load_and_preprocess_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image = image.resize((224, 224))  # Resize to 224x224 pixels
            images.append(image)
    return images

cats_images = load_and_preprocess_images(cats_folder)
dogs_images = load_and_preprocess_images(dogs_folder)

def extract_hog_features(images): # HOG features
    features = []
    for image in images:
        image_gray = image.convert('L')
        hog_features = hog(image_gray, pixels_per_cell=(16, 16), cells_per_block=(4, 4), block_norm='L2-Hys', visualize=False)
        features.append(hog_features)
    return np.array(features)


cats_features = extract_hog_features(cats_images)
dogs_features = extract_hog_features(dogs_images)
X = np.concatenate((cats_features, dogs_features))

# Label --> (cats = 0,  dogs = 1)
cats_labels = np.zeros(len(cats_features))
dogs_labels = np.ones(len(dogs_features))
y = np.concatenate((cats_labels, dogs_labels))

# training data 80% testing data 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


svm = SVC(kernel='poly', C=0.15)
svm.fit(X_train, y_train)
y_pred_train = svm.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f'Classification Accuracy: {train_accuracy}')

conf_matrix_train = confusion_matrix(y_train, y_pred_train)
plt.figure(figsize=(9, 7))
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Oranges', xticklabels=['Cats', 'Dogs'], yticklabels=['Cats', 'Dogs'])
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.title('Confusion Matrix')
plt.show()

