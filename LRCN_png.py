import os
import numpy as np
from keras.preprocessing.image import load_img
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, LSTM, ConvLSTM2D
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, recall_score, precision_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from keras.regularizers import l2
from PIL import Image, UnidentifiedImageError
import pickle
# 取消Pillow图像大小限制
# 取消Pillow图像大小限制
Image.MAX_IMAGE_PIXELS = None
def load_image(root_folder_path, target_size=(256, 256), time_steps=2):
    images = []
    labels = []

    class_folders = sorted(os.listdir(root_folder_path))
    for label, class_folder in enumerate(class_folders):
        class_path = os.path.join(root_folder_path, class_folder)
        for filename in os.listdir(class_path):
            if filename.endswith(".png"):
                img_path = os.path.join(class_path, filename)
                img = load_img(img_path, target_size=target_size)
                img_array = np.array(img)
                images.append(img_array)
                labels.append(label)  # 每个图像只添加一个标签
                print("image :", img_path)
                print("label :", label)
    images = np.array(images)
    labels = to_categorical(labels, num_classes=len(class_folders))

    # 将图像整理成时序数据，每个样本有 time_steps 个时间步骤，每个时间步骤包含一个图像
    num_chunks = images.shape[0] // time_steps
    images = images.reshape(num_chunks, time_steps, target_size[0], target_size[1], 3)
    labels = labels[:num_chunks]
    return images, labels

# Load and preprocess images from five folders
root_folder_path = 'D:/py_project/Ai_data/stft'
# Replace with the actual path to your root folder
images, labels = load_image(root_folder_path)

print(f"Shape of images: {images.shape}")
print(f"Shape of labels: {labels.shape}")
images_1 = np.copy(images)
labels_1 = np.copy(labels)
image_1_1 = np.array_split(images_1, 5)
label_1_1 = np.array_split(labels_1, 5)
images_1_2 = image_1_1[3]
labels_1_2 = label_1_1[3]
# 切分資料集
X_train, X_test, y_train, y_test = train_test_split(images_1_2, labels_1_2, test_size=0.2, random_state=42)
print(f"Shape of images: {images_1_2.shape}")
print(f"Shape of labels: {labels_1_2.shape}")

# 构建GC-LSTM模型
model = Sequential()
model.add(TimeDistributed(Conv2D(4, (3, 3), strides=(2, 2), activation='relu'), input_shape=(2, 256, 256, 3)))
model.add(TimeDistributed(Conv2D(8, (3, 3), strides=(2, 2), activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(len(os.listdir(root_folder_path)), activation='softmax'))
model.summary()

with open('history2.pkl', 'rb') as f:
    history = pickle.load(f)

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
History = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print(f"Test Accuracy: {accuracy}")

# Save the model
model.save("D2_lrcn_model.keras")

# 保存 history 对象
with open('history2.pkl', 'wb') as f:
    pickle.dump(History.history, f)

# Predict classes
predictions = model.predict(images)
predicted_classes = np.argmax(predictions, axis=1)  # Adjust axis based on your model's output

# Flatten the batch and time_steps for comparison with True labels
predicted_classes_2D = predicted_classes.flatten()
True_classes = np.argmax(labels[:len(predicted_classes_2D)], axis=1)

# Print predictions for each image
for i in range(len(True_classes)):
    print(f"Test Image {i + 1}: Actual Label - {True_classes[i]}, Predicted Label - {predicted_classes_2D[i]}")

# Print confusion matrix and classification report
confusion_mat = confusion_matrix(True_classes, predicted_classes_2D)
classification_rep = classification_report(True_classes, predicted_classes_2D, target_names=["Class 0", "Class 1", "Class 2"])
print("Confusion Matrix for predictions:\n", confusion_mat)
print("Classification Report for predictions:\n", classification_rep)

recall_png = recall_score(True_classes, predicted_classes_2D, average=None)
# Calculate recall for each class
print("Recall for each class:")
print(recall_png)
precision_png = precision_score(True_classes, predicted_classes_2D, average=None)
print("Precision for each class:")
print(precision_png)
# 计算混淆矩阵
conf_matrix = confusion_matrix(True_classes, predicted_classes_2D)
# 绘制混淆矩阵图
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=[f'Class {i}' for i in range(conf_matrix.shape[0])],
            yticklabels=[f'Class {i}' for i in range(conf_matrix.shape[0])])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
# 保存混淆矩陣圖像
plt.savefig('confusion_matrix.png')
# 顯示圖像
plt.show()

# 生成分類報告
class_report = classification_report(True_classes, predicted_classes_2D, target_names=[f'Class {i}' for i in range(conf_matrix.shape[0])])

# 打印分類報告
print(class_report)

# 保存分類報告到文本文件
with open('classification_report.txt', 'w') as f:
    f.write(class_report)

# 創建 PDF 文件
pdf_file = 'classification_report.pdf'
c = canvas.Canvas(pdf_file, pagesize=letter)
width, height = letter

# 添加混淆矩陣圖像到 PDF
c.drawImage('confusion_matrix.png', 100, 400, width=400, height=400)

# 添加分類報告到 PDF
text = c.beginText(40, 350)
text.setFont("Helvetica", 10)
for line in class_report.split('\n'):
    text.textLine(line)
c.drawText(text)

# 保存 PDF 文件
c.save()