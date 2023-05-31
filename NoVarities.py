from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage
from tensorflow import keras
from keras.utils import img_to_array
import sys
import cv2
import numpy as np


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        # initial bitmap to save our image
        self.pixmap = None
        self.filename = None

        # Load the ui file
        uic.loadUi("Image 2.ui", self)

        # Define our widgets
        self.upload_button = self.findChild(QPushButton, "upload_button")
        self.results_button = self.findChild(QPushButton, "results_button")
        self.screen_one = self.findChild(QLabel, "screen01")
        self.screen_two = self.findChild(QLabel, "screen02")
        self.bousselam = self.findChild(QLabel, "bousselam_output")
        self.gta = self.findChild(QLabel, "gta_output")
        self.oued_el_bared = self.findChild(QLabel, "oued_el_bared_output")
        self.vitron = self.findChild(QLabel, "vitron_output")
        self.avoine = self.findChild(QLabel, "avoine_output")
        self.ble_tendre = self.findChild(QLabel, "ble_tendre_output")
        self.orge = self.findChild(QLabel, "orge_output")
        self.triticale = self.findChild(QLabel, "triticale_output")
        self.total = self.findChild(QLabel, "total_output")

        # Click The Dropdown Box
        self.upload_button.clicked.connect(self.upload)
        self.results_button.clicked.connect(self.results)

        # Show The App
        self.show()

    def upload(self):
        self.filename = QFileDialog.getOpenFileName(self, "Open File", "D:\\DataSet\\Final\\Full images\\Avoine",
                                                    "All Files (*);;PNG Files (*.png);;Jpg Files (*.jpg)")

        # Open The Image
        if self.filename:
            self.pixmap = QPixmap(self.filename[0])
            # Convert the QPixmap to a QImage
            qimage = self.pixmap.toImage()

            # Convert the QImage to a numpy array
            width = qimage.width()
            height = qimage.height()
            buffer = qimage.bits().asstring(qimage.byteCount())
            image_array = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))

            # Convert the image array to OpenCV format
            image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)

            # 1.Display original image

            # Define the maximum dimensions for the resized image
            new_width = 600
            new_height = 800

            # Resize the image
            resized_image = cv2.resize(image, (new_width, new_height))

            # Convert the resized OpenCV image back to QImage
            resized_qimage = QImage(resized_image.data, new_width, new_height, QImage.Format_RGB888)

            # Convert the QImage to QPixmap
            self.pixmap = QPixmap.fromImage(resized_qimage)

            # Add Pic to label
            self.screen_one.setPixmap(self.pixmap)

    def results(self):
        if self.pixmap:

            self.pixmap = QPixmap(self.filename[0])
            # Convert the QPixmap to a QImage

            qimage = self.pixmap.toImage()

            # Convert the QImage to a numpy array
            width = qimage.width()
            height = qimage.height()
            buffer = qimage.bits().asstring(qimage.byteCount())
            image_array = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))

            # Convert the image array to OpenCV format
            image_screen = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)

            image = cv2.imread(self.filename[0])

            # Define the maximum dimensions for the resized image
            new_width = 600
            new_height = 800

            # 2. Image Segmentation

            # Define the gamma value
            gamma = 0.8

            # Apply gamma correction
            gamma_img = np.power(image / 255.0, gamma)
            gamma_img = np.uint8(gamma_img * 255)

            # Convert to grayscale
            gray = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2GRAY)

            # Define the structuring element
            kernel = np.ones((21, 21), np.uint8)

            # Perform close operation
            img_close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

            # Apply median blur
            img_blur = cv2.medianBlur(img_close, 15)  # Here, 15 is the kernel size

            # Apply Otsu's method to automatically determine the threshold value
            thresh_val, thresh_img = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find contours in the binary image
            contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            total_number = 0

            crop_image = []
            contours_final = []
            for contour in contours:
                if cv2.contourArea(contour) > 1000:  # Minimum area threshold
                    contours_final.append(contour)
                    # Get the bounding box coordinates
                    x, y, w, h = cv2.boundingRect(contour)
                    print("Area =", cv2.contourArea(contour), "x =", x, "y =", y, "w =", w, "h =", h)

                    crop_img = image[y:(y + h), x:(x + w)]
                    # Get the original height and width of the image
                    height, width = crop_img.shape[:2]

                    # Set the desired output size
                    output_size = 350

                    # Calculate the amount of padding needed on each side
                    h_pad = max(0, (output_size - height) // 2)
                    w_pad = max(0, (output_size - width) // 2)

                    # Add the padding using copyMakeBorder() function
                    output_img = cv2.copyMakeBorder(crop_img, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_CONSTANT,
                                                    value=(0, 0,
                                                           0))

                    # crop image
                    crop_image.append(output_img)

            # Load your saved model
            model = keras.models.load_model('Model/ResNet50NoVarieties50.h5')
            # print("hello")
            # Preprocess each image in the list
            preprocessed_images = []
            # i = 0
            for image_tf in crop_image:
                # print(i)
                # Convert BGR to RGB and resize the image
                image_tf = cv2.cvtColor(image_tf, cv2.COLOR_BGR2RGB)
                image_tf = cv2.resize(image_tf, (224, 224))

                # Convert the OpenCV image to a NumPy array
                image_tf = np.asarray(image_tf)

                image_tf = img_to_array(image_tf)
                normalized_img = image_tf / 255.0
                image_tf = np.expand_dims(normalized_img, axis=0)

                preprocessed_images.append(image_tf)
                # i += 1

            preprocessed_images = np.array(preprocessed_images)

            preprocessed_images = preprocessed_images.reshape(-1, 224, 224, 3)

            predictions = model.predict(preprocessed_images)

            predictions = np.argmax(predictions, axis=1)

            print(predictions)

            for prediction, contour in zip(predictions, contours_final):
                if prediction == 0:
                    cv2.drawContours(image_screen, [contour], 0, (201, 28, 28), 10)
                    total_number += 1
                elif prediction == 1:
                    cv2.drawContours(image_screen, [contour], 0, (52, 52, 171), 10)
                    total_number += 1
                elif prediction == 2:
                    cv2.drawContours(image_screen, [contour], 0, (24, 141, 24), 10)
                    total_number += 1
                elif prediction == 3:
                    cv2.drawContours(image_screen, [contour], 0, (214, 136, 0), 10)
                    total_number += 1
                elif prediction == 4:
                    cv2.drawContours(image_screen, [contour], 0, (0, 200, 200), 10)
                    total_number += 1

            # Resize the image
            resized_image = cv2.resize(image_screen, (new_width, new_height))

            # Convert the resized OpenCV image back to QImage
            resized_qimage = QImage(resized_image.data, new_width, new_height, QImage.Format_RGB888)

            # Convert the QImage to QPixmap
            self.pixmap = QPixmap.fromImage(resized_qimage)

            self.screen_two.setPixmap(self.pixmap)

            # print the final results
            self.bousselam.setText(str(np.count_nonzero(predictions == 1)))
            self.gta.setText("0")
            self.oued_el_bared.setText("0")
            self.vitron.setText("0")
            self.avoine.setText(str(np.count_nonzero(predictions == 0)))
            self.ble_tendre.setText(str(np.count_nonzero(predictions == 2)))
            self.orge.setText(str(np.count_nonzero(predictions == 3)))
            self.triticale.setText(str(np.count_nonzero(predictions == 4)))
            self.total.setText(str(total_number))

        else:
            print("No image available")


# Initialize The App
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
