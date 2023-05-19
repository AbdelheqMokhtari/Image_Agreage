from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage
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
        uic.loadUi("Image.ui", self)

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
            image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)

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

            # Filter out small contours and draw the remaining contours on the original image
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Minimum area threshold
                    cv2.drawContours(image, [contour], 0, (0, 255, 0), 5)

            # Resize the image
            resized_image = cv2.resize(image, (new_width, new_height))

            # Convert the resized OpenCV image back to QImage
            resized_qimage = QImage(resized_image.data, new_width, new_height, QImage.Format_RGB888)

            # Convert the QImage to QPixmap
            self.pixmap = QPixmap.fromImage(resized_qimage)

            self.screen_two.setPixmap(self.pixmap)

            # print the final results
            self.bousselam.setText("0")
            self.gta.setText("0")
            self.oued_el_bared.setText("0")
            self.vitron.setText("0")
            self.avoine.setText("0")
            self.ble_tendre.setText("0")
            self.orge.setText("0")
            self.triticale.setText("0")
            self.total.setText("0")

        else:
            print("No image available")


# Initialize The App
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
