from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage
import sys
import cv2
import numpy as np


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        # Load the ui file
        self.pixmap = None
        uic.loadUi("Image.ui", self)

        # Define our widgets
        self.button = self.findChild(QPushButton, "pushButton")
        self.label = self.findChild(QLabel, "label")

        # Click The Dropdown Box
        self.button.clicked.connect(self.clicker)

        # Show The App
        self.show()

    def clicker(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", "D:\DataSet\Final\Full images\Avoine",
                                            "All Files (*);;PNG Files (*.png);;Jpg Files (*.jpg)")

        # Open The Image
        if fname:
            self.pixmap = QPixmap(fname[0])
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
            max_width = 500
            max_height = 670

            # Define the maximum dimensions for the resized image
            current_height = image.shape[0] - 9
            current_width = image.shape[1] - 105
            print(current_width)
            print(current_height)
            aspect_ratio = 8
            print(aspect_ratio)
            # Get the current height and width of the image
            height, width = image.shape[:2]

            # Calculate the aspect ratio
            aspect_ratio = width / height

            # Calculate the new dimensions based on the aspect ratio
            new_width = int(min(max_width, int(max_height * aspect_ratio)))
            new_height = int(min(max_height, int(max_width / aspect_ratio)))

            new_width = int(current_width / aspect_ratio)
            new_height = int(current_height / aspect_ratio)
            print(new_height)
            print(new_width)
            # new_width = 500
            # new_height = 500

            # Resize the image
            resized_image = cv2.resize(image, (new_width, new_height))

            # Convert the resized OpenCV image back to QImage
            resized_qimage = QImage(resized_image.data, new_width, new_height, QImage.Format_RGB888)

            # Convert the QImage to QPixmap
            self.pixmap = QPixmap.fromImage(resized_qimage)

            # Add Pic to label
            self.label.setPixmap(self.pixmap)
            self.label_2.setPixmap(self.pixmap)


# Initialize The App
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
