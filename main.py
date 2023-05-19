from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog
from PyQt5 import uic
from PyQt5.QtGui import QPixmap,QImage
import sys
import cv2


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
            # Convert the pixmap to a QImage
            image = self.pixmap.toImage()

            # Convert the QImage to a numpy array
            width = image.width()
            height = image.height()
            buffer = image.constBits()
            dtype = buffer.dtype
            buffer = buffer.reshape(height, width, -1)
            image_array = buffer.copy()

            # Convert the image array to OpenCV format
            opencv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            # Define the new dimensions for the resized image
            new_width = 500
            new_height = 300

            # Resize the image
            resized_image = cv2.resize(opencv_image, (new_width, new_height))

            # Convert the OpenCV image to RGB format
            opencv_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

            # Create a QImage from the OpenCV image
            height, width, channel = opencv_image_rgb.shape
            qimage = QImage(opencv_image_rgb.data, width, height, QImage.Format_RGB888)

            # Create a QPixmap from the QImage
            self.pixmap = QPixmap.fromImage(qimage)
            # Add Pic to label
            self.label.setPixmap(self.pixmap)


# Initialize The App
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
