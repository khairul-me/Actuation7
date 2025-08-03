from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QSlider, QLabel, QFileDialog, QMainWindow, QButtonGroup, QApplication, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QCoreApplication, QObject, QRunnable, QThread, QThreadPool, pyqtSignal, QTimer, QRect


class ImageAdjustmentUI(QWidget):
    sliderValuesChanged = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        self.saturation_slider = QSlider()
        self.saturation_slider.setOrientation(1)
        self.saturation_slider.setMinimum(0)
        self.saturation_slider.setMaximum(200)
        self.saturation_slider.setValue(100)
        layout.addWidget(QLabel("Saturation"))
        layout.addWidget(self.saturation_slider)

        self.saturation_label = QLabel("Saturation: 1.00")
        layout.addWidget(self.saturation_label)

        self.hue_slider = QSlider()
        self.hue_slider.setOrientation(1)
        self.hue_slider.setMinimum(0)
        self.hue_slider.setMaximum(360)
        layout.addWidget(QLabel("Hue"))
        layout.addWidget(self.hue_slider)

        self.hue_label = QLabel("Hue: 0")
        layout.addWidget(self.hue_label)

        self.value_slider = QSlider()
        self.value_slider.setOrientation(1)
        self.value_slider.setMinimum(0)
        self.value_slider.setMaximum(200)
        self.value_slider.setValue(100)
        layout.addWidget(QLabel("Value"))
        layout.addWidget(self.value_slider)

        self.value_label = QLabel("Value: 1.00")
        layout.addWidget(self.value_label)

        self.setLayout(layout)

        self.saturation_slider.valueChanged.connect(self.update_adjustments)
        self.hue_slider.valueChanged.connect(self.update_adjustments)
        self.value_slider.valueChanged.connect(self.update_adjustments)

    def update_adjustments(self):
        saturation_factor = self.saturation_slider.value() / 100.0
        hue_shift = self.hue_slider.value()
        value_factor = self.value_slider.value() / 100.0

        self.saturation_label.setText(f"Saturation: {saturation_factor:.2f}")
        self.hue_label.setText(f"Hue: {hue_shift}")
        self.value_label.setText(f"Value: {value_factor:.2f}")

        self.sliderValuesChanged.emit((saturation_factor, hue_shift, value_factor))
