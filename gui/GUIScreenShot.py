import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import Qt
from MyWeed import Ui_OpenWeedGUI  # 导入你的UI文件
from PyQt5.QtCore import QPoint
class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 使用你的UI文件
        self.ui = Ui_OpenWeedGUI()
        self.ui.setupUi(self)

        # 添加一个按钮，用于保存界面为图片
        self.save_button = QPushButton("保存为图片", self)
        self.save_button.setGeometry(10, 10, 120, 30)
        self.save_button.clicked.connect(self.save_as_image)

    def save_as_image(self):
        # 获取界面的尺寸
        widget = self.ui.centralwidget
        width, height = widget.width(), widget.height()

        # 指定PPI（每英寸像素数）
        ppi = 700  # 这里以300 PPI为例，可以根据需要调整

        # 计算保存图片的分辨率
        resolution = ppi / 25.4  # 1英寸 = 25.4毫米

        # 创建一个Pixmap，设置分辨率
        pixmap = QPixmap(int(width * resolution), int(height * resolution))
        pixmap.setDevicePixelRatio(resolution)

        # 创建一个Painter，并将界面内容绘制到Pixmap上
        painter = QPainter(pixmap)
        widget.render(painter)
        pixmap.save("GUIShot1.jpg")
        # 结束绘制
        painter.end()


        pixmap.save("GUIShot1.jpg")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MyMainWindow()
    main_window.show()
    sys.exit(app.exec_())
