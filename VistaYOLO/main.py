import sys
from PySide6.QtWidgets import *
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from utils import detect_image

class VistaYOLO(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VistaYOLO 医学目标识别")
        self.setGeometry(100, 100, 900, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # 左侧按钮区域
        btn_layout = QVBoxLayout()
        self.btn_open = QPushButton("打开图片")
        self.btn_run = QPushButton("开始检测")
        btn_layout.addWidget(self.btn_open)
        btn_layout.addWidget(self.btn_run)
        btn_layout.addStretch()

        # 原图显示区域
        self.lbl_original = QLabel("原图")
        self.lbl_original.setMinimumSize(400, 500)
        self.lbl_original.setStyleSheet("border:1px solid #ccc")

        # 结果显示区域
        self.lbl_result = QLabel("结果")
        self.lbl_result.setMinimumSize(400, 500)
        self.lbl_result.setStyleSheet("border:1px solid #ccc")

        layout.addLayout(btn_layout)
        layout.addWidget(self.lbl_original)
        layout.addWidget(self.lbl_result)

        self.btn_open.clicked.connect(self.open_img)
        self.btn_run.clicked.connect(self.run_detect)
        self.img_path = None

    def open_img(self):
        path, _ = QFileDialog.getOpenFileName()
        if path:
            self.img_path = path
            pix = QPixmap(path).scaled(400, 500, Qt.KeepAspectRatio)
            self.lbl_original.setPixmap(pix)

    def run_detect(self):
        if not self.img_path:
            return
        res_path = detect_image(self.img_path)
        pix = QPixmap(res_path).scaled(400, 500, Qt.KeepAspectRatio)
        self.lbl_result.setPixmap(pix)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VistaYOLO()
    win.show()
    sys.exit(app.exec())