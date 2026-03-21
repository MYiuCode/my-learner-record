import sys
import os
import json
import csv
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from ultralytics import YOLO

# ==================== 全局配置 ====================
SUPPORTED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
SUPPORTED_VIDEO_FORMATS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
BOX_LINE_WIDTH = 3
TEXT_SIZE = 0.8
TEXT_THICKNESS = 2

# 颜色：暖杏色背景 + 纯黑字体 + 浅紫按钮 + 黑色边框
WARM_APRICOT = QColor(250, 244, 230)
LIGHT_PURPLE = QColor(220, 210, 230)
BUTTON_HOVER = QColor(200, 190, 210)
BUTTON_PRESSED = QColor(180, 170, 190)
BORDER_COLOR = QColor(0, 0, 0)
TEXT_COLOR = QColor(0, 0, 0)

BOX_COLORS = {'青': QColor(0, 255, 255), '红': QColor(255, 0, 0), '绿': QColor(0, 255, 0)}
BOX_SHAPES = {'方形': 'rect', '圆形': 'circle'}


# ==================== 检测线程 ====================
class DetectorThread(QThread):
    updateOriginalFrame = Signal(np.ndarray)  # 新增：发送原图
    updateDetectedFrame = Signal(np.ndarray)  # 发送检测后图
    updateResult = Signal(list)
    progress = Signal(int)
    finished = Signal()
    error = Signal(str)

    def __init__(self, model, path, conf, iou, shape, color, scale):
        super().__init__()
        self.model = model
        self.path = path
        self.conf = conf
        self.iou = iou
        self.shape = shape
        self.color = color
        self.scale = scale
        self.running = True
        self.paused = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.running = False; self.wait()

    def run(self):
        try:
            if os.path.isfile(self.path):
                if self.path.lower().endswith(SUPPORTED_IMAGE_FORMATS):
                    # 读取原图并发送
                    original_img = cv2.imread(self.path)
                    self.updateOriginalFrame.emit(original_img.copy())
                    # 检测并发送结果图
                    res = self.model.predict(self.path, conf=self.conf, iou=self.iou, imgsz=640, verbose=False)
                    detected_img = self.drawBox(res[0])
                    self.updateDetectedFrame.emit(detected_img)
                    self.updateResult.emit(self.parse(res[0]))
                elif self.path.lower().endswith(SUPPORTED_VIDEO_FORMATS):
                    cap = cv2.VideoCapture(self.path)
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cnt = 0
                    while self.running and cap.isOpened():
                        if self.paused: QThread.msleep(100); continue
                        ret, frame = cap.read()
                        if not ret: break
                        cnt += 1
                        # 发送原图
                        self.updateOriginalFrame.emit(frame.copy())
                        # 检测并发送结果图
                        res = self.model.predict(frame, conf=self.conf, iou=self.iou, imgsz=640, verbose=False)
                        detected_frame = self.drawBox(res[0])
                        self.updateDetectedFrame.emit(detected_frame)
                        self.updateResult.emit(self.parse(res[0]))
                        self.progress.emit(int(cnt / total * 100))
                        QThread.msleep(33)
                    cap.release()
            elif os.path.isdir(self.path):
                files = []
                for e in SUPPORTED_IMAGE_FORMATS: files.extend(Path(self.path).glob(f'*{e}'))
                total = len(files)
                for i, f in enumerate(files):
                    if not self.running: break
                    if self.paused: QThread.msleep(100); continue
                    # 读取原图
                    original_img = cv2.imread(str(f))
                    self.updateOriginalFrame.emit(original_img.copy())
                    # 检测并发送结果图
                    res = self.model.predict(str(f), conf=self.conf, iou=self.iou, imgsz=640, verbose=False)
                    detected_img = self.drawBox(res[0])
                    self.updateDetectedFrame.emit(detected_img)
                    self.updateResult.emit(self.parse(res[0]))
                    self.progress.emit(int((i + 1) / total * 100))
                    QThread.msleep(500)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def parse(self, r):
        out = []
        if r.boxes:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cid = int(b.cls[0])
                out.append(
                    {"box": [x1, y1, x2, y2], "class_id": cid, "class_name": r.names[cid], "conf": float(b.conf[0])})
        return out

    def drawBox(self, r):
        im = r.orig_img.copy()
        color = (self.color.blue(), self.color.green(), self.color.red())
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            label = f"{r.names[int(b.cls[0])]} {conf:.2f}"
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            nw, nh = w * self.scale, h * self.scale
            x1, y1, x2, y2 = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
            if self.shape == "rect":
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, BOX_LINE_WIDTH)
            else:
                cv2.circle(im, (int(cx), int(cy)), int(max(nw, nh) / 2), color, BOX_LINE_WIDTH)
            cv2.putText(im, label, (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, color, TEXT_THICKNESS)
        return im


# ==================== 图片显示标签（核心修改：删除放大镜图标） ====================
class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.pix = None
        self.zoom = 1.0
        self.setStyleSheet(f"background:{WARM_APRICOT.name()};")

    def setImage(self, im):
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
        self.pix = QPixmap.fromImage(qimg)
        self.zoom = 1.0
        self.refresh()

    def refresh(self):
        if self.pix:
            scaled_pix = self.pix.scaled(self.size() * self.zoom, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pix)

    def wheelEvent(self, e):
        self.zoom *= 1.15 if e.angleDelta().y() > 0 else 0.85
        self.zoom = max(0.3, min(8.0, self.zoom))
        self.refresh()
        e.accept()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.pix:
            self._drag = True
            self._p = e.pos()
            self.setCursor(Qt.ClosedHandCursor)
        e.accept()

    def mouseMoveEvent(self, e):
        if hasattr(self, '_drag') and self._drag:
            d = e.pos() - self._p
            self._p = e.pos()
            self.move(self.x() + d.x(), self.y() + d.y())
        self.update()
        e.accept()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = False
            self.setCursor(Qt.ArrowCursor)
        e.accept()

    def paintEvent(self, e):
        super().paintEvent(e)
        if not self.pix: return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        # 保留右下角缩略图
        ratio = 0.22
        tw, th = int(self.pix.width() * ratio), int(self.pix.height() * ratio)
        tr = QRect(self.width() - tw - 10, self.height() - th - 10, tw, th)
        p.drawPixmap(tr, self.pix)
        # 保留红框标记放大区域
        vw, vh = self.width() / self.pix.width() / self.zoom, self.height() / self.pix.height() / self.zoom
        vr = QRectF(tr.x(), tr.y(), tw * vw, th * vh)
        p.setPen(QPen(Qt.red, 2))
        p.drawRect(vr)
        # ========== 删除放大镜图标：注释/删除以下所有放大镜绘制代码 ==========
        # 移除卡通风放大镜图标绘制逻辑
        # ir = QRect(self.width()-58, self.height()-58, 44,44)
        # p.setPen(QPen(Qt.black, 2))
        # p.setBrush(QColor(255, 230, 100))
        # p.drawEllipse(ir)
        # p.drawRect(ir.x()+26, ir.y()+26, 10, 18)


# ==================== 圆角按钮 ====================
class RoundedButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setMinimumHeight(40)
        self.setCursor(Qt.PointingHandCursor)
        # 数据源按钮字体9号，其他12号
        if text in ["图片", "视频", "目录"]:
            f = QFont("楷体", 9, QFont.Bold)
        else:
            f = QFont("楷体", 12, QFont.Bold)
        self.setFont(f)
        self.setStyleSheet(f"""
            QPushButton {{
                background:{LIGHT_PURPLE.name()}; color:{TEXT_COLOR.name()};
                border:2px solid {BORDER_COLOR.name()}; border-radius:15px;
                padding:8px 15px; margin:8px;
            }}
            QPushButton:hover {{ background:{BUTTON_HOVER.name()}; }}
            QPushButton:pressed {{ background:{BUTTON_PRESSED.name()}; }}
            QPushButton:disabled {{ background:#e8e4e8; color:#666; }}
        """)


# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO智能目标检测可视化系统 V1.0")
        self.setMinimumSize(1800, 1000)
        self.model = None
        self.model_path = ""
        self.current_source = ""
        self.thread = None
        self.current_original_frame = None
        self.current_detected_frame = None
        self.current_results = []
        self.box_shape = "rect"
        self.box_color = BOX_COLORS['青']
        self.box_scale = 1.0

        self.loadBtn = None
        self.modelLab = None
        self.imgBtn = None
        self.videoBtn = None
        self.dirBtn = None
        self.pathLab = None
        self.sq = None
        self.cr = None
        self.sizeSlider = None
        self.sizeLab = None
        self.cyan = None
        self.red = None
        self.green = None
        self.confSpin = None
        self.iouSpin = None
        self.startBtn = None
        self.pauseBtn = None
        self.stopBtn = None
        self.prog = None
        self.saveFrameBtn = None
        self.exportBtn = None
        self.original_label = None
        self.detected_label = None
        self.detectDataText = None

        self.setupUI()
        self.connectAll()
        self.statusBar().showMessage("就绪")

    def setupUI(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background:{WARM_APRICOT.name()}; }}
            QWidget {{ background:{WARM_APRICOT.name()}; font-family: 楷体; font-size:11pt; color:{TEXT_COLOR.name()}; }}
            QGroupBox {{ font:13pt 楷体; color:{TEXT_COLOR.name()}; border:2px solid {BORDER_COLOR.name()}; border-radius:10px; margin-top:10px; padding-top:10px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left:15px; padding:0 10px; }}
            QLabel {{ font:11pt 楷体; color:{TEXT_COLOR.name()}; }}
            QRadioButton {{ font:11pt 楷体; color:{TEXT_COLOR.name()}; }}
            QTextEdit {{ border:2px solid {BORDER_COLOR.name()}; border-radius:8px; background:#fff; padding:8px; color:{TEXT_COLOR.name()}; }}
            QSlider::handle:horizontal {{ background:{LIGHT_PURPLE.name()}; border:1px solid {BORDER_COLOR.name()}; width:18px; margin:-5px 0; border-radius:9px; }}
            QDoubleSpinBox {{ border:2px solid {BORDER_COLOR.name()}; border-radius:8px; padding:5px; background:#fff; color:{TEXT_COLOR.name()}; }}
            QProgressBar {{ border:2px solid {BORDER_COLOR.name()}; border-radius:8px; background:#fff; }}
            QProgressBar::chunk {{ background:{LIGHT_PURPLE.name()}; border-radius:6px; }}
            QTabBar::tab {{ background:{LIGHT_PURPLE.name()}; padding:8px 15px; border-top-left-radius:8px; border-top-right-radius:8px; }}
            QTabBar::tab:selected {{ background:#fff; }}
            /* 标题样式 - 粗花环线框 + 放大字体 */
            .title-label {{
                font-family: 楷体;
                font-size: 20pt;  /* 字体放大 */
                font-weight: bold; /* 加粗 */
                color: #333333;
                padding: 10px;
                border: 4px dashed #666666; /* 粗花环线框 */
                border-radius: 15px; /* 圆角 */
                background-color: #f0f0f0;
                margin-bottom: 10px;
            }}
        """)

        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 左侧控制面板（完全不变）
        left_panel = QWidget()
        left_panel.setFixedWidth(260)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)

        # 模型管理
        g1 = QGroupBox("模型管理")
        v1 = QVBoxLayout(g1)
        self.loadBtn = RoundedButton("加载模型权重 (.pt)")
        self.modelLab = QLabel("未加载")
        v1.addWidget(self.loadBtn)
        v1.addWidget(self.modelLab)

        # 数据源输入
        g2 = QGroupBox("数据源输入")
        v2 = QHBoxLayout(g2)
        self.imgBtn = RoundedButton("图片")
        self.videoBtn = RoundedButton("视频")
        self.dirBtn = RoundedButton("目录")
        self.pathLab = QLabel("")
        v2.addWidget(self.imgBtn)
        v2.addWidget(self.videoBtn)
        v2.addWidget(self.dirBtn)

        # 检测框形状
        g3 = QGroupBox("检测框形状")
        v3 = QHBoxLayout(g3)
        self.sq = QRadioButton("方形")
        self.cr = QRadioButton("圆形")
        self.sq.setChecked(True)
        v3.addWidget(self.sq)
        v3.addWidget(self.cr)

        # 检测框缩放
        g4 = QGroupBox("检测框缩放")
        v4 = QHBoxLayout(g4)
        self.sizeSlider = QSlider(Qt.Horizontal)
        self.sizeSlider.setRange(50, 200)
        self.sizeSlider.setValue(100)
        self.sizeLab = QLabel("1.0x")
        v4.addWidget(self.sizeSlider)
        v4.addWidget(self.sizeLab)

        # 颜色
        g5 = QGroupBox("颜色")
        v5 = QHBoxLayout(g5)
        self.cyan = RoundedButton("青")
        self.red = RoundedButton("红")
        self.green = RoundedButton("绿")
        v5.addWidget(self.cyan)
        v5.addWidget(self.red)
        v5.addWidget(self.green)

        # 检测参数
        g6 = QGroupBox("检测参数")
        v6 = QVBoxLayout(g6)
        confLayout = QHBoxLayout()
        confLayout.addWidget(QLabel("置信度阈值:"))
        self.confSpin = QDoubleSpinBox()
        self.confSpin.setRange(0.0, 1.0)
        self.confSpin.setValue(0.25)
        self.confSpin.setSingleStep(0.01)
        confLayout.addWidget(self.confSpin)
        iouLayout = QHBoxLayout()
        iouLayout.addWidget(QLabel("IoU 阈值:"))
        self.iouSpin = QDoubleSpinBox()
        self.iouSpin.setRange(0.0, 1.0)
        self.iouSpin.setValue(0.82)
        self.iouSpin.setSingleStep(0.01)
        iouLayout.addWidget(self.iouSpin)
        v6.addLayout(confLayout)
        v6.addLayout(iouLayout)

        # 执行控制
        g7 = QGroupBox("执行控制")
        v7 = QVBoxLayout(g7)
        self.startBtn = RoundedButton("开始检测")
        self.pauseBtn = RoundedButton("暂停")
        self.stopBtn = RoundedButton("停止")
        self.prog = QProgressBar()
        v7.addWidget(self.startBtn)
        v7.addWidget(self.pauseBtn)
        v7.addWidget(self.stopBtn)
        v7.addWidget(self.prog)

        # 保存当前帧
        g8 = QGroupBox("")
        v8 = QVBoxLayout(g8)
        self.saveFrameBtn = RoundedButton("保存当前帧")
        v8.addWidget(self.saveFrameBtn)

        left_layout.addWidget(g1)
        left_layout.addWidget(g2)
        left_layout.addWidget(g3)
        left_layout.addWidget(g4)
        left_layout.addWidget(g5)
        left_layout.addWidget(g6)
        left_layout.addWidget(g7)
        left_layout.addWidget(g8)
        left_layout.addStretch()

        # 中间显示区域（完全不变）
        middle_panel = QWidget()
        middle_layout = QHBoxLayout(middle_panel)
        middle_layout.setSpacing(10)

        # 左侧：检测前图片（带美化标题）
        original_container = QWidget()
        original_layout = QVBoxLayout(original_container)
        original_title = QLabel("检测前")
        original_title.setObjectName("title-label")  # 绑定样式
        original_title.setAlignment(Qt.AlignCenter)
        original_title.setMaximumWidth(200)  # 限制宽度，让线框更美观
        original_title.setMinimumHeight(80)  # 增加高度
        original_title.setAlignment(Qt.AlignCenter)
        self.original_label = ImageLabel()
        original_layout.addWidget(original_title, alignment=Qt.AlignCenter)  # 标题居中
        original_layout.addWidget(self.original_label)

        # 右侧：检测后图片（带美化标题）
        detected_container = QWidget()
        detected_layout = QVBoxLayout(detected_container)
        detected_title = QLabel("检测后")
        detected_title.setObjectName("title-label")  # 绑定样式
        detected_title.setAlignment(Qt.AlignCenter)
        detected_title.setMaximumWidth(200)  # 限制宽度
        detected_title.setMinimumHeight(80)  # 增加高度
        detected_title.setAlignment(Qt.AlignCenter)
        self.detected_label = ImageLabel()
        detected_layout.addWidget(detected_title, alignment=Qt.AlignCenter)  # 标题居中
        detected_layout.addWidget(self.detected_label)

        middle_layout.addWidget(original_container, stretch=1)
        middle_layout.addWidget(detected_container, stretch=1)

        # 右侧数据面板（完全不变）
        right_panel = QWidget()
        right_panel.setFixedWidth(420)
        right_layout = QVBoxLayout(right_panel)
        tabWidget = QTabWidget()
        # 检测数据标签页
        detectDataTab = QWidget()
        detectDataLayout = QVBoxLayout(detectDataTab)
        self.detectDataText = QTextEdit()
        self.detectDataText.setReadOnly(True)
        detectDataLayout.addWidget(self.detectDataText)
        tabWidget.addTab(detectDataTab, "检测数据")
        # 统计信息标签页
        statInfoTab = QWidget()
        statInfoLayout = QVBoxLayout(statInfoTab)
        self.statInfoText = QTextEdit()
        self.statInfoText.setReadOnly(True)
        statInfoLayout.addWidget(self.statInfoText)
        tabWidget.addTab(statInfoTab, "统计信息")
        # 导出按钮
        self.exportBtn = RoundedButton("导出")
        right_layout.addWidget(tabWidget)
        right_layout.addWidget(self.exportBtn)

        # 组装主布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(middle_panel, stretch=3)
        main_layout.addWidget(right_panel)

        # 初始禁用按钮
        self.startBtn.setEnabled(False)
        self.pauseBtn.setEnabled(False)
        self.stopBtn.setEnabled(False)
        self.saveFrameBtn.setEnabled(False)
        self.exportBtn.setEnabled(False)

    def connectAll(self):
        self.loadBtn.clicked.connect(self.loadModel)
        self.imgBtn.clicked.connect(lambda: self.selectSource("image"))
        self.videoBtn.clicked.connect(lambda: self.selectSource("video"))
        self.dirBtn.clicked.connect(lambda: self.selectSource("dir"))
        self.sq.clicked.connect(lambda: setattr(self, "box_shape", "rect"))
        self.cr.clicked.connect(lambda: setattr(self, "box_shape", "circle"))
        self.sizeSlider.valueChanged.connect(self.onSizeChange)
        self.cyan.clicked.connect(lambda: setattr(self, "box_color", BOX_COLORS['青']))
        self.red.clicked.connect(lambda: setattr(self, "box_color", BOX_COLORS['红']))
        self.green.clicked.connect(lambda: setattr(self, "box_color", BOX_COLORS['绿']))
        self.startBtn.clicked.connect(self.startDetect)
        self.pauseBtn.clicked.connect(self.pauseDetect)
        self.stopBtn.clicked.connect(self.stopDetect)
        self.saveFrameBtn.clicked.connect(self.saveFrame)
        self.exportBtn.clicked.connect(self.exportData)

    def onSizeChange(self, v):
        self.sizeLab.setText(f"{v / 100:.1f}x")
        self.box_scale = v / 100

    def loadModel(self):
        p, _ = QFileDialog.getOpenFileName(filter="PyTorch 模型 (*.pt)")
        if p:
            self.model = YOLO(p)
            self.model_path = p
            self.modelLab.setText(f"✅ 已加载: {Path(p).name}")
            if self.current_source: self.startBtn.setEnabled(True)

    def selectSource(self, t):
        if t == "image":
            p, _ = QFileDialog.getOpenFileName(filter=f"图片 (*{' *'.join(SUPPORTED_IMAGE_FORMATS)})")
            if p:
                # 选择图片后先显示原图
                original_img = cv2.imread(p)
                self.original_label.setImage(original_img)
                self.current_original_frame = original_img
                # 清空检测后图
                self.detected_label.clear()
                self.current_detected_frame = None
        elif t == "video":
            p, _ = QFileDialog.getOpenFileName(filter=f"视频 (*{' *'.join(SUPPORTED_VIDEO_FORMATS)})")
        else:
            p = QFileDialog.getExistingDirectory()

        if p:
            self.current_source = p
            self.pathLab.setText("")
            if self.model: self.startBtn.setEnabled(True)

    def startDetect(self):
        if not self.model or not self.current_source: return
        self.thread = DetectorThread(
            self.model, self.current_source,
            self.confSpin.value(), self.iouSpin.value(),
            self.box_shape, self.box_color, self.box_scale
        )
        # 连接新的信号
        self.thread.updateOriginalFrame.connect(self.showOriginalFrame)
        self.thread.updateDetectedFrame.connect(self.showDetectedFrame)
        self.thread.updateResult.connect(self.onResult)
        self.thread.progress.connect(self.prog.setValue)
        self.thread.finished.connect(self.onDone)
        self.thread.error.connect(lambda e: QMessageBox.critical(self, "错误", e))
        self.startBtn.setEnabled(False)
        self.pauseBtn.setEnabled(True)
        self.stopBtn.setEnabled(True)
        self.saveFrameBtn.setEnabled(False)
        self.exportBtn.setEnabled(False)
        self.thread.start()

    def pauseDetect(self):
        if self.thread:
            if self.thread.paused:
                self.thread.resume()
                self.pauseBtn.setText("暂停")
            else:
                self.thread.pause()
                self.pauseBtn.setText("继续")

    def stopDetect(self):
        if self.thread:
            self.thread.stop()
            self.onDone()

    # 新增：显示原图
    def showOriginalFrame(self, im):
        self.current_original_frame = im
        self.original_label.setImage(im)

    # 新增：显示检测后图
    def showDetectedFrame(self, im):
        self.current_detected_frame = im
        self.detected_label.setImage(im)

    def onResult(self, lst):
        self.current_results = lst
        self.updateDetectData()

    def onDone(self):
        self.startBtn.setEnabled(True)
        self.pauseBtn.setEnabled(False)
        self.stopBtn.setEnabled(False)
        self.prog.setValue(100)
        self.saveFrameBtn.setEnabled(True)
        self.exportBtn.setEnabled(True)
        self.statusBar().showMessage("检测完成")

    def updateDetectData(self):
        if not self.current_results:
            self.detectDataText.clear()
            self.statInfoText.clear()
            return
        detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        class_names = self.model.names if self.model else {}
        class_count = {}
        conf_list = []
        for r in self.current_results:
            cls_name = r['class_name']
            class_count[cls_name] = class_count.get(cls_name, 0) + 1
            conf_list.append(r['conf'])

        detect_data = f"""模型路径: {self.model_path}
类别数量: {len(class_names)}
类别列表: {class_names}
----------------------------------------
检测时间: {detect_time}
数据源: {os.path.basename(self.current_source)}
----------------------------------------
检测目标数量: {len(self.current_results)}
类别分布:
"""
        for name, cnt in class_count.items():
            detect_data += f"  - {name}: {cnt}\n"
        if conf_list:
            detect_data += f"""
置信度范围:
  最小: {min(conf_list):.4f}
  最大: {max(conf_list):.4f}
  平均: {sum(conf_list) / len(conf_list):.4f}
"""
        self.detectDataText.setText(detect_data)
        self.statInfoText.setText(detect_data)

    def saveFrame(self):
        # 保存时可选保存原图或检测后图
        if self.current_original_frame is None or self.current_detected_frame is None:
            QMessageBox.warning(self, "警告", "没有可保存的帧")
            return

        save_dialog = QFileDialog()
        save_dialog.setAcceptMode(QFileDialog.AcceptSave)
        save_dialog.setNameFilter("PNG 图片 (*.png)")
        if save_dialog.exec():
            save_path = save_dialog.selectedFiles()[0]
            # 默认保存检测后图
            cv2.imwrite(save_path, self.current_detected_frame)
            QMessageBox.information(self, "成功", f"已保存到: {save_path}")

    def exportData(self):
        if not self.current_results:
            QMessageBox.warning(self, "警告", "没有可导出的数据")
            return
        p, filter = QFileDialog.getSaveFileName(filter="JSON (*.json);;CSV (*.csv)")
        if not p: return
        try:
            if filter.startswith("JSON"):
                with open(p, 'w', encoding='utf-8') as f:
                    json.dump(self.current_results, f, indent=2, ensure_ascii=False)
            else:
                with open(p, 'w', encoding='utf-8', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['x1', 'y1', 'x2', 'y2', 'class_id', 'class_name', 'confidence'])
                    for r in self.current_results:
                        w.writerow([*r['box'], r['class_id'], r['class_name'], r['conf']])
            QMessageBox.information(self, "成功", f"已导出到: {p}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())