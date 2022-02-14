#!/usr/bin/env python3
###################################################################################################
#
# Copyright (C) 2018-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################

"""This is the demo application for Face Identification which utilizes
MAX78000 EvKit to get CNN model output.
"""
import argparse
import numpy as np

from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import pyqtSlot, Qt

import cv2
from torchvision import transforms
from nms import approx_softmax
from sigmoid import generate_q_sigmoid, sigmoid_lut, q17p14, q_mul, q_div

# from PIL import Image
from camera import Camera
from cam_thread import Thread
from image_utils import cvt_qimage_to_img, cvt_img_to_qimage
from face_identifier import FaceID
from utils import load_data_arrs, get_face_image
# from mtcnn.mtcnn import MTCNN

# IndexToClassName = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person', 15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}
# IndexToClassName = {0: 'Person', 1: 'Car', 2: 'Bicycle', 3: 'Chair', 4: 'Sofa'}
IndexToClassName = {0: 'Person', 1: 'Car', 2: 'Cat', 3: 'Dog', 4: 'Airplane'}

class normalize:
    """
    Normalize input to either [-128/128, +127/128] or [-128, +127]
    """
    def __init__(self, args):
        self.args = args

    def __call__(self, img):
        if self.args.act_mode_8bit:
            return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127)
        return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127).div(128.)

class FaceIdWindow(QtWidgets.QMainWindow):
    """
    Main window for FaceID demo.
    """

    # pylint: disable=too-many-instance-attributes

    cam_num = 0
    frame_size = (240, 320)
    img_size = (480, 640, 3)
    capture_size = (56, 56)
    frame_rate = 25
    model_params = {'embedding_len': 7*4,
                    'unknown_threshold': (1.0*560.0)}
    uart_params = {'port': 'COM52', 'baud_rate': 8*115200}
    db_paths = {'embeddings': 'embeddings.bin'}

    camera = None
    face_detector = None

    def __init__(self, com_port): #pylint: disable=too-many-statements
        super().__init__()
        self.uart_params['port'] = com_port

        self.setWindowTitle('DATA Lab YOLO')
        # self.setWindowIcon(QIcon('logo.png'))
        # self.setWindowIcon(QIcon('D:\Zhimeng\Maxim\DataLab\datalab.png'))
        self.setWindowIcon(QIcon('..\datalab.png'))

        self.__load_subject_map(self.db_paths['embeddings'])

        self.face_identifier = FaceID(face_db_path=self.db_paths['embeddings'],
                                      unknown_threshold=self.model_params['unknown_threshold'])

        #create GUI screen
        self.central_widget = QtWidgets.QWidget()

        #create dropdown menu to select environment for model execution
        self.ai85_source_label = QtWidgets.QLabel('AI-85 Source: ', self.central_widget)
        self.ai85_source_combo = QtWidgets.QComboBox(self.central_widget)
        self.ai85_source_combo.addItem('<Select AI85 Device>')
        self.ai85_source_combo.addItem('Simulator')
        self.ai85_source_combo.addItem('Emulator')
        self.ai85_source_combo.addItem('EV-Kit')
        self.ai85_source_combo.activated.connect(self.__source_selected)
        self.ai85_source_combo.setCurrentIndex(3)
        self.__source_selected()
        self.ai85_source_combo.setDisabled(True)

        #create button to load image
        self.load_img_button = QtWidgets.QPushButton('Load Image', self.central_widget)
        self.load_img_button.resize(100, 32)
        self.load_img_button.clicked.connect(self.__load_image)

        #create button to start camera
        self.start_camera_button = QtWidgets.QPushButton('Start Cam', self.central_widget)
        self.start_camera_button.resize(100, 32)
        self.start_camera_button.clicked.connect(self.__init_camera)

        #create button to capture image
        self.capture_button = QtWidgets.QPushButton('Capture', self.central_widget)
        self.capture_button.resize(100, 32)
        self.capture_button.clicked.connect(self.__capture_button_pressed)
        self.capture_button.setVisible(False)
        self.capture_busy = False

        # create button to stop camera
        self.stop_camera_button = QtWidgets.QPushButton('Stop Cam', self.central_widget)
        self.stop_camera_button.resize(100, 32)
        self.stop_camera_button.clicked.connect(self.__stop_camera)
        self.stop_camera_button.setVisible(False)

        # #create view to show camera stream
        # self.preview_frame = QtWidgets.QLabel('Preview', self.central_widget)
        # self.preview_frame.resize(self.img_size[0], self.img_size[1])
        # self.preview_black_img = QPixmap(self.img_size[1], self.img_size[0])
        # self.preview_black_img.fill(Qt.black)
        # self.preview_black_img = self.preview_black_img.toImage()
        # self.__set_preview_image(self.preview_black_img)

        #create view to show captures frame
        self.captured_frame = QtWidgets.QLabel('Capture', self.central_widget)
        self.captured_frame.resize(self.capture_size[0], self.capture_size[1])
        black_img = QPixmap(8*self.capture_size[0], 8*self.capture_size[1])
        black_img.fill(Qt.black)
        black_img = black_img.toImage()
        self.__set_captured_image(black_img)

        #create the text box to show results
        # self.subject_label = QtWidgets.QLabel('Subject: ', self.central_widget)
        # self.ai85_time_label = QtWidgets.QLabel('All Time (ms): ', self.central_widget)
        # self.inf_time_label_label = QtWidgets.QLabel('Inference Time (ms): ', self.central_widget)
        # self.db_time_label_label = QtWidgets.QLabel('DB Match Dur (ms): ', self.central_widget)
        # self.ai85_energy_label = QtWidgets.QLabel('Energy (uJ): ', self.central_widget)

        # self.subject_text = QtWidgets.QLabel('Efficiency', self.central_widget)
        # self.ai85_time_text = QtWidgets.QLabel('     ', self.central_widget)
        self.inf_time_label_text = QtWidgets.QLabel('     ', self.central_widget)
        # self.db_time_label_text = QtWidgets.QLabel('     ', self.central_widget)
        # self.ai85_energy_text = QtWidgets.QLabel('     ', self.central_widget)

        self.result_box = QtWidgets.QGridLayout()
        # self.result_box.addWidget(self.subject_label, 0, 0)
        # self.result_box.addWidget(self.subject_text, 0, 1)
        # self.result_box.addWidget(self.ai85_time_label, 1, 0)
        # self.result_box.addWidget(self.ai85_time_text, 1, 1)
        # self.result_box.addWidget(self.inf_time_label_label, 1, 0)
        # self.result_box.addWidget(self.inf_time_label_text, 1, 1)
        # self.result_box.addWidget(self.db_time_label_label, 2, 0)
        # self.result_box.addWidget(self.db_time_label_text, 2, 1)
        # self.result_box.addWidget(self.ai85_energy_label, 3, 0)
        # self.result_box.addWidget(self.ai85_energy_text, 3, 1)

        # font = self.subject_text.font()
        
        font = self.captured_frame.font()
        font.setBold(True)

        self.result_table = QtWidgets.QTableWidget(self.central_widget)
        self.result_table.verticalScrollBar().setDisabled(True)
        self.result_table.verticalScrollBar().setVisible(False)
        self.result_table.horizontalScrollBar().setDisabled(True)
        self.result_table.horizontalScrollBar().setVisible(False)
        self.result_table.setRowCount(4)
        self.result_table.setColumnCount(2)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.horizontalHeader().setVisible(False)
        for i in range(2):
            self.result_table.horizontalHeader().setSectionResizeMode(i,
                                                                      QtWidgets.QHeaderView.Stretch)
        for i in range(4):
            self.result_table.verticalHeader().setSectionResizeMode(i,
                                                                    QtWidgets.QHeaderView.Stretch)
        self.result_table.setItem(0, 0, QtWidgets.QTableWidgetItem("Top-3 Objects"))
        self.result_table.setItem(0, 1, QtWidgets.QTableWidgetItem("Probability (%)"))
        self.top1_subj_text = QtWidgets.QTableWidgetItem("")
        self.top1_subj_dist = QtWidgets.QTableWidgetItem("")
        self.top2_subj_text = QtWidgets.QTableWidgetItem("")
        self.top2_subj_dist = QtWidgets.QTableWidgetItem("")
        self.top3_subj_text = QtWidgets.QTableWidgetItem("")
        self.top3_subj_dist = QtWidgets.QTableWidgetItem("")
        self.result_table.setItem(1, 0, self.top1_subj_text)
        self.result_table.setItem(1, 1, self.top1_subj_dist)
        self.result_table.setItem(2, 0, self.top2_subj_text)
        self.result_table.setItem(2, 1, self.top2_subj_dist)
        self.result_table.setItem(3, 0, self.top3_subj_text)
        self.result_table.setItem(3, 1, self.top3_subj_dist)

        self.result_table.item(0, 0).setFont(font)
        self.result_table.item(0, 1).setFont(font)
        font.setBold(False)
        self.result_table.item(1, 0).setFont(font)
        self.result_table.item(1, 1).setFont(font)
        self.result_table.item(2, 0).setFont(font)
        self.result_table.item(1, 1).setFont(font)
        self.result_table.item(3, 0).setFont(font)
        self.result_table.item(3, 1).setFont(font)

        #create layout that includes the ui components
        self.layout = QtWidgets.QGridLayout(self.central_widget)

        self.layout.addWidget(self.ai85_source_label, 0, 0)
        self.layout.addWidget(self.ai85_source_combo, 0, 1)

        self.layout.addWidget(self.load_img_button, 0, 4)
        self.layout.addWidget(self.start_camera_button, 0, 5)
        self.layout.addWidget(self.capture_button, 0, 5)
        self.layout.addWidget(self.stop_camera_button, 1, 5)

        # self.layout.addWidget(self.preview_frame, 2, 0, 21, 6)
        # self.layout.addWidget(self.captured_frame, 2, 7, 8, 4)

        self.layout.addWidget(self.captured_frame, 2, 0, 21, 6)

        self.layout.addLayout(self.result_box, 10, 6, 4, 3)
        self.layout.addWidget(self.result_table, 14, 6, 6, 4)

        self.setCentralWidget(self.central_widget)

    def __init_camera(self):
        self.start_camera_button.setText('Starting...')

        if self.camera is None:
            self.camera = Camera(cam_num=self.cam_num, frame_size=self.frame_size)

            # run thread to get frames from camera
            self.thread = Thread(parent=None, camera=self.camera, frame_rate=self.frame_rate)
            #self.thread.change_pixmap.connect(lambda p: self.__set_preview_image(p))
            # self.thread.change_pixmap.connect(self.__set_preview_image)
            self.thread.start()

            self.start_camera_button.setVisible(False)
            self.load_img_button.setVisible(False)
            self.capture_button.setVisible(True)
            self.stop_camera_button.setVisible(True)

        self.start_camera_button.setText('Start Cam')

    def __stop_camera(self):
        if self.camera is not None:
            self.thread.terminate()
            self.camera.close_camera()

            # self.__set_preview_image(self.preview_black_img)

            self.start_camera_button.setVisible(True)
            self.capture_button.setVisible(False)
            self.stop_camera_button.setVisible(False)
            self.load_img_button.setVisible(True)

            del self.camera
            self.camera = None

    def __load_image(self):
        if not self.face_identifier.has_ai85_adapter():
            self.__show_adapter_error()
            return

        img_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', directory='',
                                                            filter="Image files (*.jpg *.jpeg *.bmp *.png) ") #pylint: disable=line-too-long
        if img_path == '':
            return

        print(img_path)
        with open(img_path, 'rb') as img_file:
            content = img_file.read()
        # print(f'image_type={type(content)}')

        img = QImage()
        img.loadFromData(content)
        img_np = cvt_qimage_to_img(img)
        img.scaled(self.img_size[1], self.img_size[0], aspectRatioMode=Qt.KeepAspectRatio)

        # self.__set_preview_image(img.scaled(self.img_size[1], self.img_size[0],
        #                                     aspectRatioMode=Qt.KeepAspectRatio))
        # if self.face_detector is None:
        #     self.face_detector = MTCNN(image_size=56, margin=0, min_face_size=56,
        #                                thresholds=[0.6, 0.8, 0.92], factor=0.85,
        #                                post_process=True, device='cpu')

        # img = get_face_image(img_np, self.face_detector)

        # resize image
        dim = (224, 224)  ### need to change
        resized_img_ori = cv2.resize(img_np, dim, interpolation=cv2.INTER_AREA)
        # print(f'resized_img_ori={img_np}')

        ### ------------------------------------------------------change
        img_data = cv2.imread(img_path)
        # print(f'img_data={img_data}')

        transfrom = transforms.Compose([
        #       transforms.ToPILImage(),
        #       transforms.Resize((224, 224)),
            transforms.ToTensor(),  # hui zi dong bian huan tong dao
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        resized_img = cv2.resize(img_data, dim, interpolation=cv2.INTER_AREA)
        # print(f'resized_img={resized_img}')
        resized_img = transfrom(resized_img).permute(1, 2, 0)
        # print(f'resized_img2={resized_img}')
        # print(f'resized_img2={resized_img.shape}')
        # resized_img = (resized_img).astype(np.int)
        resized_img = resized_img.sub(0.5).mul(256.).round().clamp(min=-128, max=127).numpy()
        # print(f'resized_img3={resized_img}')
        resized_img = (resized_img).astype('uint8')
        # print(f'resized_img3={resized_img}')

        ####---------------------------------------------------------change

        if resized_img is not None:
            # if img.shape == (160, 120, 3):
            # print(f'resized_img={resized_img.shape}')

            ## change
            img_rgb = cv2.cvtColor(resized_img + 128, cv2.COLOR_BGR2RGB) ## resized_img
            # img_rgb = resized_img
            ## change

            # print(f'img_rgb={img_rgb}')
            # self.__set_captured_image(cvt_img_to_qimage(img))
            box = self.__identify_face(resized_img)
            img_rgb = cv2.rectangle(img_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
            # self.__set_captured_image(cvt_img_to_qimage(img_rgb))
            self.__set_captured_image(cvt_img_to_qimage(cv2.resize(img_rgb, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)))
            

            # preview_img = cvt_qimage_to_img(img.scaled(self.img_size[1], self.img_size[0], aspectRatioMode=Qt.KeepAspectRatio))
            # self.__set_preview_image(cvt_img_to_qimage(preview_img))

            preview_img = cvt_qimage_to_img(img)

            ### show reshaped image
            # preview_img = cvt_qimage_to_img(img.scaled(dim[1], dim[0], aspectRatioMode=Qt.KeepAspectRatio))

            preview_img_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
            y_ratio = preview_img_rgb.shape[0] // img_rgb.shape[0]
            x_ratio = preview_img_rgb.shape[1] // img_rgb.shape[1]
            # x_ratio, y_ratio = preview_img_rgb.shape[0] / img_rgb.shape[0], preview_img_rgb.shape[1] / img_rgb.shape[1]
            preview_img_rgb = cv2.rectangle(preview_img_rgb,
                                            (box[0] * x_ratio, box[1] * y_ratio),
                                            (box[2] * x_ratio, box[3] * y_ratio),
                                            (0, 255, 0), 1)
            # self.__set_preview_image(cvt_img_to_qimage(preview_img_rgb))

    def __load_subject_map(self, path):
        # print(f'load_subject_path={path}')
        subject_names_list, _, _, _ = load_data_arrs(path, load_img_prevs=False)

        self.subj_name_map = {}
        for i, subj_name in enumerate(subject_names_list):
            self.subj_name_map[i] = subj_name
        self.subj_name_map[-1] = 'Unknown'

        # print(self.subj_name_map)

    def __source_selected(self):
        if self.ai85_source_combo.currentIndex() == 0:
            self.face_identifier.set_ai85_adapter(None)
        elif self.ai85_source_combo.currentIndex() == 1:
            self.face_identifier.set_ai85_adapter('sim', model_path=self.model_params['path'])
        elif self.ai85_source_combo.currentIndex() <= 3:
            self.face_identifier.set_ai85_adapter('uart', uart_port=self.uart_params['port'],
                                                  baud_rate=self.uart_params['baud_rate'],
                                                  embedding_len=self.model_params['embedding_len'])
        else:
            print('Unknown AI85 Source selection')

    def __show_adapter_error(self):
        err_msg = QtWidgets.QErrorMessage(self)
        err_msg.setWindowTitle("Adapter Error")
        err_msg.showMessage('No AI-85 Adapter!!')
        _ = err_msg.exec_()

    def __capture_button_pressed(self):
        if not self.capture_busy:
            self.capture_busy = True

            if not self.face_identifier.has_ai85_adapter():
                self.__show_adapter_error()
                return

            capture = self.preview_frame.pixmap()
            capture = capture.toImage()

            captured_img = cvt_qimage_to_img(capture)

            cropped_img = captured_img[self.camera.start_point[1]:self.camera.end_point[1],
                                       self.camera.start_point[0]:self.camera.end_point[0]]
            cropped_img = cv2.resize(cropped_img, self.capture_size)
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB).copy()

            self.__set_captured_image(cvt_img_to_qimage(cropped_img))
            self.__identify_face(cropped_img)

            self.capture_busy = False

    def __identify_face(self, cropped_img):
        # identify face
        max_box, ai85_time, db_match_time = self.face_identifier.run(cropped_img)

        # prob = self.__cvt_dist_to_prob(dist)
    
        # prob = max_box[4] * max_box[5] / (2**28)
        # prob = max_box[5] / max_box[7] * 100
        start = 5
        end = 10
        # max_box2 = approx_softmax(max_box, start, end)
        # prob = np.max(max_box2[start:end])

        nominator = []
        max_box2 = []
        for i in range(start, end):
            e = max_box[i]
            nominator.append(pow(2, e / 16384.0))
            # nominator.append(pow(2, e >> 14))
            denominator = sum(nominator)
        for i in range(start, end):
            max_box2.append(nominator[i - start] / denominator)
        max_box2 = np.array(max_box2)

        # prob = np.max(np.array(max_box2))
        # index_max = np.argmax(np.array(max_box2))
        prob = np.sort(max_box2)[::-1]
        index_max = np.argsort(max_box2)[::-1]
        

        # self.subject_text.setText(str(max_box[6]))
        # self.ai85_time_text.setText('%.3f' % (1000 * ai85_time))
        # self.inf_time_label_text.setText('%.3f' % (max_box[11]/1000))
        # self.db_time_label_text.setText('%.3f' % (1000 * db_match_time))
        # self.ai85_energy_text.setText('N/A')

        self.top1_subj_text.setText(IndexToClassName[index_max[0]])
        self.top1_subj_dist.setText('%.2f' % prob[0])
        self.top2_subj_text.setText(IndexToClassName[index_max[1]])
        self.top2_subj_dist.setText('%.2f' % prob[1])
        self.top3_subj_text.setText(IndexToClassName[index_max[2]])
        self.top3_subj_dist.setText('%.2f' % prob[2])
        return max_box

    def __cvt_dist_to_prob(self, dist): #pylint: disable=no-self-use
        prob = 1.0 / np.power(dist, 4)
        prob /= np.sum(prob)
        prob *= 100.0
        return prob


    @pyqtSlot(QImage)
    def __set_preview_image(self, image):
        self.preview_frame.setPixmap(QPixmap.fromImage(image))

    def __set_captured_image(self, image):
        self.captured_frame.setPixmap(QPixmap.fromImage(image))

    def __del__(self):
        self.__stop_camera()


def run_face_id_app():
    """
    Main function to initiate demo GUI.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--uart_com_port', '-c', type=str, help='com port for UART communication',
                        default="COM52")
    args = parser.parse_args()
    print(f'start')
    app = QtWidgets.QApplication([])
    app_window = FaceIdWindow(args.uart_com_port)
    print(f'show window')
    app_window.show()
    print(f'show window done!')
    app.exit(app.exec_())

    # try:
    #     app = QtWidgets.QApplication([])
    #     app_window = FaceIdWindow(args.uart_com_port)
    #     print(f'show window')
    #     app_window.show()
    #     app.exit(app.exec_())
    # except Exception as ex: #pylint: disable=broad-except
    #     template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    #     message = template.format(type(ex).__name__, ex.args)
    #     print(message)
    #     print(f'message')


if __name__ == "__main__":
    run_face_id_app()
    # import cv2
    # img = cv2.imread("../000006.jpg")
    # x, y, _ = img.shape
    # x = int(x / 2)
    # y = int(y / 2)
    # h = w = 56
    # crop_img = img[y:y+h, x:x+w]
    # cv2.imshow("cropped", crop_img)
    # cv2.waitKey(0)

    # print('Original Dimensions : ',img.shape)
    # scale_percent = 60 # percent of original size
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    # dim = (56, 56)
    #
    # # resize image
    # resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    #
    # print('Resized Dimensions : ',resized.shape)
    #
    # cv2.imshow("Resized image", resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

