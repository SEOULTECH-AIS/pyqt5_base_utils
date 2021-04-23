from PyQt5.QtWidgets import\
    QWidget, QDialog, QMessageBox, QGroupBox, QFileDialog, QLabel,\
    QFrame, QSizePolicy,\
    QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QHeaderView,\
    QShortcut

from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QKeySequence

from PyQt5.QtCore import Qt, pyqtSignal

from ais_utils import _base
from ais_utils import _cv2

# from ais_utils import _error

MESSAGE_ICON_FLAG = {
    "no": QMessageBox.NoIcon,
    "question": QMessageBox.Question,
    "info": QMessageBox.Information,
    "warning": QMessageBox.Warning,
    "critical": QMessageBox.Critical}
MESSAGE_BUTTON = {
    "OK": QMessageBox.Ok,
    "NO": QMessageBox.No,
    "RE": QMessageBox.Retry}

ALIGN = {
    "Center": Qt.AlignCenter}

_DRAW_LINE = {
    "Solid": Qt.SolidLine
}

DRAW_LINE_STATE = ["Solid", ]
DRAW_SHAPE = ["Line", "Polygon", "Rectan", "Circle"]

"""
CUSTOM WIDGET
====================
"""


class page(QWidget):
    def __init__(self, title):
        super().__init__()
        self.set_the_GUI(title)

    def initUI(self):
        make_message_box(
            title="GUI ERROR",
            message="You don't make {} GUI init. please make the init function".format(self.title),
            icon_flag="critical",
            bt_flags=["OK", ])
        return None

    def set_the_GUI(self, title):
        _layout = self.initUI()
        if type(_layout) is not None:
            self.setLayout(_layout)
            self.setWindowTitle(title)


class subpage(QDialog):
    def __init__(self, title):
        super().__init__()
        self.set_the_GUI(title)

    def onOKButtonClicked(self):
        self.accept()

    def onCancelButtonClicked(self):
        self.reject()

    def showModal(self):
        return super().exec_()

    def initUI(self):
        make_message_box(
            title="GUI ERROR",
            message="You don't make {} GUI init. please make the init function".format(self.title),
            icon_flag="critical",
            bt_flags=["OK", ])
        return None

    def set_the_GUI(self, title):
        _layout = self.initUI()
        if type(_layout) is not None:
            self.setLayout(_layout)
            self.setWindowTitle(title)


class sub_section(QGroupBox):
    def __init__(self, name, default_check_option=None, is_flat=True):
        super().__init__(name)
        if default_check_option is not None:
            self.setCheckable(True)
            self.setChecked(default_check_option)

        self.set_the_GUI(is_flat)

    def initUI(self):
        make_message_box(
            title="GUI ERROR",
            message="You don't make {} GUI init. please make the init function".format(self.title),
            icon_flag="critical",
            bt_flags=["OK", ])

        return None

    def set_the_GUI(self, is_flat):
        _layout = self.initUI()
        if type(_layout) is not None:
            self.setLayout(_layout)
            self.setFlat(is_flat)


class table_module(QTableWidget):
    def __init__(self, header_text):
        super().__init__()
        self.refresh(row=0, header=header_text)

    def refresh(self, row, header=None):
        if header is not None:
            # when use table init
            self.clear()
            self.setColumnCount(len(header))
            self.setHorizontalHeaderLabels(header)
            self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        else:
            # when use table clear
            self.clearContents()

        self.setRowCount(row)

    def data_insert(self, row_ct, col_ct, text):
        self.setItem(row_ct, col_ct, QTableWidgetItem(text))

    def get_selected_row_list(self, not_selected_is_all=True):
        tmp_list = list(self.selectionModel().selection())
        return_list = []

        if len(tmp_list):
            for _range_data in tmp_list:
                _top = _range_data.top()
                _bottom = _range_data.bottom() + 1
                for _tmp_ct in range(_top, _bottom):
                    return_list.append(_tmp_ct)
        elif not_selected_is_all:
            for _ct in range(self.rowCount()):
                return_list.append(_ct)

        return return_list


class tree_module(QTreeWidget):
    def __init__(self, header_text):
        super().__init__()
        self.refresh(row=0, header=header_text)

    def refresh(self, row, header=None):
        if header is not None:
            # when use table init
            self.clear()
            self.setHeaderLabels(header)
            self.header().setStretchLastSection(False)
            self.header().setSectionResizeMode(QHeaderView.Stretch)
        else:
            # when use table clear
            self.clearContents()

    def data_insert(self, parent_widget, texts):
        _item = QTreeWidgetItem(parent_widget)

        for _ct, text in enumerate(texts):
            _item.setText(_ct, text)

        return _item

    def data_loaction(self, selected_item):
        def salmon(item, location_list):
            _p = item.parent()
            if _p is not None:
                for _ct_ct in range(_p.childCount()):
                    if item == _p.child(_ct_ct):
                        location_list.append(_ct_ct)
                        break
                _top_p = salmon(_p, location_list)
                return _top_p
            else:
                return item

        _location = []
        _selected_top_item = salmon(selected_item, _location)

        for _item_ct in range(self.topLevelItemCount()):
            if _selected_top_item == self.topLevelItem(_item_ct):
                _location.append(_item_ct)
                break
        _location.reverse()
        return _location


class h_line(QFrame):
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(1)
        self.setFixedHeight(20)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        return


class v_line(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(20)
        self.setMinimumHeight(1)
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        return


class image_module(QLabel):
    draw_data_add = pyqtSignal()

    def __init__(self):
        super().__init__("")
        self.padding = 20
        self.setMouseTracking(True)
        self.image_np_data = None

        self._past_x = []
        self._past_y = []

        self._present_x = None
        self._present_y = None

        self._draw_data = []

        self._draw_flag = "Polygon"
        self._pen_info = {
            "color": QColor(0x00, 0x00, 0x00),
            "thick": 3,
            "line_style": Qt.SolidLine
        }

        QShortcut(QKeySequence(Qt.Key_C), self, activated=self._polygon_close)

    def file_to_np(self, img_file):
        self.image_np_data = _cv2.read_img(
            file_dir=img_file,
            color_type=_cv2.COLOR_BGR)

    def set_image(self, pixmap=None):
        self.setPixmap(self._make_pixmap() if pixmap is None else pixmap)

    def set_option(self, flag, **kwarg):
        self._draw_flag = flag
        for _data in kwarg.keys():
            if _data == "color":
                _b = kwarg[_data][0]
                _g = kwarg[_data][1]
                _r = kwarg[_data][2]
                self._pen_info[_data] = QColor(_b, _g, _r)

            elif _data == "line_style":
                self._pen_info[_data] = _DRAW_LINE[kwarg[_data]]

            elif _data in self._pen_info.keys():
                self._pen_info[_data] = kwarg[_data]

    def get_image(self, is_pad=False):
        _tmp_pixmap = self.pixmap()
        _tmp_qimg = _tmp_pixmap.toImage()

        _tmp_h = _tmp_qimg.height()
        _tmp_w = _tmp_qimg.width()
        _tmp_format = _tmp_qimg.format()

        if _tmp_format == 4:  # QImage::Format_RGB32
            _tmp_c = 4
            _string_img = _tmp_qimg.bits().asstring(_tmp_w * _tmp_h * _tmp_c)
            _restore_img = _cv2.np.fromstring(
                _string_img,
                dtype=_cv2.np.uint8).reshape((_tmp_h, _tmp_w, _tmp_c))

        _pad = self.padding
        return _restore_img if is_pad else _restore_img[_pad: -_pad, _pad: -_pad, :]

    def _make_pixmap(self):
        _h, _w, _c = self.image_np_data.shape
        _new_h, _new_w = _h + 2 * self.padding, _w + 2 * self.padding
        img = _cv2.cv2.cvtColor(
            self.image_np_data, _cv2.cv2.COLOR_BGR2RGB)

        _pad_img = _cv2.np.zeros((_new_h, _new_w, _c)).astype(_cv2.np.uint8)
        _pad_img[self.padding: -self.padding, self.padding: -self.padding, :] = img

        return QPixmap.fromImage(
            QImage(_pad_img.data, _new_w, _new_h, _new_h * _c, QImage.Format_RGB888))

    def _polygon_close(self):
        if self._draw_flag == "Polygon" and len(self._past_x) >= 2:
            self._past_x.append(self._past_x[0])
            self._past_y.append(self._past_y[0])

            _, draw_image = self._draw()

            self._draw_data.append({
                "style": self._draw_flag,
                "x": self._past_x,
                "y": self._past_y})
            self.draw_data_add.emit()
            self._past_x = []
            self._past_y = []

            self.set_image(draw_image)
            self.image_np_data = self.get_image()[:, :, :3]

    def _draw(self, flag=None):
        is_end = False
        _img = self._make_pixmap()
        if len(self._past_x):
            _painter = QPainter(_img)
            _painter.setPen(
                QPen(self._pen_info["color"], self._pen_info["thick"], self._pen_info["line_style"])
            )

            _draw_option = flag if flag is not None else self._draw_flag

            # draw line
            if _draw_option == "Line":
                if len(self._past_x) == 1:  # preview
                    _p1_x = self._past_x[0]
                    _p1_y = self._past_y[0]

                    _p2_x = self._present_x
                    _p2_y = self._present_y

                elif len(self._past_x) == 2:  # draw
                    _p1_x = self._past_x[0]
                    _p1_y = self._past_y[0]

                    _p2_x = self._past_x[1]
                    _p2_y = self._past_y[1]

                    is_end = True

                _painter.drawLine(_p1_x, _p1_y, _p2_x, _p2_y)

            # draw polygon
            elif _draw_option == "Polygon":
                if len(self._past_x) == 3:
                    _start = [self._past_x[0], self._past_y[0]]
                    _end = [self._past_x[-1], self._past_y[-1]]
                    is_end = _start == _end

                if is_end:
                    _xs = self._past_x
                    _ys = self._past_y

                else:
                    _xs = self._past_x + [self._present_x, ]
                    _ys = self._past_y + [self._present_y, ]

                _st_x = _xs[0]
                _st_y = _ys[0]

                for [_x, _y] in zip(_xs[1:], _ys[1:]):
                    _painter.drawLine(_st_x, _st_y, _x, _y)
                    _st_x = _x
                    _st_y = _y

            # draw retangle
            elif _draw_option == "Rectan":
                if len(self._past_x) == 1:  # preview
                    _p1_x = self._past_x[0]
                    _p1_y = self._past_y[0]

                    _p2_x = self._present_x
                    _p2_y = self._present_y

                elif len(self._past_x) == 2:  # draw
                    _p1_x = self._past_x[0]
                    _p1_y = self._past_y[0]

                    _p2_x = self._past_x[1]
                    _p2_y = self._past_y[1]

                    is_end = True

                _left = min(_p1_x, _p2_x)
                _right = max(_p1_x, _p2_x)

                _top = min(_p1_y, _p2_y)
                _bottom = max(_p1_y, _p2_y)

                _painter.drawLine(_left, _top, _right, _top)
                _painter.drawLine(_right, _top, _right, _bottom)
                _painter.drawLine(_right, _bottom, _left, _bottom)
                _painter.drawLine(_left, _bottom, _left, _top)

            # draw circle
            elif _draw_option == "Circle":
                if len(self._past_x) == 1:  # preview
                    _p1_x = self._past_x[0]
                    _p1_y = self._past_y[0]

                    _p2_x = self._present_x
                    _p2_y = self._present_y

                elif len(self._past_x) == 2:  # draw
                    _p1_x = self._past_x[0]
                    _p1_y = self._past_y[0]

                    _p2_x = self._past_x[1]
                    _p2_y = self._past_y[1]

                    is_end = True

                _R_x = abs(_p1_x - _p2_x)
                _R_y = abs(_p1_y - _p2_y)

                _painter.drawEllipse(_p1_x - _R_x, _p1_y - _R_y, 2 * _R_x, 2 * _R_y)

            _painter.end()

        return is_end, QPixmap(_img)

    def mousePressEvent(self, QMouseEvent):
        # make draw data
        if QMouseEvent.button() == Qt.LeftButton:
            self._past_x.append(self._present_x)
            self._past_y.append(self._present_y)

        elif QMouseEvent.button() == Qt.RightButton:
            if len(self._past_x):
                self._past_x.pop()
                self._past_y.pop()

        is_end, draw_image = self._draw()
        self.set_image(draw_image)

        if is_end:
            self._draw_data.append({
                "style": self._draw_flag,
                "x": self._past_x,
                "y": self._past_y})
            self.draw_data_add.emit()
            self._past_x = []
            self._past_y = []

            self._present_x = None
            self._present_y = None

            self.image_np_data = self.get_image()[:, :, :3]

    def mouseMoveEvent(self, event):
        # make preview data
        self._present_x = event.x()
        self._present_y = event.y()

        # make preview
        _, draw_image = self._draw()
        self.set_image(draw_image)

    def draw_info_clear(self):
        self._past_x = []
        self._past_y = []

        self._present_x = None
        self._present_y = None

        self._draw_data = []

        self._draw_flag = "Polygon"
        self._pen_info = {
            "color": QColor(0x00, 0x00, 0x00),
            "thick": 3,
            "line_style": Qt.SolidLine
        }


"""
CUSTOM FUNCTION
====================
"""


"""
CUSTOM DIALOG
====================
"""


def file_n_dir_dialog(parent_widget, dialog_title, default_dir, ext_filter, error_massage):
    # extract file list
    if default_dir is not None:
        default_dir = default_dir if default_dir[-1] == _base.SLASH else default_dir + _base.SLASH
    else:
        default_dir = "." + _base.SLASH

    if ext_filter == "dir":
        _get_data = QFileDialog.getExistingDirectory(
            parent=parent_widget,
            caption=dialog_title,
            directory=default_dir)
    else:
        _get_data = QFileDialog.getOpenFileNames(
            parent=parent_widget,
            caption=dialog_title,
            directory=default_dir,
            filter=ext_filter)[0]

    if type(_get_data) == list:
        for _ct in range(len(_get_data)):
            _get_data[_ct] = _get_data[_ct].replace("/", _base.SLASH)
    else:
        _get_data = _get_data.replace("/", _base.SLASH)

    if not len(_get_data):
        _answer = make_message_box(
            title="Get File List Error" if ext_filter != "dir" else "Get Directory Error",
            message=error_massage,
            icon_flag="warning",
            bt_flags=["OK", "RE"]
        )
        if _answer == QMessageBox.Retry:
            default_dir, _get_data = file_n_dir_dialog(
                parent_widget,
                dialog_title,
                default_dir,
                ext_filter,
                error_massage)

    return default_dir, _get_data


def make_message_box(title, message, icon_flag, bt_flags):

    assert icon_flag in MESSAGE_ICON_FLAG.keys()
    assert all([_tmp in MESSAGE_BUTTON.keys() for _tmp in bt_flags])

    msg = QMessageBox()
    msg.setIcon(MESSAGE_ICON_FLAG[icon_flag])
    msg.setWindowTitle(title)
    msg.setText(message)

    bt_flag = 0
    for _tmp_bt_flag in bt_flags:
        bt_flag = bt_flag | MESSAGE_BUTTON[_tmp_bt_flag]

    msg.setStandardButtons(bt_flag)
    return msg.exec_()


# def shorcut_letter_add(label_text, shorcut_latter):
#     _latter = shorcut_latter.split("+")[-1] if "+" in shorcut_latter\
#         else shorcut_latter


def load_success():
    print("!!! custom python module ais_gui _qt__utils load Success !!!")
