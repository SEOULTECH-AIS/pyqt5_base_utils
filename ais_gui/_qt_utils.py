from PyQt5.QtWidgets import\
    QWidget, QDialog, QMessageBox, QGroupBox, QFileDialog, QLabel\
    QFrame, QSizePolicy,\
    QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QHeaderView

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore.Qt import AlignCenter

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

    def get_selected_row_list(self):
        tmp_list = list(self.selectionModel().selection())
        return_list = []

        if len(tmp_list):
            for _range_data in tmp_list:
                _top = _range_data.top()
                _bottom = _range_data.bottom() + 1
                for _tmp_ct in range(_top, _bottom):
                    return_list.append(_tmp_ct)
        else:
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
    def __init__(self):
        super().__init__("")
        self.image_np_data = None

    def file_to_np(self, img_file):
        self.image_np_data = _cv2.read_img(
            file_dir=img_file,
            color_type=_cv2.COLOR_BGR)

    def set_image(self):
        _h, _w, _c = self.image_np_data.shape
        img = _cv2.cv2.cvtColor(
            self.image_np_data, _cv2.cv2.COLOR_BGR2RGB)
        qImg = QImage(img.data, _w, _h, _w * _c, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qImg))


"""
CUSTOM FUNCTION
====================
"""


def opencv_img_converter(img_file):
    img = get_image(img_file)
    return numpy_to_qimg(img)


def numpy_to_qimg(img):
    _h, _w, _c = img.shape
    img = _cv2.cv2.cvtColor(img, _cv2.cv2.COLOR_BGR2RGB)
    qImg = QImage(img.data, _w, _h, _w * _c, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)


def get_image(img_file):
    img = _cv2.read_img(
        file_dir=img_file,
        color_type=_cv2.COLOR_BGR)
    return img


def table_init(table_widget, row, H_header=None):
    if H_header is not None:
        # when use table init
        table_widget.clear()
        table_widget.setColumnCount(len(H_header))
        table_widget.setHorizontalHeaderLabels(H_header)
        table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    else:
        # when use table clear
        table_widget.clearContents()

    table_widget.setRowCount(row)


def tree_init(tree_widget, row, H_header=None):
    if H_header is not None:
        # when use table init
        tree_widget.clear()
        tree_widget.setHorizontalHeaderLabels(H_header)
        tree_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    else:
        # when use table clear
        tree_widget.clearContents()

    tree_widget.setRowCount(row)


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


"""
CUSTOM DIALOG
====================
"""


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
