from collections import namedtuple
from functools import partial
import os
import sys

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, \
    QLabel, QComboBox, QFileDialog, QMessageBox, QDialog, QTableWidget, QTableWidgetItem, \
    QApplication, QProgressBar, QListWidget, QListWidgetItem, QCheckBox, \
    QAbstractItemView, QLineEdit
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot, QSize
from PyQt5.QtGui import QFont, QImage, QPixmap

from PIL.ImageQt import ImageQt

class ClickableQLabel(QLabel):
    clicked = pyqtSignal()
    def __init__(self, *args):
        QLabel.__init__(self, *args)
    def mousePressEvent(self, ev):
        self.clicked.emit()

def start_app(controller):
    app = QApplication(sys.argv)
    main_view = MainWidget(controller)
    sys.exit(app.exec_())

class MainWidget(QWidget):
    def __init__(self, controller):
        super().__init__()

        self.controller = controller

        self._init_UI()
        self.show()
    
    def _init_device_combobox(self, box):
        if self.controller is not None:
            self.torch_devices, device_names = self.controller.get_available_device_list()
        else:
            self.torch_devices = ["cpu", "cuda:0"]
            device_names = ["CPU", "GPU0"]
        box.addItems(device_names)


    def _select_plugin_dialog(self):
        path, _ = QFileDialog.getOpenFileName(None, "Choose plugin file", 
            os.path.join(os.path.dirname(__file__), "..", "plugins"),
            "Python script (*.py)")
        if len(path) == 0: # No file selected.
            return
        if self.controller is not None:
            self.controller.select_plugin(self, path)
        else:
            print(path)
    
    def update_plugin(self, fn):
        self.plugin_label.setText(fn)
    
    def show_error_dialog(self, title, message):
        QMessageBox.about(self, title, message)

    def _run_identification(self):
        self.controller.start_identification(self)

    def _select_device(self, idx):
        torch_device = self.torch_devices[idx]
        if self.controller is not None:
            self.controller.select_device(self, torch_device)
        else:
            print(torch_device)

    def _init_UI(self):
        self.setWindowTitle("Main window")

        # Configuration window
        layout = QVBoxLayout(self)

        ## Plugin
        plugin_group = QGroupBox("Plugin", self)
        plugin_vbox = QVBoxLayout(plugin_group)

        self.plugin_btn = QPushButton("Select plugin", self)
        self.plugin_btn.clicked.connect(self._select_plugin_dialog)
        self.plugin_label = QLabel("Please select plugin.", self)

        plugin_vbox.addWidget(self.plugin_btn)
        plugin_vbox.addWidget(self.plugin_label)

        plugin_group.setLayout(plugin_vbox)
        layout.addWidget(plugin_group)

        ## Device
        device_group = QGroupBox("Device", self)
        device_vbox = QVBoxLayout(self)

        self.device_combo = QComboBox(self)
        self._init_device_combobox(self.device_combo)
        self.device_combo.currentIndexChanged.connect(self._select_device)

        device_vbox.addWidget(self.device_combo)
        
        device_group.setLayout(device_vbox)
        layout.addWidget(device_group)

        ## Command buttons
        self.ident_btn = QPushButton("Identification", self)
        self.ident_btn.clicked.connect(self._run_identification)

        layout.addWidget(self.ident_btn)

        ##
        self.setLayout(layout)

class IdentificationView(object):
    @classmethod
    def init_view(cls, parent_view, controller):
        view = cls(parent_view, controller)
        view.show()
        return view

    def __init__(self, parent, controller):
        self.controller = controller

        # GUI
        self.w = QDialog(parent)

        self.w.setWindowTitle("Setup window for identification mode")

        # Top HBoxLayout
        hbox_layout = QHBoxLayout(self.w)
        btn_vbox_layout = QVBoxLayout(self.w)
        table_group = QGroupBox("Directories of character references", self.w)

        hbox_layout.addLayout(btn_vbox_layout)
        hbox_layout.addWidget(table_group)

        # Buttons
        btn_load = QPushButton("Load JSON", self.w)
        btn_load.clicked.connect(self._load_button)
        btn_save = QPushButton("Save JSON", self.w)
        btn_save.clicked.connect(self._save_button)
        btn_load_dir = QPushButton("Select image directory", self.w)
        btn_load_dir.clicked.connect(self._load_image_directory)
        self.label_load_dir = QLabel("No image directory selected.", self.w)
        btn_run = QPushButton("Run identification", self.w)
        btn_run.clicked.connect(self._run_button)

        # Limit number of enroll images
        limit_group = QGroupBox("Limit number of reference images:", self.w)
        limit_group_layout = QVBoxLayout(limit_group)
        limit_check = QCheckBox("Enable", self.w)
        limit_check.stateChanged.connect(self._limit_state)
        limit_edit = QLineEdit(self.w)
        limit_edit.setText("100")
        limit_edit.resize(150, 50)
        limit_edit.textEdited.connect(self._limit_val)
        limit_group_layout.addWidget(limit_check)
        limit_group_layout.addWidget(limit_edit)
        limit_group.setLayout(limit_group_layout)

        btn_vbox_layout.addWidget(btn_load)
        btn_vbox_layout.addWidget(btn_save)
        btn_vbox_layout.addWidget(btn_load_dir)
        btn_vbox_layout.addWidget(self.label_load_dir)
        btn_vbox_layout.addWidget(limit_group)
        btn_vbox_layout.addWidget(btn_run)

        # Directory group
        table_group_layout = QVBoxLayout(table_group)

        table_btn_layout = QHBoxLayout(table_group)

        btn_table_add = QPushButton("Add new character", table_group)
        btn_table_add.clicked.connect(self._add_character_row)
        btn_table_delete = QPushButton("Delete selected characters", table_group)
        btn_table_delete.clicked.connect(self._delete_button_clicked)

        table_btn_layout.addWidget(btn_table_add)
        table_btn_layout.addWidget(btn_table_delete)

        table_group_layout.addLayout(table_btn_layout)

        self.table = QTableWidget(table_group)
        self.char_dir_lst = []
        self._init_table(self.char_dir_lst)

        self.table.itemClicked.connect(self._item_clicked)
        self.table.itemChanged.connect(self._item_changed)

        table_group_layout.addWidget(self.table)
    
    def get_view(self):
        return self.w
    
    def show(self):
        self.w.show()
    
    def close(self):
        self.w.close()
    
    def sync_table(self, char_dir_lst):
        self.char_dir_lst = char_dir_lst
        self._init_table(self.char_dir_lst)

    def _limit_state(self, val):
        flag = (val > 0)
        self.controller.set_limit_state(flag)
    
    def _limit_val(self, val):
        try:
            val = int(val)
            if val > 0:
                self.controller.set_limit_number(val)
        except:
            pass

    # Button
    def _run_button(self):
        self.controller.run()
    
    # Button
    def _load_image_directory(self):
        path = QFileDialog.getExistingDirectory(None,
            "Choose a directory which contains source images", 
            os.path.dirname(__file__))
        if len(path) == 0: # No file selected.
            return
        else:
            self.controller.register_src_dir(path)
            self.label_load_dir.setText(path)



    # Button
    def _load_button(self):
        path, _ = QFileDialog.getOpenFileName(None, "Choose config file", 
            os.path.join(os.path.dirname(__file__), ".."),
            "JSON file (*.json)")
        if len(path) == 0: # No file selected.
            return
        
        self.controller.load_json(path)

    # Button
    def _save_button(self):
        path, _ = QFileDialog.getSaveFileName(None, "Save config file",
            os.path.join(os.path.dirname(__file__), ".."),
            "JSON file (*.json)")
        if len(path) == 0:
            return
        
        self.controller.save_json(path)
    
    # Button
    def _add_character_row(self):
        self.char_dir_lst.append(["",""])
        self.table.insertRow(len(self.char_dir_lst))
        self._add_table_row(len(self.char_dir_lst), "", "")

    # Button
    def _delete_button_clicked(self):
        rows = list(set([idx.row() for idx in self.table.selectedIndexes()]))
        self._delete_character_rows(rows)
    
    # Table
    def _item_clicked(self, item):
        idx = self.table.indexFromItem(item)
        if (idx.row() > 0) and (idx.column() == 1):
            path = QFileDialog.getExistingDirectory(None,
                "Choose a directory which contains character reference images", 
                os.path.dirname(__file__))
            if len(path) == 0: # No file selected.
                return
            else:
                item.setText(path)
                self.char_dir_lst[idx.row()-1][1] = path

                self._updated_table()
    
    # Table
    def _item_changed(self, item):
        idx = self.table.indexFromItem(item)
        if (idx.row() > 0) and (idx.column() == 0):
            self.char_dir_lst[idx.row()-1][0] = item.text()
            self._updated_table()
    
    # Table manipulation
    def _updated_table(self):
        self.table.resizeRowsToContents()
        self.table.resizeColumnsToContents()

        self.controller.update_char_dir_lst(self.char_dir_lst)

    def _delete_character_rows(self, index_rows):
        index_rows = sorted(index_rows, reverse=True)
        for idx in index_rows:
            self.char_dir_lst.pop(idx-1)
            self.table.removeRow(idx)

        self._updated_table()
    
    def _add_table_row(self, i_row, char, dirname):
        item = QTableWidgetItem(char)
        self.table.setItem(i_row, 0, item)
        item = QTableWidgetItem(dirname)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable) # not editable
        self.table.setItem(i_row, 1, item)

        self._updated_table()

    def _init_table(self, char_dir_lst):
        self.table.setColumnCount(2)
        self.table.setRowCount(len(self.char_dir_lst)+1)

        # headers
        bold_font = QFont()
        bold_font.setBold(True)

        item = QTableWidgetItem("Character")
        item.setFlags(Qt.ItemIsEnabled)
        item.setFont(bold_font)
        self.table.setItem(0, 0, item)

        item = QTableWidgetItem("Directory of reference images")
        item.setFlags(Qt.ItemIsEnabled)
        item.setFont(bold_font)
        self.table.setItem(0, 1, item)

        # Contents
        for i_char, (char, dirname) in enumerate(char_dir_lst):
            self._add_table_row(i_char+1, char, dirname)

        self._updated_table()
    
    def show_error_dialog(self, title, message):
        QMessageBox.about(self.w, title, message)
        
class ProgressBarWidget(object):
    class Thread(QThread):
        str_signal = pyqtSignal(str)
        int_signal = pyqtSignal(int)
        def __init__(self, finish_func):
            super().__init__()
            self.finished.connect(finish_func)
        def run(self):
            raise NotImplementedError()
        def update_str(self, txt):
            self.str_signal.emit(txt)
        def update_int(self, val):
            if not isinstance(val, int):
                val = int(val)
            self.int_signal.emit(val)
    
    def __init__(self, parent_view):
        self.w = QDialog(parent_view)

        self.w.setWindowTitle("Preprocessing...")

        vbox_layout = QVBoxLayout(self.w)

        self.label = QLabel("", self.w)

        self.bar = QProgressBar(self.w)
        self.bar.setGeometry(0, 0, 300, 25)
        self.bar.setMaximum(100)

        vbox_layout.addWidget(self.label)
        vbox_layout.addWidget(self.bar)

        self.w.setLayout(vbox_layout)
    
    def register_thread(self, thread):
        thread.str_signal.connect(self.label.setText)
        thread.int_signal.connect(self.bar.setValue)
        
    def show(self):
        self.w.show()

    def close(self):
        self.w.close()

class IdentificationVisualizeView(object):
    def __init__(self, parent_widget, controller):
        self.controller = controller

        self.cur_i_char = None
        
        # GUI
        self.w = QDialog(parent_widget)
        self.w.setWindowTitle("Identification result")

        hbox = QHBoxLayout(self.w)
        # Left panel
        vbox_1_widget = QWidget()
        vbox_1 = QVBoxLayout(self.w)
        vbox_1_widget.setLayout(vbox_1)
        vbox_1_widget.setFixedWidth(200)
        hbox.addWidget(vbox_1_widget)
        ## Done button
        btn_done = QPushButton("Save images", self.w)
        btn_done.clicked.connect(self._btn_done)
        vbox_1.addWidget(btn_done)
        ## Character selecting list
        self.char_lst = QListWidget(self.w)
        self.char_lst.currentRowChanged.connect(self._char_lst_row_changed)
        vbox_1.addWidget(self.char_lst)

        # Right panel
        vbox_2 = QVBoxLayout(self.w)
        hbox.addLayout(vbox_2)
        # ## Top buttons
        # hbox_21 = QHBoxLayout(self.w)
        # vbox_2.addLayout(hbox_21)
        # ### Check button
        # btn_check = QPushButton("Check", self.w)
        # btn_check.clicked.connect(self._btn_check)
        # hbox_21.addWidget(btn_check)
        # ### Uncheck button
        # btn_uncheck = QPushButton("Uncheck", self.w)
        # btn_uncheck.clicked.connect(self._btn_uncheck)
        # hbox_21.addWidget(btn_uncheck)
        ## Image table
        self.img_table = QTableWidget(self.w)
        self.img_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.img_table.setSelectionMode(
            QAbstractItemView.NoSelection
        )
        vbox_2.addWidget(self.img_table)
    
    # slot
    def _btn_done(self):
        path = QFileDialog.getExistingDirectory(None,
            "Choose a directory to save images.", 
            os.path.dirname(__file__))
        if len(path) == 0: # No file selected.
            return
        else:
            ret = QMessageBox.question(self.w, "",
                "Do you want to move images? [Yes->move images, No->copy images]",
                QMessageBox.Yes | QMessageBox.No)
            self.controller.save_selected_images(path, (ret==QMessageBox.Yes))

    def _char_lst_row_changed(self, row_idx):
        self.controller.show_new_character(row_idx)
    
    # public utility
    def init_character_list(self, lst_char_names):
        for i_char, charname in enumerate(lst_char_names):
            item = QListWidgetItem(charname)
            self.char_lst.insertItem(i_char, item)
        self.char_lst.setCurrentRow(0)
    
    def _checked_func(self, val, i_state):
        flag = (val > 0)
        self.controller.set_check_states(self.cur_i_char, i_state, flag)
    
    def _select_target(self, val, i_state):
        self.controller.set_selected_target(self.cur_i_char, i_state, val)
    
    def _image_clicked(self, i_state):
        cbx = self.img_table.cellWidget(i_state, 1)
        state = cbx.isChecked()
        cbx.setChecked(state ^ True)

    def init_table(self, new_i_char, state_lst):
        self.cur_i_char = new_i_char

        self.img_table.clear()
        self.img_table.setColumnCount(4)
        self.img_table.setRowCount(len(state_lst))

        for i_state, state in enumerate(state_lst):
            # Image
            thumb = state.thumb
            image = ImageQt(thumb)
            pixmap = QPixmap.fromImage(image)
            img_label = ClickableQLabel("",self.img_table)
            img_label.setPixmap(pixmap)
            img_label.clicked.connect(partial(self._image_clicked, i_state=i_state))
            self.img_table.setCellWidget(i_state, 0, img_label)

            # Checkbox
            cbx = QCheckBox("Save", self.img_table)
            cbx.setChecked(state.check)
            cbx.stateChanged.connect(partial(self._checked_func, i_state=i_state))
            self.img_table.setCellWidget(i_state, 1, cbx)

            # Combobox
            combo = QComboBox(self.img_table)
            combo.addItems(state.get_char_name_list())
            combo.setCurrentIndex(state.selected_id)
            combo.currentIndexChanged.connect(
                partial(self._select_target, i_state=i_state)
            )
            self.img_table.setCellWidget(i_state, 2, combo)

            # Filename
            fn_label = QLabel(state.filename, self.img_table)
            self.img_table.setCellWidget(i_state, 3, fn_label)
        
        self.img_table.resizeRowsToContents()
        self.img_table.resizeColumnsToContents()
    
    def show(self):
        self.w.exec_()
    
    def close(self):
        self.w.close()
    
    def show_dialog(self, title, message):
        QMessageBox.about(self.w, title, message)