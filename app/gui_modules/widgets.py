from collections import namedtuple
import os
import sys

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, \
    QLabel, QComboBox, QFileDialog, QMessageBox, QDialog, QTableWidget, QTableWidgetItem, \
    QApplication, QProgressBar
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont


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

class IdentificationWidget(object):
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

        btn_vbox_layout.addWidget(btn_load)
        btn_vbox_layout.addWidget(btn_save)
        btn_vbox_layout.addWidget(btn_load_dir)
        btn_vbox_layout.addWidget(self.label_load_dir)
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
    
    def sync_table(self, char_dir_lst):
        self.char_dir_lst = char_dir_lst
        self._init_table(self.char_dir_lst)

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
        def _run(self):
            raise NotImplementedError()
        def run(self):
            print("kiee2")
            self._run()
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