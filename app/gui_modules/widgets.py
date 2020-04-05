from collections import namedtuple
import os
import sys

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QPushButton, QLabel,\
    QComboBox, QFileDialog, QMessageBox, QDialog
from PyQt5.QtWidgets import QApplication

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

class IdentificationWidget(QWidget):
    def __init__(self, parent):
        self.w = QDialog(parent)
    
    def show(self):
        self.w.exec_()