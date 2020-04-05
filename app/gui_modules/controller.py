from importlib import import_module
import os
import sys

import torch

from .widgets import MainWidget, start_app

class Controller(object):
    def __init__(self):
        self.device = "cpu"
        self.plugin = None
        self.module = None
    
    def start_app(self):
        start_app(self)

    ## Configurations
    def get_available_device_list(self):
        torch_devices = ["cpu"]
        device_names = ["CPU"]
        for i in range(torch.cuda.device_count()):
            torch_devices.append(f"cuda:{i}")
            device_names.append(f"GPU{i}")
        return torch_devices, device_names

    def select_device(self, view, torch_device_name):
        self.device = torch_device_name
    
    def select_plugin(self, view, plugin_fn):
        plugin_dir = os.path.dirname(plugin_fn)
        plugin_name = os.path.basename(plugin_fn)[:-3]

        sys.path.append(plugin_dir)
        plugin = import_module(plugin_name)

        if "MainModule" in dir(plugin):
            self.plugin = plugin
            view.update_plugin(plugin_fn)
        else:
            view.show_error_dialog(
                "Plugin error",
                "Selected plugin does not contain MainModule class.")
    
    # Identifications
    def start_identification(self, view):
        if self.plugin is None:
            view.show_error_dialog(
                "Error",
                "Please select plugin!"
            )
        self.module = self.plugin.MainModule(device=self.device)

        print("IDENTIFICATIOn")