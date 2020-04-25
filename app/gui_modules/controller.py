from importlib import import_module
import os
import sys

from collections import defaultdict
import shutil
import json

import numpy as np
import torch

from .widgets import MainWidget, start_app, IdentificationView, \
    ProgressBarWidget, IdentificationVisualizeView

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

        Identification_Controller(self.module, self, view)

        print("IDENTIFICATIOn")

class Identification_Controller(object):
    def __init__(self, module, parent_controller, parent_view):
        self.parent = parent_controller
        self.module = module

        self.char_dir_lst = []

        self.image_src_dir = None
        self.filenames = None
        self.src_filenames = None
        self.scores = None
        self.thumbs = None

        self.states = None

        self.parent_view = parent_view
        self.view = IdentificationView.init_view(parent_view, self)
        self.progress_view = None
        self.visualize_view = None
    
    ## Configuration window
    def update_char_dir_lst(self, new_lst):
        self.char_dir_lst = new_lst
    
    def save_json(self, path):
        with open(path, "w") as h:
            json.dump(self.char_dir_lst, h)

    def load_json(self, path):
        with open(path) as h:
            self.char_dir_lst = json.load(h)
        self.view.sync_table(self.char_dir_lst)
    
    def register_src_dir(self, path):
        self.image_src_dir = path

    ## Preprocessing window
    class PreProcess(ProgressBarWidget.Thread):
        def __init__(self, finish_func, controller):
            super().__init__(finish_func)
            self.controller = controller
        def run(self):
            self.controller._load_filename_list(self)
            self.controller._load_thumbnails(self)
            self.controller._compute_scores(self)
    
    def _load_thumbnails(self, thread):
        def callback(i_fn, n_fn):
            thread.update_int(int(i_fn*100/n_fn))
        thread.update_str("Loading source image thumbnails.")
        thread.update_int(0)
        self.thumbs = self.module._load_thumbnail(self.src_filenames, size=256,
            callback=callback)

    def _load_filename_list(self, thread):
        self.filenames = []
        thread.update_str("Loading character images.")
        thread.update_int(0)
        n_char = len(self.char_dir_lst)
        for i_char, (char, directory) in enumerate(self.char_dir_lst):
            lst_fn = self.module._get_list_of_all_image_files_directory(directory)
            self.filenames.append(lst_fn)
            thread.update_int(int(i_char*100/n_char))
        
        thread.update_str("Loading source images.")
        thread.update_int(0)
        self.src_filenames = self.module._get_list_of_all_image_files_directory(self.image_src_dir)
    
    def _compute_scores(self, thread):
        def callback(*args):
            if args[0]==0:
                thread.update_int(int(args[1]*100/args[2]))
                if args[1] + 1 == args[2]:
                    thread.update_str("Computing embeddings of source images.")
                    thread.update_int(0)
            elif args[0] == 1:
                thread.update_str("Computing scores between images.")
                thread.update_int(0)
            elif args[0]==2:
                thread.update_int(int(args[1]*100/args[2]))
                if args[1] + 1 == args[2]:
                    thread.update_str("Computing average scores.")
                    thread.update_int(0)
        
        thread.update_str("Computing embeddings of character images.")
        thread.update_int(0)
        self.scores = self.module.get_identification_result(
            self.filenames, self.src_filenames, mode="Avg", callback=callback)

    def run(self):
        if self.image_src_dir is None:
            self.view.show_error_dialog("Error", "Please select source image directory.")
            return
        self.view.close()
        self.thread = self.PreProcess(self._start_identification_UI, self)
        self.progress_view = ProgressBarWidget(self.parent_view)
        self.progress_view.show()
        self.progress_view.register_thread(self.thread)
        self.thread.start()

    ## Idenfitication window
    class ItemState(object):
        def __init__(self, thumb, check, selected_id, char_lst, char_id_lst, 
            score, filename):
            self.thumb = thumb
            self.check = check
            self.selected_id = selected_id
            self.char_lst = char_lst # Unsorted. Original.
            self.char_id_lst = char_id_lst # Sorted
            self.score = score
            self.filename = filename
        def get_char_name_list(self):
            return [self.char_lst[_] for _ in self.char_id_lst]

    def _start_identification_UI(self):
        self.progress_view.close()

        charnames = [_[0] for _ in self.char_dir_lst]

        # Construct state map
        image_ids = defaultdict(list) # i_char -> list of (score, i_img)
        sorted_idx = []
        for i_img in range(self.scores.shape[0]):
            sorted_i_char = np.argsort(-self.scores[i_img])
            sorted_idx.append(sorted_i_char.tolist())

            i_char = sorted_i_char[0]
            image_ids[i_char].append((self.scores[i_img, i_char], i_img))
        
        for key in image_ids:
            image_ids[key] = sorted(image_ids[key], reverse=True)
        
        self.states = [[] for _ in range(self.scores.shape[1])]
        for i_char, sorted_lst in image_ids.items():
            for score, i_img in sorted_lst:
                state = self.ItemState(
                    thumb = self.thumbs[i_img],
                    check = False,
                    selected_id = 0,
                    char_lst = charnames,
                    char_id_lst = sorted_idx[i_img],
                    score = score,
                    filename = self.src_filenames[i_img]
                )
                self.states[i_char].append(state)
        
        # Initialize view
        self.visualize_view = IdentificationVisualizeView(
            self.parent_view, self)
        self.visualize_view.init_character_list(charnames)
        self.visualize_view.init_table(0, self.states[0])
        self.visualize_view.show()
    
    def set_check_states(self, i_char, i_img, new_bool):
        self.states[i_char][i_img].check = new_bool
    
    def set_selected_target(self, i_char, i_img, new_idx):
        self.states[i_char][i_img].selected_id = new_idx

    def show_new_character(self, i_char):
        self.visualize_view.init_table(i_char, self.states[i_char])
    
    def save_selected_images(self, path, delete_source=False):
        charnames = [_[0] for _ in self.char_dir_lst]
        for charname in charnames:
            target_dir = os.path.join(path, charname)
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
        
        for i_char, lst_states in enumerate(self.states):
            for state in lst_states:
                if state.check:
                    src_fn = state.filename
                    src_base_fn = os.path.basename(src_fn)
                    dest_fn = os.path.join(
                        path, charnames[i_char], src_base_fn
                    )

                    if delete_source:
                        # Move
                        shutil.move(src_fn, dest_fn)
                    else:
                        # Copy
                        shutil.copy(src_fn, dest_fn)
        
        self.visualize_view.show_dialog("Info", "Finished!!")
        self.visualize_view.close()

            
