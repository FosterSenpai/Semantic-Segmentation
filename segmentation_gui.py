"""GUI that loads a single MRI image and detects abnormalities using best model from training."""

import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog

import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

class SegmentationGUI:
    def __init__(self) -> None:
        self.window = tk.Tk()
        self.root = Path(__file__).resolve().parent
        self.image_size = 480
        self.model_path = self.root / "models" / "pretrained_focal_tversky_finetuned.keras"
        
        self.model = None
        self.image = None
        self.mask = None
        self.original = None
        self.photo = None
        self.busy = False
        self.worker = ThreadPoolExecutor(max_workers=1)
        
        self.colour_bg = "#202020"
        self.colour_panel = "#292929"
        self.colour_text = "#F3F3F3"
        self.colour_muted = "#AAAAAA"
        self.colour_blue = "#60CDFF"
        
        self.prepare_window(self.window)
        self.build_ui()
        
        # Loading model once
        self.wait(self.worker.submit(
            tf.keras.model.load_model,
            self.model_path,
            compile=False
            
        ) self.model_loaded)
        
    def prepare_window(self, window: tk.Tk):
        window.title("MRI Segmentation")
        window.configure(bg=self.colour_bg)
        window.resizable(False, False)
        width, height = 540, 830
        x = max(0, (window.winfo_screenwidth() - width) // 2)
        y = max(0, (window.winfo_screenheight() - height) // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")
        window.protocol("WM_DELETE_WINDOW", self.close)
        
    def label(self, parent, text="", colour=None, size=10, bold=False):
        pass
    
    def build_ui(self):
        pass
    
    def show_image(self, picture):
        pass
    
    def wait(self, future, callback):
        pass
    
    def update_buttons(self):
        pass
    
    def model_loaded(self, model):
        pass
    
    def load_image(self):
        pass
    
    def predict(self):
        pass
    
    def run_prediction(self):
        pass
    
    def display_result(self, prediction):
        pass
    
    def close(self):
        pass
    
    if __name__ == "__main__":
        gui = SegmentationGUI()
        gui.window.mainloop()