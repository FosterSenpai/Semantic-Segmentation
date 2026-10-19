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
        self.model_path = (
            self.root / "models" / "pretrained_focal_tversky_finetuned.keras"
        )

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
        self.wait(
            self.worker.submit(
                tf.keras.models.load_model, self.model_path, compile=False
            ),
            self.model_loaded,
        )

    def prepare_window(self, window: tk.Tk):
        window.title("MRI Segmentation")
        window.configure(bg=self.colour_bg)
        window.resizable(False, False)
        width, height = 540, 830
        x = max(0, (window.winfo_screenwidth() - width) // 2)
        y = max(0, (window.winfo_screenheight() - height) // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")
        window.protocol("WM_DELETE_WINDOW", self.close)

    def label(self, parent, text="", colour=None, size=10, bold=False, bg=None):
        widget = tk.Label(
            parent,
            text=text,
            bg=bg or self.colour_bg,
            fg=colour or self.colour_text,
            font=("Segoe UI", size, "bold" if bold else "normal"),
            justify="center",
            wraplength=self.image_size,
        )
        widget.pack(pady=3)
        return widget

    def build_ui(self):
        # Heading
        self.label(self.window, "MRI Segmentation", size=22, bold=True).pack_configure(
            pady=(16, 10)
        )

        # Buttons
        buttons = tk.Frame(self.window, bg=self.colour_bg)
        buttons.pack()
        self.load_button = tk.Button(
            buttons,
            text="Load image",
            command=self.load_image,
            bg="#383838",
            fg=self.colour_text,
            activebackground="#484848",
            relief="flat",
            bd=0,
            font=("Segoe UI", 10),
            padx=18,
            pady=8,
        )
        self.load_button.pack(side="left", padx=4)
        self.predict_button = tk.Button(
            buttons,
            text="Predict",
            command=self.predict,
            bg=self.colour_blue,
            fg="#10212B",
            activebackground="#99DFFF",
            relief="flat",
            bd=0,
            font=("Segoe UI", 10),
            padx=18,
            pady=8,
            state="disabled",
        )
        self.predict_button.pack(side="left", padx=4)

        # File selection
        self.filename = self.label(
            self.window, "No image selected", self.colour_muted, 9
        )
        self.filename.pack_configure(pady=10)

        # Using one image panel, swapping out with overlays after prediction
        self.canvas = tk.Canvas(
            self.window,
            width=self.image_size,
            height=self.image_size,
            bg="#151515",
            highlightthickness=0,
        )
        self.canvas.pack()
        self.canvas.create_text(
            self.image_size // 2,
            self.image_size // 2,
            text="Load an MRI image to begin",
            fill=self.colour_muted,
        )

        # Legend for overlay colours
        legend = tk.Frame(self.window, bg=self.colour_bg)
        legend.pack(pady=8)
        for colour, text in [
            ("#00FF00", "Correct Overlap"),
            ("#FFAE00", "Missed"),
            ("#F10336", "False Positive"),
        ]:
            tk.Label(
                legend,
                text=f"- {text}",
                bg=self.colour_bg,
                fg=colour,
                font=("Segoe UI", 9),
            ).pack(side="left", padx=8)

        # Results info panel
        card = tk.Frame(self.window, bg=self.colour_panel, padx=20, pady=8)
        card.pack(fill="x", padx=30)
        self.label(card, "Prediction Results", bold=True, bg=self.colour_panel)
        rows = tk.Frame(card, bg=self.colour_panel)
        rows.pack()
        self.metrics = {}
        for row, name in enumerate(
            [
                "Dice",
                "IoU / Jaccard",
                "Abnormality Covered",
                "Abnormality Missed",
                "Predicted Area Incorrect",
            ]
        ):
            tk.Label(
                rows,
                text=name,
                bg=self.colour_panel,
                fg=self.colour_text,
                font=("Segoe UI", 10),
            ).grid(row=row, column=0, sticky="e", padx=10, pady=2)
            value = tk.Label(
                rows,
                text="-",
                bg=self.colour_panel,
                fg=self.colour_blue,
                width=8,
                font=("Segoe UI", 10, "bold"),
            )
            value.grid(row=row, column=1)
            self.metrics[name] = value

        # Status
        self.status = self.label(self.window, "Loading model...", self.colour_muted, 9)
        self.status.pack_configure(pady=10)

    def show_image(self, picture):
        """Swap new image into image display"""
        self.photo = ImageTk.PhotoImage(
            picture.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        )
        self.canvas.delete("all")
        self.canvas.create_image(
            self.image_size // 2, self.image_size // 2, image=self.photo
        )

    def wait(self, future, callback):
        """Check for bg work without blocking gui"""
        # If not finished, schedule to run again in 100ms
        if not future.done():
            self.window.after(100, self.wait, future, callback)
            return

        # When bg work finished, try get result
        try:
            callback(future.result())
        except Exception as error:
            self.status.config(text=str(error))

        self.busy = False
        self.update_buttons()

    def update_buttons(self):
        """Disable controls during prediction and check if all is present to predict"""
        self.load_button.config(state="disabled" if self.busy else "normal")
        ready = self.model is not None and self.mask is not None and not self.busy
        self.predict_button.config(state="normal" if ready else "disabled")

    def model_loaded(self, model):
        self.model = model
        self.status.config(
            text="Ready to predict."
            if self.mask is not None
            else "Load an MRI with its matching mask."
        )

    def load_image(self):
        # Get MRI and clear old results
        selected = filedialog.askopenfilename(
            initialdir=self.root / "data",
            filetypes=[("MRI images", ("*.tif", "*.tiff"))],
        )
        if not selected:
            return

        self.mask = None
        self.update_buttons()
        path = Path(selected)
        self.filename.config(text=path.name)

        for value in self.metrics.values():
            value.config(text="-")

        try:
            with Image.open(path) as file:
                self.image = np.asarray(file, dtype=np.float32) / 255.0  # Scale

            # Shape check
            if self.image.shape != (256, 256, 3):
                raise ValueError("Select a 256 x 256 three channel MRI image.")

            # Using normalized FLAIR channel for display
            flair = self.image[:, :, 1]
            flair = (flair - flair.min()) / max(float(np.ptp(flair)), 1e-8)
            self.original = Image.fromarray((flair * 255).astype("uint8")).convert(
                "RGB"
            )
            self.show_image(self.original)

            # Loading matching mask
            mask_path = path.with_name(f"{path.stem}_mask{path.suffix}")
            with Image.open(mask_path) as file:
                mask = np.asarray(file) > 0

            # Mask shape check
            if mask.shape != self.image.shape[:2]:
                raise ValueError("The mask dimensions do not match the image.")

            self.mask = mask
            self.status.config(
                text="Ready to predict."
                if self.model is not None
                else "Image loaded. Waiting for the model.."
            )
        except FileNotFoundError:
            self.status.config(text="Matching mask missing. Prediction disabled.")
        except Exception as error:
            self.status.config(text=str(error))

        self.update_buttons()

    def predict(self):
        """Run model prediction on background thread."""
        self.busy = True
        self.update_buttons()
        self.status.config(text="Predicting...")
        self.wait(self.worker.submit(self.run_prediction), self.display_result)

    def run_prediction(self):
        if self.model is None:
            raise ValueError("Model is not loaded")
        if self.image is None:
            raise ValueError("Image not prepared")
        probabilities = self.model(self.image[None], training=False)
        return np.asarray(probabilities)[0, :, :, 0] > 0.5

    def display_result(self, prediction):
        if self.mask is None:
            raise ValueError("No mask loaded")
        if prediction is None:
            raise ValueError("No prediction available.")

        # Calculate overlap, correct/missed/fp
        overlap = self.mask & prediction
        missed = self.mask & ~prediction
        false_positive = prediction & ~self.mask

        tp, actual, predicted = map(np.count_nonzero, (overlap, self.mask, prediction))
        total = actual + predicted
        union = total - tp

        # Matching scores from notebook for empty masks
        results = {
            "Dice": f"{2 * tp / total if total else 1:.3f}",
            "IoU / Jaccard": f"{tp / union if union else 1:.3f}",
            "Abnormality Covered": f"{100 * tp / actual:.1f}%" if actual else "N/A",
            "Abnormality Missed": f"{100 * (actual - tp) / actual:.1f}%"
            if actual
            else "N/A",
            "Predicted Area Incorrect": f"{100 * (predicted - tp) / predicted:.1f}%"
            if predicted
            else "N/A",
        }
        for name, value in results.items():
            self.metrics[name].config(text=value)

        # Building overlay, Green = correct, Orange = missed, Red = False pos
        overlay = np.zeros((*self.mask.shape, 4), dtype="uint8")
        overlay[overlap] = [0, 255, 0, 128]
        overlay[missed] = [255, 174, 0, 70]
        overlay[false_positive] = [241, 3, 54, 70]

        self.show_image(
            Image.alpha_composite(
                self.original.convert("RGBA"), Image.fromarray(overlay)
            )
        )
        self.status.config(text="Prediction complete.")

    def close(self):
        self.worker.shutdown(wait=False, cancel_futures=True)
        self.window.destroy()


if __name__ == "__main__":
    gui = SegmentationGUI()
    gui.window.mainloop()
