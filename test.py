
import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import pytesseract
import cv2
import numpy as np
from openpyxl import Workbook

# Configure Tesseract on Windows
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


class SAPTextExtractor:
    def __init__(self, root):
        self.root = root
        self.root.title("SAP Image Text Extractor")
        self.root.geometry("1050x750")
        self.root.minsize(750, 550)

        self.image_path = None
        self.original_image = None
        self.preview_image = None

        self.mode = tk.StringVar(value="SAP Field Names")
        self.status = tk.StringVar(value="Ready - Select an image")

        self.create_gui()

    def create_gui(self):
        header = ttk.Frame(self.root, padding=12)
        header.pack(fill="x")

        ttk.Label(
            header,
            text="SAP Image Text Extractor",
            font=("Segoe UI", 20, "bold")
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="Extract SAP technical names or all text from screenshots"
        ).pack(anchor="w", pady=(4, 0))

        toolbar = ttk.Frame(self.root, padding=10)
        toolbar.pack(fill="x")

        ttk.Button(
            toolbar,
            text="Select Image",
            command=self.select_image
        ).pack(side="left", padx=4)

        ttk.Button(
            toolbar,
            text="Extract Text",
            command=self.extract_text
        ).pack(side="left", padx=4)

        ttk.Button(
            toolbar,
            text="Copy Text",
            command=self.copy_text
        ).pack(side="left", padx=4)

        ttk.Button(
            toolbar,
            text="Export Excel",
            command=self.export_excel
        ).pack(side="left", padx=4)

        ttk.Button(
            toolbar,
            text="Clear",
            command=self.clear_text
        ).pack(side="left", padx=4)

        options = ttk.Frame(self.root, padding=(14, 5))
        options.pack(fill="x")

        ttk.Label(options, text="Extraction Mode:").pack(
            side="left", padx=(0, 10)
        )

        ttk.Combobox(
            options,
            textvariable=self.mode,
            values=["SAP Field Names", "All Text"],
            state="readonly",
            width=22
        ).pack(side="left")

        main = ttk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL
        )
        main.pack(fill="both", expand=True, padx=12, pady=10)

        image_frame = ttk.LabelFrame(
            main, text="Image Preview", padding=8
        )
        text_frame = ttk.LabelFrame(
            main, text="Extracted Text", padding=8
        )

        main.add(image_frame, weight=1)
        main.add(text_frame, weight=1)

        self.image_label = ttk.Label(
            image_frame,
            text="No image selected",
            anchor="center"
        )
        self.image_label.pack(fill="both", expand=True)

        self.text_box = tk.Text(
            text_frame,
            font=("Consolas", 12),
            wrap="none",
            undo=True
        )

        scroll_y = ttk.Scrollbar(
            text_frame,
            orient="vertical",
            command=self.text_box.yview
        )

        scroll_x = ttk.Scrollbar(
            text_frame,
            orient="horizontal",
            command=self.text_box.xview
        )

        self.text_box.configure(
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set
        )

        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        self.text_box.pack(fill="both", expand=True)

        ttk.Label(
            self.root,
            textvariable=self.status,
            relief="sunken",
            anchor="w",
            padding=7
        ).pack(fill="x", side="bottom")

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select SAP Screenshot",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All Files", "*.*")
            ]
        )

        if not path:
            return

        try:
            self.original_image = Image.open(path).convert("RGB")
            self.image_path = path

            preview = self.original_image.copy()
            preview.thumbnail((480, 550))

            self.preview_image = ImageTk.PhotoImage(preview)
            self.image_label.configure(
                image=self.preview_image,
                text=""
            )

            self.status.set(f"Selected: {os.path.basename(path)}")

        except Exception as e:
            messagebox.showerror("Image Error", str(e))

    def preprocess_image(self):
        image = cv2.imread(self.image_path)

        if image is None:
            raise ValueError("Unable to read the image.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enlarge small text for OCR
        gray = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        # Improve contrast
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        processed = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )

        return processed

    def extract_sap_fields(self, image):
        # OCR optimized for short SAP field identifiers
        config = (
            "--oem 3 --psm 11 "
            "-c tessedit_char_whitelist="
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
        )

        data = pytesseract.image_to_data(
            image,
            config=config,
            output_type=pytesseract.Output.DICT
        )

        fields = []

        # Known SAP technical fields from the example screenshot
        # This list is used only for recognizing candidates,
        # not for automatically inserting them into the output.
        known_fields = {
            "MATNR", "ERSDA", "ERNAM", "LAEDA",
            "AENAM", "MTART", "XCHPF", "MSTAE",
            "MSTAV", "KZUMW", "PROFL", "HAZMAT"
        }

        image_width = image.shape[1]

        for i, word in enumerate(data["text"]):
            word = word.strip().upper()

            if not word:
                continue

            try:
                confidence = float(data["conf"][i])
            except (ValueError, TypeError):
                continue

            if confidence < 25:
                continue

            # Keep words that resemble technical field names.
            if not re.fullmatch(r"[A-Z][A-Z0-9_]{2,29}", word):
                continue

            # SAP selection screens usually place labels
            # in the left portion of the screenshot.
            left = data["left"][i]

            if word in known_fields or left < image_width * 0.48:
                fields.append((
                    data["top"][i],
                    left,
                    word
                ))

        # Sort top-to-bottom, then left-to-right
        fields.sort(key=lambda item: (item[0], item[1]))

        # Remove duplicate field names
        unique = []
        seen = set()

        for _, _, word in fields:
            if word not in seen:
                seen.add(word)
                unique.append(word)

        return unique

    def extract_text(self):
        if not self.image_path:
            messagebox.showwarning(
                "Missing Image",
                "Please select an image first."
            )
            return

        try:
            self.status.set("Extracting text...")
            self.root.update_idletasks()

            processed = self.preprocess_image()

            if self.mode.get() == "SAP Field Names":
                fields = self.extract_sap_fields(processed)
                result = "\n".join(fields)

            else:
                result = pytesseract.image_to_string(
                    processed,
                    config="--oem 3 --psm 11"
                )

            self.text_box.delete("1.0", tk.END)
            self.text_box.insert("1.0", result)

            count = len(result.splitlines())
            self.status.set(
                f"Extraction completed - {count} lines"
            )

        except Exception as e:
            messagebox.showerror(
                "Extraction Error",
                str(e)
            )
            self.status.set("Extraction failed")

    def copy_text(self):
        text = self.text_box.get("1.0", "end-1c")

        if not text.strip():
            messagebox.showwarning(
                "No Text",
                "There is no text to copy."
            )
            return

        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()

        self.status.set("Text copied to clipboard")

    def export_excel(self):
        text = self.text_box.get("1.0", "end-1c")

        if not text.strip():
            messagebox.showwarning(
                "No Text",
                "Extract text before exporting."
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save Excel File",
            defaultextension=".xlsx",
            filetypes=[("Excel Workbook", "*.xlsx")]
        )

        if not path:
            return

        try:
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "SAP Extracted Data"

            sheet["A1"] = (
                "Technical Field Name"
                if self.mode.get() == "SAP Field Names"
                else "Extracted Text"
            )

            lines = [
                line.strip()
                for line in text.splitlines()
                if line.strip()
            ]

            for row, line in enumerate(lines, start=2):
                sheet.cell(row=row, column=1, value=line)

            sheet.column_dimensions["A"].width = 40
            sheet.freeze_panes = "A2"

            workbook.save(path)

            self.status.set(f"Excel saved: {path}")
            messagebox.showinfo(
                "Success",
                f"Excel file saved successfully:\n{path}"
            )

        except Exception as e:
            messagebox.showerror(
                "Excel Export Error",
                str(e)
            )

    def clear_text(self):
        self.text_box.delete("1.0", tk.END)
        self.status.set("Text cleared")


if __name__ == "__main__":
    root = tk.Tk()
    app = SAPTextExtractor(root)
    root.mainloop()
