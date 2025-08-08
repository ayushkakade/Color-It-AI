import os
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageFilter
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from ttkbootstrap import ttk
from ttkbootstrap.dialogs import Messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from concurrent.futures import ThreadPoolExecutor

class ImageColorizerApp(TkinterDnD.Tk):
    """
    A Tkinter application for colorizing black and white images using a Caffe model.

    This class encapsulates the entire application, including the GUI, model loading,
    and image processing logic. It uses multithreading to ensure the GUI remains
    responsive during the computationally intensive colorization process.
    """
    def __init__(self):
        super().__init__()

        # Model paths (Note: These paths are relative and must exist
        # in the same directory as the Python script)
        self.PROTOTXT = "colorization_deploy_v2.prototxt"
        self.POINTS = "pts_in_hull.npy"
        self.MODEL = "colorization_release_v2.caffemodel"

        # Application state variables
        self.colorized_image_array = None
        self.original_image_array = None
        self.filename = ""
        # Using a ThreadPoolExecutor to run tasks in the background
        self.executor = ThreadPoolExecutor(max_workers=1)

        # GUI setup
        self.style = tb.Style("darkly")
        self.title("Color It AI")
        self.geometry("1000x700")
        self.resizable(True, True)
        self.withdraw()

        # Try to load the deep learning model.
        self._load_model()
        self._setup_splash_screen()

        # Set up the main UI after the splash screen
        self.after(2000, self._setup_main_ui)

    def _load_model(self):
        """
        Loads the pre-trained Caffe model and associated data.
        Exits the application if the model files are not found.
        """
        try:
            self.net = cv2.dnn.readNetFromCaffe(self.PROTOTXT, self.MODEL)
            pts = np.load(self.POINTS)
            pts = pts.transpose().reshape(2, 313, 1, 1)
            class8 = self.net.getLayerId("class8_ab")
            conv8 = self.net.getLayerId("conv8_313_rh")
            self.net.getLayer(class8).blobs = [pts.astype("float32")]
            self.net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        except Exception as e:
            # Using the correct Messagebox syntax
            Messagebox.showerror("Error", f"Failed to load model files: {e}\n\n"
                                          f"Please ensure these files are in the same directory as the script:\n"
                                          f"- {self.PROTOTXT}\n"
                                          f"- {self.POINTS}\n"
                                          f"- {self.MODEL}")
            self.quit()

    def _setup_splash_screen(self):
        """
        Creates and displays a splash screen while the application loads.
        """
        splash = tb.Toplevel(self)
        splash.geometry("400x300")
        splash.overrideredirect(True)
        splash.update_idletasks()

        # Center the splash screen
        width = splash.winfo_width()
        height = splash.winfo_height()
        x = (splash.winfo_screenwidth() // 2) - (width // 2)
        y = (splash.winfo_screenheight() // 2) - (height // 2)
        splash.geometry(f"{width}x{height}+{x}+{y}")

        splash.configure(bg='#1c1c1c')

        try:
            bg_img = Image.open("splash_bg.jpg").convert("RGBA")
            blurred = bg_img.filter(ImageFilter.GaussianBlur(10))
            splash_img = ImageTk.PhotoImage(blurred)
            bg_label = tb.Label(splash, image=splash_img)
            bg_label.image = splash_img
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except:
            pass

        tb.Label(splash, text="Color It AI", font=("Segoe UI", 18, "bold"), foreground="white", background="#1c1c1c").pack(pady=40)
        spinner = ttk.Progressbar(splash, mode='indeterminate', length=250)
        spinner.pack(pady=10)
        spinner.start(10)

        def close():
            splash.destroy()
            self.deiconify()

        self.after(2000, close)

    def _configure_styles(self):
        """
        Configures the custom styles for the application widgets to match Gemini's UI.
        This method is called on initial setup and on theme change.
        """
        # Determine panel color based on the current theme
        current_theme = self.style.theme.name
        if current_theme in ["flatly", "litera", "minty", "pulse", "lumen", "cosmo", "yeti", "journal", "sandstone"]:
            panel_bg_color = '#f0f0f0' # Light gray for light themes
        else:
            panel_bg_color = '#242526' # Dark gray for dark themes

        # Configure a custom style for the panels and labels
        self.style.configure('Custom.TFrame', background=panel_bg_color, borderwidth=0, relief='flat')
        self.style.configure('Custom.TLabel', background=panel_bg_color)
        self.style.configure('TButton', font=('Helvetica', 12), padding=10)

        self._set_panel_themes()

    def _set_panel_themes(self):
        """Dynamically sets the style for the panels."""
        if hasattr(self, 'original_panel'):
            self.original_panel.config(style='Custom.TFrame')
            self.original_label_title.config(style='Custom.TLabel')
            self.bw_image_label.config(style='Custom.TLabel')
        if hasattr(self, 'colorized_panel'):
            self.colorized_panel.config(style='Custom.TFrame')
            self.colorized_label_title.config(style='Custom.TLabel')
            self.color_image_label.config(style='Custom.TLabel')
            
    def _setup_main_ui(self):
        """
        Sets up the main application window and its widgets with a grid-based layout.
        """
        # Configure grid to be responsive
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1) # Row for images will expand
        
        # Configure the custom styles
        self._configure_styles()
        
        # Top header frame for title and theme button
        header_frame = tb.Frame(self)
        header_frame.grid(row=0, column=0, columnspan=2, pady=(10, 5), padx=20, sticky="ew")
        header_frame.columnconfigure(0, weight=1)
        header_frame.columnconfigure(1, weight=1)

        # App title label in the header
        app_title_label = tb.Label(header_frame, text="Color It AI", bootstyle="info", font=("Helvetica", 18, "bold"))
        app_title_label.grid(row=0, column=0, sticky="w")
        
        # Theme button in the top right corner
        self.theme_btn = tb.Button(header_frame, text="🌙", bootstyle="info-outline", command=self._toggle_theme)
        self.theme_btn.grid(row=0, column=1, sticky="e")

        # Progress bar and status label (placed below the header)
        self.progressbar = ttk.Progressbar(self, orient="horizontal", mode="indeterminate")
        self.progressbar.grid(row=1, column=0, columnspan=2, pady=5, padx=20, sticky="ew")
        self.progress_label = tb.Label(self, text="", bootstyle="info")
        self.progress_label.grid(row=1, column=0, columnspan=2)

        # Main frame to hold both image panels, ensuring they have equal weight
        image_container_frame = tb.Frame(self)
        image_container_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=20, pady=20)
        image_container_frame.grid_columnconfigure(0, weight=1)
        image_container_frame.grid_columnconfigure(1, weight=1)
        image_container_frame.grid_rowconfigure(0, weight=1)

        # Left side: Original Image
        self.original_panel = tb.Frame(image_container_frame, style='Custom.TFrame')
        self.original_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        # Ensure minimum size for the panel's content area
        self.original_panel.grid_rowconfigure(1, weight=1, minsize=300)
        self.original_panel.grid_columnconfigure(0, weight=1, minsize=400)
        self.original_label_title = tb.Label(self.original_panel, text="Original Image", font=("Helvetica", 14), style='Custom.TLabel')
        self.original_label_title.grid(row=0, column=0, pady=5)
        self.bw_image_label = tb.Label(self.original_panel, text="Drag & Drop a B&W Image or click 'Select Image'", anchor="center", font=("Helvetica", 12), style='Custom.TLabel')
        self.bw_image_label.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))

        # Right side: Colorized Image
        self.colorized_panel = tb.Frame(image_container_frame, style='Custom.TFrame')
        self.colorized_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        # Ensure minimum size for the panel's content area
        self.colorized_panel.grid_rowconfigure(1, weight=1, minsize=300)
        self.colorized_panel.grid_columnconfigure(0, weight=1, minsize=400)
        self.colorized_label_title = tb.Label(self.colorized_panel, text="Colorized Image", font=("Helvetica", 14), style='Custom.TLabel')
        self.colorized_label_title.grid(row=0, column=0, pady=5)
        self.color_image_label = tb.Label(self.colorized_panel, text="Colorized Image will appear here", anchor="center", font=("Helvetica", 12), style='Custom.TLabel')
        self.color_image_label.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        
        # Footer frame for select and save buttons
        footer_frame = tb.Frame(self)
        footer_frame.grid(row=3, column=0, columnspan=2, pady=10, padx=20, sticky="ew")
        footer_frame.columnconfigure(0, weight=1)
        footer_frame.columnconfigure(1, weight=1)
        
        # Select and save buttons in the footer
        self.select_btn = tb.Button(footer_frame, text="📂 Select Image", bootstyle="info-outline", command=self._select_image_from_dialog)
        self.select_btn.grid(row=0, column=0, padx=(0, 10), sticky="ew")

        self.save_btn = tb.Button(footer_frame, text="💾 Save Image", bootstyle="success-outline", command=self._save_image)
        self.save_btn.grid(row=0, column=1, padx=(10, 0), sticky="ew")
        
        # Apply the initial theme settings to the panels
        self._set_panel_themes()

        # Bind the entire window resize event to update images
        self.bind("<Configure>", self._resize_images)

        # Drag and drop functionality
        self.drop_target_register(DND_FILES)
        self.dnd_bind("<<Drop>>", lambda e: self._load_image(e.data.strip('{}')))
        
        # Update the UI state to disable the save button initially
        self._update_ui_state(False)
        
        # Force an immediate redraw to fix initial layout sizing
        self.update_idletasks()
        self.after(100, self._resize_images)

    def _start_spinner(self):
        """Starts the progress bar animation and updates the label."""
        self.progress_label.config(text="Colorizing...", bootstyle="warning")
        self.progressbar.start(10)
        self.update_idletasks()

    def _stop_spinner(self, message="Done", bootstyle="success"):
        """Stops the progress bar and updates the label."""
        self.progressbar.stop()
        self.progress_label.config(text=message, bootstyle=bootstyle)

    def _update_ui_state(self, is_enabled):
        """Enables or disables the save button and updates its cursor."""
        if is_enabled:
            self.save_btn.config(state="normal", cursor="hand2")
        else:
            self.save_btn.config(state="disabled", cursor="arrow")
    
    def _colorize_image_thread(self, path):
        """
        Loads and colorizes an image in a separate thread.
        """
        try:
            image = cv2.imread(path)
            if image is None:
                self.after(0, lambda: Messagebox.showerror("Error", "Failed to load image."))
                self.after(0, lambda: self._stop_spinner("Error", "danger"))
                return

            # Store original image for display
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.original_image_array = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            # Colorization logic
            scaled = image.astype("float32") / 255.0
            lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
            resized = cv2.resize(lab, (224, 224))
            L = cv2.split(resized)[0]
            L -= 50
            self.net.setInput(cv2.dnn.blobFromImage(L))
            ab = self.net.forward()[0, :, :, :].transpose((1, 2, 0))
            ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
            L = cv2.split(lab)[0]
            colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
            colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
            colorized = np.clip(colorized, 0, 1)
            self.colorized_image_array = (255 * colorized).astype("uint8")
            
            self.after(0, lambda: self._show_images(os.path.basename(path)))
        except Exception as e:
            self.after(0, lambda: Messagebox.showerror("Error", f"An error occurred during colorization: {e}"))
            self.after(0, lambda: self._stop_spinner("Error", "danger"))

    def _select_image_from_dialog(self):
        """Opens a file dialog to select an image."""
        from tkinter import filedialog
        file = filedialog.askopenfilename(title="Select Black & White Image",
                                          filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if file:
            self._load_image(file)

    def _load_image(self, path):
        """
        Initiates the image colorization process in a separate thread.
        """
        self._start_spinner()
        self.executor.submit(self._colorize_image_thread, path)

    def _show_images(self, name):
        """
        Updates the UI to display the original and colorized images.
        """
        self.filename = name
        self.original_label_title.config(text=f"Original: {self.filename}")
        self.colorized_label_title.config(text=f"Colorized: {self.filename}")
        self._resize_images()
        self._stop_spinner()
        # Enable the save button after successful colorization
        self._update_ui_state(True)

    def _resize_images(self, event=None):
        """
        Resizes images to fit the panels while maintaining aspect ratio.
        """
        # If no image is loaded, show placeholder text
        if self.original_image_array is None or self.colorized_image_array is None:
            self.bw_image_label.config(image='', text="Drag & Drop a B&W Image or click 'Select Image'")
            self.color_image_label.config(image='', text="Colorized Image will appear here")
            return

        # Get the panel dimensions dynamically
        panel_width = self.original_panel.winfo_width() - 20
        panel_height = self.original_panel.winfo_height() - 50 # Account for title label
        if panel_width <= 0 or panel_height <= 0:
            return
        
        # Original image
        bw_pil = Image.fromarray(self.original_image_array)
        bw_width, bw_height = bw_pil.size
        
        ratio = min(panel_width / bw_width, panel_height / bw_height)
        new_bw_width = int(bw_width * ratio)
        new_bw_height = int(bw_height * ratio)

        resized_bw_pil = bw_pil.resize((new_bw_width, new_bw_height), Image.LANCZOS)
        self.bw_img_tk = ImageTk.PhotoImage(resized_bw_pil)
        self.bw_image_label.config(image=self.bw_img_tk)

        # Colorized image
        color_pil = Image.fromarray(cv2.cvtColor(self.colorized_image_array, cv2.COLOR_BGR2RGB))
        color_width, color_height = color_pil.size

        ratio = min(panel_width / color_width, panel_height / color_height)
        new_color_width = int(color_width * ratio)
        new_color_height = int(color_height * ratio)

        resized_color_pil = color_pil.resize((new_color_width, new_color_height), Image.LANCZOS)
        self.color_img_tk = ImageTk.PhotoImage(resized_color_pil)
        self.color_image_label.config(image=self.color_img_tk)

    def _save_image(self):
        """Saves the colorized image to a file."""
        from tkinter import filedialog
        default_filename = f"colorized_{self.filename}" if self.filename else "colorized_image"
        path = filedialog.asksaveasfilename(initialfile=default_filename,
                                            defaultextension=".jpg",
                                            filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")])
        if path:
            try:
                cv2.imwrite(path, self.colorized_image_array)
                Messagebox.showinfo("Saved", f"Image saved to:\n{path}")
            except Exception as e:
                Messagebox.showerror("Error", f"Failed to save image: {e}")

    def _toggle_theme(self):
        """Toggles between the 'darkly' and 'flatly' ttkbootstrap themes."""
        new_theme = "flatly" if self.style.theme.name == "darkly" else "darkly"
        self.style.theme_use(new_theme)
        self._configure_styles()

        icon = "🌞" if new_theme == "flatly" else "🌙"
        self.theme_btn.config(text=icon)

if __name__ == "__main__":
    app = ImageColorizerApp()
    app.mainloop()
