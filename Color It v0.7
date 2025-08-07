import os
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageFilter
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from ttkbootstrap import ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
import threading

# Model paths
PROTOTXT = r"C:\Users\AYUSH\Desktop\Projects\model\colorization_deploy_v2.prototxt"
POINTS = r"C:\Users\AYUSH\Desktop\Projects\model\pts_in_hull.npy"
MODEL = r"C:\Users\AYUSH\Desktop\Projects\model\colorization_release_v2.caffemodel"

# Load model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)
pts = pts.transpose().reshape(2, 313, 1, 1)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Globals
colorized_image = None
filename = ""

# Splash screen setup
def show_splash():
    splash = tb.Toplevel()
    splash.geometry("400x300")
    splash.overrideredirect(True)
    splash.update_idletasks()

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
        app.deiconify()

    splash.after(2000, close)

# GUI setup
app = TkinterDnD.Tk()
style = tb.Style("darkly")
app.title("Black & White to Color Image Colorizer")
app.geometry("1000x700")
app.resizable(True, True)
app.withdraw()
show_splash()

# Functions
def start_spinner():
    progress_label.config(text="Colorizing...", bootstyle="warning")
    progressbar.start(10)
    app.update_idletasks()

def stop_spinner():
    progressbar.stop()
    progress_label.config(text="Done", bootstyle="success")

def select_image_from_dialog():
    from tkinter import filedialog
    file = filedialog.askopenfilename(title="Select Black & White Image",
                                      filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
    if file:
        load_image(file)

def load_image(path):
    global colorized_image, filename
    filename = os.path.basename(path)
    image = cv2.imread(path)
    if image is None:
        tb.dialogs.messagebox.showerror("Error", "Failed to load image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    start_spinner()
    app.after(50, lambda: colorize_and_display(image, gray_rgb, filename))

def colorize_and_display(image, gray_rgb, name):
    global colorized_image
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    colorized_image = colorized

    show_images(gray_rgb, colorized, name)
    stop_spinner()

def show_images(bw_array, color_array, name):
    for widget in img_frame.winfo_children():
        widget.destroy()

    bw_pil = Image.fromarray(bw_array)
    color_pil = Image.fromarray(cv2.cvtColor(color_array, cv2.COLOR_BGR2RGB))

    bw_img = ImageTk.PhotoImage(bw_pil)
    color_img = ImageTk.PhotoImage(color_pil)

    side_frame = tb.Frame(img_frame)
    side_frame.pack(pady=10, fill=X)

    tb.Label(side_frame, text=f"Original: {name}", bootstyle="info").grid(row=0, column=0, padx=10, pady=5)
    tb.Label(side_frame, text=f"Colorized: {name}", bootstyle="success").grid(row=0, column=1, padx=10, pady=5)

    tb.Label(side_frame, image=bw_img).grid(row=1, column=0, padx=10)
    tb.Label(side_frame, image=color_img).grid(row=1, column=1, padx=10)

    img_frame.bw_img = bw_img
    img_frame.color_img = color_img

def save_image():
    from tkinter import filedialog
    if colorized_image is None:
        tb.dialogs.messagebox.showerror("Error", "No image to save.")
        return
    path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                        filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")])
    if path:
        cv2.imwrite(path, colorized_image)
        tb.dialogs.messagebox.showinfo("Saved", f"Image saved to:\n{path}")

def toggle_theme():
    new_theme = "flatly" if style.theme.name == "darkly" else "darkly"
    style.theme_use(new_theme)
    icon = "🌞" if new_theme == "flatly" else "🌙"
    theme_btn.config(text=icon)

# Top buttons
btn_frame = tb.Frame(app)
btn_frame.pack(pady=10)

select_btn = tb.Button(btn_frame, text="📂 Select Image", bootstyle="primary-outline", command=select_image_from_dialog)
select_btn.pack(side=LEFT, padx=5)

save_btn = tb.Button(btn_frame, text="💾 Save Image", bootstyle="success-outline", command=save_image)
save_btn.pack(side=LEFT, padx=5)

theme_btn = tb.Button(btn_frame, text="🌙", width=3, bootstyle="info-outline", command=toggle_theme)
theme_btn.pack(side=LEFT, padx=5)

# Progress bar
progressbar = ttk.Progressbar(app, orient="horizontal", length=500, mode="indeterminate")
progressbar.pack(pady=5)
progress_label = tb.Label(app, text="", bootstyle="info")
progress_label.pack()

# Scrollable output frame
main_frame = tb.Frame(app)
main_frame.pack(fill=BOTH, expand=True)

canvas = tb.Canvas(main_frame, highlightthickness=0)
y_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
x_scrollbar = ttk.Scrollbar(main_frame, orient="horizontal", command=canvas.xview)
canvas.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set, yscrollincrement=10, xscrollincrement=10)

y_scrollbar.pack(side=RIGHT, fill=Y)
x_scrollbar.pack(side=BOTTOM, fill=X)
canvas.pack(side=LEFT, fill=BOTH, expand=True)

scrollable_window = tb.Frame(canvas)
canvas_window = canvas.create_window((0, 0), window=scrollable_window, anchor="nw")

scrollable_window.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
canvas.bind_all("<Shift-MouseWheel>", lambda e: canvas.xview_scroll(int(-1 * (e.delta / 120)), "units"))

img_frame = tb.Frame(scrollable_window)
img_frame.pack(pady=20, fill=X)

# Drag and drop
app.drop_target_register(DND_FILES)
app.dnd_bind("<<Drop>>", lambda e: load_image(e.data.strip('{}')))

app.mainloop()
