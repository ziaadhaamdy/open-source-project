import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
import threading

from color_conversion import rgb_to_gray, rgb_to_ycbcr
from glcm_features   import glcm_features
from dct_features    import dct_features
from dwt_features    import dwt_features
from lbp_features    import lbp_features
from show_results    import show_results

# ── Color palette (light / white theme) ──────────────────────────────────────
BG        = '#ffffff'   # app background (white)
PANEL     = '#f5f6fa'   # header / status bar background
PANEL_ALT = '#f0f1f6'   # panel column background
BORDER    = '#dcdfe8'   # subtle separators

BTN_BLUE  = '#2f6fed'
BTN_GRAY  = '#6b7280'
BTN_ORG   = '#ef7d2c'
BTN_GRN   = '#2f9e6f'
BTN_PURP  = '#7c5cf0'
BTN_TEAL  = '#17a3b8'
BTN_RED   = '#e6495c'
BTN_ALL   = '#3f4557'

BTN_HOVER_DIM = '#0a0c10'  # darkened overlay feel via activebackground

FG        = '#1c2233'   # main text — dark on light bg
FG_DIM    = '#6b7280'   # secondary text
ACCENT    = '#2f6fed'   # accent — darker blue reads better on white

# Particle colors for the animated background strip
PARTICLE_COLORS = ['#2f6fed', '#7c5cf0', '#17a3b8', '#ef7d2c', '#2f9e6f']

FONT_TITLE  = ('Segoe UI Semibold', 15)
FONT_BTN    = ('Segoe UI', 10, 'bold')
FONT_STATUS = ('Segoe UI', 9)
FONT_PANEL  = ('Segoe UI', 8, 'bold')


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📸  Lumira — Vision Feature Lab")
        self.root.configure(bg=BG)
        self.root.geometry("1300x780")
        self.root.resizable(True, True)

        # State
        self.original = None
        self.gray     = None
        self.ycbcr    = None
        self.glcm_f   = None
        self.dct_f    = None
        self.dwt_f    = None
        self.lbp_f    = None

        self._build_ui()

    # ── UI Construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # Title bar — now an animated canvas with floating particles behind the text
        title_bar = tk.Canvas(self.root, bg=PANEL, height=56, highlightthickness=0, bd=0)
        title_bar.pack(fill='x', side='top')

        # Title text + subtitle drawn as canvas items so they sit above the particles
        title_bar.create_text(18, 28, text="📸  Lumira — Vision Feature Lab",
                               fill=ACCENT, font=FONT_TITLE, anchor='w')
        self._subtitle_id = title_bar.create_text(0, 28, text="GLCM · DCT · DWT · LBP",
                               fill=FG_DIM, font=('Segoe UI', 9), anchor='e')

        self.title_canvas = title_bar
        self.root.update_idletasks()
        self._init_particles()
        self._animate_particles()

        # keep subtitle pinned to the right edge on resize
        title_bar.bind('<Configure>', self._reposition_subtitle)

        # thin accent line under header
        tk.Frame(self.root, bg=ACCENT, height=2).pack(fill='x')

        # Button row
        btn_outer = tk.Frame(self.root, bg=BG)
        btn_outer.pack(fill='x')
        btn_frame = tk.Frame(btn_outer, bg=BG, pady=12)
        btn_frame.pack(padx=14)

        buttons = [
            ("📂  Load Image",    BTN_BLUE, self.load_image),
            ("⬜  → Grayscale",   BTN_GRAY, self.to_gray),
            ("🎨  → YCbCr",       BTN_ORG,  self.to_ycbcr),
            ("📊  GLCM",          BTN_GRN,  lambda: self._run_in_thread(self.run_glcm)),
            ("〰  DCT",           BTN_PURP, lambda: self._run_in_thread(self.run_dct)),
            ("🌊  DWT",           BTN_TEAL, lambda: self._run_in_thread(self.run_dwt)),
            ("🔵  LBP",           BTN_RED,  lambda: self._run_in_thread(self.run_lbp)),
            ("📋  All Results",   BTN_ALL,  lambda: self._run_in_thread(self.run_all)),
        ]

        for label, color, cmd in buttons:
            b = tk.Button(btn_frame, text=label, bg=color, fg='white',
                          font=FONT_BTN, relief='flat', bd=0,
                          padx=14, pady=9, cursor='hand2',
                          activebackground=color, activeforeground=ACCENT,
                          highlightthickness=0,
                          command=cmd)
            b.pack(side='left', padx=5)
            self._add_hover(b, color)

        # Status bar
        self.status_var = tk.StringVar(value="Ready — load an image to begin.")
        status_bar = tk.Frame(self.root, bg=PANEL)
        status_bar.pack(fill='x', side='bottom')
        tk.Label(status_bar, textvariable=self.status_var,
                 bg=PANEL, fg=ACCENT, font=FONT_STATUS,
                 anchor='w', padx=14, pady=6).pack(fill='x')

        # ── 5 matplotlib panels ───────────────────────────────────────────────
        panels_frame = tk.Frame(self.root, bg=BG)
        panels_frame.pack(fill='both', expand=True, padx=10, pady=(4, 10))

        self.figs = []
        self.axes = []
        self.canvases = []
        self.ax_titles = [
            'Original RGB',
            'Gray / YCbCr',
            'Algorithm — Output 1',
            'Algorithm — Output 2',
            'Algorithm — Output 3',
        ]

        for i in range(5):
            col_outer = tk.Frame(panels_frame, bg=BORDER)
            col_outer.pack(side='left', fill='both', expand=True, padx=4)

            col_frame = tk.Frame(col_outer, bg=PANEL_ALT)
            col_frame.pack(fill='both', expand=True, padx=1, pady=1)

            fig = plt.Figure(figsize=(2.6, 5.2), facecolor=PANEL_ALT)
            ax  = fig.add_subplot(111)
            ax.set_facecolor(PANEL_ALT)
            ax.set_title(self.ax_titles[i], color=FG, fontsize=9, fontweight='bold', pad=6)
            ax.axis('off')
            for sp in ax.spines.values():
                sp.set_visible(False)

            canvas = FigureCanvasTkAgg(fig, master=col_frame)
            canvas.get_tk_widget().configure(bg=PANEL_ALT, highlightthickness=0)
            canvas.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=6)
            canvas.draw()

            self.figs.append(fig)
            self.axes.append(ax)
            self.canvases.append(canvas)

    # ── Animated background particles (title bar) ───────────────────────────
    def _init_particles(self):
        import random
        self._particles = []
        w = self.title_canvas.winfo_width() or 1300
        h = 56
        for _ in range(28):
            x = random.uniform(0, w)
            y = random.uniform(0, h)
            r = random.uniform(1.2, 2.8)
            speed = random.uniform(1.4, 3.2)
            color = random.choice(PARTICLE_COLORS)
            pid = self.title_canvas.create_oval(x - r, y - r, x + r, y + r,
                                                 fill=color, outline='', stipple='')
            self._particles.append({'id': pid, 'x': x, 'y': y, 'r': r, 'speed': speed})

    def _animate_particles(self):
        w = self.title_canvas.winfo_width() or 1300
        h = 56
        for p in self._particles:
            p['x'] += p['speed']
            if p['x'] - p['r'] > w:
                p['x'] = -p['r']
            self.title_canvas.coords(p['id'], p['x'] - p['r'], p['y'] - p['r'],
                                      p['x'] + p['r'], p['y'] + p['r'])
        # loop every ~20ms for a lively, energetic drift
        self.root.after(20, self._animate_particles)

    def _reposition_subtitle(self, event=None):
        w = self.title_canvas.winfo_width()
        self.title_canvas.coords(self._subtitle_id, w - 20, 28)

    # ── Small UI helper: hover effect ───────────────────────────────────────
    def _add_hover(self, widget, base_color):
        def lighten(hex_color, factor=1.15):
            hex_color = hex_color.lstrip('#')
            r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            r = min(255, int(r * factor))
            g = min(255, int(g * factor))
            b = min(255, int(b * factor))
            return f'#{r:02x}{g:02x}{b:02x}'

        hover_color = lighten(base_color)
        widget.bind('<Enter>', lambda e: widget.configure(bg=hover_color))
        widget.bind('<Leave>', lambda e: widget.configure(bg=base_color))

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    def _run_in_thread(self, func):
        threading.Thread(target=func, daemon=True).start()

    def _refresh(self, idx):
        self.canvases[idx].draw()

    def _show_img(self, ax_idx, img, title, cmap=None):
        ax = self.axes[ax_idx]
        ax.cla()
        ax.set_facecolor(PANEL_ALT)
        if cmap:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.set_title(title, color=FG, fontsize=9, fontweight='bold', pad=6)
        ax.axis('off')
        self._refresh(ax_idx)

    def _show_plot(self, ax_idx, plot_func, title):
        ax = self.axes[ax_idx]
        ax.cla()
        ax.set_facecolor(PANEL_ALT)
        ax.tick_params(colors=FG_DIM, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        plot_func(ax)
        ax.set_title(title, color=FG, fontsize=9, fontweight='bold', pad=6)
        self._refresh(ax_idx)

    def _clear_panels(self, indices):
        for i in indices:
            ax = self.axes[i]
            ax.cla()
            ax.set_facecolor(PANEL_ALT)
            ax.set_title(self.ax_titles[i], color=FG, fontsize=9, fontweight='bold', pad=6)
            ax.axis('off')
            self._refresh(i)

    def _ensure_gray(self):
        if self.original is None:
            messagebox.showerror("Error", "Please load an image first.")
            return False
        if self.gray is None:
            self.gray = rgb_to_gray(self.original)
            self._show_img(1, self.gray, 'Grayscale', cmap='gray')
        return True

    # ── Button Callbacks ──────────────────────────────────────────────────────
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if not path:
            return
        try:
            pil_img = Image.open(path).convert('RGB')
            self.original = np.array(pil_img)
            self.gray = self.ycbcr = None
            self.glcm_f = self.dct_f = self.dwt_f = self.lbp_f = None
            self._show_img(0, self.original, 'Original RGB')
            self._clear_panels([1, 2, 3, 4])
            self._status(f"Loaded: {path.split('/')[-1]}  |  Size: {self.original.shape[1]}×{self.original.shape[0]}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def to_gray(self):
        if self.original is None:
            messagebox.showerror("Error", "Load an image first."); return
        self._status("Converting to Grayscale...")
        self.gray = rgb_to_gray(self.original)
        self._show_img(1, self.gray, 'Grayscale', cmap='gray')

        def plot_hist(ax):
            for ch, col, lbl in zip(range(3), ['#e57373','#81c784','#4fc3f7'], ['R','G','B']):
                ax.hist(self.original[:,:,ch].ravel(), bins=64, color=col,
                        alpha=0.55, label=lbl, histtype='stepfilled')
            ax.hist(self.gray.ravel(), bins=64, color=FG, alpha=0.35, label='Gray')
            ax.legend(labelcolor=FG, facecolor=PANEL_ALT, fontsize=6)
            ax.set_xlabel('Value', color=FG_DIM, fontsize=7)

        self._show_plot(2, plot_hist, 'RGB + Gray Histogram')
        self._clear_panels([3, 4])
        self._status("✅ Grayscale conversion done.")

    def to_ycbcr(self):
        if self.original is None:
            messagebox.showerror("Error", "Load an image first."); return
        self._status("Converting to YCbCr...")
        self.ycbcr = rgb_to_ycbcr(self.original)
        self._show_img(1, self.ycbcr, 'YCbCr')
        self._show_img(2, self.ycbcr[:,:,0], 'Y  Channel', cmap='gray')
        self._show_img(3, self.ycbcr[:,:,1], 'Cb Channel', cmap='gray')
        self._show_img(4, self.ycbcr[:,:,2], 'Cr Channel', cmap='gray')
        self._status("✅ YCbCr conversion done.")

    def run_glcm(self):
        if not self._ensure_gray(): return
        self._status("Running GLCM... (may take a moment)")
        self.glcm_f = glcm_features(self.gray)
        f = self.glcm_f

        self._show_img(2, f['matrix'], 'GLCM Matrix')

        def bar_plot(ax):
            names = ['Contrast', 'Corr.', 'Energy', 'Homog.']
            vals  = [f['contrast'], f['correlation'], f['energy'], f['homogeneity']]
            ax.bar(names, vals, color=['#4fc3f7','#81c784','#ffb74d','#e57373'])
            ax.tick_params(axis='x', labelrotation=15, colors=FG_DIM, labelsize=7)
            ax.tick_params(axis='y', colors=FG_DIM, labelsize=7)
        self._show_plot(3, bar_plot, 'GLCM Features')

        def text_panel(ax):
            ax.axis('off')
            lines = [
                (f"Contrast:    {f['contrast']:.4f}",    '#4fc3f7'),
                (f"Correlation: {f['correlation']:.4f}", '#81c784'),
                (f"Energy:      {f['energy']:.4f}",      '#ffb74d'),
                (f"Homogeneity: {f['homogeneity']:.4f}", '#e57373'),
            ]
            for i, (txt, col) in enumerate(lines):
                ax.text(0.05, 0.80 - i*0.18, txt, color=col, fontsize=9,
                        transform=ax.transAxes, family='monospace')
        self._show_plot(4, text_panel, 'GLCM Values')
        self._status("✅ GLCM done.")

    def run_dct(self):
        if not self._ensure_gray(): return
        self._status("Running DCT...")
        self.dct_f = dct_features(self.gray)
        f = self.dct_f

        dct_log = np.log(np.abs(f['dct_matrix']) + 1)
        self._show_img(2, dct_log, 'DCT Spectrum (log)')
        self._show_img(3, f['reconstructed'], 'DCT Reconstruct (Low)', cmap='gray')

        def pie_plot(ax):
            te = f['total_energy']
            sizes = [f['low_freq_energy'], f['mid_freq_energy'], f['high_freq_energy']]
            pct = [100*s/te for s in sizes] if te > 0 else [33,33,34]
            ax.pie(pct, labels=['Low','Mid','High'],
                   colors=['#4fc3f7','#ffb74d','#e57373'],
                   autopct='%1.1f%%', textprops={'color': FG, 'fontsize': 8})
        self._show_plot(4, pie_plot, 'DCT Energy %')
        self._status("✅ DCT done.")

    def run_dwt(self):
        if not self._ensure_gray(): return
        self._status("Running DWT...")
        self.dwt_f = dwt_features(self.gray)
        f = self.dwt_f

        # Subband mosaic
        LL1=f['LL1']; LH1=f['LH1']; HL1=f['HL1']; HH1=f['HH1']
        r2, c2 = LL1.shape
        rows, cols = self.gray.shape
        norm = lambda x: (x-x.min())/(x.max()-x.min()+1e-8)
        wdisp = np.zeros((rows, cols))
        wdisp[:r2, :c2]   = norm(LL1)
        wdisp[:r2, c2:]   = norm(np.abs(LH1[:, :cols-c2]))
        wdisp[r2:, :c2]   = norm(np.abs(HL1[:rows-r2, :]))
        wdisp[r2:, c2:]   = norm(np.abs(HH1[:rows-r2, :cols-c2]))
        self._show_img(2, wdisp, 'DWT Level-1 Subbands', cmap='gray')
        self._show_img(3, norm(f['LL2']), 'LL Level-2 Approx', cmap='gray')

        def bar_plot(ax):
            x = np.arange(4)
            w = 0.25
            ax.bar(x-w, f['energy_L1'], w, label='L1', color='#4fc3f7')
            ax.bar(x,   f['energy_L2'], w, label='L2', color='#81c784')
            ax.bar(x+w, f['energy_L3'], w, label='L3', color='#ffb74d')
            ax.set_xticks(x)
            ax.set_xticklabels(['LL','LH','HL','HH'], color=FG_DIM, fontsize=8)
            ax.legend(labelcolor=FG, facecolor=PANEL_ALT, fontsize=7)
            ax.tick_params(colors=FG_DIM)
        self._show_plot(4, bar_plot, 'DWT Subband Energies')
        self._status("✅ DWT done.")

    def run_lbp(self):
        if not self._ensure_gray(): return
        self._status("Running LBP... (may take a moment for large images)")
        self.lbp_f = lbp_features(self.gray)
        f = self.lbp_f

        self._show_img(2, f['lbp_map'], 'LBP Map', cmap='gray')

        def hist_plot(ax):
            ax.plot(f['histogram'], color='#4fc3f7', linewidth=1.2)
            ax.fill_between(range(256), f['histogram'], alpha=0.25, color='#4fc3f7')
            ax.set_xlabel('Pattern', color=FG_DIM, fontsize=7)
            ax.tick_params(colors=FG_DIM, labelsize=7)
        self._show_plot(3, hist_plot, 'LBP Histogram')

        def text_panel(ax):
            ax.axis('off')
            lines = [
                (f"Uniformity:  {f['uniformity']:.4f}",   '#ffb74d'),
                (f"Entropy:     {f['entropy']:.4f} bits",  '#4fc3f7'),
                (f"Mean:        {f['mean_pattern']:.2f}",  '#81c784'),
                (f"Std:         {f['std_pattern']:.2f}",   '#e57373'),
            ]
            for i, (txt, col) in enumerate(lines):
                ax.text(0.05, 0.80 - i*0.18, txt, color=col, fontsize=9,
                        transform=ax.transAxes, family='monospace')
        self._show_plot(4, text_panel, 'LBP Descriptors')
        self._status("✅ LBP done.")

    def run_all(self):
        if not self._ensure_gray(): return
        self._status("Running ALL algorithms — please wait...")
        if not self.glcm_f: self.glcm_f = glcm_features(self.gray)
        if not self.dct_f:  self.dct_f  = dct_features(self.gray)
        if not self.dwt_f:  self.dwt_f  = dwt_features(self.gray)
        if not self.lbp_f:  self.lbp_f  = lbp_features(self.gray)
        show_results(self.original, self.gray, self.glcm_f, self.dct_f, self.dwt_f, self.lbp_f)
        self._status("✅ All results displayed and saved to results_all.png")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    root = tk.Tk()
    app  = ImageApp(root)
    root.mainloop()