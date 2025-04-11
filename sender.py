import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading, logging

from protocol import DEFAULT_CONFIG, encode_message, modulate_bits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("matplotlib").setLevel(logging.INFO)

class SenderGUI:
    def __init__(self, master):
        # initialize GUI: text input, sent messages log, parameter controls, spectrum plot, status label
        self.master = master
        master.title("Acoustic Modem Sender")
        self.config = DEFAULT_CONFIG
        
        # set up vars that will be used in the UI so that they exist before any UI elements try to access them
        self.volume_var = tk.DoubleVar(value=self.config.volume)
        self.symdur_var = tk.DoubleVar(value=self.config.symbol_duration)
        self.freq0_var = tk.DoubleVar(value=self.config.freq0)
        self.freq1_var = tk.DoubleVar(value=self.config.freq1)
        
        # new 4-FSK variables
        self.use_mary_var = tk.BooleanVar(value=self.config.use_mary_fsk)
        self.freq00_var = tk.DoubleVar(value=self.config.freq00)
        self.freq01_var = tk.DoubleVar(value=self.config.freq01)
        self.freq10_var = tk.DoubleVar(value=self.config.freq10)
        self.freq11_var = tk.DoubleVar(value=self.config.freq11)
        
        # UI elements
        tk.Label(master, text="Enter message:").pack(pady=5)
        self.message_text = tk.Text(master, height=4, width=50)
        self.message_text.pack(pady=5)
        
        tk.Label(master, text="Sent messages:").pack(pady=(10,0))
        self.sent_log = tk.Text(master, height=8, width=50,
                            state="disabled", bg="black", fg="white")
        self.sent_log.pack(pady=5)
        
        param_frame = tk.Frame(master)
        param_frame.pack(pady=5)
        
        # original controls (first two rows)
        tk.Label(param_frame, text="Volume (0-1):").grid(row=0, column=0, sticky="e")
        tk.Entry(param_frame, textvariable=self.volume_var, width=5).grid(row=0, column=1, padx=5)
        
        tk.Label(param_frame, text="Symbol Duration (s):").grid(row=0, column=2, sticky="e")
        tk.Entry(param_frame, textvariable=self.symdur_var, width=5).grid(row=0, column=3, padx=5)
        
        tk.Label(param_frame, text="Freq0 (Hz):").grid(row=1, column=0, sticky="e")
        tk.Entry(param_frame, textvariable=self.freq0_var, width=5).grid(row=1, column=1, padx=5)
        
        tk.Label(param_frame, text="Freq1 (Hz):").grid(row=1, column=2, sticky="e")
        tk.Entry(param_frame, textvariable=self.freq1_var, width=5).grid(row=1, column=3, padx=5)
        
        # 4-FSK controls
        tk.Label(param_frame, text="Use 4-FSK:").grid(row=2, column=0, sticky="e")
        tk.Checkbutton(param_frame, variable=self.use_mary_var).grid(row=2, column=1, padx=5)
        
        tk.Label(param_frame, text="Freq00 (Hz):").grid(row=3, column=0, sticky="e")
        tk.Entry(param_frame, textvariable=self.freq00_var, width=5).grid(row=3, column=1, padx=5)
        
        tk.Label(param_frame, text="Freq01 (Hz):").grid(row=3, column=2, sticky="e")
        tk.Entry(param_frame, textvariable=self.freq01_var, width=5).grid(row=3, column=3, padx=5)
        
        tk.Label(param_frame, text="Freq10 (Hz):").grid(row=4, column=0, sticky="e")
        tk.Entry(param_frame, textvariable=self.freq10_var, width=5).grid(row=4, column=1, padx=5)
        
        tk.Label(param_frame, text="Freq11 (Hz):").grid(row=4, column=2, sticky="e")
        tk.Entry(param_frame, textvariable=self.freq11_var, width=5).grid(row=4, column=3, padx=5)
        
        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.pack(pady=5)
        
        self.progress = ttk.Progressbar(master, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(5,2))
        self.ax.set_title("Signal Spectrum")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Magnitude")
        self.ax.set_xlim(0, 2000)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(pady=5)
        
        self.status_label = tk.Label(master, text="Ready")
        self.status_label.pack(pady=5)
    

    def update_config(self):
        # update modem config using input parameters from GUI
        try:
            # original params
            self.config.volume = float(self.volume_var.get())
            self.config.symbol_duration = float(self.symdur_var.get())
            self.config.N = int(self.config.sample_rate * self.config.symbol_duration)
            self.config.freq0 = float(self.freq0_var.get())
            self.config.freq1 = float(self.freq1_var.get())
            
            # 4-FSK params
            self.config.use_mary_fsk = self.use_mary_var.get()
            
            # only update 4-FSK frequencies if the feature is enabled
            if self.config.use_mary_fsk:
                self.config.freq00 = float(self.freq00_var.get())
                self.config.freq01 = float(self.freq01_var.get())
                self.config.freq10 = float(self.freq10_var.get())
                self.config.freq11 = float(self.freq11_var.get())
                
                # keep these in sync for binary FSK backward compatibility
                self.config.freq0 = self.config.freq00
                self.config.freq1 = self.config.freq11
        except Exception as e:
            messagebox.showerror("Parameter Error", str(e))

    def send_message(self):
        # read text, update config, encode and modulate message, update spectrum, and start audio playback
        msg = self.message_text.get("1.0", tk.END).strip()
        if not msg:
            messagebox.showwarning("No Message", "Please enter some text.")
            return
        self.update_config()
        self.status_label.config(text="Encoding message...")
        bits = encode_message(msg, self.config)
        waveform = modulate_bits(bits, self.config)
        self.plot_spectrum(waveform)
        threading.Thread(
            target=self.play_audio,
            args=(waveform, len(waveform)/self.config.sample_rate),
            daemon=True
        ).start()
        self.message_text.delete("1.0", tk.END)
        self.sent_log.config(state="normal")
        self.sent_log.insert(tk.END, msg + "\n")
        self.sent_log.config(state="disabled")
        self.sent_log.see(tk.END)

    def play_audio(self, waveform, duration):
        # play the waveform using a callback to output stream & update progress bar
        self.status_label.config(text="Transmitting...")
        total_samples = len(waveform)
        def callback(outdata, frames, time_info, status):
            nonlocal waveform
            if status:
                logging.warning(f"OutputStream status: {status}")
            start = callback.current
            end = start + frames
            chunk = waveform[start:end]
            if len(chunk) < frames:
                chunk = np.pad(chunk, (0, frames - len(chunk)), 'constant')
            outdata[:] = chunk.reshape(-1, 1)
            callback.current += frames
            pct = 100 * (callback.current / total_samples)
            self.progress["value"] = pct
            self.master.update_idletasks()
            if callback.current >= total_samples:
                raise sd.CallbackStop
        callback.current = 0
        try:
            with sd.OutputStream(channels=1, samplerate=self.config.sample_rate, callback=callback):
                sd.sleep(int(duration * 1000) + 200)
        except Exception as e:
            logging.error(f"Audio playback error: {e}")
        self.progress["value"] = 100
        self.status_label.config(text="Transmission complete!")

    def plot_spectrum(self, waveform):
        # compute FFT of waveform and update spectrum plot with vertical lines at target frequencies
        N = len(waveform)
        window = np.hanning(N)
        fft_vals = np.fft.rfft(waveform * window)
        fft_freqs = np.fft.rfftfreq(N, 1.0 / self.config.sample_rate)
        magnitude = np.abs(fft_vals)
        self.ax.clear()
        self.ax.plot(fft_freqs, magnitude)
        self.ax.set_title("Signal Spectrum")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Magnitude")
        self.ax.set_xlim(0, 2000)
        self.ax.axvline(self.config.freq0, color='red', linestyle='--', label="Freq0")
        self.ax.axvline(self.config.freq1, color='green', linestyle='--', label="Freq1")
        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = SenderGUI(root)
    root.mainloop()