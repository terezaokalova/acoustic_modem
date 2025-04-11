#!/usr/bin/env python3
# sender.py

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
        self.master = master
        master.title("Acoustic Modem Sender")
        self.config = DEFAULT_CONFIG
        self.volume_var = tk.DoubleVar(value=self.config.volume)
        self.symdur_var = tk.DoubleVar(value=self.config.symbol_duration)
        self.freq0_var = tk.DoubleVar(value=self.config.freq0)
        self.freq1_var = tk.DoubleVar(value=self.config.freq1)
        # FSK mode selection - 8 for 8-fsk, 4 for 4-fsk
        self.fsk_mode_var = tk.IntVar(value=8)
        # 4-FSK parameters
        self.freq00_var = tk.DoubleVar(value=self.config.freq00)
        self.freq01_var = tk.DoubleVar(value=self.config.freq01)
        self.freq10_var = tk.DoubleVar(value=self.config.freq10)
        self.freq11_var = tk.DoubleVar(value=self.config.freq11)
        # 8-FSK parameters
        self.freq000_var = tk.DoubleVar(value=self.config.freq000)
        self.freq001_var = tk.DoubleVar(value=self.config.freq001)
        self.freq010_var = tk.DoubleVar(value=self.config.freq010)
        self.freq011_var = tk.DoubleVar(value=self.config.freq011)
        self.freq100_var = tk.DoubleVar(value=self.config.freq100)
        self.freq101_var = tk.DoubleVar(value=self.config.freq101)
        self.freq110_var = tk.DoubleVar(value=self.config.freq110)
        self.freq111_var = tk.DoubleVar(value=self.config.freq111)
        # UI elements
        tk.Label(master, text="Enter Message:").pack(pady=5)
        self.message_text = tk.Text(master, height=4, width=50)
        self.message_text.pack(pady=5)
        tk.Label(master, text="Sent Messages:").pack(pady=(10,0))
        self.sent_log = tk.Text(master, height=8, width=50, state="disabled", bg="black", fg="white")
        self.sent_log.pack(pady=5)
        # notebook for params
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(pady=5, fill="both", expand=True)
        basic_frame = ttk.Frame(self.notebook)
        self.notebook.add(basic_frame, text="Basic Parameters")
        fsk4_frame = ttk.Frame(self.notebook)
        self.notebook.add(fsk4_frame, text="4-FSK")
        fsk8_frame = ttk.Frame(self.notebook)
        self.notebook.add(fsk8_frame, text="8-FSK")
        tk.Label(basic_frame, text="Volume (0-1):").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(basic_frame, textvariable=self.volume_var, width=5).grid(row=0, column=1, padx=5, pady=5)
        tk.Label(basic_frame, text="Symbol Duration (s):").grid(row=0, column=2, sticky="e", padx=5, pady=5)
        tk.Entry(basic_frame, textvariable=self.symdur_var, width=5).grid(row=0, column=3, padx=5, pady=5)
        tk.Label(basic_frame, text="Freq0 (Hz):").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(basic_frame, textvariable=self.freq0_var, width=5).grid(row=1, column=1, padx=5, pady=5)
        tk.Label(basic_frame, text="Freq1 (Hz):").grid(row=1, column=2, sticky="e", padx=5, pady=5)
        tk.Entry(basic_frame, textvariable=self.freq1_var, width=5).grid(row=1, column=3, padx=5, pady=5)
        # mode selection
        mode_frame = tk.Frame(basic_frame)
        mode_frame.grid(row=2, column=0, columnspan=4, pady=10)
        tk.Label(mode_frame, text="FSK Mode:").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="4-FSK", variable=self.fsk_mode_var, value=4,
                       command=self.update_fsk_mode).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(mode_frame, text="8-FSK", variable=self.fsk_mode_var, value=8,
                       command=self.update_fsk_mode).pack(side=tk.LEFT, padx=10)
        # 4-FSK params
        tk.Label(fsk4_frame, text="Freq00 (Hz):").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(fsk4_frame, textvariable=self.freq00_var, width=7).grid(row=0, column=1, padx=5, pady=5)
        tk.Label(fsk4_frame, text="Freq01 (Hz):").grid(row=0, column=2, sticky="e", padx=5, pady=5)
        tk.Entry(fsk4_frame, textvariable=self.freq01_var, width=7).grid(row=0, column=3, padx=5, pady=5)
        tk.Label(fsk4_frame, text="Freq10 (Hz):").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(fsk4_frame, textvariable=self.freq10_var, width=7).grid(row=1, column=1, padx=5, pady=5)
        tk.Label(fsk4_frame, text="Freq11 (Hz):").grid(row=1, column=2, sticky="e", padx=5, pady=5)
        tk.Entry(fsk4_frame, textvariable=self.freq11_var, width=7).grid(row=1, column=3, padx=5, pady=5)
        # 8-fsk params
        tk.Label(fsk8_frame, text="Freq000 (Hz):").grid(row=0, column=0, sticky="e", padx=5, pady=3)
        tk.Entry(fsk8_frame, textvariable=self.freq000_var, width=7).grid(row=0, column=1, padx=5, pady=3)
        tk.Label(fsk8_frame, text="Freq001 (Hz):").grid(row=0, column=2, sticky="e", padx=5, pady=3)
        tk.Entry(fsk8_frame, textvariable=self.freq001_var, width=7).grid(row=0, column=3, padx=5, pady=3)
        tk.Label(fsk8_frame, text="Freq010 (Hz):").grid(row=1, column=0, sticky="e", padx=5, pady=3)
        tk.Entry(fsk8_frame, textvariable=self.freq010_var, width=7).grid(row=1, column=1, padx=5, pady=3)
        tk.Label(fsk8_frame, text="Freq011 (Hz):").grid(row=1, column=2, sticky="e", padx=5, pady=3)
        tk.Entry(fsk8_frame, textvariable=self.freq011_var, width=7).grid(row=1, column=3, padx=5, pady=3)
        tk.Label(fsk8_frame, text="Freq100 (Hz):").grid(row=2, column=0, sticky="e", padx=5, pady=3)
        tk.Entry(fsk8_frame, textvariable=self.freq100_var, width=7).grid(row=2, column=1, padx=5, pady=3)
        tk.Label(fsk8_frame, text="Freq101 (Hz):").grid(row=2, column=2, sticky="e", padx=5, pady=3)
        tk.Entry(fsk8_frame, textvariable=self.freq101_var, width=7).grid(row=2, column=3, padx=5, pady=3)
        tk.Label(fsk8_frame, text="Freq110 (Hz):").grid(row=3, column=0, sticky="e", padx=5, pady=3)
        tk.Entry(fsk8_frame, textvariable=self.freq110_var, width=7).grid(row=3, column=1, padx=5, pady=3)
        tk.Label(fsk8_frame, text="Freq111 (Hz):").grid(row=3, column=2, sticky="e", padx=5, pady=3)
        tk.Entry(fsk8_frame, textvariable=self.freq111_var, width=7).grid(row=3, column=3, padx=5, pady=3)
        # send button and progress bar
        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.pack(pady=5)
        self.progress = ttk.Progressbar(master, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=5)
        # plotting spectrum
        self.fig, self.ax = plt.subplots(figsize=(5, 2))
        self.ax.set_title("Signal Spectrum")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Magnitude")
        self.ax.set_xlim(0, 3000)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(pady=5)
        self.status_label = tk.Label(master, text="Ready")
        self.status_label.pack(pady=5)

    # update FSK mode notebook visuals based on selection
    def update_fsk_mode(self):
        mode = self.fsk_mode_var.get()
        if mode == 4:
            self.notebook.select(1)
        elif mode == 8:
            self.notebook.select(2)

    # update modem configuration from GUI controls
    def update_config(self):
        try:
            # force use of M-ary FSK with 8-FSK settings
            self.config.use_mary_fsk = True
            self.config.fsk_mode = self.fsk_mode_var.get()
            self.config._update_bits_per_symbol()
            self.config.volume = float(self.volume_var.get())
            self.config.symbol_duration = float(self.symdur_var.get())
            self.config.N = int(self.config.sample_rate * self.config.symbol_duration)
            # update binary FSK frequencies
            self.config.freq0 = float(self.freq0_var.get())
            self.config.freq1 = float(self.freq1_var.get())
            if self.config.fsk_mode == 4:
                self.config.freq00 = float(self.freq00_var.get())
                self.config.freq01 = float(self.freq01_var.get())
                self.config.freq10 = float(self.freq10_var.get())
                self.config.freq11 = float(self.freq11_var.get())
                self.config.freq0 = self.config.freq00
                self.config.freq1 = self.config.freq11
            elif self.config.fsk_mode == 8:
                self.config.freq000 = float(self.freq000_var.get())
                self.config.freq001 = float(self.freq001_var.get())
                self.config.freq010 = float(self.freq010_var.get())
                self.config.freq011 = float(self.freq011_var.get())
                self.config.freq100 = float(self.freq100_var.get())
                self.config.freq101 = float(self.freq101_var.get())
                self.config.freq110 = float(self.freq110_var.get())
                self.config.freq111 = float(self.freq111_var.get())
                self.config.freq0 = self.config.freq000
                self.config.freq1 = self.config.freq111
        except Exception as e:
            messagebox.showerror("Parameter Error", str(e))

    # send message by encoding, modulating, and playing audio
    def send_message(self):
        msg = self.message_text.get("1.0", tk.END).strip()
        if not msg:
            messagebox.showwarning("No Message", "Please enter some text")
            return
        self.update_config()
        self.status_label.config(text="Encoding Message")
        bits = encode_message(msg, self.config)
        waveform = modulate_bits(bits, self.config)
        self.plot_spectrum(waveform)
        threading.Thread(target=self.play_audio, args=(waveform, len(waveform)/self.config.sample_rate), daemon=True).start()
        self.message_text.delete("1.0", tk.END)
        self.sent_log.config(state="normal")
        self.sent_log.insert(tk.END, msg + "\n")
        self.sent_log.config(state="disabled")
        self.sent_log.see(tk.END)

    # play the modulated audio using sd output stream
    def play_audio(self, waveform, duration):
        self.status_label.config(text="Transmitting")
        total_samples = len(waveform)
        # define callback for streaming output
        def callback(outdata, frames, time_info, status):
            nonlocal waveform
            if status:
                logging.warning(f"Outputstream Status: {status}")
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
            logging.error(f"Audio Playback Error: {e}")
        self.progress["value"] = 100
        self.status_label.config(text="Transmission Complete")
        
    # generated waveform spectrum
    def plot_spectrum(self, waveform):
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
        if self.config.use_mary_fsk and self.config.fsk_mode == 8:
            self.ax.set_xlim(0, 3000)
            colors = ['red', 'green', 'blue', 'purple', 'orange', 'magenta', 'gray', 'brown']
            freqs = [self.config.freq000, self.config.freq001, self.config.freq010, self.config.freq011,
                     self.config.freq100, self.config.freq101, self.config.freq110, self.config.freq111]
            for col, freq in zip(colors, freqs):
                self.ax.axvline(freq, color=col, linestyle="--")
        else:
            self.ax.set_xlim(0, 2000)
            self.ax.axvline(self.config.freq0, color='red', linestyle='--')
            self.ax.axvline(self.config.freq1, color='green', linestyle='--')
        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = SenderGUI(root)
    root.mainloop()
