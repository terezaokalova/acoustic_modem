#!/usr/bin/env python3
# listener.py


import tkinter as tk
from tkinter import ttk
import numpy as np
import sounddevice as sd
import matplotlib
import scipy.signal
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading, time, queue, logging
from protocol import DEFAULT_CONFIG, StreamingDecoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("matplotlib").setLevel(logging.INFO)

class ListenerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Acoustic Modem Listener") 
        self.config = DEFAULT_CONFIG
        tk.Label(master, text="Decoded Messages:").pack(pady=(5, 0))
        # text area for displaying decoded messages
        self.text_area = tk.Text(master, height=10, width=60, state="disabled", bg="black", fg="white")
        self.text_area.pack(pady=5)
        self.listen_button = tk.Button(master, text="Start Listening", command=self.toggle_listening)
        self.listen_button.pack(pady=5)
        # plotting incoming audio spectrum
        self.fig, self.ax = plt.subplots(figsize=(5, 2))
        self.ax.set_title("Incoming Audio Spectrum")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Magnitude")
        self.ax.set_xlim(0, 2000)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(pady=5)
        self.status_label = tk.Label(master, text="Not Listening")
        self.status_label.pack(pady=5)
        self.listening = False
        # queue for storing incoming audio samples
        self.audio_queue = queue.Queue()
        self.decoder = StreamingDecoder(self.config)
        self.stream = None
        # thread to update the GUI periodically
        self.update_thread = threading.Thread(target=self.gui_update_loop, daemon=True)
        self.update_thread.start()

    # process audio by applying a bandpass filter
    def process_audio(self, samples):
        # calculate Nyquist frequency
        nyquist = self.config.sample_rate / 2
        low_cutoff = 500 / nyquist
        high_cutoff = 1700 / nyquist
        b, a = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        filtered_samples = scipy.signal.filtfilt(b, a, samples)
        return filtered_samples

    # toggle between starting and stopping listening
    def toggle_listening(self):
        if not self.listening:
            self.start_listening()
            self.listen_button.config(text="Stop Listening")
        else:
            self.stop_listening()
            self.listen_button.config(text="Start Listening")

    # start listening to the input stream
    def start_listening(self):
        self.listening = True
        self.status_label.config(text="Listening")
        device_id = None
        # create an input stream with a callback
        self.stream = sd.InputStream(device=device_id, channels=1,
                                     samplerate=self.config.sample_rate,
                                     blocksize=self.config.N // 2,
                                     callback=self.audio_callback)
        self.stream.start()

    # stop the input stream and listening
    def stop_listening(self):
        self.listening = False
        self.status_label.config(text="Not Listening")
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    # callback function for the input stream to put audio samples into the queue
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            logging.warning(f"Inputstream Status: {status}")
        samples = indata[:, 0].copy()
        self.audio_queue.put(samples)

    # update the GUI based on the audio input and decoded messages
    def gui_update_loop(self):
        while True:
            try:
                while not self.audio_queue.empty():
                    chunk = self.audio_queue.get_nowait()
                    # adjust filter if using 8-fsk mode, else process normally
                    if self.config.use_mary_fsk and self.config.fsk_mode == 8:
                        filtered = self.adjust_filter(chunk, high=2700)
                    else:
                        filtered = self.process_audio(chunk)
                    # process filtered samples with the decoder
                    self.decoder.process_samples(filtered)
                    self.update_spectrum(chunk)
                msgs = self.decoder.get_messages()
                if msgs:
                    for m in msgs:
                        self.display_message(m)
                        self.master.update_idletasks()
                        logging.info(f"Displayed Message in GUI: {m}")
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"Gui_update_loop Error: {e}")

    def display_message(self, msg: str):
        self.text_area.config(state="normal")
        self.text_area.insert(tk.END, msg + "\n")
        self.text_area.config(state="disabled")
        self.text_area.see(tk.END)

    # adjust filter for a given high cutoff freq
    def adjust_filter(self, samples, high):
        nyquist = self.config.sample_rate / 2
        low_cutoff = 500 / nyquist
        high_cutoff = high / nyquist
        b, a = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        return scipy.signal.filtfilt(b, a, samples)

    # update spectrum plot with the new audio samples
    def update_spectrum(self, samples: np.ndarray):
        N = len(samples)
        if N == 0:
            return
        window = np.hanning(N)
        fft_vals = np.fft.rfft(samples * window)
        fft_freqs = np.fft.rfftfreq(N, 1.0 / self.config.sample_rate)
        magnitude = np.abs(fft_vals)
        self.ax.clear()
        self.ax.plot(fft_freqs, magnitude)
        self.ax.set_title("Incoming Audio Spectrum")
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
            self.ax.axvline(self.config.freq0, color="red", linestyle="--")
            self.ax.axvline(self.config.freq1, color="green", linestyle="--")
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ListenerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_listening)
    root.mainloop()
