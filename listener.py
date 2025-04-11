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
        # setup GUI for listening: message display, Start/Stop button, and spectrum plot
        self.master = master
        master.title("Acoustic Modem Listener")
        self.config = DEFAULT_CONFIG
        tk.Label(master, text="Decoded messages:").pack(pady=(5,0))
        self.text_area = tk.Text(
            master,
            height=10,
            width=60,
            state="disabled",
            bg="black",
            fg="white"
        )
        self.text_area.pack(pady=5)
        self.listen_button = tk.Button(master, text="Start Listening", command=self.toggle_listening)
        self.listen_button.pack(pady=5)
        self.fig, self.ax = plt.subplots(figsize=(5,2))
        self.ax.set_title("Incoming Audio Spectrum")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Magnitude")
        self.ax.set_xlim(0, 2000)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(pady=5)
        self.status_label = tk.Label(master, text="Not Listening")
        self.status_label.pack(pady=5)
        self.listening = False
        self.audio_queue = queue.Queue()
        self.decoder = StreamingDecoder(self.config)
        # self.volume_var = tk.DoubleVar(value=self.config.volume)
        self.stream = None
        self.update_thread = threading.Thread(target=self.gui_update_loop, daemon=True)
        self.update_thread.start()

    def process_audio(self, samples):
        # apply a 4th order Butterworth bandpass filter (500-1700Hz) to incoming samples
        nyquist = self.config.sample_rate / 2
        low_cutoff = 500 / nyquist  # just below 600Hz
        high_cutoff = 1700 / nyquist  # just above 1600Hz
        b, a = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        filtered_samples = scipy.signal.filtfilt(b, a, samples)
        return filtered_samples
    
    def toggle_listening(self):
        # switch listening state and update button label
        if not self.listening:
            self.start_listening()
            self.listen_button.config(text="Stop Listening")
        else:
            self.stop_listening()
            self.listen_button.config(text="Start Listening")

    def start_listening(self):
        # query devices, start audio input stream, update status label
        self.listening = True
        self.status_label.config(text="Listening...")
        devices = sd.query_devices()
        logging.info(f"Available audio devices:")
        for i, device in enumerate(devices):
            logging.info(f"Device {i}: {device['name']}")
        device_id = None  # default device (i'm using Mac)
        self.stream = sd.InputStream(
            device=device_id,  
            channels=1,
            samplerate=self.config.sample_rate,
            blocksize=self.config.N // 2,
            callback=self.audio_callback
        )
        self.stream.start()

    def stop_listening(self):
        # stop audio stream and update status label
        self.listening = False
        self.status_label.config(text="Not Listening")
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def audio_callback(self, indata, frames, time_info, status):
        # input callback: copy audio samples and enqueue them for processing
        if status:
            logging.warning(f"InputStream status: {status}")
        samples = indata[:, 0].copy()
        self.audio_queue.put(samples)

    def gui_update_loop(self):
        # continuously process audio queue, update spectrum plot, and show decoded messages
        while True:
            try:
                while not self.audio_queue.empty():
                    chunk = self.audio_queue.get_nowait()
                    self.decoder.process_samples(chunk)
                    self.update_spectrum(chunk)
                msgs = self.decoder.get_messages()
                if msgs:
                    for m in msgs:
                        self.display_message(m)
                        self.master.update_idletasks()
                        logging.info(f"Displayed message in GUI: '{m}'")
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"gui_update_loop error: {e}")

    def display_message(self, msg: str):
        # append the decoded message to the text area
        self.text_area.config(state="normal")
        self.text_area.insert(tk.END, msg + "\n")
        self.text_area.config(state="disabled")
        self.text_area.see(tk.END)

    # higher modes for FSK
    # def update_spectrum(self, samples: np.ndarray):
    #     N = len(samples)
    #     if N == 0:
    #         return
    #     window = np.hanning(N)
    #     fft_vals = np.fft.rfft(samples * window)
    #     fft_freqs = np.fft.rfftfreq(N, 1.0 / self.config.sample_rate)
    #     magnitude = np.abs(fft_vals)
    #     self.ax.clear()
    #     self.ax.plot(fft_freqs, magnitude)
    #     self.ax.set_title("Incoming Audio Spectrum")
    #     self.ax.set_xlabel("Frequency (Hz)")
    #     self.ax.set_ylabel("Magnitude")
        
    #     # Adjust xlim based on FSK mode
    #     if self.config.fsk_mode == 16:
    #         self.ax.set_xlim(0, 3500)  # Higher range for 16-FSK
    #     else:
    #         self.ax.set_xlim(0, 2500)  # Standard range
        
    #     # Add vertical lines for each frequency based on FSK mode
    #     if not self.config.use_mary_fsk:
    #         # Binary FSK
    #         self.ax.axvline(self.config.freq0, color="red", linestyle="--", label="Freq0")
    #         self.ax.axvline(self.config.freq1, color="green", linestyle="--", label="Freq1")
        
    #     elif self.config.fsk_mode == 4:
    #         # 4-FSK
    #         self.ax.axvline(self.config.freq00, color="red", linestyle="--", label="Freq00")
    #         self.ax.axvline(self.config.freq01, color="green", linestyle="--", label="Freq01")
    #         self.ax.axvline(self.config.freq10, color="blue", linestyle="--", label="Freq10")
    #         self.ax.axvline(self.config.freq11, color="purple", linestyle="--", label="Freq11")
        
    #     elif self.config.fsk_mode == 8:
    #         # 8-FSK
    #         colors = ["red", "orange", "yellow", "green", "blue", "indigo", "violet", "purple"]
    #         freqs = [
    #             self.config.freq000, self.config.freq001, self.config.freq010, self.config.freq011,
    #             self.config.freq100, self.config.freq101, self.config.freq110, self.config.freq111
    #         ]
    #         for i, (freq, color) in enumerate(zip(freqs, colors)):
    #             label = f"Freq{i:03b}"
    #             self.ax.axvline(freq, color=color, linestyle="--", label=label)
        
    #     elif self.config.fsk_mode == 16:
    #         # 16-FSK - Show only a few key frequencies to avoid cluttering
    #         freqs = self.config.freq_16fsk
    #         markers = [0, 5, 10, 15]  # Selected indices to mark
    #         colors = ["red", "green", "blue", "purple"]
            
    #         for i, color in zip(markers, colors):
    #             label = f"F{i:04b}"
    #             self.ax.axvline(freqs[i], color=color, linestyle="--", label=label)
        
    #     self.ax.legend(fontsize='small', ncol=2 if self.config.fsk_mode >= 8 else 1)
    #     self.canvas.draw()
    
    # 4-FSK only
    def update_spectrum(self, samples: np.ndarray):
        # compute FFT of the latest audio chunk and update plot with target frequency markers
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
        self.ax.set_xlim(0, 2000)
        
        if self.config.use_mary_fsk:
            self.ax.axvline(self.config.freq00, color="red", linestyle="--", label="Freq00")
            self.ax.axvline(self.config.freq01, color="green", linestyle="--", label="Freq01")
            self.ax.axvline(self.config.freq10, color="blue", linestyle="--", label="Freq10")
            self.ax.axvline(self.config.freq11, color="purple", linestyle="--", label="Freq11")
        else:
            self.ax.axvline(self.config.freq0, color="red", linestyle="--", label="Freq0")
            self.ax.axvline(self.config.freq1, color="green", linestyle="--", label="Freq1")
            
        self.ax.legend()
        self.canvas.draw()
        max_mag = np.max(magnitude) if len(magnitude) > 0 else 0

if __name__ == "__main__":
    root = tk.Tk()
    app = ListenerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_listening)
    root.mainloop()