Overview
In this project, I developed an acoustic modem system that converts text into modulated audio tones and transmits them via a computer’s speakers. A receiving module captures the transmitted audio through a microphone, demodulates the tones, and reconstructs the original text. The system employs Frequency Shift Keying (FSK) modulation with two distinct pure sine wave tones—600 Hz representing bit 0 and 1600 Hz representing bit 1—and is implemented across three Python modules: protocol.py, sender.py, and listener.py. This exercise demonstrates my abilities in programming, digital signal processing, and problem solving, all of which are skills directly relevant to my internship work.

Architecture
The protocol.py module forms the core of my system. It handles all signal processing and digital data conversion routines. In this module, I implemented functions to convert text into a bit stream and vice versa using 8‑bit ASCII (big‑endian) conversion. The data is framed by prepending a fixed 20‑bit preamble, an 8‑bit synchronization pattern, and an 8‑bit header that indicates the message length; this framing ensures that the receiver reliably synchronizes with the incoming signal and knows exactly how many bits to expect. I also implemented optional error correction using Hamming encoding and error detection using a CRC‑8 checksum. For modulation, pure sine wave tones are generated with Hanning windowing to minimize spectral leakage, while for demodulation, the Goertzel algorithm efficiently extracts the energy at the target frequencies. A streaming decoder state machine is then used to process continuous audio data and reconstruct messages in real time.

The sender.py module provides a Tkinter graphical user interface. This interface includes a text input area for message entry, controls for adjusting transmission parameters (such as volume, symbol duration, and tone frequencies), and displays both a log of sent messages and a real-time FFT spectrum plot. The sender converts text to a bit stream using the functions in protocol.py, modulates the bit stream into an audio waveform, and plays the resulting signal via the computer’s speakers. Real-time feedback is provided through a progress bar and a spectrum plot to help the user monitor transmission status.

The listener.py module implements the reception functionality via its own Tkinter interface. It captures audio from the microphone using sounddevice’s InputStream, applies a 4th order Butterworth bandpass filter to reduce out-of-band noise, and passes the processed audio to the streaming decoder (from protocol.py). The decoder employs the Goertzel algorithm and a state machine to detect the preamble, synchronize with the message, and extract the data bits before converting them back into text. The decoded text is then displayed on the GUI, along with a live FFT spectrum of the incoming audio.

Detailed Explanation of Key Algorithms
Goertzel Algorithm
To extract tone information during demodulation, I adopted the Goertzel algorithm, which efficiently computes the magnitude of a specific frequency component without having to calculate a full FFT. The math behind this algorithm is as follows:

First, given a block of $N$ samples and a target frequency, the bin index corresponding to that frequency is calculated by

k = \text{int}\left(0.5 + N \times \frac{\text{target\_freq}}{\text{sample\_rate}}\right)

Next, the angular frequency $\omega$ and the coefficient are computed using:
\omega = \frac{2\pi k}{N}, \quad \text{coeff} = 2\cos(\omega)

The algorithm processes the samples using the recurrence relation:
q(n) = s(n) + \text{coeff} \times q(n-1) - q(n-2)
with initial conditions $q(-1) = 0$ and $q(-2) = 0$.

 After processing the $N$ samples, the magnitude of the target frequency is determined using:
\text{magnitude} = \sqrt{q(N-1)^2 + q(N-2)^2 - \text{coeff} \cdot q(N-1) \cdot q(N-2)}
 
This approach isolates the energy at the target frequency while reducing computational overhead, making it ideal for real-time tone detection.

FEC with Hamming Codes
To improve transmission reliability, I implemented optional forward error correction (FEC) using Hamming codes. The process begins by splitting the bitstream into 4‑bit nibbles. For each nibble, three parity bits are computed using XOR operations as follows:
p_1 = d_1 \oplus d_2 \oplus d_4, \quad p_2 = d_1 \oplus d_3 \oplus d_4, \quad p_3 = d_2 \oplus d_3 \oplus d_4
 
The 7‑bit code word is then formed by arranging these bits as:
[p_1, p_2, d_1, p_3, d_2, d_3, d_4]

During decoding, the receiver recalculates the parity bits and computes a 3‑bit syndrome. A nonzero syndrome indicates the position of a single-bit error, which is corrected by flipping the corresponding bit. Finally, the original 4 data bits are extracted from the corrected block.

CRC‑8 Calculation
For error detection, I implemented a CRC‑8 checksum using the polynomial $0x07$. The algorithm groups the data bits into 8-bit bytes (padding if necessary) and initializes a CRC register to zero. For each byte, the CRC value is updated by XOR’ing the byte with the CRC. The register is then processed bit by bit: if the most significant bit is set, it is left-shifted and XOR’ed with the polynomial; otherwise, it is left-shifted without XOR. The final 8-bit result constitutes the CRC, which is appended to the transmitted data and verified at the receiver.

# Design Specifications
My implementation meets all the requirements of the Acoustic Modem Project. The protocol.py module provides a complete encoding/decoding scheme that translates text into acoustic signals and vice versa. It includes functions for text-to-bit conversion, message framing (using a fixed preamble, sync pattern, and header), and modulation/demodulation using sine wave tones with Hanning windowing. Optional error correction via Hamming codes and error detection via CRC‑8 have been implemented to enhance reliability, although these features are disabled by default.

The sender.py module offers a clear, user-friendly Tkinter interface for message entry, parameter adjustment, and real-time feedback through a progress bar and FFT spectrum plot. This module meets the requirement to provide a text input interface, transmission parameter options, and clear user feedback during transmission.

The listener.py module captures audio from the microphone, applies bandpass filtering, and uses a streaming decoder to demodulate the received signal. The decoder’s use of the Goertzel algorithm and a state machine to reliably detect, synchronize, and reconstruct messages aligns with the project’s guidelines for robust error handling and streaming implementation.

Overall, my solution is modular, well-documented, and demonstrates a thoughtful balance between transmission speed and reliability while addressing noise immunity. The system has been designed to work in varying noise environments and includes real-time feedback to enhance usability.

# Requirements


## Usage
To run the system, first launch the listener component to prepare for reception: python listener.py
Then, in a separate terminal or on another computer, start the sender component: python sender.py

In the sender interface, enter your message, adjust the transmission parameters if needed, and click “Send.” The sender displays the transmission spectrum and progress while playing the modulated signal through the speakers. The listener captures the audio, processes it, and displays the decoded message in real time.
