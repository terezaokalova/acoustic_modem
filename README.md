## Overview
In this project, I developed an acoustic modem system that converts text into modulated audio tones and transmits them via a computer's speakers. A receiving module captures the transmitted audio through a microphone, demodulates the tones, and reconstructs the original text. The system employs Frequency Shift Keying (FSK) modulation with two distinct pure sine wave tones—600 Hz representing bit 0 and 1600 Hz representing bit 1.

## Architecture
The protocol.py module handles all signal processing and digital data conversion routines. In this module, I implemented functions to convert text into a bit stream and vice versa using 8‑bit ASCII (big‑endian) conversion. The data is framed by prepending a fixed 20‑bit preamble, an 8‑bit synchronization pattern, and an 8‑bit header that indicates the message length. This framing ensures that the receiver synchronizes with the incoming signal and knows how many bits to expect. I also implemented optional error correction using Hamming encoding and error detection using a CRC‑8 checksum (not used for now). For modulation, pure sine wave tones are generated with Hanning windowing to minimize spectral leakage, while for demodulation, the Goertzel algorithm efficiently extracts the energy at the target frequencies. A streaming decoder state machine is then used to process continuous audio data and handle real-time reconstruction of messages.

The sender.py module contains a Tkinter graphical user interface, which includes a text input area for message entry, controls for adjusting transmission parameters (such as volume, symbol duration, and tone frequencies), and displays both a log of sent messages and a real-time FFT spectrum plot. The sender converts text to a bit stream using functions from protocol.py, modulates the bit stream into an audio waveform, and plays the resulting signal via the computer's speakers. Real-time feedback is provided through a progress bar and a spectrum plot to let the user monitor transmission status.

The listener.py module implements the reception of messages via its a Tkinter interface. It captures audio from the microphone using sounddevice's InputStream, applies a 4th order Butterworth bandpass filter to reduce out-of-band noise, and passes the processed audio to the streaming decoder (from protocol.py). The decoder employs the Goertzel algorithm and a state machine to detect the preamble, synchronize with the message, and extract the data bits before converting them back into text. The decoded text is then displayed on the GUI, along with a live FFT spectrum of the incoming audio.

## Algorithms
### Goertzel Algorithm
To extract tone information during demodulation, I adopted the Goertzel algorithm, which efficiently computes the magnitude of a specific frequency component without having to calculate a full FFT.

First, given a block of $N$ samples and a target frequency, the bin index corresponding to that frequency is calculated by

k = \text{int}\left(0.5 + N \times \frac{\text{target\_freq}}{\text{sample\_rate}}\right)

Next, the angular frequency $\omega$ and the coefficient are computed using:
\omega = \frac{2\pi k}{N}, \quad \text{coeff} = 2\cos(\omega)

The algorithm processes the samples via the recurrence relation:
q(n) = s(n) + \text{coeff} \times q(n-1) - q(n-2)
with initial conditions $q(-1) = 0$ and $q(-2) = 0$.

 After processing the $N$ samples, the magnitude of the target frequency is determined using:
\text{magnitude} = \sqrt{q(N-1)^2 + q(N-2)^2 - \text{coeff} \cdot q(N-1) \cdot q(N-2)}
 
This approach isolates the energy at the target frequency while reducing computational overhead, which justifies its use for real-time tone detection.

### FEC with Hamming Codes
I implemented optional forward error correction (FEC) using Hamming codes. It starts by splitting the bitstream into 4‑bit nibbles. For each nibble, three parity bits are computed using XOR operations as follows:
p_1 = d_1 \oplus d_2 \oplus d_4, \quad p_2 = d_1 \oplus d_3 \oplus d_4, \quad p_3 = d_2 \oplus d_3 \oplus d_4
 
The 7‑bit code word is then formed by arranging these bits as:
[p_1, p_2, d_1, p_3, d_2, d_3, d_4]

During decoding, the receiver recalculates the parity bits and computes a 3‑bit syndrome. A nonzero syndrome indicates the position of a single-bit error, which is corrected by flipping the corresponding bit. Finally, the original 4 data bits are extracted from the corrected block.

### CRC‑8 Calculation
For error detection, I implemented a CRC‑8 checksum using the polynomial $0x07$. The algorithm groups the data bits into 8-bit bytes (padding when needed for desired length) and initializes a CRC register to zero. For each byte, the CRC value is updated by XOR'ing the byte with the CRC. The register is then processed bit by bit: if the most significant bit is set, it is left-shifted and XOR'ed with the polynomial; otherwise, it is left-shifted without XOR. The final 8-bit result constitutes the CRC, which is appended to the transmitted data and verified at the receiver.

### Requirements
All dependencies are listed in the requirements.txt file. All code was written and tested on MacBook Air M1 with 8GB memory.

### Usage
To run the system, first launch the listener component to prepare for reception: python listener.py
Then, in a separate terminal or on another computer, start the sender component: python sender.py

In the sender interface, enter your message, adjust the transmission parameters if needed (with a transmission speed caveat, as further described below), and hit "Send." The sender displays the transmission spectrum and progress while playing the modulated signal through the speakers. The listener captures the audio, processes it, and displays the decoded message immediately after the full message is sent.

## Bottlenecks and Next Steps

While current implementation includes error detection (CRC-8) and error correction (Hamming codes), they're disabled by default due to several persisting problems that need to be addressed:

### Hamming
Currently, the Hamming code corrects only single-bit errors—if multiple bits are corrupted in a 7‑bit block or if the header is affected, the syndrome may point to an incorrect bit for correction, causing decoding errors.

### CRC
The CRC‑8 routine it dependent on precise byte alignment, i.e. padding any incomplete byte to 8 bits. Noise-induced bit errors or misalignment can lead to extra padding or wrong byte values, so the computed CRC won't match the transmitted checksum. Moreover, if the header or framing gets corrupted, the receiver may extract an incorrect data length, leading to failures of this check. As of now, my implementation is an all-or-nothing approach, with no mechanism to identify which part of the message is likely corrupted.

With both Hamming and CTC enabled, the performance degrades with each message, leading to gibberish outputs at the decoding step. I suspect that bit drift occurs when the receiver's symbol timing gradually misaligns with the transmitted bit boundaries, leading to a cumulative phase error in the bit detection process. This drift may be due to clock frequency differences between sender and receiver or environmental noise that distorts symbol transitions. As transmission continues, this misalignment compounds, eventually causing the decoder to sample at incorrect positions relative to the transmitted symbols, which in turn leads to increased bit error rates. Without periodic resynchronization markers or adaptive timing adjustment, longer messages become increasingly corrupted as the drift accumulates beyond the system's tolerance threshold.

### Bit Synchronization: 
At higher noise levels, occasional bit errors in the header can cause the decoder to misinterpret the message length, leading to complete decoding failure. The Hamming code can correct single-bit errors within a code word, but if synchronization is lost due to header corruption, the entire message fails.

### Trade-off with Speed: 
Enabling error correction reduces the effective data rate by approximately 75% (7 bits transmitted for every 4 data bits), which significantly impacts transmission speed.

## Possible Ways Forward

### Hamming Code Improvements
I am considering implementing an extended Hamming code (SECDED - Single Error Correction, Double Error Detection) to least detect when two-bit errors occur, combine Hamming codes with interleaving to distribute burst errors across multiple codewords, and develop a more robust block synchronization mechanism using periodic markers throughout the message.

### CRC Improvements
I might replace CRC with a stronger CRC polynomial with better error detection properties, segmenting the message and applying separate CRC checks to each segment for partial message recovery, implementing a progressive CRC scheme with intermediate checksums at regular intervals, or exploring alternative error detection codes.

#### Robust Header Protection: 
Implement stronger error correction specifically for the header portion of the message, possibly using a more powerful code like Reed-Solomon (need to do a little more self-studying on that topic first).

#### Adaptive Error Correction: 
Implement a scheme that can adjust error correction levels based on detected noise conditions, enabling more protection in noisy environments and faster transmission in clean ones. Adaptive Filter Design could be particularly helpful if the noise levels do not remain stationary throughout the full duration of transmission. A possible feature to add could be real-time signal-to-noise ratio (SNR) estimation to help adaptively adjust decoding parameters and provide feedback about transmission quality.

#### Interleaving: 
Implement bit interleaving to spread the impact of burst errors across multiple Hamming code words, improving resilience against temporally correlated noise.

### Additional Considerations
For intermediate-length messages, the decoding quality is high, even in the presence of ambient noise. This robustness comes from the choice of two sufficiently distinct frequencies that both exceed the typical noise range. With respect to transmission speed, symbol duration below 0.1s was found to be substantially less reliable, hence trading off accuracy for speed.

In one of my older implementations, I was able to transmit longer messages in chunks (whereby a message would get broken down and then reassembled at the decoding stage) which enabled accurate decoding for longer lengths at the expense of transmission speed.