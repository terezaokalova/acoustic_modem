#!/usr/bin/env python3
"""
protocol.py

Modulation/demodulation accepting preamble + header + data
FEC and CRC (disabled for now)
Goertzel's algorithm used for frequency detection
"""

import numpy as np
import logging
import sounddevice as sd

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# CONFIGURATION AND GLOBAL VARIABLES

# new - 4-FSK
class ModemConfig:
    def __init__(self):
        self.sample_rate = 44100
        self.symbol_duration = 0.1
        self.N = int(self.sample_rate * self.symbol_duration)
        
        # frequencies need to be spaced far enough apart to be distinguishable
        self.freq00 = 500.0   # For bits 00
        self.freq01 = 900.0   # For bits 01
        self.freq10 = 1300.0  # For bits 10
        self.freq11 = 1700.0  # For bits 11
        
        # Keep volume and other parameters
        self.volume = 0.8
        self.preamble = "11111000001111100000"
        self.sync_pattern = "11001100"
        self.use_hamming = False
        self.use_crc = False
        self.header_length_bits = 8
        
        # for backward compatibility, keep original freq0/freq1
        self.freq0 = self.freq00
        self.freq1 = self.freq11
        
        # flag to enable/disable M-ary FSK
        self.use_mary_fsk = True
        self.bits_per_symbol = 2  # 4-FSK encodes 2 bits per symbol
        
        # threshold for bit decision in demodulation
        self.threshold_factor = 0.5
        self.overlap = 0

DEFAULT_CONFIG = ModemConfig()

# Utility functions for Text <-> Bits conversion

def text_to_bits(text: str) -> list:
    # convert each character to 8-bit representation (big-endian)
    bits = []
    for ch in text:
        b = ord(ch)
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits

def bits_to_text(bits: list) -> str:
    # group bits into 8-bit bytes and convert to ASCII; pad if incomplete
    chars = []
    if len(bits) % 8 != 0:
        logging.warning(f"Bit length {len(bits)} not divisible by 8, padding with zeros")
        bits = bits + [0] * (8 - (len(bits) % 8))
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i+j]
            # append character if printable, else '?'
            if 32 <= byte <= 126:
                chars.append(chr(byte))
            else:
                chars.append('?')
    return "".join(chars)

# Hamming Code Functions (FEC)

def hamming_encode_nibble(nibble: list) -> list:
    # encode 4-bit nibble into 7-bit using Hamming code (computes three parity bits)
    d1, d2, d3, d4 = nibble
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return [p1, p2, d1, p3, d2, d3, d4]

def hamming_decode_block(block: list[int]) -> tuple[list[int], bool]:
    # decode 7-bit block: use syndrome to detect and correct single-bit errors
    p1, p2, d1, p3, d2, d3, d4 = block
    s1 = p1 ^ d1 ^ d2 ^ d4
    s2 = p2 ^ d1 ^ d3 ^ d4
    s3 = p3 ^ d2 ^ d3 ^ d4
    syndrome = (s1 << 2) | (s2 << 1) | s3
    corrected = False
    # if there is an error, flip the bit at the position indicated by the syndrome
    if syndrome != 0:
        pos = syndrome - 1
        if pos < len(block):
            block[pos] ^= 1  # flip erroneous bit
            corrected = True
    # each nibble is 4 bits, but the Hamming code adds 3 parity bits, making it 7 bits
    nibble = [block[2], block[4], block[5], block[6]]
    return nibble, corrected


def apply_fec(bits: list, config: ModemConfig) -> list:
    # pad bits and encode every 4-bit nibble using Hamming code
    if not config.use_hamming:
        return bits
    if len(bits) % 4 != 0:
        pad = 4 - (len(bits) % 4)
        bits += [0] * pad
    fec_bits = []
    for i in range(0, len(bits), 4):
        nibble = bits[i:i+4]
        fec_bits.extend(hamming_encode_nibble(nibble))
    return fec_bits

def remove_fec(bits: list, config: ModemConfig) -> tuple[list[int], int]:
    # decode bits in groups of 7 and count corrections made
    if not config.use_hamming:
        return bits, 0
    if len(bits) % 7 != 0:
        logging.warning("FEC mismatch: bit length not multiple of 7")
        return bits, 0
    decoded_bits = []
    error_count = 0
    for i in range(0, len(bits), 7):
        block = bits[i:i+7]
        nib, corr = hamming_decode_block(block)
        decoded_bits.extend(nib)
        if corr:
            error_count += 1
    return decoded_bits, error_count

def compute_crc8(bits: list) -> list:
    # compute CRC8 over the bits; ensures data integrity if enabled
    if len(bits) % 8 != 0:
        bits += [0] * (8 - (len(bits) % 8))
    data = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i+j]
        data.append(byte)
    crc = 0
    poly = 0x07
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) & 0xFF) ^ poly
            else:
                crc = (crc << 1) & 0xFF
    out_bits = []
    for i in range(7, -1, -1):
        out_bits.append((crc >> i) & 1)
    return out_bits

# Message Encoding/Decoding

def encode_message(text: str, config: ModemConfig = DEFAULT_CONFIG) -> list:
    # convert text to bits and build header (8-bit length)
    raw_bits = text_to_bits(text)
    length = len(text)
    header = []
    for i in range(7, -1, -1):
        header.append((length >> i) & 1)
    # apply FEC if enabled; then, append CRC 
    data_bits = apply_fec(raw_bits, config)
    if config.use_crc:
        crc_bits = compute_crc8(data_bits)
        data_bits.extend(crc_bits)
    # prepend preamble and sync pattern
    preamble_bits = [int(b) for b in config.preamble]
    sync_bits = [int(b) for b in config.sync_pattern]
    final_bits = preamble_bits + sync_bits + header + data_bits
    logging.info(f"Message encoded: '{text}' -> {len(final_bits)} bits")
    return final_bits


def decode_message(bits: list[int],
                   config: ModemConfig = DEFAULT_CONFIG
                  ) -> tuple[str, bool]:
    # verify preamble; if missing, return error
    preamble_len = len(config.preamble)
    if bits[:preamble_len] != [int(b) for b in config.preamble]:
        logging.error("Preamble not found!")
        return "", False
    bits = bits[preamble_len:]
    # skip sync pattern (assumes fixed length)
    sync_len = len(config.sync_pattern)
    if len(bits) < sync_len:
        logging.error("No room for sync pattern")
        return "", False
    bits = bits[sync_len:]
    # extract header (8 bits) for message length
    if len(bits) < 8:
        logging.error("No room for header")
        return "", False
    msg_len = 0
    header_bits = bits[:8]
    for b in header_bits:
        msg_len = (msg_len << 1) | b
    if msg_len <= 0 or msg_len > 1000:
        logging.error(f"Invalid message length: {msg_len}")
        return "", False
    data_bits = bits[8:]
    # determine expected data length based on FEC/CRC settings
    if config.use_hamming:
        expected_data = ((msg_len * 8 + 3) // 4) * 7
    else:
        expected_data = msg_len * 8
    if config.use_crc:
        expected_data += 8
    if len(data_bits) < expected_data:
        logging.error(f"Not enough data bits. Expected {expected_data}, got {len(data_bits)}")
        return "", False
    actual_data = data_bits[:expected_data]
    # verify CRC if enabled
    if config.use_crc:
        crc_recv = actual_data[-8:]
        actual_data = actual_data[:-8]
        crc_calc = compute_crc8(actual_data)
        if crc_recv != crc_calc:
            logging.error("CRC mismatch!")
            return "", False
    # remove FEC encoding if enabled
    if config.use_hamming:
        actual_data, _ = remove_fec(actual_data, config)
    actual_data = actual_data[:msg_len*8]
    text = bits_to_text(actual_data)
    logging.info(f"Message decoded: '{text}'")
    return text, True

# Frequency Detection via Goertzel Algorithm

def goertzel_mag(samples: np.ndarray, target_freq: float, sample_rate: int) -> float:
    # check if we're processing one frequency or multiple
    if isinstance(target_freq, (list, tuple, np.ndarray)):
        return _goertzel_multi_freq(samples, target_freq, sample_rate)
    
    # original single-frequency implementation
    N = len(samples)
    k = int(0.5 + (N * target_freq) / sample_rate)
    omega = 2.0 * np.pi * k / N
    cosine = np.cos(omega)
    sine = np.sin(omega)
    coeff = 2.0 * cosine
    
    # Process samples
    q1 = q2 = 0.0
    for s in samples:
        q0 = coeff * q1 - q2 + s
        q2 = q1
        q1 = q0
        
    # calculate magnitude using real and imaginary parts
    real = q1 - q2 * cosine
    imag = q2 * sine
    return np.sqrt(real*real + imag*imag)

def _goertzel_multi_freq(samples: np.ndarray, frequencies: list, sample_rate: int) -> list:
    # helper to calculate magnitude for multiple frequencies in one pass"
    N = len(samples)
    results = []
    
    # Process each frequency
    for freq in frequencies:
        k = int(0.5 + (N * freq) / sample_rate)
        omega = 2.0 * np.pi * k / N
        cosine = np.cos(omega)
        sine = np.sin(omega)
        coeff = 2.0 * cosine
        
        q1 = q2 = 0.0
        for s in samples:
            q0 = coeff * q1 - q2 + s
            q2 = q1
            q1 = q0
            
        real = q1 - q2 * cosine
        imag = q2 * sine
        results.append(np.sqrt(real*real + imag*imag))
        
    return results

# Modulation/Demodulation

def generate_tone(freq: float, N: int, sr: int, volume: float) -> np.ndarray:
    # generate sine wave for given frequency and apply Hanning window
    t = np.arange(N) / sr
    tone = np.sin(2 * np.pi * freq * t)
    window = np.hanning(N)
    return (tone * window * volume).astype(np.float32)

# m-ary alr
def modulate_bits(bit_list: list, config: ModemConfig) -> np.ndarray:
    # map each bit pair to corresponding tone
    N = config.N
    wave_chunks = []
    
    if config.use_mary_fsk:
        # create tones for 4-FSK
        tone00 = generate_tone(config.freq00, N, config.sample_rate, config.volume)
        tone01 = generate_tone(config.freq01, N, config.sample_rate, config.volume)
        tone10 = generate_tone(config.freq10, N, config.sample_rate, config.volume)
        tone11 = generate_tone(config.freq11, N, config.sample_rate, config.volume)
        
        # pad bit_list if its length is odd
        if len(bit_list) % 2 != 0:
            bit_list = bit_list + [0]
            
        # process bits in pairs
        for i in range(0, len(bit_list), 2):
            bit_pair = (bit_list[i] << 1) | bit_list[i+1]
            if bit_pair == 0:
                wave_chunks.append(tone00)
            elif bit_pair == 1:
                wave_chunks.append(tone01)
            elif bit_pair == 2:
                wave_chunks.append(tone10)
            else:  # bit_pair == 3
                wave_chunks.append(tone11)
                
            # emphasis on last few symbols
            if i >= len(bit_list) - 32:  # last 16 symbols (32 bits)
                wave_chunks[-1] = wave_chunks[-1].copy() * 1.1
    else:
        # original binary FSK implementation
        tone0 = generate_tone(config.freq0, N, config.sample_rate, config.volume)
        tone1 = generate_tone(config.freq1, N, config.sample_rate, config.volume)
        for i, bit in enumerate(bit_list):
            if i >= len(bit_list) - 16:
                tone = tone1.copy() * 1.1 if bit == 1 else tone0.copy() * 1.1
                wave_chunks.append(tone)
            else:
                wave_chunks.append(tone1 if bit == 1 else tone0)
                
    # trailing silence
    silence = np.zeros(N // 4, dtype=np.float32)
    wave_chunks.append(silence)
    waveform = np.concatenate(wave_chunks)
    logging.info(f"Waveform generated: {len(bit_list)} bits -> {len(waveform)} samples")
    return waveform

def demodulate_bits(waveform: np.ndarray, config: ModemConfig) -> list:
    N = config.N
    num_symbols = len(waveform) // N
    out_bits = []
    
    if config.use_mary_fsk:
        # frequencies list for 4-FSK
        freq_list = [config.freq00, config.freq01, config.freq10, config.freq11]
        
        for i in range(num_symbols):
            segment = waveform[i*N:(i+1)*N]
            window = np.hanning(len(segment))
            windowed_segment = segment * window
            
            # multi-frequency Goertzel algorithm
            energies = goertzel_mag(windowed_segment, freq_list, config.sample_rate)
            max_idx = np.argmax(energies)
            
            # convert to bit pair
            if max_idx == 0:   # freq00 has max energy
                out_bits.extend([0, 0])
            elif max_idx == 1: # freq01 has max energy
                out_bits.extend([0, 1])
            elif max_idx == 2: # freq10 has max energy
                out_bits.extend([1, 0])
            else:              # freq11 has max energy
                out_bits.extend([1, 1])
    else:
        # original binary FSK implementation
        for i in range(num_symbols):
            segment = waveform[i*N:(i+1)*N]
            window = np.hanning(len(segment))
            windowed_segment = segment * window
            e0 = goertzel_mag(windowed_segment, config.freq0, config.sample_rate)
            e1 = goertzel_mag(windowed_segment, config.freq1, config.sample_rate)
            out_bits.append(1 if e1 > e0 * config.threshold_factor else 0)
            
    return out_bits

# Streaming Decoder

class StreamingDecoder:
    def __init__(self, config: ModemConfig = DEFAULT_CONFIG):
        # initialize buffers and state for streaming audio decoding
        self.config = config
        self.buffer = np.array([], dtype=np.float32)
        self.symbol_samples = config.N
        self.overlap = config.overlap
        self.collected_bits = []
        self.messages = []
        self.state = "SEARCHING"        # states: SEARCHING, COLLECTING, DECODING
        self.found_preamble_at = -1
        self.total_bits_needed = 0
        self.collecting_timeout = 0
        self.collecting_timeout_limit = 5000

    def process_samples(self, new_samples: np.ndarray):
        # Add new audio samples to buffer and extract symbols sequentially
        self.buffer = np.concatenate((self.buffer, new_samples))
        
        while len(self.buffer) >= self.symbol_samples:
            symbol = self.buffer[:self.symbol_samples]
            step = max(1, self.symbol_samples - self.overlap)
            self.buffer = self.buffer[step:]
            
            # Process symbol based on FSK mode
            if self.config.use_mary_fsk:
                # Compute energies at all target frequencies using multi-frequency Goertzel
                windowed = symbol * np.hanning(len(symbol))
                freq_list = [self.config.freq00, self.config.freq01, self.config.freq10, self.config.freq11]
                energies = goertzel_mag(windowed, freq_list, self.config.sample_rate)
                
                # Find the frequency with maximum energy
                max_idx = np.argmax(energies)
                
                # Convert to bit pair
                if max_idx == 0:   # freq00 has max energy
                    self.collected_bits.extend([0, 0])
                elif max_idx == 1: # freq01 has max energy
                    self.collected_bits.extend([0, 1])
                elif max_idx == 2: # freq10 has max energy
                    self.collected_bits.extend([1, 0])
                else:              # freq11 has max energy
                    self.collected_bits.extend([1, 1])
                    
                # Call state machine update twice since we added two bits
                self._update_state_machine()
                self._update_state_machine()
            else:
                # Original binary FSK implementation
                windowed = symbol * np.hanning(len(symbol))
                e0 = goertzel_mag(windowed, self.config.freq0, self.config.sample_rate)
                e1 = goertzel_mag(windowed, self.config.freq1, self.config.sample_rate)
                bit = 1 if e1 > e0 * self.config.threshold_factor else 0
                self.collected_bits.append(bit)
                self._update_state_machine()


    def _update_state_machine(self):
        # state machine: SEARCH for preamble, then COLLECT until full message is received, then DECODE
        if self.state == "SEARCHING":
            self._search_for_preamble()
        elif self.state == "COLLECTING":
            self.collecting_timeout += 1
            if self.collecting_timeout > self.collecting_timeout_limit:
                self.state = "SEARCHING"  # reset on timeout
                self.collecting_timeout = 0
                return
            if len(self.collected_bits) >= self.found_preamble_at + self.total_bits_needed:
                self.state = "DECODING"
                self._decode_message()

    def _search_for_preamble(self):
        # scan collected bits for preamble with 90% match, then read header to determine message length
        pre = [int(b) for b in self.config.preamble]
        pre_len = len(pre)
        sy_len = len(self.config.sync_pattern)
        if len(self.collected_bits) < pre_len + sy_len + 8:
            return
        for i in range(len(self.collected_bits) - pre_len + 1):
            window = self.collected_bits[i:i+pre_len]
            if sum(a == b for a, b in zip(window, pre)) / pre_len >= 0.90:
                if i + pre_len + sy_len + 8 > len(self.collected_bits):
                    self.state = "COLLECTING"
                    self.found_preamble_at = i
                    self.collecting_timeout = 0
                    return
                hdr = self.collected_bits[i+pre_len+sy_len : i+pre_len+sy_len+8]
                msg_len = 0
                for bit in hdr:
                    msg_len = (msg_len << 1) | bit
                if not (0 < msg_len <= 1000):
                    continue
                data_bits = msg_len * 8
                if self.config.use_hamming:
                    data_bits = ((msg_len*8 + 3)//4) * 7
                if self.config.use_crc:
                    data_bits += 8
                self.total_bits_needed = pre_len + sy_len + 8 + data_bits
                self.found_preamble_at = i
                self.state = "COLLECTING"
                self.collecting_timeout = 0
                return

    def _decode_message(self):
        # extract bits for full message, rebuild stream with exact preamble and sync, decode message
        i = self.found_preamble_at
        pre_len = len(self.config.preamble)
        sy_len = len(self.config.sync_pattern)
        bits = self.collected_bits[i : i + self.total_bits_needed]
        if len(bits) < self.total_bits_needed:
            self.state = "SEARCHING"
            return
        exact_pre  = [int(b) for b in self.config.preamble]
        exact_sync = [int(b) for b in self.config.sync_pattern]
        data_hdr   = bits[pre_len + sy_len:]  # header + payload (+CRC)
        clean_bits = exact_pre + exact_sync + data_hdr
        text, valid = decode_message(clean_bits, self.config)
        if valid:
            logging.info(f"Decoded message: '{text}'")
            self.messages.append(text)
            self.collected_bits = []  # clear bits on success
        else:
            logging.error("Decode failed, retrying search")
            self.collected_bits = self.collected_bits[i:]  # drop bits up to preamble
        self.state = "SEARCHING"
        self.found_preamble_at = -1
        self.total_bits_needed = 0
        self.collecting_timeout = 0

    def get_messages(self) -> list:
        # return accumulated decoded messages and clear message list
        out = self.messages[:]
        self.messages = []
        return out

if __name__ == "__main__":
    # tests for protocol functions with several messages
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Running protocol unit tests...")
    
    special_messages = [
        "hello world",
        "it's weird: right?",
        "quotes: 'single' and \"double\"",
        "spaces  and   symbols!@#$"
    ]
    
    for msg in special_messages:
        print(f"\nTesting message: '{msg}'")
        bits = encode_message(msg, DEFAULT_CONFIG)
        wave = modulate_bits(bits, DEFAULT_CONFIG)
        rec_bits = demodulate_bits(wave, DEFAULT_CONFIG)
        rec_msg, valid = decode_message(rec_bits, DEFAULT_CONFIG)
        print(f"Direct pipeline: '{rec_msg}', valid: {valid}")
        
        decoder = StreamingDecoder(DEFAULT_CONFIG)
        chunk_size = DEFAULT_CONFIG.N // 2
        for i in range(0, len(wave), chunk_size):
            if i + chunk_size <= len(wave):
                chunk = wave[i:i+chunk_size]
                decoder.process_samples(chunk)
        stream_msgs = decoder.get_messages()
        print(f"Streaming decoder: {stream_msgs}")
        
        if msg == rec_msg and msg in stream_msgs:
            print("PASSED")
        else:
            print("FAILED")