#!/usr/bin/env python3
"""
protocol.py
acoustic modem implementation using 8-FSK
"""

import numpy as np
import logging
import sounddevice as sd

class ModemConfig:
    def __init__(self):
        self.sample_rate = 44100
        self.symbol_duration = 0.1
        self.N = int(self.sample_rate * self.symbol_duration)
        # 4-fsk frequencies
        self.freq00 = 500.0
        self.freq01 = 900.0
        self.freq10 = 1300.0
        self.freq11 = 1700.0
        # binary for backward compatibility
        self.freq0 = self.freq00
        self.freq1 = self.freq11
        # 8-fsk freqs
        self.freq000 = 500.0
        self.freq001 = 800.0
        self.freq010 = 1100.0
        self.freq011 = 1400.0
        self.freq100 = 1700.0
        self.freq101 = 2000.0
        self.freq110 = 2300.0
        self.freq111 = 2600.0
        self.volume = 0.8
        self.preamble = "11111000001111100000"
        self.sync_pattern = "11001100"
        self.use_hamming = False
        self.use_crc = False
        self.header_length_bits = 8
        # M-ary FSK config
        self.use_mary_fsk = True
        # when set to 4, first message needs to be 4-fsk, works for 8 too thereafter (and vice versa)
        self.fsk_mode = 8
        self.bits_per_symbol = 2
        self._update_bits_per_symbol()
        self.threshold_factor = 0.5
        self.overlap = 0

    # update bits per symbol based on FSK mode settings
    def _update_bits_per_symbol(self):
        if self.use_mary_fsk:
            if self.fsk_mode == 4:
                self.bits_per_symbol = 2
            elif self.fsk_mode == 8:
                self.bits_per_symbol = 3
            else:
                self.bits_per_symbol = 1
        else:
            self.bits_per_symbol = 1

DEFAULT_CONFIG = ModemConfig()

# convert text to a list of bits
def text_to_bits(text: str) -> list:
    bits = []
    for ch in text:
        b = ord(ch)
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits

# convert list of bits to text
def bits_to_text(bits: list) -> str:
    chars = []
    # modding
    if len(bits) % 8 != 0:
        logging.warning("Bit length not divisible by 8, padding")
        bits += [0] * (8 - (len(bits) % 8))
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i+j]
        if 32 <= byte <= 126:
            chars.append(chr(byte))
        else:
            chars.append('?')
    return "".join(chars)

# hamming encoding on a nibble of bits (not used)
def hamming_encode_nibble(nibble: list) -> list:
    d1, d2, d3, d4 = nibble
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return [p1, p2, d1, p3, d2, d3, d4]

# decode a 7-bit block using Hamming code and correct single-bit errors (not used)
def hamming_decode_block(block: list) -> tuple[list, bool]:
    p1, p2, d1, p3, d2, d3, d4 = block
    s1 = p1 ^ d1 ^ d2 ^ d4
    s2 = p2 ^ d1 ^ d3 ^ d4
    s3 = p3 ^ d2 ^ d3 ^ d4
    syndrome = (s1 << 2) | (s2 << 1) | s3
    corrected = False
    if syndrome != 0:
        pos = syndrome - 1
        if pos < len(block):
            block[pos] ^= 1
            corrected = True
    nibble = [block[2], block[4], block[5], block[6]]
    return nibble, corrected

# fwd error correction if enabled (not used)
def apply_fec(bits: list, config: ModemConfig) -> list:
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

# remove fwd error correction from bits (not used)
def remove_fec(bits: list, config: ModemConfig) -> tuple[list, int]:
    if not config.use_hamming:
        return bits, 0
    if len(bits) % 7 != 0:
        logging.warning("FEC mismatch")
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

# compute CRC8 checksum for a list of bits (not used)
def compute_crc8(bits: list) -> list:
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

# encode a message by converting text to bits, adding header and preamble (and applying FEC and CRC)
def encode_message(text: str, config: ModemConfig = DEFAULT_CONFIG) -> list:
    raw_bits = text_to_bits(text)
    length = len(text)
    header = [(length >> i) & 1 for i in range(7, -1, -1)]
    data_bits = apply_fec(raw_bits, config)
    if config.use_crc:
        crc_bits = compute_crc8(data_bits)
        data_bits.extend(crc_bits)
    preamble_bits = [int(b) for b in config.preamble]
    sync_bits = [int(b) for b in config.sync_pattern]
    final_bits = preamble_bits + sync_bits + header + data_bits
    logging.info(f"Message encoded: {text} -> {len(final_bits)} bits")
    return final_bits

# decode a received message from a bit stream
def decode_message(bits: list, config: ModemConfig = DEFAULT_CONFIG) -> tuple[str, bool]:
    preamble_len = len(config.preamble)
    if bits[:preamble_len] != [int(b) for b in config.preamble]:
        logging.error("Preamble not found")
        return "", False
    bits = bits[preamble_len:]
    sync_len = len(config.sync_pattern)
    if len(bits) < sync_len:
        logging.error("No room for sync pattern")
        return "", False
    bits = bits[sync_len:]
    if len(bits) < 8:
        logging.error("No room for header")
        return "", False
    msg_len = 0
    header_bits = bits[:8]
    for b in header_bits:
        msg_len = (msg_len << 1) | b
    if msg_len <= 0 or msg_len > 1000:
        logging.error(f"Invalid msg len: {msg_len}")
        return "", False
    data_bits = bits[8:]
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
    if config.use_crc:
        crc_recv = actual_data[-8:]
        actual_data = actual_data[:-8]
        crc_calc = compute_crc8(actual_data)
        if crc_recv != crc_calc:
            logging.error("CRC mismatch")
            return "", False
    if config.use_hamming:
        actual_data, _ = remove_fec(actual_data, config)
    actual_data = actual_data[:msg_len*8]
    text = bits_to_text(actual_data)
    logging.info(f"Decoded message: {text}")
    return text, True

# apply the Goertzel algorithm to compute the magnitude of a frequency component
def goertzel_mag(samples: np.ndarray, target_freq, sample_rate: int) -> float:
    if isinstance(target_freq, (list, tuple, np.ndarray)):
        return _goertzel_multi_freq(samples, target_freq, sample_rate)
    N = len(samples)
    k = int(0.5 + (N * target_freq) / sample_rate)
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
    return np.sqrt(real * real + imag * imag)

# apply the Goertzel algorithm for multiple frequencies
def _goertzel_multi_freq(samples: np.ndarray, frequencies: list, sample_rate: int) -> list:
    N = len(samples)
    results = []
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
        results.append(np.sqrt(real * real + imag * imag))
    return results

# generate a tone for a given frequency using a Hanning window
def generate_tone(freq: float, N: int, sr: int, volume: float) -> np.ndarray:
    t = np.arange(N) / sr
    tone = np.sin(2 * np.pi * freq * t)
    window = np.hanning(N)
    return (tone * window * volume).astype(np.float32)

# modulate a bit list into a waveform using FSK modulation
def modulate_bits(bit_list: list, config: ModemConfig) -> np.ndarray:
    N = config.N
    wave_chunks = []
    if config.use_mary_fsk:
        if config.fsk_mode == 4:
            # generate tones for 4-FSK mode
            tone00 = generate_tone(config.freq00, N, config.sample_rate, config.volume)
            tone01 = generate_tone(config.freq01, N, config.sample_rate, config.volume)
            tone10 = generate_tone(config.freq10, N, config.sample_rate, config.volume)
            tone11 = generate_tone(config.freq11, N, config.sample_rate, config.volume)
            if len(bit_list) % 2 != 0:
                bit_list.append(0)
            for i in range(0, len(bit_list), 2):
                pair = (bit_list[i] << 1) | bit_list[i+1]
                if pair == 0:
                    wave_chunks.append(tone00)
                elif pair == 1:
                    wave_chunks.append(tone01)
                elif pair == 2:
                    wave_chunks.append(tone10)
                else:
                    wave_chunks.append(tone11)
                if i >= len(bit_list) - 32:
                    wave_chunks[-1] = wave_chunks[-1].copy() * 1.1
        elif config.fsk_mode == 8:
            # generate tones for 8-FSK mode
            if len(bit_list) % 3 != 0:
                pad = 3 - (len(bit_list) % 3)
                bit_list += [0] * pad
            tones = [
                generate_tone(config.freq000, N, config.sample_rate, config.volume),
                generate_tone(config.freq001, N, config.sample_rate, config.volume),
                generate_tone(config.freq010, N, config.sample_rate, config.volume),
                generate_tone(config.freq011, N, config.sample_rate, config.volume),
                generate_tone(config.freq100, N, config.sample_rate, config.volume),
                generate_tone(config.freq101, N, config.sample_rate, config.volume),
                generate_tone(config.freq110, N, config.sample_rate, config.volume),
                generate_tone(config.freq111, N, config.sample_rate, config.volume)
            ]
            for i in range(0, len(bit_list), 3):
                symbol_idx = (bit_list[i] << 2) | (bit_list[i+1] << 1) | bit_list[i+2]
                wave_chunks.append(tones[symbol_idx])
                if i >= len(bit_list) - 48:
                    wave_chunks[-1] = wave_chunks[-1].copy() * 1.1
    else:
        # binary FSK modulation
        tone0 = generate_tone(config.freq0, N, config.sample_rate, config.volume)
        tone1 = generate_tone(config.freq1, N, config.sample_rate, config.volume)
        for i, bit in enumerate(bit_list):
            if i >= len(bit_list) - 16:
                tone = tone1.copy() * 1.1 if bit == 1 else tone0.copy() * 1.1
                wave_chunks.append(tone)
            else:
                wave_chunks.append(tone1 if bit == 1 else tone0)
    # add silence at the end of the waveform
    silence = np.zeros(N // 4, dtype=np.float32)
    wave_chunks.append(silence)
    waveform = np.concatenate(wave_chunks)
    logging.info(f"Waveform generated: {len(bit_list)} bits -> {len(waveform)} samples")
    return waveform

# demodulate a waveform into a bit list
def demodulate_bits(waveform: np.ndarray, config: ModemConfig) -> list:
    N = config.N
    num_symbols = len(waveform) // N
    out_bits = []
    if config.use_mary_fsk:
        if config.fsk_mode == 4:
            freq_list = [config.freq00, config.freq01, config.freq10, config.freq11]
            for i in range(num_symbols):
                segment = waveform[i * N:(i + 1) * N]
                window = np.hanning(len(segment))
                windowed_segment = segment * window
                energies = goertzel_mag(windowed_segment, freq_list, config.sample_rate)
                max_idx = np.argmax(energies)
                if max_idx == 0:
                    out_bits.extend([0, 0])
                elif max_idx == 1:
                    out_bits.extend([0, 1])
                elif max_idx == 2:
                    out_bits.extend([1, 0])
                else:
                    out_bits.extend([1, 1])
        elif config.fsk_mode == 8:
            freq_list = [
                config.freq000, config.freq001, config.freq010, config.freq011,
                config.freq100, config.freq101, config.freq110, config.freq111
            ]
            for i in range(num_symbols):
                segment = waveform[i * N:(i + 1) * N]
                window = np.hanning(len(segment))
                windowed_segment = segment * window
                energies = goertzel_mag(windowed_segment, freq_list, config.sample_rate)
                max_idx = np.argmax(energies)
                bits = [(max_idx >> 2) & 1, (max_idx >> 1) & 1, max_idx & 1]
                out_bits.extend(bits)
    else:
        for i in range(num_symbols):
            segment = waveform[i * N:(i + 1) * N]
            window = np.hanning(len(segment))
            windowed_segment = segment * window
            e0 = goertzel_mag(windowed_segment, config.freq0, config.sample_rate)
            e1 = goertzel_mag(windowed_segment, config.freq1, config.sample_rate)
            out_bits.append(1 if e1 > e0 * config.threshold_factor else 0)
    return out_bits

# for real-time decoding
class StreamingDecoder:
    def __init__(self, config: ModemConfig = DEFAULT_CONFIG):
        # initialize with default configuration and empty buffers
        self.config = config
        self.buffer = np.array([], dtype=np.float32)
        self.symbol_samples = config.N
        self.overlap = config.overlap
        self.collected_bits = []
        self.messages = []
        self.state = "SEARCHING"  # states can be searching, collecting, or decoding
        self.found_preamble_at = -1
        self.total_bits_needed = 0
        self.collecting_timeout = 0
        self.collecting_timeout_limit = 5000

    # process new audio samples and extract symbols
    def process_samples(self, new_samples: np.ndarray):
        self.buffer = np.concatenate((self.buffer, new_samples))
        while len(self.buffer) >= self.symbol_samples:
            symbol = self.buffer[:self.symbol_samples]
            # advance buffer by symbol size (with potential overlap (not for now))
            step = max(1, self.symbol_samples - self.overlap)
            self.buffer = self.buffer[step:]
            if self.config.use_mary_fsk:
                windowed = symbol * np.hanning(len(symbol))
                if self.config.fsk_mode == 4:
                    freq_list = [self.config.freq00, self.config.freq01, self.config.freq10, self.config.freq11]
                    energies = goertzel_mag(windowed, freq_list, self.config.sample_rate)
                    max_idx = np.argmax(energies)
                    if max_idx == 0:
                        self.collected_bits.extend([0, 0])
                    elif max_idx == 1:
                        self.collected_bits.extend([0, 1])
                    elif max_idx == 2:
                        self.collected_bits.extend([1, 0])
                    else:
                        self.collected_bits.extend([1, 1])
                    # update state machine (multiple times for 4-FSK)
                    self._update_state_machine()
                    self._update_state_machine()
                elif self.config.fsk_mode == 8:
                    freq_list = [
                        self.config.freq000, self.config.freq001, self.config.freq010, self.config.freq011,
                        self.config.freq100, self.config.freq101, self.config.freq110, self.config.freq111
                    ]
                    energies = goertzel_mag(windowed, freq_list, self.config.sample_rate)
                    max_idx = np.argmax(energies)
                    bits = [(max_idx >> 2) & 1, (max_idx >> 1) & 1, max_idx & 1]
                    self.collected_bits.extend(bits)
                    # update state machine (multiple times for 8-FSK)
                    self._update_state_machine()
                    self._update_state_machine()
                    self._update_state_machine()
            else:
                # process for binary FSK modulation
                windowed = symbol * np.hanning(len(symbol))
                e0 = goertzel_mag(windowed, self.config.freq0, self.config.sample_rate)
                e1 = goertzel_mag(windowed, self.config.freq1, self.config.sample_rate)
                bit = 1 if e1 > e0 * self.config.threshold_factor else 0
                self.collected_bits.append(bit)
                self._update_state_machine()

    # update the state machine based on collected bits
    def _update_state_machine(self):
        if self.state == "SEARCHING":
            self._search_for_preamble()
        elif self.state == "COLLECTING":
            self.collecting_timeout += 1
            if self.collecting_timeout > self.collecting_timeout_limit:
                self.state = "SEARCHING"
                self.collecting_timeout = 0
                return
            if len(self.collected_bits) >= self.found_preamble_at + self.total_bits_needed:
                self.state = "DECODING"
                self._decode_message()

    # search for the preamble in the collected bits
    def _search_for_preamble(self):
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
                    data_bits = ((msg_len * 8 + 3) // 4) * 7
                if self.config.use_crc:
                    data_bits += 8
                self.total_bits_needed = pre_len + sy_len + 8 + data_bits
                self.found_preamble_at = i
                self.state = "COLLECTING"
                self.collecting_timeout = 0
                return

    # decode the message from the collected bits once enough bits are gathered
    def _decode_message(self):
        i = self.found_preamble_at
        pre_len = len(self.config.preamble)
        sy_len = len(self.config.sync_pattern)
        bits = self.collected_bits[i : i + self.total_bits_needed]
        if len(bits) < self.total_bits_needed:
            self.state = "SEARCHING"
            return
        exact_pre = [int(b) for b in self.config.preamble]
        exact_sync = [int(b) for b in self.config.sync_pattern]
        data_hdr = bits[pre_len + sy_len:]
        clean_bits = exact_pre + exact_sync + data_hdr
        text, valid = decode_message(clean_bits, self.config)
        if valid:
            logging.info(f"Decoded message: {text}")
            self.messages.append(text)
            self.collected_bits = []
        else:
            logging.error("Decode failed, retrying search")
            self.collected_bits = self.collected_bits[i:]
        self.state = "SEARCHING"
        self.found_preamble_at = -1
        self.total_bits_needed = 0
        self.collecting_timeout = 0

    # get the messages that have been successfully decoded
    def get_messages(self) -> list:
        out = self.messages[:]
        self.messages = []
        return out

# unit tests for the protocol functions (should protocol.py be run)
if __name__ == "__main__":
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
        print(f"Streaming Decoder: {stream_msgs}")
        if msg == rec_msg and msg in stream_msgs:
            print("Passed")
        else:
            print("Failed")
