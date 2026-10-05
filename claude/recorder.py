"""
recorder.py — Microphone recorder with Voice Activity Detection (VAD)

Automatically starts capturing when speech is detected,
and stops after a configurable silence period.
"""

import numpy as np
import sounddevice as sd
from collections import deque


class AudioRecorder:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        silence_threshold: float = 0.015,
        silence_duration: float = 1.5,
        max_duration: float = 30.0,
        pre_buffer_duration: float = 0.5,
        chunk_duration: float = 0.05,  # 50ms chunks for responsive VAD
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.max_duration = max_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.max_silence_chunks = int(silence_duration / chunk_duration)
        self.max_chunks = int(max_duration / chunk_duration)
        self.pre_buffer_size = int(pre_buffer_duration / chunk_duration)

    def record(self, on_speech_start=None, on_speech_end=None) -> np.ndarray:
        """
        Record audio with VAD.
        - Waits silently until speech is detected
        - Records until silence_duration of silence
        - Returns float32 numpy array at sample_rate
        
        Callbacks:
          on_speech_start(): called when speech starts
          on_speech_end():   called when silence ends recording
        """
        audio_frames: list[np.ndarray] = []
        silence_chunks = 0
        speaking = False
        pre_buffer: deque[np.ndarray] = deque(maxlen=self.pre_buffer_size)

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.chunk_size,
        ) as stream:
            total_chunks = 0

            while total_chunks < self.max_chunks:
                chunk_data, _ = stream.read(self.chunk_size)
                chunk = chunk_data.flatten()
                amplitude = float(np.abs(chunk).mean())
                total_chunks += 1

                if not speaking:
                    pre_buffer.append(chunk)
                    if amplitude > self.silence_threshold:
                        # Speech started — include pre-buffer to avoid clipping
                        speaking = True
                        if on_speech_start:
                            on_speech_start()
                        for pre_chunk in pre_buffer:
                            audio_frames.append(pre_chunk)
                        audio_frames.append(chunk)
                else:
                    audio_frames.append(chunk)

                    if amplitude < self.silence_threshold:
                        silence_chunks += 1
                        if silence_chunks >= self.max_silence_chunks:
                            if on_speech_end:
                                on_speech_end()
                            break
                    else:
                        silence_chunks = 0

        if not audio_frames:
            return np.array([], dtype=np.float32)

        return np.concatenate(audio_frames, axis=0).astype(np.float32)

    def list_devices(self):
        """Print available audio input devices."""
        devices = sd.query_devices()
        print("\nAvailable audio devices:")
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                marker = "→" if i == sd.default.device[0] else " "
                print(f"  {marker} [{i}] {d['name']}")
        print()
