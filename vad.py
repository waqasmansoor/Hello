# vad.py

import torch
from silero_vad import load_silero_vad, get_speech_timestamps

# Load VAD model once
model = load_silero_vad()

def apply_vad(audio_tensor: torch.Tensor, sample_rate: int, return_seconds=False):
    """
    Apply Silero VAD to an audio tensor and return the voiced segments.

    :param audio_tensor: torch.Tensor (1, samples) at 16kHz
    :param sample_rate: sample rate (must be 16000)
    :param return_seconds: whether to return speech timestamps in seconds (default: False)
    :return: torch.Tensor with voiced audio only
    """
    if sample_rate != 16000:
        raise ValueError("Silero VAD requires audio to be 16kHz.")

    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=sample_rate,
        return_seconds=return_seconds
    )

    if not speech_timestamps:
        return audio_tensor  # fallback

    # Concatenate all speech segments
    voiced = torch.cat([
        audio_tensor[0][seg['start']:seg['end']] for seg in speech_timestamps
    ]).unsqueeze(0)

    return voiced
