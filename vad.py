import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from parameters import SAMPLE_RATE
# Load VAD model once
model = load_silero_vad()


def apply_vad(audio_tensor: torch.Tensor, sample_rate: int, audio_length,return_seconds=False):
    """
    Apply Silero VAD to an audio tensor and return the voiced segments.

    :param audio_tensor: torch.Tensor (1, samples) at 16kHz
    :param sample_rate: sample rate (must be 16000)
    :param return_seconds: whether to return speech timestamps in seconds (default: False)
    :return: torch.Tensor with voiced audio only
    """
    if sample_rate != SAMPLE_RATE:
        raise ValueError("Silero VAD requires audio to be 16kHz.")

    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=sample_rate,
        return_seconds=return_seconds
    )

    if not speech_timestamps:
        return audio_tensor  # fallback

    waveform=torch.cat([audio_tensor[0][seg['start']:seg['end']] for seg in speech_timestamps]).unsqueeze(0)
    chunk_size = int(audio_length * sample_rate) 
    chunks=[]
    r = 0
    while r + chunk_size <= waveform.shape[1]:
        chunks.append(waveform[:,r:r+chunk_size])
        r += chunk_size
    return chunks
    # chunks = []
    # chunk_size = int(AUDIO_LENGTH * sample_rate) 
    # for s in speech_timestamps:
    #     start = s['start']
    #     end = s['end']
    #     l = end - start
    #     print(start/10000,end/10000,l/10000,chunk_size/10000)
    #     if l > chunk_size:
    #         r = start
    #         while r + chunk_size <= end:
    #             print(f"Chunk from {r} to {r + chunk_size}")
    #             chunks.append(audio_tensor[:, r:r + chunk_size])  # keep channel dimension
    #             r += chunk_size
    #     elif l == chunk_size:
    #         chunks.append(audio_tensor[:, start:end])
    #     else:
    #         continue
                
    # return chunks
        
    # Concatenate all speech segments
    # voiced = torch.cat([
    #     audio_tensor[0][seg['start']:seg['end']] for seg in speech_timestamps
    # ]).unsqueeze(0)

    # return voiced
