import whisperx
import torch
import os
import argparse
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
audio_file = "audios/audio.wav"
batch_size = 16 # reduce if low on GPU mem
compute_type = "int8" if device=="cpu" else "float32" # change to "int8" if low on GPU mem (may reduce accuracy)
model="medium.en"
model_dir = os.path.join("models",model)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="WhisperX")
    parser.add_argument("filename", help="Path to the input file")
    args=parser.parse_args()

    
if os.path.exists(args.filename):
    audio = whisperx.load_audio(args.filename)
    print(f"Running Whisper {model} on {device}")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model = whisperx.load_model(model, device, compute_type=compute_type,download_root=model_dir)   
    result = model.transcribe(audio, batch_size=batch_size)

    transcript_path=os.path.join("results",args.filename)
    if not os.path.exists(transcript_path):
        os.mkdir(os.path.join("results",args.filename))
    with open(os.path.join(transcript_path,"transcript.pkl"),"wb") as fp:
        pickle.dump(result,fp)

else:
    print("No such file exists")
    


    
    


    
    

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
# model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
# result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# print(result["segments"]) # after alignment

# # delete model if low on GPU resources
# # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# # 3. Assign speaker labels
# diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token="HF_TOKEN", device=device)

# # add min/max number of speakers if known
# diarize_segments = diarize_model(audio,min_speakers=2, max_speakers=2)
# # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

# result = whisperx.assign_word_speakers(diarize_segments, result)
# print(diarize_segments)
# print(result["segments"]) # segments are now assigned speaker IDs