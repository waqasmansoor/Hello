import whisperx
import torch
import os
import argparse
import pickle
from whisperx.diarize import DiarizationPipeline,assign_word_speakers
from dotenv import load_dotenv
from alignment import load_align_model
from whisperx.alignment import align
from pathlib import Path
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"
audio_file = "audios/audio.wav"
batch_size = 4 # reduce if low on GPU mem
compute_type = "int8" if device=="cpu" else "float32" # change to "int8" if low on GPU mem (may reduce accuracy)
model="medium.en"
model_dir = "model"
language_code = "en"
transcript_path="results"
ADDRESS = "http://192.168.23.142:11434"

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "HF_TOKEN")

def run_new_model(model:str,speaker_text:str):
    
    prompt = (
        f"The following is everything said by one speaker:\n\n"
        f"{speaker_text}\n\n"
        "Instructions:\n"
        "- ONLY return the name if the speaker clearly said their own name\n"
        "- If not, return: null\n"
        "- DO NOT include anything else — no explanation, no emojis, no formatting.\n"
        "- Output must be exactly one line.\n"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0
    }
    
    response = requests.post(ADDRESS + "/api/generate", json=payload)
    response.raise_for_status()

    result = response.json()

    if "response" in result:
        return result["response"].strip()
        # print(f"Results {result['response']}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="WhisperX")
    parser.add_argument("filename", help="Path to the input file")
    parser.add_argument("-d","--diarization",action="store_true",help="Enable Speaker Diarization")
    parser.add_argument("-m","--model",type=str,default=model,help="Whiserpx model")
    parser.add_argument("-o","--output",action="store_true",help="Show Results")
    args=parser.parse_args()

    
if os.path.exists(args.filename):
    
    name= Path(args.filename).stem

    audio = whisperx.load_audio(args.filename)
    filepath=os.path.join(transcript_path,args.model+"_"+name+"_transcript.pkl")
    if not os.path.exists(filepath):

        print(f"Running Whisper {args.model} on {device}...")
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(os.path.join(model_dir,model)):
            os.mkdir(os.path.join(model_dir,model))
        model = whisperx.load_model(model, device, compute_type=compute_type,download_root=model_dir)   
        result = model.transcribe(audio, batch_size=batch_size)

        
        if not os.path.exists(transcript_path):
            os.mkdir(transcript_path)
        with open(filepath,"wb") as fp:
            pickle.dump(result,fp)
    else:

        print(f"Transcript File for {args.filename} already exists.")

    filepath2=os.path.join(transcript_path,args.model+"_"+name+"_diarize.pkl")
    if args.diarization:
        if not os.path.exists(filepath2):
            with open(os.path.join(transcript_path,name+"_transcript.pkl"),"rb") as fp:
                result=pickle.load(fp)
            print(f"Performing Diarization...")
            align_model, align_metadata = load_align_model(language_code,device, model_dir = model_dir)
            result = align(result["segments"],align_model,align_metadata,audio,device,print_progress = True)
            diarize_model = DiarizationPipeline(use_auth_token=HUGGINGFACE_TOKEN, device=device)
            diarize_segments = diarize_model(args.filename, min_speakers=2, max_speakers=4)
            result = assign_word_speakers(diarize_segments, result)

            if not os.path.exists(transcript_path):
                os.mkdir(transcript_path)
            with open(filepath2,"wb") as fp:
                pickle.dump(result,fp)
        else:
            print(f"Diarize File for {args.filename} already exists.")

    if args.output:
        if os.path.exists(filepath2):
            with open(filepath2,"rb") as diarize_file:
                text=pickle.load(diarize_file)
            def get_speakers(segments):
                return list({s["speaker"] for s in segments if "speaker" in s})
            
            def count_tokens(text:str):
                tokens=len(text)//4
                return tokens
            
            def get_speaker_text(speaker:str,full_text:str):
                '''
                INPUT:
                    speaker SPEAKER_01
                    
                    text:
                        [SPEAKER_01] speaker text...
                        [SPEAKER_02] speaker text...
                '''
                text = ""
                for line in full_text:

                    capture_speaker_name = False
                    name=""
                    for i,c in enumerate(line):
                        if c == "[":
                            capture_speaker_name=True
                        if capture_speaker_name:
                            name+=c
                        if c== "]":
                            break
                            
                    
                    name = name[1:-1] #Remove brackets
                    if name == speaker:
                        t= line[i+1:].strip()+"\n"
                        text+=t            
                        
                        # print(name,line[i+1:])
                
                print(f"{speaker}, Tokens: {count_tokens(text)}")
                return text
            
            lines = []
            speakers_data=[]
            
            for segment in text["segments"]:
                if "speaker" in segment:
                    speaker = segment["speaker"]
                else:
                    speaker = "UNKNOWN"
                speakers_data.append(
                    {
                        "speaker":speaker,
                        "text":segment["text"].strip()
                    }
                )
                lines.append(f"[{speaker}] {segment['text'].strip()}")
            
            speakers = get_speakers(text["segments"])
            print(f"Getting Speakers...")
            for s in speakers:
                speaker_text = get_speaker_text(s,lines)
                # print(speaker_text)
                speaker_name = run_new_model("mixtral",speaker_text)
                for sd in speakers_data:
                    if sd["speaker"] == s:
                        sd["speaker"]=speaker_name if speaker_name!="null" else sd["speaker"]

            for sd in speakers_data:
                print(f"[{sd['speaker']}] {sd['text']}")
            
                

else:
    print("No such file exists")
    


