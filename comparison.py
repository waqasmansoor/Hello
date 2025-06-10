
import pickle
import numpy as np

# with open("transcribe.pkl", 'rb') as f:
#     results = pickle.load(f)

# for r in results:
#     for s in r[0]['segments']:
#         print(s)
        # print(f"{s['start']} - {s['end']}: {s['text']}")

# with open("align.pkl",'rb') as f:
#     results = pickle.load(f)

# for r in results:
#     for s in r[0]['segments']:
#         print(f"{s['start']} - {s['end']}: {s['text']}, words: {s['words']}")

# with open("diarize.pkl",'rb')as f:
#     results = pickle.load(f)
#     for r in results:
        
#         for s in r[0]['segments']:
#             if "speaker" in s:
#                 print(f"{s['text']}: {s['speaker']}")
#             else:

#                 print("...",s)
            
            # speaker = ''
            # words = {}
            # for w in s["tokens"]:
                
            #     speaker = w["speaker"]
            #     if not speaker in words:
            #         words[speaker] = []
            #     words[speaker].append(w["token"])
            
            # for k,v in words.items():
            #     print(" ".join(v),k)
            # "".join(t for t in words.values())
                
            # print(f"{s['start']} - {s['end']}: {s['text']}")
from sqlalchemy import create_engine, or_, func, distinct, and_, exists
from sqlalchemy.orm import sessionmaker, scoped_session, aliased
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (Column, String, Float, ForeignKey, Integer, Text, DateTime, BOOLEAN)
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import requests

Base = declarative_base()
class DataBase:
    def __init__(self, db_path: str):
        self.engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)

        # Thread-safe session factory
        self.Session = scoped_session(sessionmaker(bind=self.engine))

class Transcript(Base):
    __tablename__ = 'transcript'

    id = Column(String, primary_key=True)  # Unique ID (filename or UUID)
    time = Column(DateTime, default=datetime.utcnow, nullable=False)
    source = Column(String, nullable=False)
    title = Column(String, default="")
    audio_file = Column(String)
    processed_audio_file = Column(String)
    direction = Column(String, nullable=True)
    incoming_phone = Column(String, nullable=True)
    outgoing_phone = Column(String, nullable=True)
    processing_model = Column(String, nullable=True)
    processing_time = Column(Float, nullable=True)
    audio_time = Column(Float, nullable=True)
    ai_did_good = Column(BOOLEAN, default=True)
    speaker_processing_model = Column(String, nullable=True)
    speaker_processing_time = Column(Float, nullable=True)
    speaker_ai_did_good = Column(BOOLEAN, default=True)

    segments = relationship("TranscriptSegment", back_populates="transcript", cascade="all, delete-orphan")
    summary = relationship(
        "Summary",
        back_populates="transcript",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="joined",  # or "selectin"
    )

    def get_speakers(self):
        return list({s.speaker for s in self.segments if s.speaker})
    
    def to_string(self):
        lines = []
        for segment in sorted(self.segments, key=lambda s: s.start):
            speaker = segment.speaker or "UNKNOWN"
            lines.append(f"[{speaker}] {segment.text.strip()}")
        return "\n".join(lines)

class TranscriptSegment(Base):
    __tablename__ = 'transcript_segment'

    id = Column(Integer, primary_key=True, autoincrement=True)
    transcript_id = Column(String, ForeignKey('transcript.id'))
    start = Column(Float)
    end = Column(Float)
    audio_file = Column(String)
    text = Column(Text, default="")
    speaker = Column(String, default="")

    transcript = relationship("Transcript", back_populates="segments")

    

class Summary(Base):
    __tablename__ = "summary"

    id = Column(Integer, primary_key=True)
    text = Column(Text)
    audio_file = Column(String)
    transcript_id = Column(String, ForeignKey("transcript.id"), nullable=False, unique=True)
    processing_model = Column(String, nullable=True)
    processing_time = Column(Float, nullable=True)
    ai_did_good = Column(BOOLEAN, default=True)

    transcript = relationship("Transcript", back_populates="summary")

t = Transcript(
    id=str(uuid.uuid4()),
    audio_file="/usr/code/AIBox/Hello/data/diarize.pkl",
    processed_audio_file="/usr/code/AIBox/Hello/data/diarize.pkl",
)

DB_PATH = "call_auditor.db"
DataBase(db_path=DB_PATH)
with open("/usr/code/AIBox/Hello/data/diarize.pkl", "rb") as f:
    result = pickle.load(f)
    result = result[0][0]
    diarization_segments = {
        "diarize_segments": None,
        "segments_with_speakers": result["segments"]
    }

for seg in diarization_segments["segments_with_speakers"]:
    speaker = seg.get("speaker", "unknown")
    line = f"[{seg['start']:.2f}–{seg['end']:.2f}] Speaker {speaker}: {seg['text']}"
    

    transcription_segment = TranscriptSegment(
        start=seg['start'],
        end=seg['end'],
        audio_file="/usr/code/AIBox/Hello/data/diarize.pkl",
        text=seg['text'],
        speaker=speaker
    )
    transcription_segment.audiofile = "/usr/code/AIBox/Hello/data/diarize.pkl"

    t.segments.append(transcription_segment)

speakers = t.get_speakers()
transcript = t

lines = t.to_string()
start = False
names = []
texts=[]
text = ""
for l in lines:
    if l == "[":
        start = True
        name = ""
        if text != "":
            texts.append(text)
        continue
    if l == "]":
        start = False
        names.append(name)
        text = ""
        continue

    if start:
        name = "".join([name,l])
    else:
        text += l
texts.append(text)

def count_tokens(part):
    tokens=0
    for i in part:        
        tokens+=len(i[1])//4
    return tokens


tt=0
tthresh=500
text_parts=[]
k=0
for i in range(len(names)):
    text = texts[i]
    name = names[i]
    tokens = len(text)//4
    tt += tokens
    if tt > tthresh:
        tt=0
        part = []
        
        for j in range(k,i):
            part.append((names[j],texts[j]))
        k=i
        tokens_in_part = count_tokens(part)
        text_parts.append(part)



def get_speakers(part):
    return np.unique([x[0] for x in part])

for i,part in enumerate(text_parts):    
    speakers=get_speakers(text_parts[i])
    dict_format = "{" + ", ".join([f"'{s}': null" for s in speakers]) + "}"
    transcript_text = "\n".join([f"[{s}] {t}" for s, t in part])
    # print(transcript_text)
# print(transcript.to_string())
# print(speakers)

    prompt = (
        "Extract full names for each speaker from the transcript below. "
        "Return only this Python dictionary:\n"
        f"{dict_format}\n\n"
        "Rules:\n"
        "- Only use full names that are explicitly stated.\n"
        "- Do not guess or make up names.\n"
        "- Use null if the full name is not mentioned.\n"
        "- Do not add any extra explanation.\n\n"
        f"Transcript:\n{transcript_text}\n"
    )

    payload = {
        "model": "mistral:7b",
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0
    }

    response = requests.post("http://127.0.0.1:11434" + "/api/generate", json=payload)
    response.raise_for_status()

    result = response.json()
    print(result)