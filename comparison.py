
import pickle


with open("transcribe.pkl", 'rb') as f:
    results = pickle.load(f)

for r in results:
    for s in r[0]['segments']:
        print(f"{s['start']} - {s['end']}: {s['text']}")

with open("align.pkl",'rb') as f:
    results = pickle.load(f)

# for r in results:
#     for s in r[0]['segments']:
#         print(f"{s['start']} - {s['end']}: {s['text']}, words: {s['words']}")

with open("diarize.pkl",'rb')as f:
    results = pickle.load(f)
    for r in results:
        
        for s in r[0]['segments']:
            speaker = ''
            words = {}
            for w in s["words"]:
                
                speaker = w["speaker"]
                if not speaker in words:
                    words[speaker] = []
                words[speaker].append(w["word"])
            
            for k,v in words.items():
                print(" ".join(v),k)
            # "".join(t for t in words.values())
                
        #     print(f"{s['start']} - {s['end']}: {s['text']}")