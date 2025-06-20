// src/App.tsx
import React, { useRef, useState, useEffect } from 'react';

declare global {
  interface Window {
    Recorder: any;
  }
}

function App() {
  const [frame, setFrame] = useState(1);
  const [isRecording, setIsRecording] = useState(false);
  const [username, setUsername] = useState<string | null>(null);
  const [url, setUrl] = useState<RequestInfo | URL>('http://localhost:5000/train')
  const [isLoading, setIsLoading] = useState(false);
  const [filepath, setFilepath] = useState<String | null>(null);


  const gumStream = useRef<MediaStream | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const recorder = useRef<any>(null);
  const input = useRef<MediaStreamAudioSourceNode | null>(null);

  useEffect(() => {
    const script = document.createElement('script')
    script.src = './recorder.js';
    script.async = true;
    document.body.appendChild(script);
    return () => {
      document.body.removeChild(script);
    }
  }, [])

  useEffect(() => {
    if (frame == 1) {
      setUrl('http://localhost:5000/save_embeddings')
    }
    else {
      setUrl('http://localhost:5000/process_audio')
    }
  }, [frame])

  const askUsername = () => {
    const name = prompt("What's your name?");
    if (name) setUsername(name.trim());
  };

  const startStreaming = async () => {
    console.log('Streaming Started');

    if (isRecording) {
      stopRecording()
    } else {
      startRecording()
    }


  }
  const checkUserAndStartRecording = async () => {
    if (!username) {
      askUsername();
      if (!username) return;
    }

    startRecording()
  }
  const startRecording = async () => {
    console.log('Recording button clicked');

    const constraints = { audio: true, video: false };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      console.log('getUserMedia() success');

      gumStream.current = stream
      audioContext.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      input.current = audioContext.current.createMediaStreamSource(stream);

      recorder.current = new window.Recorder(input.current, { numChannels: 1 });
      recorder.current.record()
      console.log("Recording started");
      setIsRecording(true);


    } catch (err) {
      console.log(`Error accessing microphone:`, err);
    }
  }
  const stopRecording = () => {
    console.log('Stop button clicked');

    recorder.current?.stop();
    gumStream.current?.getAudioTracks().forEach((track) => track.stop());

    recorder.current?.exportWAV(sendAudio);
    setIsRecording(false);
  };

  const get_timestamp = () => {
    const now = new Date();
    const pad = (n: number) => String(n).padStart(2, '0');
    const timestamp: string = now.getFullYear().toString() +
      pad(now.getMonth() + 1) +
      pad(now.getDate()) +
      '-' +
      pad(now.getHours()) +
      pad(now.getMinutes()) +
      pad(now.getSeconds());
    return timestamp;
  }
  const sendAudio = (blob: Blob) => {
    setIsLoading(true); // Start loading

    const formData = new FormData();
    const source = 'meeting';
    const num1 = Math.floor(Math.random() * 10000);
    const num2 = Math.floor(Math.random() * 10000);
    const pad = (n: number) => String(n).padStart(4, '0');
    const ts = get_timestamp();
    const filename = `${source}-${pad(num1)}-${pad(num2)}-${ts}-0.0.wav`;
    formData.append('file', blob, filename);

    if (frame === 1) {
      if (!username) {
        askUsername();
        if (!username) return;
      }
      formData.append('name', username);
    }

    fetch(url, {
      method: 'POST',
      body: formData
    })
      .then((res) => res.json())
      .then((data) => {
        console.log('✅ Server response:', data);
        const file_path = data.transcript_id;
        setFilepath(file_path)
      })
      .catch((err) => {
        console.error('❌ Upload failed:', err);
      })
      .finally(() => {
        setIsLoading(false); // Stop loading
      });
  };

  const resetUsername = () => {
    setUsername(null);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-6 relative">

      {/* Username Display with ❌ */}
      {username && (
        <div className="absolute top-4 right-4 bg-white border px-4 py-2 rounded shadow-md flex items-center gap-2">
          <span className="font-medium text-gray-800">👤 {username}</span>
          <button onClick={resetUsername} className="text-red-600 hover:text-red-800 font-bold text-lg">❌</button>
        </div>
      )}

      {/* Left Arrow on Frame 2 */}
      {frame === 2 && (
        <button
          onClick={() => setFrame(1)}
          className="absolute left-4 top-1/2 transform -translate-y-1/2 text-3xl text-gray-600 hover:text-gray-800"
        >
          ⬅️
        </button>

      )}
      {filepath && (
        <div className="mt-4">
          <a
            href={filepath.replace('./', '/')} // Make it usable if served statically
            className="text-blue-600 underline hover:text-blue-800"
            target="_blank"
            rel="noopener noreferrer"
          >
            {filepath}
          </a>
        </div>
      )}
      {isLoading && (
        <div className="mt-4 text-center">
          <div className="animate-spin h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto"></div>
          <p className="mt-2 text-gray-700">Processing audio...</p>
        </div>
      )}

      {/* Right Arrow on Frame 1 */}
      {frame === 1 && (
        <button
          onClick={() => setFrame(2)}
          className="absolute right-4 top-1/2 transform -translate-y-1/2 text-3xl text-gray-600 hover:text-gray-800"
        >
          ➡️
        </button>
      )}

      {frame === 1 ? (
        <>
          <h1 className="text-3xl font-bold mb-6">🎤 Voice Recorder</h1>
          <div className="flex gap-4 mb-6">
            <button
              onClick={checkUserAndStartRecording}
              disabled={isRecording}
              className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
            >
              Record
            </button>
            <button
              onClick={stopRecording}
              disabled={!isRecording}
              className="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
            >
              Stop
            </button>
          </div>
          {!username && (
            <button
              onClick={askUsername}
              className="text-blue-600 underline hover:text-blue-800"
            >
              Set your name
            </button>
          )}
        </>
      ) : (
        <>
          <h1 className="text-3xl font-bold mb-6">Record</h1>
          <button
            className="bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-6 rounded"
            onClick={startStreaming}
          >
            Stream
          </button>
        </>
      )}
    </div>
  );
}

export default App;
