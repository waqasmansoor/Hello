// src/App.tsx
import React, { useRef, useState, useEffect } from 'react';

declare global {
  interface Window {
    Recorder: any;
  }
}

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [username, setUsername] = useState<string | null>(null);

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

  const askUsername = () => {
    const name = prompt("What's your name?");
    if (name) setUsername(name.trim());
  };

  const startRecording = async () => {
    console.log('Recording button clicked');
    if (!username){
      askUsername();
      if (!username) return;
    }
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

    recorder.current?.exportWAV(sendAudioToServer);
    setIsRecording(false);
  };

  const sendAudioToServer = (blob: Blob) => {
    if (!username){
      askUsername();
      if (!username) return;
    }
    const formData = new FormData();
    formData.append('audio', blob, 'recording.wav');
    formData.append('name', username);
    fetch('http://localhost:5000/', {
      method: 'POST',
      body: formData
    })
      .then((res) => res.json())
      .then((data) => {
        console.log('✅ Server response:', data);
      })
      .catch((err) => {
        console.error('❌ Upload failed:', err);
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
          <button onClick={resetUsername} className="text-red-600 hover:text-red-800 font-bold text-lg">
            ❌
          </button>
        </div>
      )}

      <h1 className="text-3xl font-bold mb-6">🎤 Voice Recorder</h1>

      <div className="flex gap-4 mb-6">
        <button
          onClick={startRecording}
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
    </div>
  );
}

export default App;
