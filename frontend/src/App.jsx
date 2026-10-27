
import { useRef, useState } from "react";
import { Room, RoomEvent, Track } from "livekit-client";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [status, setStatus] = useState("Ready to connect");
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const roomRef = useRef(null);
  const audioContainerRef = useRef(null);

  async function startConversation() {
    setBusy(true);
    setStatus("Connecting to LiveKit...");

    try {
      const response = await fetch(`${API_URL}/api/voice/token`, {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error(`Token request failed (${response.status})`);
      }

      const data = await response.json();
      const room = new Room();

      room.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === Track.Kind.Audio && audioContainerRef.current) {
          const element = track.attach();
          audioContainerRef.current.appendChild(element);
        }
      });

      room.on(RoomEvent.TrackUnsubscribed, (track) => {
        track.detach().forEach((element) => element.remove());
      });

      room.on(RoomEvent.Disconnected, () => {
        setConnected(false);
        setStatus("Disconnected");
      });

      room.on(RoomEvent.ParticipantConnected, () => {
        setStatus("Connected — participant joined");
      });

      room.on(RoomEvent.TrackPublished, () => {
        setStatus("Connected — waiting for voice response");
      });

      await room.connect(data.server_url, data.participant_token);
      roomRef.current = room;

      await room.localParticipant.setMicrophoneEnabled(true);

      setConnected(true);
      setStatus("Connected — speak when the agent is ready");
    } catch (error) {
      console.error("Voice connection failed:", error);
      setStatus(`Connection failed: ${error.message}`);
      if (roomRef.current) {
        await roomRef.current.disconnect();
        roomRef.current = null;
      }
    } finally {
      setBusy(false);
    }
  }

  async function stopConversation() {
    setBusy(true);

    try {
      if (roomRef.current) {
        await roomRef.current.disconnect();
        roomRef.current = null;
      }
      if (audioContainerRef.current) {
        audioContainerRef.current.replaceChildren();
      }
      setConnected(false);
      setStatus("Conversation stopped");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app">
      <section className="voice-card">
        <div className="eyebrow">ECOMMERCE SUPPORT</div>
        <h1>Voice Assistant</h1>
        <p className="description">
          Talk naturally with your e-commerce customer service assistant.
        </p>

        <div className={`status ${connected ? "connected" : ""}`}>
          <span className="status-dot" />
          {status}
        </div>

        <div className="actions">
          <button
            className="start-button"
            onClick={startConversation}
            disabled={busy || connected}
          >
            {busy ? "Please wait..." : "Start conversation"}
          </button>

          <button
            className="stop-button"
            onClick={stopConversation}
            disabled={busy || !connected}
          >
            Stop conversation
          </button>
        </div>

        <p className="hint">
          Allow microphone access when your browser asks.
        </p>

        <div ref={audioContainerRef} />
      </section>
    </main>
  );
}

export default App;
