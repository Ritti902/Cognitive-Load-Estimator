import { useEffect, useRef, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer
} from 'recharts';

import { startCapture, stopCapture } from './capture';
import Quiz from './Quiz';

const API_URL = 'http://127.0.0.1:8000';
const COLORS = { LOW: '#1D9E75', MEDIUM: '#BA7517', OVERLOADED: '#E24B4A' };

export default function Dashboard() {
  const [loadClass, setLoadClass] = useState(null);
  const [loadScore, setLoadScore] = useState(null);
  const [predictionStatus, setPredictionStatus] = useState('waiting_for_session');
  const [modelError, setModelError] = useState('');
  const [history, setHistory] = useState([]);
  const [showBreak, setShowBreak] = useState(false);
  const [session, setSession] = useState(null);
  const [apiError, setApiError] = useState('');
  const sessionIdRef = useRef(null);
  const overloadCount = useRef(0);

  useEffect(() => {
    let mounted = true;
    let reconnectTimer = null;
    let ws = null;
    let activePoll = null;

    async function refreshSession() {
      try {
        const response = await fetch(`${API_URL}/session/active`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'FastAPI session endpoint failed');
        const nextSession = data.active ? data.session : null;
        const nextId = data.active ? data.session_id : null;
        if (!mounted) return;
        if (sessionIdRef.current !== nextId) {
          sessionIdRef.current = nextId;
          setHistory([]);
          setLoadClass(null);
          setLoadScore(null);
          setPredictionStatus(nextId ? 'waiting_for_features' : 'waiting_for_session');
          overloadCount.current = 0;
        }
        setSession(nextSession);
        setApiError('');
        if (nextId) startCapture(nextId);
        else stopCapture();
      } catch (error) {
        if (!mounted) return;
        sessionIdRef.current = null;
        setSession(null);
        setApiError(error.message);
        setPredictionStatus('api_unavailable');
        setLoadClass(null);
        setLoadScore(null);
        stopCapture();
      }
    }

    function connectDashboard() {
      if (!mounted) return;
      ws = new WebSocket('ws://127.0.0.1:8000/ws/dashboard');
      ws.onmessage = event => {
        try {
          const data = JSON.parse(event.data);
          if (data.session_id && data.session_id !== sessionIdRef.current) return;
          setPredictionStatus(data.status || 'waiting_for_features');
          setModelError(data.model?.error || data.persistence_error || data.error || '');
          if (data.status !== 'ready') {
            setLoadClass(null);
            setLoadScore(null);
            return;
          }
          setLoadClass(data.load_class);
          setLoadScore(data.load_score);
          setHistory(previous => {
            const score = data.load_class === 'LOW' ? 0 : data.load_class === 'MEDIUM' ? 1 : 2;
            return [...previous.slice(-59), { time: new Date().toLocaleTimeString(), score }];
          });
          if (data.load_class === 'OVERLOADED') {
            overloadCount.current += 1;
            if (overloadCount.current >= 2) {
              setShowBreak(true);
              overloadCount.current = 0;
            }
          } else {
            overloadCount.current = 0;
          }
        } catch (error) {
          console.error('Invalid dashboard update:', error);
        }
      };
      ws.onclose = () => {
        if (mounted) reconnectTimer = setTimeout(connectDashboard, 2000);
      };
      ws.onerror = () => ws.close();
    }

    refreshSession();
    activePoll = setInterval(refreshSession, 3000);
    connectDashboard();
    return () => {
      mounted = false;
      clearInterval(activePoll);
      clearTimeout(reconnectTimer);
      if (ws) ws.close();
      stopCapture();
    };
  }, []);

  const activeClass = loadClass || 'WAITING';
  const color = COLORS[loadClass] || '#64748b';
  const statusMessage = {
    waiting_for_session: 'Create a session to begin collecting data.',
    waiting_for_features: 'Waiting for CV, behaviour, and quiz features for this session.',
    waiting_for_fresh_features: 'Webcam or behaviour readings are stale; waiting for fresh data.',
    model_unavailable: 'Predictions are disabled because the trained ONNX model is unavailable.',
    inference_error: 'The model could not process the current feature vector.',
    api_unavailable: 'FastAPI is unavailable.'
  }[predictionStatus] || 'Waiting for a model prediction.';

  return (
    <div style={{ fontFamily: 'Arial', padding: 24, maxWidth: 900, margin: '0 auto' }}>
      <h2 style={{ color: '#185FA5' }}>Cognitive Load Monitor</h2>
      <p>
        Active session: <strong>{session?.id || 'None'}</strong>
        {session && <> · Participant {session.student_id} · {session.difficulty}</>}
      </p>
      {session && <p>
        NASA-TLX: <strong>{session.assessment_status === 'submitted'
          ? `Submitted · ${Number(session.nasa_tlx_score).toFixed(1)}/100 · ${session.load_class}`
          : session.assessment_status === 'unverified'
            ? 'Stored value is unverified'
            : session.assessment_status === 'missing'
              ? 'Missing'
              : 'Pending'}</strong>
        {['pending', 'missing'].includes(session.assessment_status) && <>
          {' · After the study, '}
          <a href={`http://127.0.0.1:5003/tlx/${encodeURIComponent(session.id)}`} target="_blank" rel="noreferrer">submit NASA-TLX</a>
          {' for this session.'}
        </>}
      </p>}
      {!session && <p>Start one with <code>python collect_sessions.py</code>.</p>}
      {apiError && <p role="alert">API: {apiError}</p>}

      <div style={{ background: color + '22', border: `2px solid ${color}`, borderRadius: 16, padding: 24, marginBottom: 24, display: 'flex', alignItems: 'center', gap: 24 }}>
        <div style={{ fontSize: 48, fontWeight: 'bold', color }}>{loadScore === null ? '—' : `${Math.round(loadScore * 100)}%`}</div>
        <div>
          <div style={{ fontSize: 28, fontWeight: 'bold', color }}>{activeClass}</div>
          <div style={{ color: '#555', fontSize: 14 }}>
            {loadScore === null ? statusMessage : 'Top class probability (softmax, uncalibrated) · updates every 5 sec'}
          </div>
          {modelError && <div style={{ color: '#8b1e1e', fontSize: 12, marginTop: 6 }}>{modelError}</div>}
        </div>
      </div>

      <h3>Session History</h3>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={history}>
          <XAxis dataKey="time" hide />
          <YAxis domain={[0, 2]} ticks={[0, 1, 2]} tickFormatter={value => ['Low', 'Med', 'High'][value]} />
          <Tooltip formatter={value => ['Low', 'Medium', 'Overloaded'][value]} />
          <Line type="monotone" dataKey="score" stroke="#185FA5" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>

      <Quiz sessionId={session?.id} difficulty={session?.difficulty} />

      {showBreak && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ background: 'white', borderRadius: 16, padding: 40, textAlign: 'center', maxWidth: 400 }}>
            <h2 style={{ color: '#E24B4A' }}>You seem overloaded!</h2>
            <p>You have been struggling for 10+ seconds.<br />Take a 2-minute break.</p>
            <button onClick={() => setShowBreak(false)} style={{ background: '#185FA5', color: 'white', border: 'none', borderRadius: 8, padding: '12px 32px', fontSize: 16, cursor: 'pointer' }}>
              OK, I will take a break
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
