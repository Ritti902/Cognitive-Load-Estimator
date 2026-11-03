import { io } from 'socket.io-client';

let eventBuffer = [];
let lastMouseMoveTime = Date.now();
let lastScrollY = window.scrollY;
let socket = null;
let intervalId = null;
let captureSessionId = null;

function handleKeydown(event) {
  // Store key categories only; never retain the key text or typed content.
  eventBuffer.push({
    type: 'key',
    key_type: event.key === 'Backspace' ? 'backspace'
      : event.key === ' ' ? 'space'
      : event.key.length === 1 ? 'letter' : 'other'
  });
}

function handleMousemove() {
  lastMouseMoveTime = Date.now();
}

function handleScroll() {
  const currentY = window.scrollY;
  eventBuffer.push({ type: 'scroll', direction: currentY > lastScrollY ? 'down' : 'up' });
  lastScrollY = currentY;
}

function computeFeatures(sessionId) {
  const now = Date.now();
  const keys = eventBuffer.filter(event => event.type === 'key');
  const scrolls = eventBuffer.filter(event => event.type === 'scroll');
  const letters = keys.filter(key => key.key_type === 'letter').length;
  const backspaces = keys.filter(key => key.key_type === 'backspace').length;
  let reversals = 0;
  for (let i = 1; i < scrolls.length; i += 1) {
    if (scrolls[i].direction !== scrolls[i - 1].direction) reversals += 1;
  }
  const mouse_idle_sec = (now - lastMouseMoveTime) / 1000;
  eventBuffer = [];
  return {
    session_id: sessionId,
    timestamp: Math.floor(now / 1000),
    // One-second window: characters/5 gives words, then *60 gives WPM.
    wpm: Math.round((letters * 12) * 10) / 10,
    backspace_rate: Math.round((keys.length ? backspaces / keys.length : 0) * 100) / 100,
    scroll_reversals: reversals,
    mouse_idle_sec: Math.round(mouse_idle_sec * 10) / 10
  };
}

export function startCapture(sessionId) {
  if (!sessionId) {
    stopCapture();
    return;
  }
  if (socket && captureSessionId === sessionId) return;
  if (socket) stopCapture();

  captureSessionId = sessionId;
  eventBuffer = [];
  lastMouseMoveTime = Date.now();
  lastScrollY = window.scrollY;
  socket = io('http://127.0.0.1:5001', { reconnection: true });
  socket.on('connect_error', error => console.error('Behaviour bridge connection error:', error.message));

  document.addEventListener('keydown', handleKeydown);
  document.addEventListener('mousemove', handleMousemove, { passive: true });
  window.addEventListener('scroll', handleScroll, { passive: true });

  intervalId = setInterval(() => {
    const features = computeFeatures(captureSessionId);
    if (socket && socket.connected) {
      socket.emit('behaviour_features', features, acknowledgement => {
        if (acknowledgement && acknowledgement.status === 'error') {
          console.error('Behaviour features were rejected:', acknowledgement.message);
        }
      });
    }
  }, 1000);
}

export function stopCapture() {
  document.removeEventListener('keydown', handleKeydown);
  document.removeEventListener('mousemove', handleMousemove);
  window.removeEventListener('scroll', handleScroll);
  if (intervalId !== null) {
    clearInterval(intervalId);
    intervalId = null;
  }
  if (socket) {
    socket.disconnect();
    socket = null;
  }
  captureSessionId = null;
  eventBuffer = [];
}
