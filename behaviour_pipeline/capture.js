// Standalone browser capture helper. The React dashboard uses src/capture.js;
// this version remains available to pages that load the original script.
(() => {
  if (window.__cognitiveLoadCapture) return;

  const state = { events: [], lastMouse: Date.now(), lastScrollY: window.scrollY, socket: null, timer: null, sessionId: null, polling: false };
  window.__cognitiveLoadCapture = state;

  function keydown(event) {
    state.events.push({ type: 'key', keyType: event.key === 'Backspace' ? 'backspace' : event.key === ' ' ? 'space' : event.key.length === 1 ? 'letter' : 'other' });
    if (state.events.length > 2000) state.events.splice(0, state.events.length - 2000);
  }
  function mousemove() { state.lastMouse = Date.now(); }
  function scroll() {
    const currentY = window.scrollY;
    state.events.push({ type: 'scroll', direction: currentY > state.lastScrollY ? 'down' : 'up' });
    if (state.events.length > 2000) state.events.splice(0, state.events.length - 2000);
    state.lastScrollY = currentY;
  }

  function featureWindow(sessionId) {
    const now = Date.now();
    const keys = state.events.filter(item => item.type === 'key');
    const scrolls = state.events.filter(item => item.type === 'scroll');
    let reversals = 0;
    for (let i = 1; i < scrolls.length; i += 1) if (scrolls[i].direction !== scrolls[i - 1].direction) reversals += 1;
    const result = {
      session_id: sessionId,
      timestamp: Math.floor(now / 1000),
      wpm: keys.filter(item => item.keyType === 'letter').length * 12,
      backspace_rate: keys.length ? keys.filter(item => item.keyType === 'backspace').length / keys.length : 0,
      scroll_reversals: reversals,
      mouse_idle_sec: Math.round(((now - state.lastMouse) / 1000) * 10) / 10
    };
    state.events = [];
    return result;
  }

  async function syncActiveSession() {
    try {
      const response = await fetch('http://127.0.0.1:8000/session/active');
      const data = await response.json();
      const sessionId = data.active ? data.session_id : null;
      if (sessionId !== state.sessionId) state.events = [];
      state.sessionId = sessionId;
      return sessionId;
    } catch (error) {
      console.error('Could not read active cognitive-load session:', error.message);
      state.sessionId = null;
      return null;
    }
  }

  async function startCapture() {
    if (state.socket) return;
    if (typeof window.io !== 'function') {
      console.error('Socket.IO client is required before behaviour_pipeline/capture.js.');
      return;
    }
    state.socket = window.io('http://127.0.0.1:5001', { reconnection: true });
    document.addEventListener('keydown', keydown);
    document.addEventListener('mousemove', mousemove, { passive: true });
    window.addEventListener('scroll', scroll, { passive: true });
    state.timer = setInterval(async () => {
      if (state.polling) return;
      state.polling = true;
      try {
        const sessionId = await syncActiveSession();
        if (sessionId && state.socket && state.socket.connected) {
          state.socket.emit('behaviour_features', featureWindow(sessionId));
        } else {
          state.events = [];
        }
      } finally {
        state.polling = false;
      }
    }, 1000);
  }

  function stopCapture() {
    document.removeEventListener('keydown', keydown);
    document.removeEventListener('mousemove', mousemove);
    window.removeEventListener('scroll', scroll);
    if (state.timer) clearInterval(state.timer);
    if (state.socket) state.socket.disconnect();
    state.timer = null;
    state.socket = null;
    state.sessionId = null;
    state.events = [];
  }

  window.startCognitiveLoadCapture = startCapture;
  window.stopCognitiveLoadCapture = stopCapture;
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', startCapture, { once: true });
  else startCapture();
})();
