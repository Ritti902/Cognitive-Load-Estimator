import { render, screen, waitFor } from '@testing-library/react';
import App from './App';
import { startCapture } from './capture';

jest.mock('./capture', () => ({ startCapture: jest.fn(), stopCapture: jest.fn() }));

class MockWebSocket {
  constructor() { this.readyState = 1; }
  close() { this.readyState = 3; }
}

beforeEach(() => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ active: false, session_id: null, session: null })
  });
  global.WebSocket = MockWebSocket;
});

afterEach(() => {
  jest.clearAllMocks();
});

test('shows honest waiting state when no session/model prediction is available', async () => {
  render(<App />);
  expect(screen.getByText('Cognitive Load Monitor')).toBeInTheDocument();
  await waitFor(() => expect(screen.getByText('Create a session to begin collecting data.')).toBeInTheDocument());
  expect(screen.getByText('WAITING')).toBeInTheDocument();
  expect(screen.queryByText('LOW')).not.toBeInTheDocument();
  expect(screen.getByText('Create and activate a session to start the quiz.')).toBeInTheDocument();
});

test('displays the API active session and starts behavior capture with that ID', async () => {
  global.fetch.mockResolvedValue({
    ok: true,
    json: async () => ({
      active: true,
      session_id: 'session-123',
      session: { id: 'session-123', student_id: 'p-1', difficulty: 'easy' }
    })
  });
  render(<App />);
  await waitFor(() => expect(screen.getByText(/Active session:/).textContent).toContain('session-123'));
  expect(startCapture).toHaveBeenCalledWith('session-123');
});

test('shows the active session assessment state and session-bound form link', async () => {
  global.fetch.mockResolvedValue({
    ok: true,
    json: async () => ({
      active: true,
      session_id: 'session-456',
      session: {
        id: 'session-456', student_id: 'participant-b', difficulty: 'medium',
        assessment_status: 'pending', nasa_tlx_score: null, load_class: null
      }
    })
  });
  render(<App />);
  const link = await screen.findByRole('link', { name: 'submit NASA-TLX' });
  expect(link.href).toBe('http://127.0.0.1:5003/tlx/session-456');
  expect(screen.getByText(/NASA-TLX:/).textContent).toContain('Pending');
});

test('shows a submitted assessment and removes the duplicate submission link', async () => {
  global.fetch.mockResolvedValue({
    ok: true,
    json: async () => ({
      active: true,
      session_id: 'session-789',
      session: {
        id: 'session-789', student_id: 'participant-c', difficulty: 'hard',
        assessment_status: 'submitted', nasa_tlx_score: 50, load_class: 'MEDIUM'
      }
    })
  });
  render(<App />);
  await waitFor(() => expect(screen.getByText(/NASA-TLX:/).textContent).toContain('Submitted · 50.0/100 · MEDIUM'));
  expect(screen.queryByRole('link', { name: 'submit NASA-TLX' })).not.toBeInTheDocument();
});
