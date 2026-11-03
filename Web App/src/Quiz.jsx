import { useEffect, useRef, useState } from 'react';

const QUIZ_URL = 'http://127.0.0.1:5002';

export default function Quiz({ sessionId, difficulty }) {
  const [index, setIndex] = useState(0);
  const [question, setQuestion] = useState(null);
  const [choice, setChoice] = useState('');
  const [confidence, setConfidence] = useState('');
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [reload, setReload] = useState(0);
  const currentSession = useRef(sessionId);
  currentSession.current = sessionId;

  useEffect(() => {
    setIndex(0);
    setQuestion(null);
    setChoice('');
    setConfidence('');
    setMessage('');
  }, [sessionId, difficulty]);

  useEffect(() => {
    if (!sessionId || !difficulty) return undefined;
    let cancelled = false;
    setLoading(true);
    fetch(`${QUIZ_URL}/question/${difficulty}/${index}?session_id=${encodeURIComponent(sessionId)}`)
      .then(async response => {
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Could not load quiz question');
        return data;
      })
      .then(data => {
        if (cancelled) return;
        setQuestion(data);
        setChoice('');
        if (data.done) setMessage('Quiz complete. Your answers were saved to this session.');
        else setMessage('');
      })
      .catch(error => {
        if (!cancelled) {
          setQuestion(null);
          setMessage(error.message);
        }
      })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [sessionId, difficulty, index, reload]);

  async function submitAnswer(event) {
    event.preventDefault();
    if (!question || question.done || choice === '') return;
    const submittedSessionId = sessionId;
    const submittedQuestion = question;
    setLoading(true);
    try {
      const response = await fetch(`${QUIZ_URL}/answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          difficulty,
          index,
          chosen_option: Number(choice),
          confidence: Number(confidence),
          attempt_id: submittedQuestion.attempt_id
        })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not submit answer');
      if (currentSession.current !== submittedSessionId) return;
      setMessage(data.is_correct ? `Correct · ${data.response_time}s` : `Answer recorded · ${data.response_time}s`);
      setQuestion(null);
      setIndex(value => value + 1);
    } catch (error) {
      if (currentSession.current !== submittedSessionId) return;
      setMessage(error.message);
      // The one-time timer token may have been consumed by a rejected request.
      setReload(value => value + 1);
    } finally {
      if (currentSession.current === submittedSessionId) setLoading(false);
    }
  }

  return (
    <section style={{ marginTop: 24, padding: 20, border: '1px solid #ddd', borderRadius: 12 }}>
      <h3>Quiz</h3>
      {!sessionId && <p>Create and activate a session to start the quiz.</p>}
      {sessionId && loading && !question && <p>Loading question…</p>}
      {sessionId && question && !question.done && (
        <form onSubmit={submitAnswer}>
          <p>{question.text}</p>
          {question.options.map((option, optionIndex) => (
            <label key={optionIndex} style={{ display: 'block', margin: '8px 0' }}>
              <input
                type="radio"
                name="quiz-answer"
                value={optionIndex}
                checked={choice === String(optionIndex)}
                onChange={event => setChoice(event.target.value)}
              />{' '}{option}
            </label>
          ))}
          <label>
            Confidence&nbsp;
            <select value={confidence} required onChange={event => setConfidence(event.target.value)}>
              <option value="" disabled>Choose 1–5</option>
              {[1, 2, 3, 4, 5].map(value => <option key={value} value={value}>{value}</option>)}
            </select>
          </label>
          <button type="submit" disabled={loading || choice === '' || confidence === ''} style={{ marginLeft: 12 }}>Submit answer</button>
        </form>
      )}
      {message && <p role="status">{message}</p>}
      {sessionId && !question && !loading && <button type="button" onClick={() => setReload(value => value + 1)}>Retry quiz connection</button>}
    </section>
  );
}
