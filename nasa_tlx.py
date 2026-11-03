"""Participant NASA-TLX collection service (run after each study session)."""

from __future__ import annotations

from flask import Flask, jsonify, render_template_string, request

from database import NASA_FIELDS, get_session, save_nasa_tlx

app = Flask(__name__)

NASA_FORM = """
<!doctype html><html><head><meta charset="utf-8"><title>NASA-TLX</title>
<style>body{font-family:Arial;max-width:640px;margin:40px auto;padding:0 16px}label{display:block;margin:18px 0 6px}input{width:100%;padding:8px}button{margin-top:20px;padding:10px 18px}#result{margin-top:18px}</style></head><body>
<h2>NASA-TLX Assessment</h2>
<p>Session <strong>{{ session.id }}</strong> · Participant <strong>{{ session.student_id }}</strong> · Difficulty {{ session.difficulty }}</p>
{% if session.assessment_status == 'submitted' %}
<p role="status">This assessment was already submitted. Score: {{ '%.1f'|format(session.nasa_tlx_score) }}/100 ({{ session.load_class }}).</p>
{% elif session.nasa_tlx_score is not none %}
<p role="alert">A score is stored for this session, but it is unverified. This form cannot overwrite it.</p>
{% else %}
<h3>Session Complete — Rate Your Experience</h3>
<p>Rate the session you just completed. Enter a number from 0 (low) to 20 (high) for each item.</p>
<form id="tlxForm">
{% for name, label in fields %}<label for="{{ name }}">{{ label }}</label>
<input type="number" id="{{ name }}" name="{{ name }}" min="0" max="20" step="1" required>{% endfor %}
<button type="submit">Submit ratings</button></form><div id="result" role="status"></div>
<script>
const form=document.getElementById('tlxForm');
form.addEventListener('submit',async event=>{
  event.preventDefault();
  const result=document.getElementById('result');
  const data={session_id:{{ session.id|tojson }}};
  for(const name of ['mental','physical','temporal','performance','effort','frustration']) data[name]=Number(form.elements[name].value);
  try {
    const response=await fetch('/submit_tlx',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const body=await response.json();
    if(!response.ok) throw new Error(body.error||'Submission failed');
    result.textContent=`Thank you. NASA-TLX score: ${body.nasa_score.toFixed(1)}/100 (${body.load_class}).`;
    form.querySelector('button').disabled=true;
  } catch(error) { result.textContent=error.message; }
});
</script>
{% endif %}
</body></html>
"""

FIELD_LABELS = [
    ("mental", "Mental demand"),
    ("physical", "Physical demand"),
    ("temporal", "Temporal demand"),
    ("performance", "Performance shortfall (how unsuccessful were you?)"),
    ("effort", "Effort"),
    ("frustration", "Frustration"),
]


@app.get("/tlx/<session_id>")
def show_tlx(session_id: str):
    session = get_session(session_id)
    if not session:
        return "Unknown session ID", 404
    return render_template_string(NASA_FORM, session=session, fields=FIELD_LABELS)


@app.post("/submit_tlx")
def submit_tlx():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON body is required"}), 400
    session_id = data.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        return jsonify({"error": "session_id is required"}), 400
    scores = {field: data.get(field) for field in NASA_FIELDS}
    try:
        nasa_score, load_class = save_nasa_tlx(session_id, scores)
    except ValueError as exc:
        status = 409 if "already" in str(exc) else 422 if "ratings" in str(exc) else 404
        return jsonify({"error": str(exc)}), status
    return jsonify({
        "session_id": session_id,
        "nasa_score": nasa_score,
        "load_class": load_class,
        "label_source": "NASA-TLX",
        "assessment_status": "submitted",
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5003, debug=False)
