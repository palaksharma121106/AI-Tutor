import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
import streamlit as st

# ---------------------------------------------------------------------------
# Step 1: Environment & client setup
# ---------------------------------------------------------------------------

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"


def get_groq_client() -> Groq:
    """
    Creates and returns a Groq client using the API key from .env

    Test it standalone:
        >>> client = get_groq_client()
        >>> print(client)
    """
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please add it to your .env file "
            "or Streamlit Cloud Secrets."
        )
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Step 2: Prompt building
# ---------------------------------------------------------------------------

def build_prompt(topic: str, course_context: str = "", level: str = "Beginner") -> str:
    """
    Builds a structured prompt asking the LLM to return STRICT JSON.

    Parameters
    ----------
    topic : the subject the student wants explained
    course_context : optional free text, e.g. "B.Tech CS - Data Structures",
                     "Class 10 Physics", "Preparing for interviews" — used
                     to tailor depth, vocabulary, and references.
    level : "Beginner" | "Intermediate" | "Advanced" — controls how deep
            the explanation and example go.

    The schema always asks for a `code_example` block. The model is told
    to leave it empty (language: "none") when the topic has no natural
    code representation (e.g. "Photosynthesis"), and to fill it in fully
    for programming / CS / math-with-algorithms topics (e.g. "Recursion",
    "Variables in C", "Binary Search").

    Test it standalone:
        >>> prompt = build_prompt("Recursion", "B.Tech CS", "Intermediate")
        >>> print(prompt)
    """
    context_line = (
        f'The student is studying: "{course_context}". Tailor vocabulary, '
        f"depth, and any references/analogies to that context where useful."
        if course_context.strip()
        else "No specific course context was given — keep it broadly accessible."
    )

    prompt = f"""
You are an expert tutor who explains topics clearly to students.

Topic: "{topic}"
Student level: {level}
{context_line}

Return ONLY a valid JSON object (no markdown, no code fences, no extra text
before or after it) with EXACTLY the following structure:

{{
  "topic_type": "technical" or "conceptual",
  "explanation": "A clear explanation of the topic, matched to the student's level (5-8 sentences). If technical, explain the underlying mechanism/logic, not just a surface definition.",
  "bullet_points": ["key point 1", "key point 2", "key point 3", "key point 4", "key point 5"],
  "example": "One concrete, real-world example that illustrates the topic in plain language",
  "code_example": {{
    "language": "python, c, java, etc. — or \\"none\\" if this topic has no natural code representation",
    "code": "A short, complete, runnable code snippet demonstrating the topic. Empty string if not applicable.",
    "code_explanation": "A clear walkthrough of what the code does, ideally referencing specific lines/parts (e.g. 'The base case on line 2 stops the recursion...'). Empty string if not applicable."
  }},
  "quiz": [
    {{
      "question": "A quiz question about the topic",
      "options": ["option A", "option B", "option C", "option D"],
      "answer": "the correct option text (must exactly match one of the options)"
    }},
    {{
      "question": "A second quiz question testing a different aspect of the topic",
      "options": ["option A", "option B", "option C", "option D"],
      "answer": "the correct option text (must exactly match one of the options)"
    }},
    {{
      "question": "A third quiz question, slightly harder than the first two",
      "options": ["option A", "option B", "option C", "option D"],
      "answer": "the correct option text (must exactly match one of the options)"
    }}
  ],
  "follow_up_question": "One thought-provoking follow-up question the student could explore next to go deeper",
  "graph_data": {{
    "title": "A short title relevant to the topic",
    "x_label": "label for the x-axis",
    "y_label": "label for the y-axis",
    "labels": ["A", "B", "C", "D"],
    "values": [10, 20, 30, 40]
  }},
  "graph_insight": "2-4 sentences explaining what this specific chart shows, how to read it, and why it's relevant to understanding the topic (e.g. explain what the growth pattern or comparison means in context)."
}}

Rules:
- graph_data must represent something genuinely meaningful and topic-relevant
  (e.g. time/space complexity growth across input sizes, stages of a process,
  before/after comparison, category comparison). If nothing numeric truly
  fits, invent a simple, sensible illustrative comparison and say so in
  graph_insight.
- "values" must be numbers (int or float), not strings.
- If topic_type is "technical", code_example MUST be filled in with real,
  correct, runnable code — do not leave it empty.
- If topic_type is "conceptual" and code truly does not apply, set
  code_example.language to "none" and leave code/code_explanation as "".
- Respond with the JSON object ONLY. Do not wrap it in ```json fences and
  do not add any commentary before or after it.
"""
    return prompt.strip()


# ---------------------------------------------------------------------------
# Step 3: Calling the LLM
# ---------------------------------------------------------------------------

def call_llm(prompt: str, temperature: float = 0.6) -> str:
    """
    Sends the prompt to Groq's chat completion endpoint and
    returns the raw text response (a string, ideally JSON).

    Test it standalone:
        >>> raw = call_llm(build_prompt("Photosynthesis"))
        >>> print(raw)
    """
    client = get_groq_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful tutor that always responds with "
                    "strict, valid JSON only, matching the schema given "
                    "by the user."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=2200,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Step 4: Cleaning & parsing the response
# ---------------------------------------------------------------------------

def clean_json_text(raw_text: str) -> str:
    """
    Strips markdown code fences (```json ... ```) and any stray text
    the model sometimes adds around the JSON object, so json.loads()
    doesn't choke on it.

    Test it standalone:
        >>> clean_json_text('```json\\n{"a": 1}\\n```')
        '{"a": 1}'
    """
    text = raw_text.strip()

    text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)

    return text


def parse_response(raw_text: str) -> dict:
    """
    Parses the cleaned LLM output into a Python dict.
    Falls back to a safe default structure if parsing fails,
    so the Streamlit UI never crashes on a malformed response.

    Test it standalone:
        >>> data = parse_response(raw_json_string)
        >>> print(data["explanation"])
    """
    cleaned = clean_json_text(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = {
            "topic_type": "conceptual",
            "explanation": (
                "Sorry, the AI response could not be parsed. Please try again."
            ),
            "bullet_points": [],
            "example": "",
            "code_example": {"language": "none", "code": "", "code_explanation": ""},
            "quiz": [],
            "follow_up_question": "",
            "graph_data": {
                "title": "No data available",
                "x_label": "",
                "y_label": "",
                "labels": [],
                "values": [],
            },
            "graph_insight": "",
            "raw_response": raw_text,
        }

    # Defensively fill in any keys the model might have skipped
    defaults = {
        "topic_type": "conceptual",
        "explanation": "",
        "bullet_points": [],
        "example": "",
        "code_example": {"language": "none", "code": "", "code_explanation": ""},
        "quiz": [],
        "follow_up_question": "",
        "graph_data": {
            "title": "",
            "x_label": "",
            "y_label": "",
            "labels": [],
            "values": [],
        },
        "graph_insight": "",
    }
    for key, default_value in defaults.items():
        data.setdefault(key, default_value)

    # Make sure nested code_example dict has all sub-keys too
    for key, default_value in defaults["code_example"].items():
        data["code_example"].setdefault(key, default_value)

    return data


# ---------------------------------------------------------------------------
# Step 5: Orchestrator - the full pipeline in a single call
# ---------------------------------------------------------------------------

def get_tutor_response(topic: str, course_context: str = "", level: str = "Beginner") -> dict:
    """
    Full pipeline: topic + context -> prompt -> LLM call -> parsed structured data.
    This is the ONLY function main.py (the Streamlit UI) should call.

    Test it standalone:
        >>> result = get_tutor_response("Variables in C", "B.Tech CS - Semester 1", "Beginner")
        >>> print(result["code_example"]["code"])
    """
    prompt = build_prompt(topic, course_context, level)
    raw_text = call_llm(prompt)
    structured_data = parse_response(raw_text)
    return structured_data


# ---------------------------------------------------------------------------
# Step 6: Manual test block 
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_topic = "Recursion"
    test_context = "B.Tech CS - Data Structures"
    test_level = "Intermediate"

    print(f"Testing AI Tutor pipeline for topic: {test_topic}\n")

    result = get_tutor_response(test_topic, test_context, test_level)

    print("TOPIC TYPE:", result["topic_type"])
    print("\nEXPLANATION:\n", result["explanation"])

    print("\nBULLET POINTS:")
    for point in result["bullet_points"]:
        print(" -", point)

    print("\nEXAMPLE:\n", result["example"])

    print("\nCODE EXAMPLE:")
    print(" Language:", result["code_example"]["language"])
    print(result["code_example"]["code"])
    print("Explanation:", result["code_example"]["code_explanation"])

    print("\nQUIZ:")
    for i, q in enumerate(result["quiz"], start=1):
        print(f" Q{i}: {q['question']}")
        print("    Options:", q["options"])
        print("    Answer:", q["answer"])

    print("\nFOLLOW-UP QUESTION:\n", result["follow_up_question"])
    print("\nGRAPH DATA:\n", result["graph_data"])
    print("\nGRAPH INSIGHT:\n", result["graph_insight"])
