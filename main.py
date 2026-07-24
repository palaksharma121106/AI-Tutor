import streamlit as st
import plotly.graph_objects as go

from utils import get_tutor_response

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Tutor",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }
    .tutor-hero {
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #2d3748;
        margin-bottom: 1.2rem;
    }
    .tutor-hero h1 {
        margin-bottom: 0.2rem;
        font-size: 1.9rem;
    }
    .tutor-hero p {
        color: #9ca3af;
        margin-bottom: 0;
    }
    .section-card {
        padding: 1.1rem 1.3rem;
        border-radius: 12px;
        background: #161b22;
        border: 1px solid #2d3748;
        margin-bottom: 1rem;
    }
    .pill {
        display: inline-block;
        padding: 0.15rem 0.7rem;
        border-radius: 999px;
        background: #2563eb22;
        color: #60a5fa;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "tutor_data" not in st.session_state:
    st.session_state.tutor_data = None

if "topic" not in st.session_state:
    st.session_state.topic = ""

# ---------------------------------------------------------------------------
# Personalization controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Personalize your tutor")

    course_context = st.text_input(
        "What are you studying? (optional)",
        placeholder="e.g. B.Tech CS - Data Structures, Class 10 Physics...",
        help="The AI tailors depth, vocabulary, and references to this.",
    )

    level = st.select_slider(
        "Level",
        options=["Beginner", "Intermediate", "Advanced"],
        value="Beginner",
    )

    st.divider()
    st.caption(
        "💡 Tip: for programming topics (e.g. 'Recursion', "
        "'Variables in C', 'Pointers'), the tutor auto-generates a "
        "runnable code example with a walkthrough."
    )
    st.caption("Model: `llama-3.1-8b-instant` via Groq")

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="tutor-hero">
        <h1>📘 AI Tutor</h1>
        <p>Enter any topic — get a tailored explanation, code, quiz, and visual insight.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

input_col, button_col = st.columns([4, 1])

with input_col:
    topic_input = st.text_input(
        "Topic",
        placeholder="e.g. Recursion, Variables in C, Photosynthesis, Newton's Second Law...",
        label_visibility="collapsed",
    )

with button_col:
    generate_clicked = st.button("Generate Lesson", type="primary", use_container_width=True)

if generate_clicked:
    if not topic_input.strip():
        st.warning("Please enter a topic first.")
    else:
        with st.spinner(f"Building your lesson on '{topic_input}'..."):
            try:
                st.session_state.tutor_data = get_tutor_response(
                    topic=topic_input.strip(),
                    course_context=course_context.strip(),
                    level=level,
                )
                st.session_state.topic = topic_input.strip()
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Something went wrong while calling the AI: {e}")

# ---------------------------------------------------------------------------
# Display section 
# ---------------------------------------------------------------------------

data = st.session_state.tutor_data

if data:
    st.divider()

    topic_type = data.get("topic_type", "conceptual")
    badge = "🖥️ Technical" if topic_type == "technical" else "🧩 Conceptual"
    st.markdown(f'<span class="pill">{badge} · {level}</span>', unsafe_allow_html=True)
    st.subheader(st.session_state.topic)

    has_code = bool(data["code_example"].get("code", "").strip())

    tab_labels = ["🧠 Explanation", "📝 Quiz", "📊 Visual Insight", "🔍 Go Deeper"]
    if has_code:
        tab_labels.insert(1, "💻 Code")

    tabs = st.tabs(tab_labels)
    tab_map = dict(zip(tab_labels, tabs))

    # --- Explanation tab ---
    with tab_map["🧠 Explanation"]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.write(data["explanation"])
        st.markdown("</div>", unsafe_allow_html=True)

        if data["bullet_points"]:
            st.markdown("#### 🔑 Key Points")
            for point in data["bullet_points"]:
                st.markdown(f"- {point}")

        if data["example"]:
            st.markdown("#### 💡 Real-world Example")
            st.info(data["example"])

    # --- Code tab (only if applicable) ---
    if has_code:
        with tab_map["💻 Code"]:
            code = data["code_example"]
            lang = code.get("language", "text") or "text"
            st.markdown(f"**Language:** `{lang}`")
            st.code(code["code"], language=lang)
            if code.get("code_explanation"):
                st.markdown("#### Walkthrough")
                st.write(code["code_explanation"])

    # --- Quiz tab ---
    with tab_map["📝 Quiz"]:
        if data["quiz"]:
            st.write("Test your understanding:")

            user_answers = {}
            for i, q in enumerate(data["quiz"]):
                st.markdown(f"**Q{i + 1}. {q['question']}**")
                user_answers[i] = st.radio(
                    label="Choose an answer:",
                    options=q["options"],
                    key=f"quiz_{st.session_state.topic}_{i}",
                    index=None,
                    label_visibility="collapsed",
                )

            if st.button("Check Answers"):
                score = 0
                for i, q in enumerate(data["quiz"]):
                    selected = user_answers.get(i)
                    correct = q["answer"]
                    if selected == correct:
                        score += 1
                        st.success(f"Q{i + 1}: Correct! ✅")
                    else:
                        st.error(f"Q{i + 1}: Incorrect ❌ — correct answer: **{correct}**")
                st.metric("Score", f"{score} / {len(data['quiz'])}")
        else:
            st.info("No quiz was generated for this topic.")

    # --- Visual insight tab ---
    with tab_map["📊 Visual Insight"]:
        graph = data.get("graph_data", {})
        if graph.get("labels") and graph.get("values"):
            if graph.get("title"):
                st.markdown(f"**{graph['title']}**")

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=graph["labels"],
                        y=graph["values"],
                        marker_color="#3b82f6",
                        text=graph["values"],
                        textposition="outside",
                    )
                ]
            )
            fig.update_layout(
                xaxis_title=graph.get("x_label", ""),
                yaxis_title=graph.get("y_label", ""),
                template="plotly_dark",
                margin=dict(l=20, r=20, t=30, b=20),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            if data.get("graph_insight"):
                st.markdown("#### What this shows")
                st.write(data["graph_insight"])
        else:
            st.info("No graph data was generated for this topic.")

    # --- Go deeper tab ---
    with tab_map["🔍 Go Deeper"]:
        if data["follow_up_question"]:
            st.markdown("#### Follow-up question to explore next")
            st.write(data["follow_up_question"])
        else:
            st.info("No follow-up question was generated.")

else:
    st.info("Enter a topic above and click **Generate Lesson** to get started.")