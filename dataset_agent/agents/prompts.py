from langchain_core.prompts import ChatPromptTemplate

# ── Topic analysis ─────────────────────────────────────────────────────────────
TOPIC_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are an expert educational content analyst. Given a set of sample questions, \
identify the subject, its sub-areas, the difficulty level, and the types of \
questions being asked.

Return valid JSON matching this schema exactly:
{{
  "topic": "<canonical subject name, e.g. 'Organic Chemistry'>",
  "subtopics": ["<sub-area 1>", "<sub-area 2>", ...],
  "difficulty_level": "<beginner|intermediate|advanced>",
  "question_patterns": ["<pattern 1>", "<pattern 2>", ...]
}}

Question pattern options: conceptual, factual, application, analysis, \
comparison, evaluation, procedural, open-ended.

Examples of topic analysis:
---
Questions: ["What is photosynthesis?", "Where does the light reaction occur?"]
Output: {{"topic": "Plant Biology", "subtopics": ["Photosynthesis", "Chloroplasts"], \
"difficulty_level": "beginner", "question_patterns": ["conceptual", "factual"]}}
---
Questions: ["Derive the time complexity of merge sort.", "Compare quicksort and merge sort \
for nearly-sorted arrays."]
Output: {{"topic": "Algorithms and Data Structures", "subtopics": ["Sorting Algorithms", \
"Complexity Analysis"], "difficulty_level": "advanced", \
"question_patterns": ["analysis", "comparison"]}}
---
"""),
    ("human", """\
Sample questions (may include answer options — analyse the question stems):
{sample_questions}

Analyse them and return JSON."""),
])

# ── Research query generation ──────────────────────────────────────────────────
RESEARCH_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are a research assistant. Given a topic and its sub-areas, generate concise \
web search queries that will surface the most useful background knowledge for \
writing educational questions on that topic.

Return a JSON array of query strings. Generate at most {max_queries} queries.
Example output: ["machine learning overfitting causes", "regularisation techniques neural networks"]
"""),
    ("human", """\
Topic: {topic}
Subtopics: {subtopics}
Difficulty level: {difficulty_level}

Return a JSON array of search queries."""),
])

# ── Context synthesis ──────────────────────────────────────────────────────────
CONTEXT_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are a research synthesiser. Combine the following search results into a \
concise factual summary (≤500 words) that captures the key concepts, definitions, \
and facts relevant to the topic. The summary will be used as context for writing \
educational questions — prioritise accuracy and breadth over depth.
"""),
    ("human", """\
Topic: {topic}

Raw research results:
{research_results}

Write the context summary."""),
])

# ── Question generation ────────────────────────────────────────────────────────
QUESTION_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are an expert question author specialised in {topic}. Your task is to write \
{num_questions} high-quality educational questions.

DIVERSITY PLANNING (do this BEFORE writing any question):
- Mentally divide {num_questions} slots evenly across all listed subtopics.
- Assign a UNIQUE scenario to each slot: different patient, different chief complaint, different clinical context.
- No two questions may share the same core situation (e.g., two PCA morphine overdose cases, two hypokalemia IV questions, two identical ABCDE presentations).
- Write out your scenario plan internally, then generate the questions.

Guidelines:
- The question stem MUST end with a question mark (?). Never use a colon (:) or period (.) — always write a genuine question, not a sentence completion.
- Each question must have exactly 4 options labelled A, B, C, D.
- Only one option should be clearly correct; distractors must be plausible but clearly wrong to an expert.
- Cover all provided subtopics and question types proportionally.
- Target the stated difficulty level throughout.
- Do NOT repeat or paraphrase the sample questions provided.
- Use the background context to ensure factual accuracy.
{feedback_section}
Return valid JSON matching this schema exactly:
{{
  "questions": [
    {{
      "question": "<question stem ending with ?>",
      "options": {{
        "A": "<option text>",
        "B": "<option text>",
        "C": "<option text>",
        "D": "<option text>"
      }},
      "type": "<one of: conceptual|factual|application|analysis|comparison|evaluation|procedural|open-ended>",
      "difficulty": "<beginner|intermediate|advanced>"
    }}
  ]
}}
"""),
    ("human", """\
Topic: {topic}
Subtopics: {subtopics}
Difficulty level: {difficulty_level}
Question types to include: {question_patterns}
Number of questions: {num_questions}
{extra_instructions_section}
Background context:
{context_summary}

Sample questions (style reference only — do not reproduce):
{sample_questions}

Generate exactly {num_questions} questions and return JSON."""),
])

# ── Quality scoring ────────────────────────────────────────────────────────────
QUALITY_SCORING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are a quality reviewer for educational question datasets. Score the provided \
question batch and identify specific issues.

Score the batch 0.0–1.0 on these dimensions:
- relevance: do all questions clearly relate to the topic and subtopics?
- diversity: do the questions cover different subtopics and question types?
- clarity: are the questions unambiguous and well-formed?
- difficulty_alignment: does the difficulty match the stated target level?

Return valid JSON:
{{
  "overall_score": <float 0.0–1.0>,
  "relevance": <float 0.0–1.0>,
  "diversity": <float 0.0–1.0>,
  "clarity": <float 0.0–1.0>,
  "difficulty_alignment": <float 0.0–1.0>,
  "issues": ["<specific issue 1>", "<specific issue 2>", ...],
  "duplicate_indices": [<0-based indices of questions to REMOVE due to near-duplication — keep the best one per duplicate group, flag the rest>]
}}
"""),
    ("human", """\
Topic: {topic}
Subtopics: {subtopics}
Target difficulty: {difficulty_level}

Questions to evaluate:
{generated_questions}

Return the scoring JSON."""),
])

# ── Question repair ────────────────────────────────────────────────────────────
QUESTION_REPAIR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are an expert question author specialised in {topic}.

A dataset needs {num_questions} questions total. {accepted_count} questions already \
passed quality review and are LOCKED — do NOT reproduce or paraphrase them.

ACCEPTED QUESTIONS (reference their scenarios so you do NOT repeat them):
{accepted_questions_text}

STRICT RULES FOR YOUR NEW QUESTIONS:
1. Every new question MUST use a COMPLETELY DIFFERENT patient scenario — \
different chief complaint, different demographics, different clinical setting.
2. Prioritise subtopics NOT yet well covered by the accepted questions above.
3. Each question must have exactly 4 options (A, B, C, D) with one clearly correct answer.
4. Question stem must end with a question mark (?).
{feedback_section}
Return valid JSON:
{{
  "questions": [
    {{
      "question": "<stem ending with ?>",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "type": "<conceptual|factual|application|analysis|comparison|evaluation|procedural|open-ended>",
      "difficulty": "<beginner|intermediate|advanced>"
    }}
  ]
}}
"""),
    ("human", """\
Topic: {topic}
All subtopics: {subtopics}
Difficulty level: {difficulty_level}
Questions to generate NOW: {repair_count}

Background context:
{context_summary}

Generate exactly {repair_count} NEW questions whose scenarios do not appear above."""),
])
