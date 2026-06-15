# MailMind AI — Presentation Content

Speaker-ready content for a 14-slide presentation.
Each slide lists concise talking points, the figure to embed, and a short speaker note.

---

## Slide 1 — MailMind AI: Your Inbox, Intelligently Organized

- Agentic AI email assistant for final-year project
- Subtitle: NLP + Machine Learning + autonomous agent
- Six categories, prioritised, summarised, action-ready
- Team: [team members]
- Guide: [guide name]

FIGURE: none
NOTES: Open with the one-line promise — an inbox that organises, prioritises, and acts for you. Keep it to 20 seconds before diving in.

---

## Slide 2 — Problem Statement

- Email overload — too many messages, too little time
- High cognitive load triaging every message manually
- Important and urgent mail buried under noise
- Promotions, social, spam crowd out real priorities
- No prioritisation within legitimate mail

FIGURE: none
NOTES: Frame the pain everyone feels: the inbox is a flat list with no sense of what actually matters. Set up the need for intelligence.

---

## Slide 3 — Existing System

- Manual sorting — slow, repetitive, error-prone
- Simple keyword and rule-based filters only
- Brittle to new wording and phrasing
- Cannot prioritise within legitimate mail
- No learning from user behaviour
- No summaries or suggested actions

FIGURE: none
NOTES: Stress that traditional filters are static rules — they tag, but they do not understand, rank, or act.

---

## Slide 4 — Proposed System

- MailMind AI — an agentic email assistant
- Value prop: understands, prioritises, and acts on your inbox
- Pillar 1: ML classification into six categories
- Pillar 2: NLP signals (keywords, intent, sentiment, urgency)
- Pillar 3: context-aware priority scoring
- Pillars 4-5: behavioural adaptation + agentic output

FIGURE: none
NOTES: Position MailMind as five pillars working in one pipeline — not just a smarter filter, but an assistant that takes the next step.

---

## Slide 5 — Objectives

- Reduce cognitive load of inbox triage
- Prioritise mail by true importance and urgency
- Personalise ranking from user behaviour over time
- Be proactive — summaries, flags, suggested actions
- Improve productivity by surfacing what matters first

FIGURE: none
NOTES: Five objectives map directly to the five pillars. Each is measurable through automation rate and classification accuracy.

---

## Slide 6 — System Architecture

- Layered Python package (src/mailmind)
- Config, schema, text utils foundation
- Data, ML, NLP, behavioural, context, agent modules
- SQLite persistence; FastAPI + Streamlit interfaces
- 44 pytest tests, all passing

FIGURE: docs/screenshots/architecture.png
NOTES: Walk left to right through the architecture diagram — data in, intelligence in the middle, agentic insights out.

---

## Slide 7 — Workflow

- Step 1: ML classification into one of six categories
- Step 2: NLP signals — keywords, intent, sentiment, urgency
- Step 3: context priority scoring → 0-100 score
- Step 4: behavioural adaptation from past actions
- Step 5: agentic output — summary, actions, urgent/VIP/spam flags
- process_inbox() returns insights sorted by priority

FIGURE: none
NOTES: Emphasise this is a per-email pipeline; record_feedback() closes the loop so the system adapts over time.

---

## Slide 8 — AI Components

- Classification: ML model sorts mail into six categories
- NLP: extracts keywords, intent, sentiment, urgency cues
- Behavioural: learns engagement from replies, opens, deletes
- Contextual: weighted priority score across five signals
- Agentic: extractive summary plus suggested actions and flags

FIGURE: none
NOTES: One bullet per pillar. Make clear these components are complementary stages, not competing models.

---

## Slide 9 — Technologies Used

- Language: Python 3.12
- ML/Data: scikit-learn, numpy, pandas, scipy, joblib
- NLP: NLTK (VADER, punkt, WordNet), optional spaCy
- API: FastAPI, uvicorn, pydantic
- UI/Viz: Streamlit, altair, matplotlib, seaborn
- Storage/Test: SQLite (stdlib), pytest

FIGURE: none
NOTES: Group by layer so the audience sees a coherent, production-shaped stack rather than a random tool list.

---

## Slide 10 — Dataset

- 4,200 synthetic emails, 700 per category (balanced)
- Six categories: Important, Work, Personal, Social, Promotions, Spam
- 14 subject × 14 body templates with slot-filling
- 12% ambiguity fraction borrows confusable neighbour bodies
- Deterministic (seed=42); rich schema with metadata
- Stratified 80/20 split: 3,360 train / 840 test

FIGURE: docs/screenshots/category_distribution.png
NOTES: Highlight that the 12% ambiguity makes classes realistically overlapping — not artificially separable.

---

## Slide 11 — Model Training

- Features: TF-IDF (1,2)-grams + 8 engineered numeric features
- Combined via ColumnTransformer with MaxAbsScaler
- Four models compared on the 840-email test set
- Logistic Regression best: F1 0.9095 / acc 0.9095
- Beat Linear SVM, Complement NB, Random Forest
- Trained on 3,360 emails, 80/20 stratified split

FIGURE: docs/screenshots/model_comparison.png
NOTES: Logistic Regression edged out a calibrated Linear SVM and clearly beat the tree and NB baselines.

---

## Slide 12 — Results & Findings

- Accuracy 90.9%, macro-F1 0.9095
- Per-class F1 range 0.8978 (Work) to 0.9187 (Personal)
- Confusions on overlapping pairs: Work/Important, Promotions/Spam
- vs rule-based baseline: +18.8 accuracy points
- 67.5% relative reduction in misclassification error
- Rules cannot prioritise, learn, summarise, or suggest

FIGURE: docs/screenshots/confusion_matrix.png
NOTES: The confusion matrix shows errors cluster on genuinely ambiguous pairs, which is expected and acceptable.

---

## Slide 13 — Screenshots / Demo

- Working Streamlit dashboard, inbox sorted by priority
- Critical/High/Medium/Low bands from 0-100 score
- Per-email category, urgency, intent, suggested action
- Example: "URGENT: Production outage" → Important, High (79.1)
- Spam and promotions flagged with delete/unsubscribe actions
- Analytics and detail views also available

FIGURE: docs/screenshots/dashboard_inbox.png
NOTES: Demo the live inbox — point out the priority ordering and the one-click suggested actions per email.

---

## Slide 14 — Future Scope & Conclusion

- IMAP integration for live, real-world inboxes
- BERT/transformer models for deeper understanding
- Generative AI for drafted reply suggestions
- Multilingual support for global users
- Illustrative: high automation rate implies real time savings
- Conclusion: MailMind turns a flat inbox into an intelligent assistant

FIGURE: none
NOTES: Be honest — time-saving is an illustrative estimate from the measured automation rate, not a human user study. Close on the vision.
