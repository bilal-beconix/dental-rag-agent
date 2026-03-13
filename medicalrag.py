"""
Metro East Dental Clinic — RAG Agent
Built on top of Agent10.py template

CHANGES FROM TEMPLATE:
- Dual document loader (PDF + DOCX)
- Category-aware chunking with metadata
- 6 Chroma collections (one per category)
- 6 specialized retriever tools
- Safety node with HIPAA guardrails
- Human handoff logic
- Smart query router node
- Medical-grade system prompt
"""
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.documents import Document
import re

# Remove any existing OPENAI_API_KEY from environment to ensure .env file takes precedence
if 'OPENAI_API_KEY' in os.environ:
    del os.environ['OPENAI_API_KEY']

load_dotenv('medical.env', override=True)

# ─────────────────────────────────────────────
# 1. MODELS  (same as your template)
# ─────────────────────────────────────────────
llm_base = ChatOpenAI(model="gpt-4o", temperature=0)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# ─────────────────────────────────────────────
# 2. DOCUMENT LOADING  (upgraded: PDF + DOCX)
# ─────────────────────────────────────────────
PDF_PATH  = r"C:\Users\bilal\OneDrive\Desktop\LangGraph\LangGraph_Agents\knwoledge_base\Medical_knowledge_base.pdf"

def load_documents():
    all_docs = []

    # --- Load PDF ---
    if os.path.exists(PDF_PATH):
        print(f"Loading PDF: {PDF_PATH}")
        pdf_loader = PyPDFLoader(PDF_PATH)
        pdf_pages = pdf_loader.load()
        # Tag every page with its source
        for doc in pdf_pages:
            doc.metadata["source_type"] = "original_dataset"
        all_docs.extend(pdf_pages)
        print(f"  ✅ PDF loaded — {len(pdf_pages)} pages")
    else:
        print(f"  ⚠️  PDF not found at {PDF_PATH}")

    return all_docs

raw_docs = load_documents()


# ─────────────────────────────────────────────
# 3. CATEGORY DETECTION
#    Assigns every chunk to one of 6 namespaces
# ─────────────────────────────────────────────
CATEGORIES = {
    "insurance_billing": [
        "insurance", "copay", "deductible", "cdt", "cpt", "coverage",
        "billing", "claim", "ppo", "hmo", "medicaid", "medicare",
        "pre-auth", "aerohealth", "carefirst", "statecare",
        "senioradvantage", "out-of-network", "oon", "balance billing",
        "coordination of benefits", "cob", "lifetime max", "fee schedule",
        "ucr", "mac", "remittance", "adjudication", "clearinghouse"
    ],
    "clinic_policies": [
        "office hours", "holiday", "cancellation", "no-show", "payment plan",
        "carecredit", "sunbit", "emergency", "after-hours", "intake",
        "late arrival", "pediatric", "waitlist", "reminder", "membership",
        "financing", "appointment", "reschedule", "in-house plan"
    ],
    "procedures_faqs": [
        "procedure", "cleaning", "filling", "root canal", "crown",
        "implant", "extraction", "invisalign", "veneer", "gum graft",
        "bone graft", "srp", "scaling", "pre-procedure", "post-procedure",
        "recovery", "sedation", "anesthesia", "pricing", "prophylaxis",
        "composite", "endodontic", "periodontal", "osseointegration",
        "dry socket", "gauze", "fluoride", "sealant", "nitrous"
    ],
    "provider_info": [
        "dr.", "doctor", "provider", "specialist", "hygienist", "languages",
        "rahman", "chen", "rostova", "gonzalez", "patel", "staff",
        "coordinator", "assistant", "schedule", "availability"
    ],
    "appointment_logistics": [
        "parking", "transit", "subway", "location", "accessibility",
        "telehealth", "duration", "first visit", "walk-in", "ada",
        "booking", "online", "phone", "sms", "74th", "jackson heights",
        "wheelchair", "ramp", "zoom", "directions", "bus", "train"
    ],
    "compliance_legal": [
        "hipaa", "consent", "legal", "privacy", "financial responsibility",
        "refund", "dispute", "rights", "phi", "disclosure", "informed consent",
        "authorization", "records", "correction", "appeal", "denial"
    ]
}

def detect_category(text: str) -> str:
    text_lower = text.lower()
    scores = {cat: 0 for cat in CATEGORIES}
    for cat, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in text_lower:
                scores[cat] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "clinic_policies"  # default fallback


# ─────────────────────────────────────────────
# 4. CHUNKING  (upgraded: metadata per chunk)
#    Same RecursiveCharacterTextSplitter as your
#    template but with tighter sizes + metadata
# ─────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # tighter than template's 1000 — one idea per chunk
    chunk_overlap=100    # same ratio as template
)

raw_chunks = text_splitter.split_documents(raw_docs)

# Enrich every chunk with category metadata
for chunk in raw_chunks:
    chunk.metadata["category"] = detect_category(chunk.page_content)

print(f"\n📦 Created {len(raw_chunks)} chunks")

# Group chunks by category for separate Chroma collections
chunks_by_category = {cat: [] for cat in CATEGORIES}
for chunk in raw_chunks:
    cat = chunk.metadata.get("category", "clinic_policies")
    if cat in chunks_by_category:
        chunks_by_category[cat].append(chunk)

for cat, chunks in chunks_by_category.items():
    print(f"   {cat:<30} {len(chunks)} chunks")


# ─────────────────────────────────────────────
# 5. VECTOR STORES  (upgraded: 6 collections)
#    Same Chroma as your template,
#    one collection per category
# ─────────────────────────────────────────────
PERSIST_DIR = "chroma_db_dental"
os.makedirs(PERSIST_DIR, exist_ok=True)

print("\n🔄 Building Chroma vector stores...")
vector_stores = {}

for category, chunks in chunks_by_category.items():
    if not chunks:
        print(f"   ⚠️  No chunks for {category} — skipping")
        continue
    try:
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name=category
        )
        vector_stores[category] = vs
        print(f"   ✅ {category} — {len(chunks)} chunks indexed")
    except Exception as e:
        print(f"   ❌ Error building {category}: {e}")

# Create retrievers — k=4 per category (same search type as template)
retrievers = {
    cat: vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    for cat, vs in vector_stores.items()
}


# ─────────────────────────────────────────────
# 6. TOOLS  (upgraded: 6 specialized tools)
#    Same @tool pattern as your template
# ─────────────────────────────────────────────

def _search(category: str, query: str) -> str:
    """Shared search logic — used by all 6 tools."""
    if category not in retrievers:
        return f"No data available for category: {category}"
    docs = retrievers[category].invoke(query)
    if not docs:
        return f"No relevant information found in {category}."
    results = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source_type", "document")
        results.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(results)


@tool
def search_insurance_billing(query: str) -> str:
    """Search insurance coverage rules, copays, deductibles, billing codes (CDT/CPT),
    pre-authorization requirements, and out-of-network policies."""
    return _search("insurance_billing", query)


@tool
def search_clinic_policies(query: str) -> str:
    """Search office hours, cancellation policy, payment plans (CareCredit, Sunbit),
    new patient intake process, emergency protocol, and pediatric policies."""
    return _search("clinic_policies", query)


@tool
def search_procedures_faqs(query: str) -> str:
    """Search procedure explanations, pre and post care instructions, recovery timelines,
    pricing guide, sedation options, and Invisalign FAQs."""
    return _search("procedures_faqs", query)


@tool
def search_provider_info(query: str) -> str:
    """Search doctor specializations, provider schedules, languages spoken,
    which provider accepts which insurance, and staff directory."""
    return _search("provider_info", query)


@tool
def search_appointment_logistics(query: str) -> str:
    """Search booking methods, location, parking, subway/transit directions,
    accessibility info, telehealth availability, and appointment durations."""
    return _search("appointment_logistics", query)


@tool
def search_compliance_legal(query: str) -> str:
    """Search HIPAA privacy practices, patient rights, consent form summaries,
    financial responsibility agreement, and billing dispute/refund policies."""
    return _search("compliance_legal", query)


tools = [
    search_insurance_billing,
    search_clinic_policies,
    search_procedures_faqs,
    search_provider_info,
    search_appointment_logistics,
    search_compliance_legal,
]

tools_dict = {t.name: t for t in tools}

# Bind tools to LLM
llm = llm_base.bind_tools(tools)


# ─────────────────────────────────────────────
# 7. SAFETY LAYER  (new — not in template)
#    Detects queries that must be blocked or
#    escalated before the LLM even sees them
# ─────────────────────────────────────────────
BLOCKED_PATTERNS = [
    r"\bdiagnos\w*\b",
    r"\bprescri\w*\b",
    r"\bmedication\b",
    r"\bdosage\b",
    r"\btreat(?:ment)?\s+my\b",
    r"\bdo i have\b",
    r"\bis this cancer\b",
    r"\bshould i take\b",
    r"\bwhat drug\b",
]

ESCALATE_PATTERNS = [
    r"\burgent\b",
    r"\bemergency\b",
    r"\bsevere pain\b",
    r"\bcan't stop bleeding\b",
    r"\bswelling\b.*\bjaw\b",
    r"\bfever\b",
    r"\bcan'?t breathe\b",
    r"\baccident\b",
]

def check_safety(text: str) -> tuple[str, str]:
    """
    Returns (action, message)
    action = "block" | "escalate" | "proceed"
    """
    text_lower = text.lower()

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text_lower):
            return (
                "block",
                "I'm not able to provide medical diagnoses, prescriptions, or treatment "
                "recommendations. For medical advice, please speak directly with one of our "
                "providers. You can reach us at (123) 456-789 or book an appointment online."
            )

    for pattern in ESCALATE_PATTERNS:
        if re.search(pattern, text_lower):
            return (
                "escalate",
                "⚠️ This sounds like it may be an urgent situation.\n\n"
                "**During business hours:** Please call us immediately at (123) 456-7890. "
                "Walk-ins with severe pain, uncontrolled bleeding, or acute swelling are "
                "triaged immediately.\n\n"
                "**After hours:** Call our main line and you will be routed to the on-call "
                "provider who will return your call within 30 minutes.\n\n"
                "**If you are having difficulty breathing or swallowing, go to the nearest "
                "emergency room immediately.**"
            )

    return ("proceed", "")


# ─────────────────────────────────────────────
# 8. AGENT STATE  (same TypedDict as template
#    + two extra flags)
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    messages:      Annotated[Sequence[BaseMessage], add_messages]
    safety_status: str   # "proceed" | "block" | "escalate"
    safety_msg:    str   # message to return if blocked/escalated


# ─────────────────────────────────────────────
# 9. SYSTEM PROMPT  (upgraded: medical guardrails)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a helpful virtual assistant for Metro East Dental Clinic, located at
37-15 74th Street, Jackson Heights, NY 11372.

Your job is to answer patient and staff questions using ONLY the information
in the clinic's knowledge base. Use the six retriever tools available to find
the most relevant information before answering.

═══════════════════════════════════════════════
WHAT YOU CAN ANSWER
═══════════════════════════════════════════════
✅ Insurance coverage, copays, deductibles, billing codes
✅ Office hours, cancellation policy, payment plans
✅ What procedures the clinic offers and what they involve
✅ Pre and post care instructions for procedures
✅ Procedure pricing ranges (general estimates only)
✅ Provider specializations, languages spoken, availability
✅ How to book, location, parking, transit directions
✅ HIPAA rights, consent form summaries, financial policy

═══════════════════════════════════════════════
HARD RULES — NEVER VIOLATE THESE
═══════════════════════════════════════════════
❌ NEVER diagnose a patient or interpret their symptoms
❌ NEVER recommend a specific medication or dosage
❌ NEVER confirm or deny whether a specific person is a patient
❌ NEVER ask for or repeat a patient's name, DOB, or insurance ID
❌ NEVER access or reference individual patient records
❌ NEVER speculate beyond the provided documents

═══════════════════════════════════════════════
WHEN YOU DON'T KNOW
═══════════════════════════════════════════════
If the answer is not in the knowledge base, say:
"I don't have that information — please call our office at (123) 456-7890
or speak with our front desk team for the most accurate answer."

Do NOT guess or make up information, especially about medical topics,
insurance coverage, or pricing.

═══════════════════════════════════════════════
TONE & FORMAT
═══════════════════════════════════════════════
- Be warm, clear, and concise
- For complex answers, use bullet points or numbered steps
- Always cite which area of the knowledge base your answer comes from
  (e.g., "According to our insurance coverage guide...")
- For urgent symptoms, always direct the patient to call or visit immediately
"""


# ─────────────────────────────────────────────
# 10. GRAPH NODES  (upgraded template nodes
#     + new safety_check node)
# ─────────────────────────────────────────────

def safety_check(state: AgentState) -> AgentState:
    """
    NEW NODE — not in original template.
    Runs before the LLM. Blocks or escalates dangerous queries.
    """
    last_message = state["messages"][-1]
    user_text = last_message.content if hasattr(last_message, "content") else ""

    action, msg = check_safety(user_text)

    return {
        "messages":      state["messages"],
        "safety_status": action,
        "safety_msg":    msg
    }


def call_llm(state: AgentState) -> AgentState:
    """Same pattern as template — calls LLM with full message history."""
    messages = list(state["messages"])
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm.invoke(messages)
    return {
        "messages":      [response],
        "safety_status": state.get("safety_status", "proceed"),
        "safety_msg":    state.get("safety_msg", "")
    }


def take_action(state: AgentState) -> AgentState:
    """Same pattern as template — executes tool calls."""
    tool_calls = state["messages"][-1].tool_calls
    results = []

    for t in tool_calls:
        tool_name = t["name"]
        query     = t["args"].get("query", "")
        print(f"  🔍 Calling: {tool_name}  |  Query: {query[:60]}...")

        if tool_name not in tools_dict:
            result = "Tool not found. Please retry with a valid tool."
        else:
            result = tools_dict[tool_name].invoke(query)
            print(f"     Retrieved {len(str(result))} chars")

        results.append(
            ToolMessage(tool_call_id=t["id"], name=tool_name, content=str(result))
        )

    print("  ✅ Tool execution complete — returning to LLM")
    return {
        "messages":      results,
        "safety_status": state.get("safety_status", "proceed"),
        "safety_msg":    state.get("safety_msg", "")
    }


def return_safety_response(state: AgentState) -> AgentState:
    """
    NEW NODE — returns the pre-written safety message
    directly without hitting the LLM.
    """
    safety_response = AIMessage(content=state["safety_msg"])
    return {
        "messages":      [safety_response],
        "safety_status": state["safety_status"],
        "safety_msg":    state["safety_msg"]
    }


# ─────────────────────────────────────────────
# 11. ROUTING FUNCTIONS  (upgraded)
# ─────────────────────────────────────────────

def route_after_safety(state: AgentState) -> Literal["llm", "safety_response"]:
    """After safety check: proceed to LLM or short-circuit to safety response."""
    if state.get("safety_status") in ("block", "escalate"):
        return "safety_response"
    return "llm"


def should_continue(state: AgentState) -> bool:
    """Same logic as template — check if LLM wants to call a tool."""
    last = state["messages"][-1]
    return hasattr(last, "tool_calls") and len(last.tool_calls) > 0


# ─────────────────────────────────────────────
# 12. BUILD THE GRAPH  (upgraded template graph)
#
#  safety_check
#      ↓
#  route_after_safety
#    ├─ block/escalate → safety_response → END
#    └─ proceed ──────→ llm
#                          ↓
#                    should_continue?
#                       ├─ Yes → take_action → llm (loop)
#                       └─ No  → END
# ─────────────────────────────────────────────

graph = StateGraph(AgentState)

# Add all nodes
graph.add_node("safety_check",      safety_check)
graph.add_node("safety_response",   return_safety_response)
graph.add_node("llm",               call_llm)
graph.add_node("retriever_agent",   take_action)

# Entry point — always run safety check first
graph.set_entry_point("safety_check")

# After safety check: proceed or block
graph.add_conditional_edges(
    "safety_check",
    route_after_safety,
    {
        "llm":             "llm",
        "safety_response": "safety_response"
    }
)

# Safety response always ends
graph.add_edge("safety_response", END)

# After LLM: use tools or end (same as template)
graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)

# After tool use: always go back to LLM (same as template)
graph.add_edge("retriever_agent", "llm")

# Compile
rag_agent = graph.compile()
print("\n✅ Dental RAG Agent compiled successfully")


# ─────────────────────────────────────────────
# 13. RUNNER  (same pattern as template)
# ─────────────────────────────────────────────

def running_agent():
    print("\n" + "═"*50)
    print("  Metro East Dental Clinic — RAG Assistant")
    print("  Jackson Heights, NY  |  (718) 555-0192")
    print("═"*50)
    print("  Type 'exit' or 'quit' to stop\n")

    while True:
        user_input = input("Patient/Staff: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        messages = [HumanMessage(content=user_input)]

        try:
            result = rag_agent.invoke({
                "messages":      messages,
                "safety_status": "proceed",
                "safety_msg":    ""
            })

            print("\nAssistant:", result["messages"][-1].content)
            print()

        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            print("Please try rephrasing your question.\n")


if __name__ == "__main__":
    running_agent()