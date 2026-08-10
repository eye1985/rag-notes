# LangGraph

Problem with simple chain
- Cant loop back
- No state between steps
- No conditional routing
- Crashes lose all

LangGraph solves this.

- Cycles and loops
- Persistent state
- Conditional routing
- Crash recovery


## Setup

### State 

Something that holds a state
```python
class MultiStepState(TypedDict):
    input: str
    analyzed: str
    enhanced: str
    final: str
```

### Node 

A step or a destination
```python
    def analyze_node(state: MultiStepState) -> dict:
        response = llm.invoke(
            [
                HumanMessage(
                    content=f"Analyze the following input and summarize it in one sentence: {state['input']}"
                )
            ]
        )
        return {
            "analyzed": response.content,
        }
```

### Edge

Like a road to the node

```python
    graph = StateGraph(MultiStepState)
    graph.add_node("analyze", analyze_node)
    graph.add_node("enhance", enhance)
    graph.add_node("finalize", finalize)

    # Edge start here
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "enhance")
    graph.add_edge("enhance", "finalize")
    graph.add_edge("finalize", END)
    # Edge ends here

    app = graph.compile()
    result = app.invoke({"input": "Artificial intelligence"})
```

## Conditional Edges

### Example

```python
def demo_conditional_edges():
    def classify_query(state: RouterState) -> dict:
        response = llm.invoke(
            f"Classify this query as 'question', 'command', or 'statement'. "
            f"Reply with just the world.\n\n{state['query']}"
        )

        return {"query_type": response.content.lower().strip()}

    def handle_question(state: RouterState) -> dict:
        response = llm.invoke(f"Answer this question: {state['query']}")
        return {"response": f"[Answer] {response.content}"}

    def handle_command(state: RouterState) -> dict:
        return {"response": f"[Executing I'll help you with: {state['query']}]"}

    def handle_statement(state: RouterState) -> dict:
        return {"response": f"[Acknowledged] Thanks for sharing: {state['query']}"}

    def route_by_type(
        state: RouterState,
    ) -> Literal["question", "command", "statement"]:
        qt = state["query_type"]

        if "question" in qt:
            return "question"
        elif "command" in qt:
            return "command"
        else:
            return "statement"

    graph = StateGraph(RouterState)

    graph.add_node("classify", classify_query)
    graph.add_node("handle_question", handle_question)
    graph.add_node("handle_command", handle_command)
    graph.add_node("handle_statement", handle_statement)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_type,
        {
            "question": "handle_question",
            "command": "handle_command",
            "statement": "handle_statement",
        },
    )

    graph.add_edge("handle_question", END)
    graph.add_edge("handle_command", END)
    graph.add_edge("handle_statement", END)

    app = graph.compile()

    queries = [
        "What is the capital of France?",
        "Send an email to John",
        "I love programming",
    ]

    for query in queries:
        result = app.invoke({"query": query})
        print(f"Query: {query}")
        print(f"Type: {result['query_type']}")
        print(f"Response: {result['response']}")
        print("-" * 40)
```

### Loop back

Just use the edge to call back to a previous edge to loop back.

### Cycles and loops example

```python
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
import operator
from dotenv import load_dotenv

load_dotenv()

"""
Cycles and Loops in LangGraph
Self-correcting agents and iterative refinement
"""

llm = init_chat_model("gpt-4o-mini", temperature=0.0)


class CodeGenState(TypedDict):
    task: str
    code: str
    errors: Annotated[list[str], operator.add]
    iteration: int
    max_iterations: int
    success: bool


def demo_self_correcting_code():
    """Self-correcting code generator."""

    def generate_code(state: CodeGenState) -> dict:
        if state["iteration"] == 0:
            # First attempt
            prompt = f"Write Python code for: {state['task']}\nReturn only the code."
        else:
            # Correction attempt
            prompt = (
                f"Fix this Python code:\n{state['code']}\n\n"
                f"Errors:\n{state['errors'][-1]}\n\n"
                "Return only the corrected code."
            )

        response = llm.invoke(prompt)
        code = response.content.strip()

        # Clean up markdown code blocks if present
        if code.startswith("```"):
            code = code.split("```")[1]
            if code.startswith("python"):
                code = code[6:]

        return {"code": code, "iteration": state["iteration"] + 1}

    def validate_code(state: CodeGenState) -> dict:
        code = state["code"]

        # Step 1: Does it compile?
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            return {"errors": [f"SyntaxError: {e}"], "success": False}

        # Step 2: Does it RUN and produce correct results?
        test_cases = [
            ([3, 1, 4, 1, 5, 9], 5),  # normal case
            ([1, 1, 1], None),  # all same → no second largest
            ([7], None),  # single element
            ([3, -1, 3, 5, 5], 3),  # duplicates at top
        ]

        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            return {"errors": [f"Runtime error: {e}"], "success": False}

        if "solve" not in namespace:
            return {"errors": ["Function 'solve' not found in code"], "success": False}

        for inputs, expected in test_cases:
            try:
                result = namespace["solve"](inputs)
                if result != expected:
                    return {
                        "errors": [
                            f"solve({inputs}) returned {result}, expected {expected}"
                        ],
                        "success": False,
                    }
            except Exception as e:
                return {"errors": [f"solve({inputs}) raised {e}"], "success": False}

        return {"success": True}

    def should_continue(state: CodeGenState) -> Literal["generate", "end"]:
        if state["success"]:
            return "end"
        elif state["iteration"] >= state["max_iterations"]:
            return "end"
        else:
            return "generate"

    def finalize(state: CodeGenState) -> dict:
        return state

    graph = StateGraph(CodeGenState)

    graph.add_node("generate", generate_code)
    graph.add_node("validate", validate_code)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate", should_continue, {"generate": "generate", "end": "finalize"}
    )  # Loop back to "generate" if not successful and under max iterations, otherwise go to "finalize"
    graph.add_edge("finalize", END)

    app = graph.compile()

    # visualize the graph
    print("\n--- Mermaid Graph ---")
    # print(app.get_graph().draw_mermaid())

    # save as PNG
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("graph_code.png", "wb") as f:
        f.write(png_bytes)
    print("\nGraph saved to graph_code.png")

    print("Self-Correcting Code Generator:\n")

    result = app.invoke(
        {
            "task": "a function that calculates factorial recursively",
            "code": "",
            "errors": [],
            "iteration": 0,
            "max_iterations": 3,
            "success": False,
        }
    )

    print(f"Task: {result['task']}")
    print(f"Iterations: {result['iteration']}")
    print(f"Success: {result['success']}")
    print(f"Final Code:\n{result['code']}")


class ResearchState(TypedDict):
    topic: str
    findings: Annotated[list[str], operator.add]
    questions: list[str]
    iteration: int
    max_depth: int
    summary: str


def demo_iterative_research():
    """Iterative research that goes deeper based on findings."""

    def research(state: ResearchState) -> dict:
        print(f"\n{'─' * 50}")
        print(f"📚 [RESEARCH] Depth {state['iteration'] + 1}/{state['max_depth']}")

        if state["iteration"] == 0:
            query = f"Give me 3 key facts about: {state['topic']}"
            print(f"   Starting fresh on: {state['topic']}")
        else:
            question = state["questions"][-1] if state["questions"] else "elaborate"
            query = f"Based on these findings:\n{state['findings'][-1]}\n\nGo deeper: {question}"
            print(f"   Following up on: {question}")

        response = llm.invoke(query)
        print(f"   ✅ Found {len(response.content.splitlines())} lines of findings")
        print(f"   Preview: {response.content[:120]}...")
        return {"findings": [response.content]}

    def generate_questions(state: ResearchState) -> dict:
        print(f"\n{'─' * 50}")
        print(f"🤔 [QUESTIONING] Analyzing latest findings...")

        response = llm.invoke(
            f"Based on this finding:\n{state['findings'][-1]}\n\n"
            "What's one deeper question to explore? Reply with just the question."
        )

        print(f"   Next question: {response.content.strip()}")

        return {"questions": [response.content], "iteration": state["iteration"] + 1}

    def synthesize(state: ResearchState) -> dict:
        print(f"\n{'─' * 50}")
        print(
            f"🧬 [SYNTHESIZE] Combining {len(state['findings'])} rounds of findings..."
        )

        all_findings = "\n\n".join(state["findings"])
        response = llm.invoke(
            f"Synthesize these findings into a coherent summary:\n\n{all_findings}"
        )

        print(f"   ✅ Summary generated ({len(response.content.split())} words)")
        return {"summary": response.content}

    def should_continue(state: ResearchState) -> Literal["research", "synthesize"]:
        if state["iteration"] >= state["max_depth"]:
            print(
                f"\n🏁 [ROUTER] Max depth reached ({state['iteration']}/{state['max_depth']}) → synthesizing"
            )
            return "synthesize"
        print(
            f"\n🔄 [ROUTER] Depth {state['iteration']}/{state['max_depth']} → going deeper"
        )
        return "research"

    graph = StateGraph(ResearchState)

    graph.add_node("research", research)
    graph.add_node("generate_questions", generate_questions)
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "research")
    graph.add_edge("research", "generate_questions")
    graph.add_conditional_edges(
        "generate_questions",
        should_continue,
        {"research": "research", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)

    app = graph.compile()

    print("=" * 50)
    print("🔬 ITERATIVE RESEARCH WORKFLOW")
    print("=" * 50)

    result = app.invoke(
        {
            "topic": "quantum computing applications",
            "findings": [],
            "questions": [],
            "iteration": 0,
            "max_depth": 2,
            "summary": "",
        }
    )

    print(f"\n{'=' * 50}")
    print(f"📊 RESEARCH COMPLETE")
    print(f"   Topic: {result['topic']}")
    print(f"   Depth reached: {result['iteration']}")
    print(f"   Findings collected: {len(result['findings'])}")
    print(f"   Questions explored: {len(result['questions'])}")
    print(f"\n📝 Final Summary:\n{result['summary']}")


if __name__ == "__main__":
    # demo_self_correcting_code()
    demo_iterative_research()

```

### Checkpoints using Langgraph memory

```python
"""
Human-in-the-Loop Patterns in LangGraph
Interrupt, review, modify, and resume
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from typing import Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import time

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ─── Helper for visual separation ───
def phase_banner(phase_num: int, title: str):
    print(f"\n{'=' * 55}")
    print(f"  PHASE {phase_num}: {title}")
    print(f"{'=' * 55}")


def step_print(icon: str, label: str, detail: str = ""):
    print(f"\n{icon} [{label}] {detail}")


# ════════════════════════════════════════════════════════
# DEMO 1: Interrupt for Approval
# ════════════════════════════════════════════════════════


class ApprovalState(TypedDict):
    request: str
    draft: str
    approved: bool
    feedback: str
    final: str


def demo_interrupt_for_approval():
    """Interrupt execution for human approval."""

    def create_draft(state: ApprovalState) -> dict:
        step_print("📝", "DRAFT NODE", "Entering create_draft node...")
        print(f"   Request: \"{state['request']}\"")
        print(f"   Calling LLM to generate draft...")

        response = llm.invoke(f"Create a professional response for: {state['request']}")

        print(f"   Draft generated ({len(response.content.split())} words)")
        print(f"   Preview: {response.content[:100]}...")
        return {"draft": response.content}

    def wait_for_approval(state: ApprovalState) -> dict:
        step_print("👁️", "APPROVAL NODE", "Entering wait_for_approval node...")
        print(f"   Approved: {state['approved']}")
        print(
            f"   Feedback: '{state['feedback']}'"
            if state["feedback"]
            else "   Feedback: (none yet)"
        )
        # This node is where we'll interrupt
        return state

    def finalize(state: ApprovalState) -> dict:
        step_print("📦", "FINALIZE NODE", "Entering finalize node...")
        print(f"   Approved: {state['approved']}")

        if state["approved"]:
            print(f"   Action: Using draft as-is (human approved)")
            return {"final": state["draft"]}
        else:
            print(f"   Action: Revising draft based on feedback...")
            print(f"   Feedback: \"{state['feedback']}\"")
            # Incorporate feedback
            response = llm.invoke(
                f"Revise this draft based on feedback:\n\n"
                f"Draft: {state['draft']}\n\n"
                f"Feedback: {state['feedback']}"
            )
            print(f"   Revised draft generated ({len(response.content.split())} words)")
            return {"final": response.content}

    graph = StateGraph(ApprovalState)

    graph.add_node("draft", create_draft)
    graph.add_node("approval", wait_for_approval)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "draft")
    graph.add_edge("draft", "approval")
    graph.add_edge("approval", "finalize")
    graph.add_edge("finalize", END)

    # Compile with checkpointer and interrupt
    memory = MemorySaver()
    app = graph.compile(
        checkpointer=memory, interrupt_before=["approval"]  # Pause before this node
    )

    print("\n" + "=" * 55)
    print("  HUMAN-IN-THE-LOOP: APPROVAL WORKFLOW")
    print("=" * 55)

    print("\n   Graph: START -> [draft] -> ⏸️ -> [approval] -> [finalize] -> END")
    print("   Interrupt set BEFORE: 'approval' node")

    # Configuration for this thread
    config = {"configurable": {"thread_id": "demo-1"}}

    # ─── PHASE 1: Run until interrupt ───
    phase_banner(1, "RUN UNTIL INTERRUPT")
    print("   Calling app.invoke() with initial state...")
    print("   The graph will run until it hits the interrupt point.\n")

    result = app.invoke(
        {
            "request": "Write a thank-you email for a job interview",
            "draft": "",
            "approved": False,
            "feedback": "",
            "final": "",
        },
        config,
    )

    step_print("⏸️", "PAUSED", "Graph execution interrupted!")
    print(f"   Draft is ready: {result['draft'][:150]}...")
    print(f"   Final is empty: '{result['final']}'")
    print(f"\n   The graph is now FROZEN. Waiting for human input.")
    print(f"   In a real app, your frontend would show the draft here.")

    # ─── PHASE 2: Inspect paused state ───
    phase_banner(2, "INSPECT PAUSED STATE")

    current_state = app.get_state(config)
    print(f"   app.get_state(config) tells us:")
    print(f"   Next node(s): {current_state.next}")
    print(f"   State keys: {list(current_state.values.keys())}")
    print(f"   Draft filled: {'Yes' if current_state.values['draft'] else 'No'}")
    print(f"   Approved: {current_state.values['approved']}")
    print(f"   Final filled: {'Yes' if current_state.values['final'] else 'No'}")

    # ─── PHASE 3: Human provides feedback and resume ───
    phase_banner(3, "HUMAN INJECTS FEEDBACK + RESUME")

    feedback_text = (
        "Make it more concise and add specific mention of the company culture"
    )
    print(f"   Human decision: REJECT (request changes)")
    print(f'   Human feedback: "{feedback_text}"')
    print(f"\n   Calling app.update_state() to inject human input...")

    # Update state with human input
    app.update_state(
        config, {"approved": False, "feedback": feedback_text}  # Request changes
    )

    print(f"   State updated. approved=False, feedback set.")
    print(f"\n   Calling app.invoke(None, config) to RESUME...")
    print(f"   (None means 'no new input, just continue from checkpoint')\n")

    # Continue execution
    final_result = app.invoke(None, config)

    # ─── RESULT ───
    step_print("✅", "WORKFLOW COMPLETE", "")
    print(f"   Final result ({len(final_result['final'].split())} words):")
    print(f"   {final_result['final'][:200]}...")
    print(f"\n   Graph path taken:")
    print(
        f"   START -> [draft] -> ⏸️ PAUSE -> human feedback -> [approval] -> [finalize] -> END"
    )


# ════════════════════════════════════════════════════════
# DEMO 2: Iterative Review (Human-in-the-Loop + Cycles)
# ════════════════════════════════════════════════════════


class ReviewState(TypedDict):
    document: str
    review_comments: list[str]
    revision_count: int
    status: str


def demo_iterative_review():
    """Multiple rounds of human review."""

    def submit_for_review(state: ReviewState) -> dict:
        step_print("📋", "SUBMIT NODE", f"Round {state['revision_count'] + 1}")
        print(f"   Status incoming: '{state['status']}'")
        print(f"   Setting status to 'pending_review'")
        print(f"   Document preview: {state['document'][:100]}...")
        return {"status": "pending_review"}

    def apply_feedback(state: ReviewState) -> dict:
        step_print(
            "🔧", "APPLY FEEDBACK NODE", f"Revision #{state['revision_count'] + 1}"
        )

        if not state["review_comments"]:
            print(f"   No comments to apply. Passing through.")
            return state

        feedback = state["review_comments"][-1]
        print(f'   Feedback to apply: "{feedback}"')
        print(f"   Current document: {state['document'][:80]}...")
        print(f"   Calling LLM to revise...")

        response = llm.invoke(
            f"Revise this document based on feedback:\n\n"
            f"Document: {state['document']}\n\n"
            f"Feedback: {feedback}"
        )

        print(f"   Revised document ({len(response.content.split())} words)")
        print(f"   Preview: {response.content[:100]}...")

        return {
            "document": response.content,
            "revision_count": state["revision_count"] + 1,
            "status": "revised",
        }

    def route_after_review(state: ReviewState) -> Literal["apply", "done"]:
        step_print("🔀", "ROUTER", f"Checking status: '{state['status']}'")
        if state["status"] == "approved":
            print(f"   Decision: APPROVED -> routing to 'done' node")
            return "done"
        print(f"   Decision: NOT APPROVED -> routing to 'apply' node")
        return "apply"

    def finalize(state: ReviewState) -> dict:
        step_print("🏁", "DONE NODE", "Finalizing document")
        print(f"   Total revisions: {state['revision_count']}")
        print(f"   Final document: {state['document'][:100]}...")
        return {"status": "finalized"}

    graph = StateGraph(ReviewState)

    graph.add_node("submit", submit_for_review)
    graph.add_node("apply", apply_feedback)
    graph.add_node("done", finalize)

    graph.add_edge(START, "submit")
    graph.add_conditional_edges(
        "submit", route_after_review, {"apply": "apply", "done": "done"}
    )
    graph.add_edge("apply", "submit")  # Loop for more reviews
    graph.add_edge("done", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory, interrupt_before=["submit"])

    print("\n" + "=" * 55)
    print("  HUMAN-IN-THE-LOOP: ITERATIVE REVIEW WORKFLOW")
    print("=" * 55)

    print("\n   Graph: START -> ⏸️ -> [submit] -> [ROUTER]")
    print("                                       ├── approved -> [done] -> END")
    print(
        "                                       └── else -> [apply] -> ⏸️ [submit] (LOOP)"
    )
    print("   Interrupt set BEFORE: 'submit' node (fires EVERY loop)")

    config = {"configurable": {"thread_id": "review-1"}}

    # ─── ROUND 0: Initial submission ───
    phase_banner(0, "INITIAL SUBMISSION")
    print("   Sending initial document into the graph...")
    print('   Document: "AI is technology that helps computers think."\n')

    result = app.invoke(
        {
            "document": "AI is technology that helps computers think.",
            "review_comments": [],
            "revision_count": 0,
            "status": "",
        },
        config,
    )

    step_print("⏸️", "PAUSED", "Graph hit interrupt_before='submit'")
    print(f"   Document ready for review: \"{result['document']}\"")
    print(f"   Revisions so far: {result['revision_count']}")

    current_state = app.get_state(config)
    print(f"   Next node: {current_state.next}")
    print(f"\n   Waiting for human reviewer...")

    # ─── ROUND 1: Reviewer wants changes ───
    phase_banner(1, "REVIEWER REQUESTS CHANGES")

    feedback_1 = "Add more technical depth and examples"
    print(f'   Reviewer says: "{feedback_1}"')
    print(f"   Reviewer sets status: 'needs_revision'")
    print(f"\n   Calling app.update_state() to inject review...")

    app.update_state(
        config, {"review_comments": [feedback_1], "status": "needs_revision"}
    )

    print(f"   State updated. Calling app.invoke(None) to resume...\n")

    result = app.invoke(None, config)

    step_print("⏸️", "PAUSED AGAIN", "Graph looped back to 'submit' and paused")
    print(f"   Revised document: {result['document'][:150]}...")
    print(f"   Revisions so far: {result['revision_count']}")

    current_state = app.get_state(config)
    print(f"   Next node: {current_state.next}")
    print(f"\n   Waiting for human reviewer again...")

    # ─── ROUND 2: Reviewer wants more changes ───
    phase_banner(2, "REVIEWER REQUESTS MORE CHANGES")

    feedback_2 = "Good improvement! Now add a concrete example of neural networks"
    print(f'   Reviewer says: "{feedback_2}"')
    print(f"   Reviewer sets status: 'needs_revision'")

    app.update_state(
        config, {"review_comments": [feedback_2], "status": "needs_revision"}
    )

    print(f"   Resuming graph...\n")

    result = app.invoke(None, config)

    step_print("⏸️", "PAUSED AGAIN", "Graph looped back to 'submit' and paused")
    print(f"   Revised document: {result['document'][:150]}...")
    print(f"   Revisions so far: {result['revision_count']}")

    # ─── ROUND 3: Reviewer approves ───
    phase_banner(3, "REVIEWER APPROVES")

    print(f'   Reviewer says: "Looks great!"')
    print(f"   Reviewer sets status: 'approved'")

    app.update_state(config, {"status": "approved"})

    print(f"   Resuming graph for final time...\n")

    final = app.invoke(None, config)

    # ─── FINAL SUMMARY ───
    step_print("✅", "WORKFLOW COMPLETE", "")
    print(f"   Final status: {final['status']}")
    print(f"   Total revisions: {final['revision_count']}")
    print(f"   Final document: {final['document'][:200]}...")
    print(f"\n   Full timeline:")
    print(f"   Round 0: START -> ⏸️ (human reviews initial doc)")
    print(f"   Round 1: resume -> [submit] -> [apply] -> ⏸️ (human reviews revision 1)")
    print(f"   Round 2: resume -> [submit] -> [apply] -> ⏸️ (human reviews revision 2)")
    print(f"   Round 3: resume -> [submit] -> [done] -> END (human approved!)")


if __name__ == "__main__":
    print("=" * 55)
    print("  Demo 1: Interrupt for Approval")
    print("=" * 55)
    demo_interrupt_for_approval()

    print("\n" + "=" * 55)
    print("  Demo 2: Iterative Review")
    print("=" * 55)
    demo_iterative_review()

```

### Save in SQLITE example

python```
"""
Checkpointing and Persistence in LangGraph
Save and resume agent state
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import operator
import tempfile
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


def demo_memory_saver():
    """In-memory checkpointing for development."""

    def chat(state: ChatState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)

    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    saver = MemorySaver()
    app = graph.compile(checkpointer=saver)

    # Configuration with thread_id
    config = {"configurable": {"thread_id": "user-123"}}

    print("Memory Saver Demo (Multi-turn conversation):\n")

    # Turn 1
    result = app.invoke(
        {"messages": [HumanMessage(content="My name is Paulo")]}, config
    )
    print(f"Turn 1 - AI: {result['messages'][-1].content}")

    # Turn 2 - Conversation continues
    result = app.invoke({"messages": [HumanMessage(content="What's my name?")]}, config)
    print(f"Turn 2 - AI: {result['messages'][-1].content}")

    # Check full history
    state = app.get_state(config)
    print(f"\nTotal messages in state: {len(state.values['messages'])}")


def demo_sqlite_persistence():
    """SQLite persistence for durable storage."""

    def chat(state: ChatState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    print(f"\nSQLite Persistence Demo:")
    print(f"Database: {db_path}\n")

    # First session
    with SqliteSaver.from_conn_string(db_path) as saver:
        app = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "persistent-user"}}

        result = app.invoke(
            {
                "messages": [
                    HumanMessage(content="Remember: The secret code is ALPHA-7")
                ]
            },
            config,
        )
        print(f"Session 1 - Stored secret code")

        # PostgresSaver with a real database!
        # Simulate app restart - new session
    with SqliteSaver.from_conn_string(db_path) as saver:
        app = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "persistent-user"}}

        result = app.invoke(
            {"messages": [HumanMessage(content="What was the secret code?")]}, config
        )
        print(f"Session 2 - AI: {result['messages'][-1].content}")


def demo_state_inspection():
    """Inspect and manipulate checkpoint state."""

    def chat(state: ChatState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "inspect-demo"}}

    print("\nState Inspection Demo:\n")

    # Build up some state
    app.invoke({"messages": [HumanMessage(content="Hello!")]}, config)
    app.invoke({"messages": [HumanMessage(content="How are you?")]}, config)

    # Get current state
    state = app.get_state(config)

    print("Current state:")
    print(f"  Next node: {state.next}")
    print(f"  Message count: {len(state.values['messages'])}")

    # Get state history
    print("\nState history:")
    for i, snapshot in enumerate(app.get_state_history(config)):
        print(f"  Checkpoint {i}: {len(snapshot.values['messages'])} messages")
        if i >= 3:
            print("  ...")
            break


def demo_branching_conversations():
    """Branch conversations from checkpoints."""

    def chat(state: ChatState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    print("\nBranching Conversations Demo:\n")

    # Main conversation
    main_config = {"configurable": {"thread_id": "main"}}
    app.invoke(
        {"messages": [HumanMessage(content="What's the weather like?")]}, main_config
    )

    # Get checkpoint to branch from
    main_state = app.get_state(main_config)

    # Branch A - Beach vacation
    branch_a_config = {"configurable": {"thread_id": "branch-beach"}}
    # Copy state to new thread
    app.update_state(branch_a_config, main_state.values)

    result_a = app.invoke(
        {"messages": [HumanMessage(content="What about a beach vacation?")]},
        branch_a_config,
    )
    print(f"Branch A (Beach): {result_a['messages'][-1].content[:100]}...")

    # Branch B - Mountain adventure
    branch_b_config = {"configurable": {"thread_id": "branch-mountain"}}
    app.update_state(branch_b_config, main_state.values)

    result_b = app.invoke(
        {"messages": [HumanMessage(content="What about mountain hiking?")]},
        branch_b_config,
    )
    print(f"Branch B (Mountain): {result_b['messages'][-1].content[:100]}...")


def demo_checkpoint_internals():
    """
    Peek inside a checkpoint — see exactly what LangGraph saves.

    Uses a 2-node graph so we generate multiple checkpoints,
    then walks through every field in the checkpoint object.
    """

    # ── Build a 2-node graph so we get several checkpoints ──

    class TaskState(TypedDict):
        messages: Annotated[list[BaseMessage], operator.add]
        step: str

    def analyze(state: TaskState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response], "step": "analyzed"}

    def summarize(state: TaskState) -> dict:
        summary_prompt = [
            HumanMessage(
                content=f"Summarize this in one sentence: {state['messages'][-1].content}"
            )
        ]
        response = llm.invoke(summary_prompt)
        return {"messages": [response], "step": "summarized"}

    graph = StateGraph(TaskState)
    graph.add_node("analyze", analyze)
    graph.add_node("summarize", summarize)
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "summarize")
    graph.add_edge("summarize", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "internals-demo"}}

    print("\nCheckpoint Internals Demo")
    print("=" * 55)
    print("Graph: START -> [analyze] -> [summarize] -> END")
    print("=" * 55)

    # ── Run the graph ──

    app.invoke(
        {"messages": [HumanMessage(content="Explain why the sky is blue")], "step": ""},
        config,
    )

    # ════════════════════════════════════════════════════════
    # PART 1: What's in the CURRENT state snapshot?
    # ════════════════════════════════════════════════════════

    print("\n--- PART 1: Current State Snapshot (app.get_state) ---\n")

    state = app.get_state(config)

    # state.values — your actual TypedDict data
    print("1) state.values (your state data):")
    print(f"   step: '{state.values['step']}'")
    print(f"   messages: {len(state.values['messages'])} total")
    for i, msg in enumerate(state.values["messages"]):
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"     [{i}] {role}: {msg.content[:80]}...")

    # state.next — which node runs next (empty = graph finished)
    print(f"\n2) state.next (pending node):")
    print(f"   {state.next if state.next else '() — graph finished, no pending nodes'}")

    # state.config — the config that produced this snapshot
    print(f"\n3) state.config (thread + checkpoint IDs):")
    print(f"   thread_id:     {state.config['configurable']['thread_id']}")
    print(f"   checkpoint_id: {state.config['configurable']['checkpoint_id']}")

    # state.metadata — who created this checkpoint
    print(f"\n4) state.metadata (provenance info):")
    print(f"   source:  {state.metadata.get('source', 'N/A')}")
    print(f"   step:    {state.metadata.get('step', 'N/A')}")
    print(f"   writes:  {state.metadata.get('writes', 'N/A')}")

    # state.parent_config — pointer to the PREVIOUS checkpoint
    print(f"\n5) state.parent_config (previous checkpoint):")
    if state.parent_config:
        print(
            f"   parent checkpoint_id: {state.parent_config['configurable']['checkpoint_id']}"
        )
    else:
        print(f"   None — this is the very first checkpoint")

    # state.created_at — timestamp
    print(f"\n6) state.created_at (when saved):")
    print(f"   {state.created_at}")

    # ════════════════════════════════════════════════════════
    # PART 2: Walk through ALL checkpoints (time travel)
    # ════════════════════════════════════════════════════════

    print("\n--- PART 2: Full Checkpoint History (app.get_state_history) ---\n")
    print("LangGraph saves a checkpoint at EACH step. Let's see them all:\n")

    for i, snapshot in enumerate(app.get_state_history(config)):
        step_num = snapshot.metadata.get("step", "?")
        source = snapshot.metadata.get("source", "?")
        writes = snapshot.metadata.get("writes", {})
        msg_count = len(snapshot.values.get("messages", []))
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
        current_step = snapshot.values.get("step", "")

        # Which node just wrote to this checkpoint?
        node_name = list(writes.keys())[0] if writes else "—"

        print(f"  Checkpoint {i}:")
        print(f"    id:         {checkpoint_id[:30]}...")
        print(f"    source:     {source}")
        print(f"    step:       {step_num}")
        print(f"    written by: {node_name}")
        print(f"    state.step: '{current_step}'")
        print(f"    messages:   {msg_count}")
        print(f"    next:       {snapshot.next if snapshot.next else '() — finished'}")
        print(f"    created_at: {snapshot.created_at}")
        print()

    # ════════════════════════════════════════════════════════
    # PART 3: Jump to a specific checkpoint (rewind)
    # ════════════════════════════════════════════════════════

    print("--- PART 3: Rewind — Jump to a Previous Checkpoint ---\n")

    # Find the checkpoint right after the "analyze" node ran
    target_snapshot = None
    for snapshot in app.get_state_history(config):
        writes = snapshot.metadata.get("writes", {})
        if "analyze" in writes:
            target_snapshot = snapshot
            break

    if target_snapshot:
        target_id = target_snapshot.config["configurable"]["checkpoint_id"]
        print(f"  Found checkpoint after 'analyze' node: {target_id[:30]}...")
        print(f"  Messages at that point: {len(target_snapshot.values['messages'])}")
        print(f"  state.step at that point: '{target_snapshot.values.get('step', '')}'")

        # You can resume from this exact checkpoint
        rewind_config = {
            "configurable": {"thread_id": "internals-demo", "checkpoint_id": target_id}
        }

        rewound_state = app.get_state(rewind_config)
        print(f"\n  Loaded checkpoint — next node would be: {rewound_state.next}")
        print(f"  We're back to BEFORE 'summarize' ran!")
        print(
            f"  Calling invoke(None) from here would re-run 'summarize' with fresh output."
        )
    else:
        print("  Could not find target checkpoint.")

    # ════════════════════════════════════════════════════════
    # SUMMARY: Anatomy of a checkpoint
    # ════════════════════════════════════════════════════════

    print("\n" + "=" * 55)
    print("  CHECKPOINT ANATOMY — What Gets Saved")
    print("=" * 55)
    print(
        """
    state.values        → Your TypedDict data (messages, step, etc.)
    state.next          → Tuple of nodes that run next (() if done)
    state.config        → thread_id + checkpoint_id (unique address)
    state.parent_config → Previous checkpoint's address (linked list)
    state.metadata      → source, step number, which node wrote
    state.created_at    → Timestamp of when this checkpoint was saved

    Checkpoints are saved:
      1. BEFORE the first node runs (initial input state)
      2. AFTER each node completes (with updated state)
      3. At interrupt points (frozen state for human-in-the-loop)

    Think of it as a linked list of snapshots:
      [initial] --> [after analyze] --> [after summarize]
         ^               ^                    ^
       parent          parent              current (latest)
    """
    )


if __name__ == "__main__":
    # demo_memory_saver()
    # demo_sqlite_persistence()
    # demo_state_inspection()
    # demo_branching_conversations()
    demo_checkpoint_internals()
```
