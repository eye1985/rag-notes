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
