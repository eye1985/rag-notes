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
