# Robust Memory System for Large Language Models

![llm-memory-system](https://github.com/user-attachments/assets/2612c618-d00b-433b-a285-b114eaf23069)

Large language models (LLMs) are inherently stateless. In order to provide historical information to models, we need a robust memory system that will keep track of previous conversations and context. Memory is a concept that refers to how LLMs store and retrieve information to generate accurate and coherent outputs. The memory can be divided into two broad categories:

1. Internal memory (what the model "remembers" during a conversation). This is based on the **context Window**.
2. External memory (what the model can access externally, beyond its immediate context window).

Model's internal memory is limited by context window size. Context windows is a fixed number of tokens (words, characters, or sub-word units) that allows LLMs to "remember" what has been said in a conversation or the text they have processed. However, this window is limited. For example, GPT-3 has a context window of around 2048 tokens, while GPT-4 can handle up to 8,192 tokens. This limitation means that the model can only remember or consider information within that window. Anything beyond that falls out of the model's immediate memory, and it can no longer retain or use that information directly.

Unlike in-context learning, which relies on the LLM's internal context window, structured memory stores information outside of the model's immediate memory.

---

## Building a Chatbot Memory System

Building a robust chatbot memory system boils down to a few core patterns:

1. Sliding-Window (Raw Chat History)

- What it is: Keep the last `N` messages in the prompt.
- How it works: On each turn, you append the new **user+assistant** exchange to a buffer and trim it once it exceeds the token limit.
- This is trivial to implement, and it is always exact.
- However, you quickly runs out of **context** capacity. Plus, early history is lost as you exceed N messages.

2. Summarization / Condensation Memory

- What it is: Periodically compress old history into a short summary.
- How it works: After every `M` turns (or whenever the buffer grows too big), send the oldest chunk to the LLM with a "summarize these into bullet points" prompt.

3. Embedding-Based Retrieval Memory (RAG-Style)
4. Structured / Key-Value Memory
5. **Hybrid Approaches**:
- Often the best systems mix two or more techniques:
- Raw buffer for the most recent few turns
- Summaries for medium-term context
- Vector retrieval for important long-term facts
- Key-value for user-specific preferences or profile data

Many LLM frameworks provide abstractions/wrappers for robust llm memory system. E.g. LangChain:

- ConversationBufferMemory (sliding window)
- ConversationSummaryMemory (auto-summaries)
- VectorStoreRetrieverMemory (RAG)
- CombinedMemory (mix & match)
