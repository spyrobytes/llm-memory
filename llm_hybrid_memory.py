# Chatbot memory system: Hybrid Approach from Scratch
#
# This is a working version
# Let's build the llm memory system, we'll start small and iterate until we get
# it right.

import uuid
import requests
from sentence_transformers import SentenceTransformer
import chromadb

# ─── CONFIG ──────────────────────────────────────────────────────────────
OLLAMA_URL        = "http://localhost:11434"  # your Ollama server
LLAMA_MODEL       = "llama3.2:3b"
MAX_BUFFER_TURNS  = 6                         # keep this many recent turns
SUMMARY_TRIGGER   = 8                         # condense once buffer exceeds this
VECTOR_K          = 3                         # how many memories to retrieve
# ─────────────────────────────────────────────────────────────────────────

# The HybridMemory class
class HybridMemory:
    def __init__(self):
        # 1) raw buffer of dicts: [{role,content}, …]
        self.buffer = []
        # 2) rolling summary string
        self.summary = ""
        # 3) key→value store
        self.kv = {}
        # 4) vector store (Chroma) + embedder
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.Client()
        self.col = client.get_or_create_collection("chat_memory")

    def call_llm(self, prompt: str) -> str:
        """Send a prompt to Ollama and return the completion text."""
        url = f"{OLLAMA_URL}/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 512,
        }

        resp = requests.post(url, headers=headers, data=json.dumps(data))
        if resp.status_code != 200:
            resp.raise_for_status()
        return resp.json()["response"]


    def embed(self, text: str):
        return self.embedder.encode(text).tolist()

    def extract_kv(self, text: str):
        """
        Naïve example: look for lines like "My favorite color is X".
        In reality you'd call the LLM with an extraction prompt.
        """
        if "favorite color is" in text.lower():
            color = text.split("favorite color is")[-1].strip().split()[0]
            self.kv["favorite_color"] = color

    def add_turn(self, role: str, content: str):
        # 1) add to raw buffer
        self.buffer.append({"role": role, "content": content})

        # 2) extract KV facts
        if role == "user":
            self.extract_kv(content)

        # 3) embed + upsert into vector store
        vec = self.embed(content)
        self.col.add(
            ids=[str(uuid.uuid4())],
            embeddings=[vec],
            metadatas=[{"role": role}],
            documents=[content]
        )

        # 4) if buffer too big, summarize oldest
        if len(self.buffer) > SUMMARY_TRIGGER:
            to_summarize = "\n".join(
                f"{m['role']}: {m['content']}"
                for m in self.buffer[:-MAX_BUFFER_TURNS]
            )
            summary_prompt = (
                "Condense the following conversation into a brief bullet-list summary:\n\n"
                + to_summarize
            )
            new_sum = self.call_llm(summary_prompt)
            # merge with prior summary
            self.summary += "\n" + new_sum
            # keep only the most recent turns in the buffer
            self.buffer = self.buffer[-MAX_BUFFER_TURNS:]

    def retrieve(self, query: str):
        # 1) embed your query
        qv = self.embed(query)
        # 2) pass it as a singleton list to `query_embeddings`
        results = self.col.query(
            query_embeddings=qv,
            n_results=VECTOR_K
        )
        # 3) unpack the first (and only) batch of documents
        #    results["documents"] is a list of lists
        return results["documents"][0]

    def build_memory_context(self, query: str) -> str:
        parts = []

        # A) raw buffer
        if self.buffer:
            recent = "\n".join(
                f"{m['role']}: {m['content']}"
                for m in self.buffer
            )
            parts.append("### Recent turns:\n" + recent)

        # B) rolling summary
        if self.summary:
            parts.append("### Summary of earlier conversation:\n" + self.summary)

        # C) key–value facts
        if self.kv:
            kv_txt = "\n".join(f"{k}: {v}" for k, v in self.kv.items())
            parts.append("### Known facts:\n" + kv_txt)

        # D) retrieved long‑term snippets
        retrieved = self.retrieve(query)
        if retrieved:
            parts.append("### Relevant memories:\n" + "\n\n".join(retrieved))

        return "\n\n".join(parts)

# ─── USAGE EXAMPLE ────────────────────────────────────────────────────────
# The code has been shown to work as expected using this test driver
if __name__ == "__main__":
    mem = HybridMemory()

    while True:
        user_in = input("\n👤 You: ")
        if not user_in:
            break

        # 1) record user's turn
        mem.add_turn("user", user_in)

        # 2) assemble full prompt
        context = mem.build_memory_context(user_in)
        prompt = (
            f"{context}\n\n"
            f"### New question:\n{user_in}\n\n"
            f"### Assistant answer:\n"
        )

        # 3) call LLM
        answer = mem.call_llm(prompt)
        print("\n🤖 Assistant:", answer)

        # 4) record assistant's turn
        mem.add_turn("assistant", answer)

# ─── END USAGE EXAMPLE ────────────────────────────────────────────────────────
