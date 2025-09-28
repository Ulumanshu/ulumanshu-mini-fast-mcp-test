from typing import List, Dict, Optional
from fastmcp import FastMCP
from fastmcp.server.auth import JWTVerifier
from fastmcp.server.auth.providers.jwt import RSAKeyPair

# === Auth setup ===
key_pair = RSAKeyPair.generate()
SERVER_LABEL = "ulumanshu-hpt-tiny"
token = key_pair.create_token(audience=SERVER_LABEL)
print("\n==== COPY THIS TOKEN TO YOUR GPT CONFIGURATION ====")
print(f"Authorization: Bearer {token}")
print("=================================================\n")

auth = JWTVerifier(public_key=key_pair.public_key, audience=SERVER_LABEL)
mcp = FastMCP(name="Ulumanshu Tiny Tool Server", auth=auth)

# === Mock corpus ===
MOCK_DATA = {
    "python": {
        "title": "Python Docs",
        "url": "https://docs.python.org/",
        "snippet": "Official Python documentation covering syntax, stdlib, and tutorials.",
        "content": "Welcome to Python official documentation..."
    },
    "fastmcp": {
        "title": "FastMCP Guide",
        "url": "https://example.local/fastmcp",
        "snippet": "Learn how to build MCP servers with FastMCP.",
        "content": "FastMCP is a library for quickly building MCP servers..."
    },
    "chatgpt": {
        "title": "ChatGPT Overview",
        "url": "https://openai.com/chatgpt",
        "snippet": "Conversational AI model developed by OpenAI.",
        "content": "ChatGPT is an advanced language model..."
    },
}

# --- Your original example tool ---
@mcp.tool
def multiply(x: int, y: int) -> int:
    """Multiplies two integers."""
    return x * y

# --- Mock 'search' tool ---
@mcp.tool
def search(query: str, limit: int = 5) -> List[Dict]:
    """
    Mock search: looks for the query keyword in the MOCK_DATA keys.
    Returns list of {title, url, snippet}.
    """
    results = []
    q = query.lower()
    for key, row in MOCK_DATA.items():
        if q in key:
            results.append({
                "title": row["title"],
                "url": row["url"],
                "snippet": row["snippet"],
            })
            if len(results) >= limit:
                break
    return results or [{
        "title": "No results",
        "url": None,
        "snippet": f"No mock entries matched query: {query}"
    }]

# --- Mock 'fetch' tool ---
@mcp.tool
def fetch(url: str) -> Dict[str, Optional[str]]:
    """
    Mock fetch: returns 'content' if the url matches one in MOCK_DATA.
    Otherwise returns status='not-found'.
    """
    for row in MOCK_DATA.values():
        if row["url"] == url:
            return {"url": url, "status": "200", "content": row["content"]}
    return {"url": url, "status": "not-found", "content": None}

if __name__ == "__main__":
    # Start HTTP transport → endpoint available at /mcp/ (note trailing slash!)
    mcp.run(transport="http", port=8000)
