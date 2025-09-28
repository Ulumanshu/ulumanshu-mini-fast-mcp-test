# ulumanshu_tiny_mcp.py
from typing import List, Dict, Optional
from fastmcp import FastMCP
from fastmcp.server.auth import JWTVerifier
from fastmcp.server.auth.providers.jwt import RSAKeyPair

# === Auth (dev: ephemeral RSA; prod: load your own keys/secrets) ===
SERVER_LABEL = "ulumanshu-hpt-tiny"
_key_pair = RSAKeyPair.generate()
TOKEN = _key_pair.create_token(audience=SERVER_LABEL)

print("\n==== COPY THIS TOKEN TO YOUR GPT CONFIGURATION ====")
print(f"Authorization: Bearer {TOKEN}")
print("=================================================\n")

auth = JWTVerifier(public_key=_key_pair.public_key, audience=SERVER_LABEL)
mcp = FastMCP(name="Ulumanshu Tiny Tool Server", auth=auth)

# === In-memory mock corpus (edit as you like) ===
MOCK_DATA: Dict[str, Dict[str, str]] = {
    "python": {
        "title": "Python Docs",
        "url": "https://docs.python.org/",
        "snippet": "Official Python documentation covering syntax, stdlib, and tutorials.",
        "content": "Welcome to the official Python documentation..."
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

# === Example tool preserved ===
@mcp.tool
def multiply(x: int, y: int) -> int:
    """Multiplies two integers."""
    return x * y

# === Mock 'search' (keyword match on MOCK_DATA keys) ===
@mcp.tool
def search(query: str, limit: int = 5) -> List[Dict]:
    """
    Mock search: returns entries whose key contains the query substring.
    Output items: {title, url, snippet}.
    """
    q = (query or "").lower().strip()
    results: List[Dict] = []
    if not q:
        return [{"title": "No query", "url": None, "snippet": "Provide a search term."}]
    for key, row in MOCK_DATA.items():
        if q in key.lower():
            results.append({
                "title": row.get("title"),
                "url": row.get("url"),
                "snippet": row.get("snippet"),
            })
            if len(results) >= max(1, min(limit, 25)):
                break
    return results or [{
        "title": "No results",
        "url": None,
        "snippet": f"No mock entries matched query: {query}"
    }]

# === Mock 'fetch' (exact URL match to an entry in MOCK_DATA) ===
@mcp.tool
def fetch(url: str) -> Dict[str, Optional[str]]:
    """
    Mock fetch: returns {url, status, content} from MOCK_DATA if URL matches.
    Otherwise returns status='not-found'.
    """
    for row in MOCK_DATA.values():
        if row.get("url") == url:
            return {"url": url, "status": "200", "content": row.get("content")}
    return {"url": url, "status": "not-found", "content": None}

# === Convenience: manage the mock corpus at runtime ===
@mcp.tool
def add_mock(key: str, title: str, url: str, snippet: str, content: str) -> Dict[str, str]:
    """
    Add or replace a MOCK_DATA entry.
    """
    MOCK_DATA[key] = {"title": title, "url": url, "snippet": snippet, "content": content}
    return {"status": "ok", "message": f"Added/updated mock: {key}"}

@mcp.tool
def delete_mock(key: str) -> Dict[str, str]:
    """
    Delete a MOCK_DATA entry by key.
    """
    if key in MOCK_DATA:
        del MOCK_DATA[key]
        return {"status": "ok", "message": f"Deleted mock: {key}"}
    return {"status": "not-found", "message": f"No mock with key: {key}"}

@mcp.tool
def list_mocks() -> List[Dict[str, str]]:
    """
    List current mock entries (key, title, url).
    """
    return [{"key": k, "title": v.get("title"), "url": v.get("url")} for k, v in MOCK_DATA.items()]

# === Simple health check ===
@mcp.tool
def ping() -> str:
    """Basic liveness check."""
    return "pong"

if __name__ == "__main__":
    # HTTP transport → MCP endpoint is http://<host>:8000/mcp/  (note the trailing slash!)
    mcp.run(transport="http", port=8000)
