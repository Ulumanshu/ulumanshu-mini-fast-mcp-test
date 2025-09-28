"""
Ulumanshu MCP Server — "ulumanshu-mcp-mid-term-memory"

Implements MCP tools backed by an OpenAI Vector Store that is created automatically if absent:
- search(query): semantic search over the vector store
- fetch(id): retrieve full content for a file in the vector store
- add_memory(text, title?, filename?): add new text as a file into the vector store
- multiply(x, y): demo arithmetic
"""

import io
import logging
import os
from typing import Dict, List, Any, Optional

from fastmcp import FastMCP
from starlette.responses import JSONResponse
from starlette.requests import Request
from openai import OpenAI

# ---------- Branding / constants ----------
BRAND_NAME = "Ulumanshu"
VS_NAME = "ulumanshu-mcp-mid-term-memory"   # persistent vector store name

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ulumanshu-mcp")

# ---------- OpenAI client ----------
# Uses OPENAI_API_KEY from env (standard OpenAI SDK behavior)
openai_client = OpenAI()

SERVER_INSTRUCTIONS = f"""
{BRAND_NAME} MCP Server provides mid-term memory via an OpenAI Vector Store named "{VS_NAME}".
Use `search` to find semantically relevant documents, `fetch` for full content, and `add_memory`
to insert new text into the store at runtime.
"""

def ensure_vector_store_named(name: str) -> str:
    """
    Find a vector store by name; create it if missing. Returns vector_store_id.
    Adjust method paths if your SDK uses beta endpoints.
    """
    logger.info(f"[init] Ensuring vector store named: {name!r}")

    # Try to locate by listing (basic single-page list; expand if you have many stores)
    try:
        listing = openai_client.vector_stores.list()
        if hasattr(listing, "data"):
            for vs in listing.data:
                vs_name = getattr(vs, "name", None)
                vs_id = getattr(vs, "id", None)
                if vs_name == name and vs_id:
                    logger.info(f"[init] Reusing existing vector store: {vs_name} ({vs_id})")
                    return vs_id
    except Exception as e:
        logger.warning(f"[init] Unable to list vector stores, will attempt create. Detail: {e}")

    # Not found → create
    created = openai_client.vector_stores.create(name=name)
    vs_id = getattr(created, "id", None)
    if not vs_id:
        raise RuntimeError("Failed to create vector store (no id in response).")
    logger.info(f"[init] Created vector store: {name} ({vs_id})")
    return vs_id


def create_server() -> FastMCP:
    """Create and configure the MCP server with search/fetch/add_memory/multiply tools."""
    mcp = FastMCP(name=f"{BRAND_NAME} MCP Server", instructions=SERVER_INSTRUCTIONS)

    # Ensure the vector store exists and capture its id for all tools
    VECTOR_STORE_ID = ensure_vector_store_named(VS_NAME)

    # --- Health route so ChatGPT's Refresh (GET /mcp/) returns 200 instead of 405 ---
    @mcp.custom_route("/mcp/", methods=["GET"])
    async def mcp_health(_request: Request):
        return JSONResponse({
            "status": "ok",
            "name": mcp.name,
            "vector_store_name": VS_NAME,
            "vector_store_id": VECTOR_STORE_ID,
            "tools": sorted(mcp.tools.keys()),
        })

    @mcp.custom_route("/mcp/", methods=["OPTIONS"])
    async def mcp_options(_request: Request):
        headers = {
            "Allow": "GET, POST, OPTIONS",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        }
        return JSONResponse({"status": "ok"}, headers=headers)

    # -----------------------------
    # Tools
    # -----------------------------

    @mcp.tool()
    async def search(query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Semantic search over the OpenAI Vector Store.

        Args:
            query: Natural-language search string.

        Returns:
            {"results": [ { "id": str, "title": str, "text": str, "url": str | None }, ... ]}
        """
        q = (query or "").strip()
        if not q:
            return {"results": []}

        logger.info(f"[search] VS={VECTOR_STORE_ID} query={q!r}")

        # Mirrors your earlier example; adjust to your SDK if needed.
        response = openai_client.vector_stores.search(
            vector_store_id=VECTOR_STORE_ID,
            query=q
        )

        results: List[Dict[str, Any]] = []
        if hasattr(response, "data") and response.data:
            for i, item in enumerate(response.data):
                item_id = getattr(item, "file_id", f"vs_{i}")
                item_filename = getattr(item, "filename", f"Document {i+1}")

                # Attempt to pull a short text snippet
                content_list = getattr(item, "content", [])
                text_content = ""
                if content_list:
                    first = content_list[0]
                    if hasattr(first, "text"):
                        text_content = first.text
                    elif isinstance(first, dict):
                        text_content = first.get("text", "") or ""

                if not text_content:
                    text_content = "No content available"

                snippet = (text_content[:200] + "...") if len(text_content) > 200 else text_content

                results.append({
                    "id": item_id,
                    "title": item_filename,
                    "text": snippet,
                    "url": f"https://platform.openai.com/storage/files/{item_id}",
                })

        logger.info(f"[search] returned {len(results)} result(s)")
        return {"results": results}

    @mcp.tool()
    async def fetch(id: str) -> Dict[str, Any]:
        """
        Retrieve the full document content from the Vector Store by file ID.

        Args:
            id: Vector Store file ID (e.g., 'file_abc123').

        Returns:
            { "id": str, "title": str, "text": str, "url": str, "metadata": Any | None }
        """
        if not id:
            raise ValueError("Document ID is required")

        logger.info(f"[fetch] VS={VECTOR_STORE_ID} file_id={id}")

        content_response = openai_client.vector_stores.files.content(
            vector_store_id=VECTOR_STORE_ID,
            file_id=id
        )
        file_info = openai_client.vector_stores.files.retrieve(
            vector_store_id=VECTOR_STORE_ID,
            file_id=id
        )

        full_text = ""
        if hasattr(content_response, "data") and content_response.data:
            parts: List[str] = []
            for part in content_response.data:
                if hasattr(part, "text"):
                    parts.append(part.text)
                elif isinstance(part, dict):
                    t = part.get("text")
                    if t:
                        parts.append(t)
            full_text = "\n".join(parts)
        if not full_text:
            full_text = "No content available"

        title = getattr(file_info, "filename", f"Document {id}")
        metadata: Optional[Any] = getattr(file_info, "attributes", None) if hasattr(file_info, "attributes") else None

        return {
            "id": id,
            "title": title,
            "text": full_text,
            "url": f"https://platform.openai.com/storage/files/{id}",
            "metadata": metadata
        }

    @mcp.tool()
    async def add_memory(text: str, title: Optional[str] = None, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Add new text into the mid-term memory vector store as a file.

        Args:
            text: The textual content to store (plain text).
            title: Optional human-readable title (stored as filename if provided).
            filename: Optional filename override; defaults to a safe derived name.

        Returns:
            {
              "status": "ok",
              "file_id": "file_xxx",
              "vector_store_id": "vs_xxx",
              "title": "...",
              "filename": "...",
              "bytes": <int>
            }
        """
        payload = text or ""
        if not payload.strip():
            raise ValueError("`text` must be non-empty.")

        # Prefer filename > title > default
        fname = filename or (f"{title}.txt" if title else "memory.txt")
        file_bytes = payload.encode("utf-8")

        logger.info(f"[add_memory] VS={VECTOR_STORE_ID} uploading {len(file_bytes)} bytes as {fname!r}")

        # 1) Create a File in OpenAI
        # Many SDKs accept file-like objects; adjust if your SDK requires different args.
        up = openai_client.files.create(
            file=(fname, io.BytesIO(file_bytes)),
            purpose="assistants"
        )
        file_id = getattr(up, "id", None)
        if not file_id:
            raise RuntimeError("File upload failed (no id returned).")

        # 2) Attach the file to the Vector Store
        attach = openai_client.vector_stores.files.create(
            vector_store_id=VECTOR_STORE_ID,
            file_id=file_id
        )
        # Optional: check attach result if your SDK returns a status

        return {
            "status": "ok",
            "file_id": file_id,
            "vector_store_id": VECTOR_STORE_ID,
            "title": title or fname,
            "filename": fname,
            "bytes": len(file_bytes),
        }

    @mcp.tool()
    async def multiply(x: int, y: int) -> int:
        """Multiply two integers."""
        return x * y

    return mcp


def main():
    """Start the Ulumanshu MCP Server (HTTP JSON-RPC at /mcp/)."""
    # Gentle heads-up if the API key is missing; health route will still work.
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set — tools using OpenAI will fail until you set it.")

    server = create_server()
    logger.info("Starting Ulumanshu MCP Server on 0.0.0.0:8000 over HTTP")
    logger.info("MCP endpoint: http://<host>:8000/mcp/  (use trailing slash in ChatGPT)")
    server.run(transport="http", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
