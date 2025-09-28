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
import json

import asyncio
import decimal
from datetime import datetime, date
from databricks import sql as databricks_sql
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
# --- NEW: minimal Databricks + Azure Function envs ---
DATABRICKS_SERVER_HOSTNAME = os.getenv("DATABRICKS_SERVER_HOSTNAME")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")


# Only one schema-related env (since your column names are fixed below):
# Set this to the *actual* parcel id column in test.gold.gold_report (e.g., "parcel_id" or "parcelId").
GOLD_REPORT_PARCEL_ID_COLUMN = "value"
GOLD_REPORT_TABLE = "test.gold.gold_report"
COL_REPORT_ID = "report_id"
COL_CREATE_DATE = "createDate"
COL_LAYER_ID = "layerId"
COL_FIELD_ID = "fieldId"

openai_client = OpenAI()

SERVER_INSTRUCTIONS = f"""
{BRAND_NAME} MCP Server provides mid-term memory via an OpenAI Vector Store named "{VS_NAME}".
Use `search` to find semantically relevant documents, `fetch` for full content, and `add_memory`
to insert new text into the store at runtime.
"""


def _db_connect():
    missing = [k for k, v in {
        "DATABRICKS_SERVER_HOSTNAME": DATABRICKS_SERVER_HOSTNAME,
        "DATABRICKS_HTTP_PATH": DATABRICKS_HTTP_PATH,
        "DATABRICKS_TOKEN": DATABRICKS_TOKEN,
        "GOLD_REPORT_PARCELID_COLUMN": GOLD_REPORT_PARCEL_ID_COLUMN,
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

    return databricks_sql.connect(
        server_hostname=DATABRICKS_SERVER_HOSTNAME,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN,
    )


def _db_query_dicts(sql_text: str, params: tuple = ()) -> List[Dict[str, Any]]:
    with _db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_text, params)
            cols = [c[0] for c in cur.description]
            out = []
            for row in cur.fetchall():
                if isinstance(row, dict):
                    out.append(row)
                else:
                    out.append({cols[i]: row[i] for i in range(len(cols))})
            return out


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

    # -----------------------------
    # NEW TOOLS (Databricks-only; parcel_id lives in `value`)
    # -----------------------------

    @mcp.tool()
    async def parcels_latest_unique(limit: Optional[int] = 20) -> Dict[str, Any]:
        """
        Return latest-unique parcels.
        Logic mirrors the working Genie query:
          - partition by `value`, order by `createDate` desc
          - keep rn = 1 (latest row for that value)
        Outputs: [{parcel_id, report_id, createDate}], ordered by createDate desc.
        """
        sql_text = f"""
            WITH ranked AS (
                SELECT
                    value                 AS parcel_id,
                    {COL_REPORT_ID}       AS report_id,
                    {COL_CREATE_DATE}     AS createDate,
                    ROW_NUMBER() OVER (PARTITION BY value ORDER BY {COL_CREATE_DATE} DESC) AS rn
                FROM {GOLD_REPORT_TABLE}
                WHERE value IS NOT NULL
                  AND {COL_CREATE_DATE} IS NOT NULL
            )
            SELECT parcel_id, report_id, createDate
            FROM ranked
            WHERE rn = 1
            ORDER BY createDate DESC
        """
        rows = _db_query_dicts(sql_text)
        if limit:
            rows = rows[: int(limit)]
        return {"count": len(rows), "parcels": rows}

    @mcp.tool()
    async def parcel_latest_full(
            parcel_id: str,
            report_id: str,
            limit: Optional[int] = 200,
            offset: Optional[int] = 0,
            byte_budget_kb: Optional[int] = 512,
    ) -> Dict[str, Any]:
        """
        Single-query, size-capped:
          SELECT * FROM test.gold.gold_report
          WHERE report_id = ? AND value = ?
          ORDER BY layerId, fieldId
        - Returns at most `limit` rows (default 200), starting at `offset`
        - Aborts row accumulation when approx JSON size exceeds `byte_budget_kb` (default 512 KB)
        - Ensures `attributes` is a string; stringifies non-JSON-native types
        """
        # Clamp inputs defensively
        safe_limit = int(limit or 200)
        safe_limit = max(1, min(safe_limit, 5000))
        safe_offset = int(offset or 0)
        safe_offset = max(0, safe_offset)
        budget_bytes = int(byte_budget_kb or 512) * 1024
        budget_bytes = max(32 * 1024, min(budget_bytes, 8 * 1024 * 1024))  # 32KB..8MB

        # Build SQL with literal LIMIT/OFFSET (many drivers don't bind these well)
        sql = f"""
                SELECT *
                FROM {GOLD_REPORT_TABLE}
                WHERE {COL_REPORT_ID} = ? AND value = ?
                ORDER BY {COL_LAYER_ID}, {COL_FIELD_ID}
                LIMIT {safe_limit} OFFSET {safe_offset}
            """

        rows = await asyncio.to_thread(_db_query_dicts, sql, (report_id, parcel_id))

        # Transform rows and enforce byte budget
        out_rows: List[Dict[str, Any]] = []
        approx = 0

        for r in rows:
            rr = dict(r)

            # attributes -> string (prefer JSON string)
            a = rr.get("attributes")
            if not isinstance(a, str):
                try:
                    rr["attributes"] = json.dumps(a, default=str, ensure_ascii=False)
                except Exception:
                    rr["attributes"] = "" if a is None else str(a)

            # Minimal safety for other columns (timestamps/decimals/bytes etc.)
            for k, v in list(rr.items()):
                if k == "attributes":
                    continue
                if isinstance(v, (datetime, date, decimal.Decimal, bytes, bytearray)):
                    rr[k] = str(v)

            # Estimate size if we add this row
            row_json = json.dumps(rr, ensure_ascii=False)
            row_bytes = len(row_json.encode("utf-8"))

            # If nothing added yet and a single row exceeds budget, include it anyway
            # (caller can raise budget if needed)
            if out_rows and approx + row_bytes > budget_bytes:
                break

            out_rows.append(rr)
            approx += row_bytes

        next_offset = safe_offset + len(out_rows)
        # If we hit either the limit or the budget early, expose a next_offset for paging
        more_available = len(rows) == safe_limit

        return {
            "parcel_id": parcel_id,
            "report_id": report_id,
            "offset": safe_offset,
            "limit": safe_limit,
            "byte_budget_kb": budget_bytes // 1024,
            "approx_bytes": approx,
            "rows_returned": len(out_rows),
            "next_offset": next_offset if more_available else None,
            "rows": out_rows,
        }

    return mcp


# --- Expose server object(s) for FastMCP Cloud inspector ---
mcp = create_server()   # primary export expected by the inspector
server = mcp            # optional alias
app = mcp               # optional alias

# Local runner only; Cloud will manage the process
if __name__ == "__main__":
    # Use 8080 if your platform expects that port; otherwise 8000 is fine.
    mcp.run(transport="http", host="0.0.0.0", port=8000)
