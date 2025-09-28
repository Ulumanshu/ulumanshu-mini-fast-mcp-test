from fastmcp import FastMCP
from fastmcp.server.auth import JWTVerifier
from fastmcp.server.auth.providers.jwt import RSAKeyPair

# === Generate RSA Key Pair (for dev use) ===
key_pair = RSAKeyPair.generate()

# === Generate Bearer Token ===
# This token must be used by any client (e.g., OpenAI) to access the tools
server_label = "ulumanshu-hpt-tiny"
token = key_pair.create_token(audience=server_label)

print("\n==== COPY THIS TOKEN TO YOUR GPT CONFIGURATION ====")
print(f"Authorization: Bearer {token}")
print("=================================================\n")

# === JWT Verifier for Access Control ===
auth = JWTVerifier(
    public_key=key_pair.public_key,
    audience=server_label,
)

# === Define your MCP server ===
mcp = FastMCP(name="Ulumanshu Tiny Tool Server", auth=auth)

# === Example Tool ===
@mcp.tool
def multiply(x: int, y: int) -> int:
    """Multiplies two integers."""
    return x * y

# === Run the MCP server ===
if __name__ == "__main__":
    mcp.run(transport="http", port=8000)  # Or change to SSE if needed

