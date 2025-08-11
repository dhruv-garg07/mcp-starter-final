import asyncio
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl

import markdownify
import httpx
import readabilipy

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        """
        (Using DuckDuckGo because Google blocks most programmatic scraping.)
        """
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]

# --- MCP Server Setup ---
mcp = FastMCP(
    "Summarize Text MCP",
    auth=SimpleBearerAuthProvider(TOKEN),
)


@mcp.tool
async def about() -> dict:
    return {"name": mcp.name, "description": "First MCP Server"}


# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool: summarize_text (first N lines + TL;DR, no LLM) ---
SummarizeTextDescription = RichToolDescription(
    description="Summarize plain text or a URL: returns the first N non-empty lines and a TL;DR made from leading content (no LLM).",
    use_when="Use to quickly skim long text or pages without model calls.",
    side_effects="Fetches the URL if provided and simplifies HTML to Markdown before summarizing.",
)

@mcp.tool(description=SummarizeTextDescription.model_dump_json())
async def summarize_text(
    text: Annotated[str | None, Field(description="Plain text to summarize.")] = None,
    url: Annotated[AnyUrl | None, Field(description="URL to fetch and summarize.")] = None,
    lines: Annotated[int, Field(description="Number of leading non-empty lines to include.")] = 10,
    tldr_chars: Annotated[int, Field(description="Max characters for TL;DR.")] = 500,
) -> str:
    """
    Behavior:
    - If `text` given: summarize that.
    - Else if `url` given: fetch via Fetch.fetch_url (HTML→Markdown), then summarize.
    - Output includes: First N lines + TL;DR (trimmed from the start of the content).
    """
    if not text and not url:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Provide either `text` or `url`."))

    # Get content
    if text:
        content = str(text).strip()
        source = "text"
        source_ref = ""
    else:
        fetched, _ = await Fetch.fetch_url(str(url), Fetch.USER_AGENT, force_raw=False)
        content = fetched.strip()
        source = "url"
        source_ref = f"{url}"

    # Normalize params
    if lines is None or lines < 1:
        lines = 1
    if tldr_chars is None or tldr_chars < 1:
        tldr_chars = 1

    # First N non-empty lines
    non_empty = [ln for ln in content.splitlines() if ln.strip()]
    first_part = "\n".join(non_empty[:lines]).strip()

    # TL;DR from leading content (whitespace-collapsed), truncated to tldr_chars
    collapsed = " ".join(content.split())
    if len(collapsed) <= tldr_chars:
        tldr = collapsed
    else:
        cut = collapsed[:tldr_chars].rstrip()
        # try to cut at the last space for nicer ending
        sp = cut.rfind(" ")
        tldr = (cut[:sp] if sp != -1 and sp > tldr_chars // 2 else cut) + "…"

    return (
        f"🧾 **Summarize Text**\n\n"
        f"- **Mode:** {source}{(' (' + source_ref + ')') if source_ref else ''}\n"
        f"- **First {lines} lines:**\n"
        f"---\n{first_part}\n---\n\n"
        f"**TL;DR**\n{tldr}"
    )

# --- Run MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
