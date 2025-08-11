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
        Perform a scoped DuckDuckGo search and return a list of job posting URLs.
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
    "Job Finder MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)


@mcp.tool
async def about() -> dict:
    """
    Return a compact, self-describing manifest so clients can show quick help.
    """
    return {
        "name": mcp.name,
        "description": "A tiny, production-friendly MCP server with a couple of practical tools.",
        "tools": [
            {
                "name": "about",
                "what": "Returns name/description and available tools.",
                "usage": "Call with no args.",
            },
            {
                "name": "validate",
                "what": "Simple health check used by clients to verify connectivity.",
                "usage": "Call with no args; returns your configured MY_NUMBER.",
            },
            {
                "name": "job_finder",
                "what": "Summarize a job post from text or URL, or (best-effort) return search links.",
                "usage": "Provide one of: job_description, job_url, or a search-y user_goal.",
            },
            {
                "name": "make_img_black_and_white",
                "what": "Convert a base64 image to grayscale and return it.",
                "usage": "Pass base64-encoded PNG/JPEG via puch_image_data.",
            },
        ],
    }


# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    """
    Keep behavior Puch expects (string), but normalize whitespace.
    """
    try:
        return str(MY_NUMBER).strip()
    except Exception:
        # still guarantee a string
        return ""


# --- Tool: job_finder (simple + sturdy) ---
JobFinderDescription = RichToolDescription(
    description="Summarize/parse a job from text or a URL, or (best-effort) return search links.",
    use_when="Use for quick job intel: paste description, pass a URL, or ask to 'find' roles.",
    side_effects="Returns concise markdown: title/experience/skills bullets and a short TL;DR or links.",
)

@mcp.tool(description=JobFinderDescription.model_dump_json())
async def job_finder(
    user_goal: Annotated[str, Field(description="The user's goal (can be a description, intent, or freeform query)")],
    job_description: Annotated[str | None, Field(description="Full job description text, if available.")] = None,
    job_url: Annotated[AnyUrl | None, Field(description="A URL to fetch a job description from.")] = None,
    raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
) -> str:
    """
    Simple, dependency-light behavior:
    - If job_description given: extract rough title, experience, and skills via regex/keyword scans + produce TL;DR.
    - If job_url given: fetch + simplify to Markdown (or raw), trim to a safe length, add TL;DR (first lines).
    - Else if user_goal looks like a search: attempt DuckDuckGo links (gracefully degrade if bs4 isn't installed).
    """
    import re
    from textwrap import shorten

    def _extract_title(text: str) -> str:
        # Heuristic: first non-empty line under ~80 chars, otherwise a common title pattern
        for line in (ln.strip() for ln in text.splitlines()):
            if 3 <= len(line) <= 80 and not line.lower().startswith(("responsibilit", "about", "who we", "requirements")):
                return line
        m = re.search(r"(Data Scientist|ML Engineer|Software Engineer|SDE|Backend Engineer|Frontend Engineer|MLOps|Product Manager)", text, re.I)
        return m.group(0) if m else "Role"

    def _extract_experience(text: str) -> str:
        yrs = re.findall(r"(\d+)\s*\+?\s*years?", text, re.I)
        if yrs:
            mx = max(int(x) for x in yrs)
            return f"{mx}+ years mentioned"
        return "Experience not specified"

    def _extract_location(text: str) -> str:
        loc_hits = re.findall(r"(Remote|Hybrid|Bengaluru|Bangalore|Mumbai|Pune|Hyderabad|Chennai|Delhi|Gurgaon|Noida|India|US|USA|UK|Europe|APAC)", text, re.I)
        return ", ".join(sorted(set(x.title() for x in loc_hits))) if loc_hits else "Location not specified"

    def _extract_skills(text: str) -> list[str]:
        kws = [
            "python","java","c++","golang","rust","typescript","javascript","react","node",
            "docker","kubernetes","linux","git","sql","nosql","postgres","mysql","mongodb",
            "aws","gcp","azure","terraform",
            "ml","machine learning","deep learning","pytorch","tensorflow","nlp","cv","llm","rag",
            "spark","hadoop","kafka","airflow"
        ]
        found = []
        low = text.lower()
        for k in kws:
            if k in low:
                found.append(k.upper() if k in {"aws","gcp","sql","nlp","cv","llm","rag"} else k)
        # dedupe, keep original order
        seen = set()
        out = []
        for k in found:
            if k not in seen:
                out.append(k)
                seen.add(k)
        return out[:20]

    def _tldr(md: str, max_chars: int = 500) -> str:
        # Take first few lines as a quick TL;DR
        head = "\n".join(md.strip().splitlines()[:12]).strip()
        return shorten(head, width=max_chars, placeholder="…")

    if job_description:
        text = job_description.strip()
        title = _extract_title(text)
        exp = _extract_experience(text)
        loc = _extract_location(text)
        skills = _extract_skills(text)
        tldr = _tldr(text, 600)
        return (
            f"📝 **Job Description Analysis**\n\n"
            f"**Title (guess):** {title}\n"
            f"**Experience:** {exp}\n"
            f"**Location:** {loc}\n"
            f"**Skills (detected):** {', '.join(skills) if skills else '—'}\n\n"
            f"**User Goal:** {user_goal}\n\n"
            f"**TL;DR:**\n{tldr}"
        )

    if job_url:
        content, preface = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
        # Trim to keep responses lightweight
        max_len = 6000 if raw else 4000
        body = content.strip()[:max_len]
        tldr = _tldr(content, 600) if not raw else "Raw mode requested; TL;DR skipped."
        return (
            f"🔗 **Fetched**: {job_url}\n\n"
            f"{preface}{body}\n\n"
            f"---\n**User Goal:** {user_goal}\n"
            f"**TL;DR:** {tldr}"
        )

    lg = user_goal.lower()
    if any(w in lg for w in ["look for", "find ", "search", "open roles", "hiring"]):
        try:
            links = await Fetch.google_search_links(user_goal)
        except Exception as e:
            links = [f"<error>Search unavailable ({e.__class__.__name__}). Provide a job_url instead.</error>"]
        return "🔍 **Search Results**\n\n" + "\n".join(f"- {link}" for link in links)

    raise McpError(ErrorData(code=INVALID_PARAMS, message="Provide a job_description, a job_url, or phrase your user_goal like a search (e.g., 'find ML engineer jobs in India')."))


# Image inputs and sending images

MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
    description="Convert a base64 image (PNG/JPEG) to grayscale and return it.",
    use_when="User provides base64 image data and wants a quick grayscale version.",
    side_effects="Returns both a short status message and the grayscale image as base64 PNG.",
)

@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")] = None,
) -> list[TextContent | ImageContent]:
    import base64
    import io
    from PIL import Image, ImageOps

    if not puch_image_data:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="puch_image_data is required (base64 string)."))

    try:
        image_bytes = base64.b64decode(puch_image_data, validate=True)
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)  # auto-orient
        bw_image = image.convert("L")

        buf = io.BytesIO()
        bw_image.save(buf, format="PNG")
        bw_bytes = buf.getvalue()
        bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")

        info = TextContent(
            type="text",
            text=f"Converted to grayscale. Original: {image.width}x{image.height}, Mode: {image.mode}. Returned PNG.",
        )
        img = ImageContent(type="image", mimeType="image/png", data=bw_base64)
        return [info, img]
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

# --- Run MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
