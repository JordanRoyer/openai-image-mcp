import os
import base64

from mcp.server.fastmcp import FastMCP, Image
from openai import OpenAI

mcp = FastMCP("OpenAI Image Generator")


@mcp.tool()
def generate_image(prompt: str) -> Image:
    """
    Generate an image using OpenAI GPT-Image-2.

    Use this tool ONLY when the user asks to create or generate an image.
    Do not use it for normal questions, writing, research, calculations,
    or other tasks that do not require an image.
    """

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    result = client.images.generate(
        model="gpt-image-2",
        prompt=prompt,
    )

    image_data = base64.b64decode(result.data[0].b64_json)

    return Image(
        data=image_data,
        format="png",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=port,
    )
