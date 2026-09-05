import os
import base64
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

mcp = FastMCP("OpenAI Image Generator")

@mcp.tool()
def generate_image(prompt: str) -> str:
    """
    Generate an image using OpenAI.

    Use this tool ONLY when the user asks to create or generate an image.
    Do not use it for normal questions, writing, research, calculations,
    or other tasks that do not require an image.
    """
    
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    result = client.images.generate(
        model="gpt-image-2",
        prompt=prompt
    )

    image_data = base64.b64decode(result.data[0].b64_json)

    filename = "/tmp/generated_image.png"

    with open(filename, "wb") as f:
        f.write(image_data)

    return f"Image generated successfully and saved to {filename}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
