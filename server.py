from fastmcp import FastMCP
import os

app = FastMCP("Keyword Search MCP Server")

@app.tool()
def search_keyword_in_file(file_path: str, keyword:str) -> str:

    if not os.path.exists(file_path):
        return f"{file_path} not found"
    
    results = []

    with open(file_path, "r", encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):  # iterate over lines, not characters
            if keyword.lower() in line.lower():
                results.append(f"Line {number} : {line.strip()}")

    if results:
        return "\n".join(results)
    else:
        return f"No occurrences of '{keyword}' found in '{file_path}'"
    
if __name__ == "__main__":
    app.run(transport = "http", host="0.0.0.0", port=3000)