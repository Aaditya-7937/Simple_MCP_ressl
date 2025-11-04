#!/usr/bin/env python3
"""
Advanced File Search MCP Server
Provides comprehensive file search capabilities with multiple search modes and file operations.
"""

import asyncio
import aiofiles
import os
import re
import fnmatch
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

import mcp
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    EXACT = "exact"
    CASE_INSENSITIVE = "case_insensitive"
    REGEX = "regex"
    WORD_BOUNDARY = "word_boundary"


class FileType(str, Enum):
    TEXT = "text"
    CODE = "code"
    LOG = "log"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    ALL = "all"


class SearchRequest(BaseModel):
    file_path: str = Field(..., description="Path to the file to search")
    keyword: str = Field(..., description="Keyword or pattern to search for")
    mode: SearchMode = Field(default=SearchMode.CASE_INSENSITIVE, description="Search mode")
    max_results: int = Field(default=100, ge=1, le=1000, description="Maximum number of results to return")
    context_lines: int = Field(default=0, ge=0, le=5, description="Number of context lines around matches")


class BatchSearchRequest(BaseModel):
    directory: str = Field(..., description="Directory to search in")
    keyword: str = Field(..., description="Keyword or pattern to search for")
    file_patterns: List[str] = Field(default=["*"], description="File patterns to include (e.g., ['*.py', '*.txt'])")
    exclude_dirs: List[str] = Field(default=[".git", "__pycache__", "node_modules"], description="Directories to exclude")
    mode: SearchMode = Field(default=SearchMode.CASE_INSENSITIVE, description="Search mode")
    max_files: int = Field(default=50, ge=1, le=200, description="Maximum number of files to search")
    max_results_per_file: int = Field(default=10, ge=1, le=50, description="Maximum results per file")


class FileInfoRequest(BaseModel):
    file_path: str = Field(..., description="Path to the file")


@dataclass
class SearchResult:
    line_number: int
    content: str
    start_pos: int
    end_pos: int
    context_before: List[str] = None
    context_after: List[str] = None


class AdvancedFileSearchServer:
    """Advanced MCP server for file search operations."""
    
    def __init__(self):
        self.server = Server("advanced-file-search-server")
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Setup all MCP tool handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools():
            return [
                types.Tool(
                    name="search_file",
                    description="Search for keywords in a file with advanced options",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to the file"},
                            "keyword": {"type": "string", "description": "Text to search for"},
                            "mode": {
                                "type": "string", 
                                "enum": [mode.value for mode in SearchMode],
                                "description": "Search mode: exact, case_insensitive, regex, or word_boundary"
                            },
                            "max_results": {
                                "type": "integer", 
                                "minimum": 1, 
                                "maximum": 1000,
                                "description": "Maximum results to return"
                            },
                            "context_lines": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 5,
                                "description": "Context lines around matches"
                            }
                        },
                        "required": ["file_path", "keyword"]
                    }
                ),
                types.Tool(
                    name="batch_search",
                    description="Search for keywords across multiple files in a directory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "directory": {"type": "string", "description": "Directory path to search in"},
                            "keyword": {"type": "string", "description": "Text to search for"},
                            "file_patterns": {
                                "type": "array", 
                                "items": {"type": "string"},
                                "description": "File patterns like ['*.py', '*.txt']"
                            },
                            "exclude_dirs": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Directories to exclude"
                            },
                            "mode": {
                                "type": "string", 
                                "enum": [mode.value for mode in SearchMode],
                                "description": "Search mode"
                            },
                            "max_files": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 200,
                                "description": "Maximum files to search"
                            },
                            "max_results_per_file": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 50,
                                "description": "Maximum results per file"
                            }
                        },
                        "required": ["directory", "keyword"]
                    }
                ),
                types.Tool(
                    name="get_file_info",
                    description="Get information about a file (size, lines, encoding detection)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to the file"}
                        },
                        "required": ["file_path"]
                    }
                ),
                types.Tool(
                    name="find_files",
                    description="Find files by name pattern in a directory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "directory": {"type": "string", "description": "Directory to search in"},
                            "pattern": {"type": "string", "description": "Filename pattern (supports wildcards)"},
                            "recursive": {"type": "boolean", "description": "Search recursively"},
                            "file_type": {
                                "type": "string",
                                "enum": [ft.value for ft in FileType],
                                "description": "Filter by file type"
                            }
                        },
                        "required": ["directory", "pattern"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict):
            try:
                if name == "search_file":
                    request = SearchRequest(**arguments)
                    return await self._handle_search_file(request)
                elif name == "batch_search":
                    request = BatchSearchRequest(**arguments)
                    return await self._handle_batch_search(request)
                elif name == "get_file_info":
                    request = FileInfoRequest(**arguments)
                    return await self._handle_get_file_info(request)
                elif name == "find_files":
                    return await self._handle_find_files(**arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                return [types.TextContent(
                    type="text", 
                    text=f"Error in {name}: {str(e)}"
                )]
    
    async def _handle_search_file(self, request: SearchRequest) -> List[types.TextContent]:
        """Handle single file search request."""
        try:
            results = await self._search_in_file(
                request.file_path, 
                request.keyword, 
                request.mode,
                request.max_results,
                request.context_lines
            )
            
            if not results:
                return [types.TextContent(
                    type="text", 
                    text=f"No matches found for '{request.keyword}' in {request.file_path}"
                )]
            
            result_text = self._format_single_file_results(results, request.file_path, request.keyword)
            return [types.TextContent(type="text", text=result_text)]
            
        except FileNotFoundError:
            return [types.TextContent(type="text", text=f"File not found: {request.file_path}")]
        except PermissionError:
            return [types.TextContent(type="text", text=f"Permission denied: {request.file_path}")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error searching file: {str(e)}")]
    
    async def _handle_batch_search(self, request: BatchSearchRequest) -> List[types.TextContent]:
        """Handle batch search across multiple files."""
        try:
            all_results = await self._batch_search_files(request)
            
            if not all_results:
                return [types.TextContent(
                    type="text", 
                    text=f"No matches found for '{request.keyword}' in {request.directory}"
                )]
            
            result_text = self._format_batch_results(all_results, request.keyword)
            return [types.TextContent(type="text", text=result_text)]
            
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error in batch search: {str(e)}")]
    
    async def _handle_get_file_info(self, request: FileInfoRequest) -> List[types.TextContent]:
        """Get detailed information about a file."""
        try:
            info = await self._get_file_info(request.file_path)
            return [types.TextContent(type="text", text=info)]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error getting file info: {str(e)}")]
    
    async def _handle_find_files(self, directory: str, pattern: str, recursive: bool = True, 
                               file_type: FileType = FileType.ALL) -> List[types.TextContent]:
        """Find files by pattern."""
        try:
            files = await self._find_files_by_pattern(directory, pattern, recursive, file_type)
            
            if not files:
                return [types.TextContent(
                    type="text", 
                    text=f"No files found matching '{pattern}' in {directory}"
                )]
            
            result_text = f"Found {len(files)} files matching '{pattern}':\n\n"
            for file_path in sorted(files):
                result_text += f"• {file_path}\n"
                
            return [types.TextContent(type="text", text=result_text)]
            
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error finding files: {str(e)}")]
    
    async def _search_in_file(self, file_path: str, keyword: str, mode: SearchMode, 
                            max_results: int, context_lines: int) -> List[SearchResult]:
        """Search for keyword in a single file."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = await file.read()
            
            lines = content.split('\n')
            results = []
            
            for line_num, line in enumerate(lines, 1):
                if len(results) >= max_results:
                    break
                    
                matches = self._find_matches_in_line(line, keyword, mode)
                for start, end in matches:
                    result = SearchResult(
                        line_number=line_num,
                        content=line,
                        start_pos=start,
                        end_pos=end
                    )
                    
                    # Add context if requested
                    if context_lines > 0:
                        result.context_before = lines[max(0, line_num - context_lines - 1):line_num - 1]
                        result.context_after = lines[line_num:min(len(lines), line_num + context_lines)]
                    
                    results.append(result)
            
            return results
            
        except Exception:
            # Fallback to different encoding
            async with aiofiles.open(file_path, 'r', encoding='latin-1', errors='ignore') as file:
                content = await file.read()
            
            lines = content.split('\n')
            results = []
            
            for line_num, line in enumerate(lines, 1):
                if len(results) >= max_results:
                    break
                    
                matches = self._find_matches_in_line(line, keyword, mode)
                for start, end in matches:
                    results.append(SearchResult(
                        line_number=line_num,
                        content=line,
                        start_pos=start,
                        end_pos=end
                    ))
            
            return results
    
    def _find_matches_in_line(self, line: str, keyword: str, mode: SearchMode) -> List[tuple[int, int]]:
        """Find all matches in a single line based on search mode."""
        matches = []
        
        try:
            if mode == SearchMode.EXACT:
                start = 0
                while True:
                    pos = line.find(keyword, start)
                    if pos == -1:
                        break
                    matches.append((pos, pos + len(keyword)))
                    start = pos + 1
                    
            elif mode == SearchMode.CASE_INSENSITIVE:
                lower_line = line.lower()
                lower_keyword = keyword.lower()
                start = 0
                while True:
                    pos = lower_line.find(lower_keyword, start)
                    if pos == -1:
                        break
                    matches.append((pos, pos + len(keyword)))
                    start = pos + 1
                    
            elif mode == SearchMode.REGEX:
                try:
                    for match in re.finditer(keyword, line):
                        matches.append((match.start(), match.end()))
                except re.error:
                    # Fallback to literal search if regex is invalid
                    return self._find_matches_in_line(line, re.escape(keyword), SearchMode.EXACT)
                    
            elif mode == SearchMode.WORD_BOUNDARY:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                for match in re.finditer(pattern, line, re.IGNORECASE):
                    matches.append((match.start(), match.end()))
                    
        except Exception:
            pass
            
        return matches
    
    async def _batch_search_files(self, request: BatchSearchRequest) -> Dict[str, List[SearchResult]]:
        """Search across multiple files in a directory."""
        all_results = {}
        searched_files = 0
        
        for root, dirs, files in os.walk(request.directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in request.exclude_dirs]
            
            for file in files:
                if searched_files >= request.max_files:
                    break
                    
                # Check file patterns
                if not any(fnmatch.fnmatch(file, pattern) for pattern in request.file_patterns):
                    continue
                
                file_path = os.path.join(root, file)
                try:
                    results = await self._search_in_file(
                        file_path, 
                        request.keyword, 
                        request.mode,
                        request.max_results_per_file,
                        0  # No context for batch search
                    )
                    
                    if results:
                        all_results[file_path] = results
                    
                    searched_files += 1
                    
                except Exception:
                    continue
        
        return all_results
    
    async def _get_file_info(self, file_path: str) -> str:
        """Get comprehensive file information."""
        try:
            stat = os.stat(file_path)
            
            # Get file size in human-readable format
            size = stat.st_size
            if size < 1024:
                size_str = f"{size} bytes"
            elif size < 1024 * 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size/(1024*1024):.1f} MB"
            
            # Count lines
            line_count = 0
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                async for _ in file:
                    line_count += 1
            
            # Get file type based on extension
            ext = Path(file_path).suffix.lower()
            file_type = {
                '.py': 'Python', '.js': 'JavaScript', '.html': 'HTML', '.css': 'CSS',
                '.json': 'JSON', '.xml': 'XML', '.txt': 'Text', '.md': 'Markdown',
                '.csv': 'CSV', '.log': 'Log', '.java': 'Java', '.cpp': 'C++', '.c': 'C',
                '.rs': 'Rust', '.go': 'Go', '.php': 'PHP', '.rb': 'Ruby'
            }.get(ext, 'Unknown')
            
            info = f"""File Information:
• Path: {file_path}
• Size: {size_str}
• Lines: {line_count:,}
• Type: {file_type}
• Extension: {ext or 'None'}
• Created: {stat.st_ctime:.0f}
• Modified: {stat.st_mtime:.0f}
• Permissions: {oct(stat.st_mode)[-3:]}"""
            
            return info
            
        except Exception as e:
            return f"Error getting file info: {str(e)}"
    
    async def _find_files_by_pattern(self, directory: str, pattern: str, recursive: bool, 
                                   file_type: FileType) -> List[str]:
        """Find files matching pattern and type."""
        found_files = []
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if fnmatch.fnmatch(file, pattern):
                        file_path = os.path.join(root, file)
                        if self._matches_file_type(file_path, file_type):
                            found_files.append(file_path)
        else:
            try:
                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)
                    if os.path.isfile(item_path) and fnmatch.fnmatch(item, pattern):
                        if self._matches_file_type(item_path, file_type):
                            found_files.append(item_path)
            except OSError:
                pass
        
        return found_files
    
    def _matches_file_type(self, file_path: str, file_type: FileType) -> bool:
        """Check if file matches the specified type."""
        if file_type == FileType.ALL:
            return True
        
        ext = Path(file_path).suffix.lower()
        type_mapping = {
            FileType.TEXT: ['.txt', '.md', '.rst', '.log'],
            FileType.CODE: ['.py', '.js', '.java', '.cpp', '.c', '.rs', '.go', '.php', '.rb', '.html', '.css', '.xml'],
            FileType.LOG: ['.log', '.txt'],
            FileType.CSV: ['.csv'],
            FileType.JSON: ['.json'],
            FileType.XML: ['.xml']
        }
        
        return ext in type_mapping.get(file_type, [])
    
    def _format_single_file_results(self, results: List[SearchResult], file_path: str, keyword: str) -> str:
        """Format single file search results."""
        output = [f"Search results for '{keyword}' in {file_path}:"]
        output.append(f"Found {len(results)} matches\n")
        
        for i, result in enumerate(results, 1):
            output.append(f"Match {i} (Line {result.line_number}):")
            
            # Add context before if available
            if result.context_before:
                for ctx_line in result.context_before:
                    output.append(f"  {ctx_line}")
            
            # Highlight the match in the line
            line = result.content
            if result.start_pos < len(line):
                highlighted = (line[:result.start_pos] + 
                             ">>>" + line[result.start_pos:result.end_pos] + "<<<" + 
                             line[result.end_pos:])
                output.append(f"  {highlighted}")
            else:
                output.append(f"  {line}")
            
            # Add context after if available
            if result.context_after:
                for ctx_line in result.context_after:
                    output.append(f"  {ctx_line}")
            
            output.append("")  # Empty line between matches
        
        return "\n".join(output)
    
    def _format_batch_results(self, all_results: Dict[str, List[SearchResult]], keyword: str) -> str:
        """Format batch search results."""
        total_matches = sum(len(results) for results in all_results.values())
        output = [f"Batch search results for '{keyword}':"]
        output.append(f"Found {total_matches} matches across {len(all_results)} files\n")
        
        for file_path, results in sorted(all_results.items()):
            output.append(f"📁 {file_path} ({len(results)} matches):")
            for result in results[:5]:  # Show first 5 matches per file
                line_preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
                output.append(f"  Line {result.line_number}: {line_preview}")
            if len(results) > 5:
                output.append(f"  ... and {len(results) - 5} more matches")
            output.append("")
        
        return "\n".join(output)


async def main():
    """Main entry point for the MCP server."""
    server = AdvancedFileSearchServer()
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="advanced-file-search-server",
                server_version="2.0.0",
                capabilities=server.server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                )
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())