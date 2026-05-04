"""
MCP Server — Model Context Protocol tool dispatcher for Maternal Instinct AI.

This server acts as the central hub for Layer 2 specialist agents to call
structured tools. It follows the workflow diagram:

    Layer 2 Agents  ←→  MCP Server  ←→  MCP Tools
                         ↑
                    (schedule_tools, notes_tools, material_tools)

The server exposes:
  - call_tool(tool_name, **kwargs)   : Execute any registered tool
  - list_tools()                     : List all available tools with metadata
  - get_tool_metadata(tool_name)     : Get parameter schema for a specific tool
  - get_workload_context()           : Convenience — workload + upcoming deadlines
  - get_full_context()               : Full context bundle for agent injection
"""
import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger("backend.mcp_server.server")


class MCPServer:
    """
    Central MCP tool dispatcher.

    Aggregates all tool registries (schedule, notes, materials) into
    a unified interface for Layer 2 agents to call.

    Usage:
        mcp = MCPServer()
        result = await mcp.call_tool("get_schedule")
        result = await mcp.call_tool("search_materials", query="teknik belajar")
        result = await mcp.call_tool("add_task", task="Essay Fisika", deadline="2026-03-10")
    """

    def __init__(self):
        self._registry: dict[str, dict] = {}
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load all tool registries to avoid circular imports at startup."""
        if self._loaded:
            return
        try:
            from backend.mcp_server.schedule_tools import SCHEDULE_TOOLS
            from backend.mcp_server.notes_tools import NOTES_TOOLS
            from backend.mcp_server.material_tools import MATERIAL_TOOLS

            self._registry.update(SCHEDULE_TOOLS)
            self._registry.update(NOTES_TOOLS)
            self._registry.update(MATERIAL_TOOLS)

            self._loaded = True
            logger.info(f"MCP Server loaded {len(self._registry)} tools.")
        except Exception as e:
            logger.error(f"MCP Server failed to load tool registries: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    async def call_tool(self, tool_name: str, **kwargs) -> dict:
        """
        Execute a registered tool by name.

        Args:
            tool_name: Name of the tool (e.g. 'get_schedule', 'search_materials').
            **kwargs:  Tool-specific parameters.

        Returns:
            Tool result dict. Always includes "success" key.
            On unknown tool: { "success": False, "error": "Unknown tool: ..." }
        """
        self._ensure_loaded()

        if tool_name not in self._registry:
            available = list(self._registry.keys())
            return {
                "success": False,
                "error": f"Unknown tool: '{tool_name}'. Available: {available}",
            }

        tool_entry = self._registry[tool_name]
        fn = tool_entry["fn"]

        try:
            # Tools may be sync or async
            if asyncio.iscoroutinefunction(fn):
                result = await fn(**kwargs)
            else:
                result = fn(**kwargs)
            return result
        except TypeError as e:
            logger.error(f"Tool '{tool_name}' called with wrong parameters: {e}")
            return {
                "success": False,
                "error": f"Parameter error for tool '{tool_name}': {e}",
            }
        except Exception as e:
            logger.error(f"Tool '{tool_name}' raised an exception: {e}")
            return {
                "success": False,
                "error": f"Tool '{tool_name}' failed: {e}",
            }

    def list_tools(self) -> list[dict]:
        """
        Return a list of all available tools with their descriptions and parameter schemas.

        Returns:
            [ { "name": str, "description": str, "parameters": dict, "destructive": bool } ]
        """
        self._ensure_loaded()
        return [
            {
                "name": name,
                "description": entry.get("description", ""),
                "parameters": entry.get("parameters", {}),
                "destructive": entry.get("destructive", False),
            }
            for name, entry in self._registry.items()
        ]

    def get_tool_metadata(self, tool_name: str) -> Optional[dict]:
        """
        Get metadata for a specific tool.

        Returns:
            { "name": str, "description": str, "parameters": dict, "destructive": bool,
              "confirm_message": str | None } or None
        """
        self._ensure_loaded()
        entry = self._registry.get(tool_name)
        if not entry:
            return None
        return {
            "name": tool_name,
            "description": entry.get("description", ""),
            "parameters": entry.get("parameters", {}),
            "destructive": entry.get("destructive", False),
            "confirm_message": entry.get("confirm_message"),
        }

    def is_destructive(self, tool_name: str) -> bool:
        """
        Return True if the named tool is marked as destructive (requires confirmation).

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if destructive, False if not found or not destructive.
        """
        self._ensure_loaded()
        entry = self._registry.get(tool_name, {})
        return bool(entry.get("destructive", False))

    def get_confirm_message(self, tool_name: str) -> str:
        """
        Return the human-readable confirmation message for a destructive tool.

        Args:
            tool_name: Name of the tool.

        Returns:
            Confirmation message string, or a generic fallback.
        """
        self._ensure_loaded()
        entry = self._registry.get(tool_name, {})
        return entry.get(
            "confirm_message",
            f"Ara akan menjalankan tindakan '{tool_name}'. Apakah kamu yakin?"
        )

    # ── Context Bundles for Agent Injection ───────────────────────────────────

    async def get_workload_context(self) -> dict:
        """
        Convenience method: fetch workload + upcoming deadlines in one call.

        Used by Layer 2 agents and Layer 3 policy checker to make decisions.

        Returns:
            {
                "workload": { total_hours, overloaded, ... },
                "upcoming": { tasks: [...], count: int },
            }
        """
        workload, upcoming = await asyncio.gather(
            self.call_tool("get_daily_workload"),
            self.call_tool("get_upcoming_deadlines", days=3),
        )
        return {"workload": workload, "upcoming": upcoming}

    async def get_full_context(self, user_query: str = "") -> dict:
        """
        Full context bundle: schedule + workload + notes + relevant materials.

        Designed to be injected into Layer 2 agent prompts for rich context.

        Args:
            user_query: The user's message — used for semantic material search.

        Returns:
            {
                "schedule": { tasks: [...], count: int },
                "workload": { total_hours, overloaded, ... },
                "upcoming_deadlines": { tasks: [...] },
                "relevant_materials": [ { rank, text, source, relevance_score } ],
                "notes_count": int,
            }
        """
        tasks = [
            self.call_tool("get_schedule"),
            self.call_tool("get_daily_workload"),
            self.call_tool("get_upcoming_deadlines", days=3),
            self.call_tool("get_notes", limit=5),
        ]

        if user_query:
            tasks.append(self.call_tool("search_materials", query=user_query, top_k=3))
        else:
            async def _empty():
                return {"success": True, "results": []}
            tasks.append(_empty())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        schedule_result = results[0] if not isinstance(results[0], Exception) else {}
        workload_result = results[1] if not isinstance(results[1], Exception) else {}
        upcoming_result = results[2] if not isinstance(results[2], Exception) else {}
        notes_result = results[3] if not isinstance(results[3], Exception) else {}
        materials_result = results[4] if not isinstance(results[4], Exception) else {}

        return {
            "schedule": schedule_result,
            "workload": workload_result,
            "upcoming_deadlines": upcoming_result,
            "notes_count": notes_result.get("count", 0),
            "relevant_materials": materials_result.get("results", []),
        }

    def format_schedule_for_prompt(self, schedule_result: dict) -> str:
        """
        Format a schedule tool result into a readable string for LLM injection.

        Args:
            schedule_result: Result from get_schedule() tool call.

        Returns:
            Formatted markdown string, or empty string if no tasks.
        """
        tasks = schedule_result.get("tasks", [])
        if not tasks:
            return ""

        lines = ["📋 **DATA JADWAL USER SAAT INI:**"]
        for t in tasks:
            lines.append(
                f"  - {t['task']} "
                f"(Deadline: {t['deadline']} | "
                f"Estimasi: {t['est_hours']} jam | "
                f"Prioritas: {t['priority']} | "
                f"Status: {t.get('status', 'pending')})"
            )
        return "\n".join(lines)

    def format_workload_for_prompt(self, workload_result: dict) -> str:
        """
        Format a workload tool result into a readable string for LLM injection.

        Returns:
            One-line workload summary string.
        """
        if not workload_result.get("success"):
            return ""

        total = workload_result.get("total_hours", 0)
        count = workload_result.get("task_count", 0)
        overloaded = workload_result.get("overloaded", False)
        overload_by = workload_result.get("overload_by_hours", 0)

        status = (
            f"⚠️ OVERLOADED (+{overload_by:.1f} jam melebihi batas)"
            if overloaded
            else "✅ Dalam batas aman"
        )

        return (
            f"📊 **BEBAN KERJA:** {total:.1f} jam dari {count} tugas aktif — {status}"
        )


# ─── Module-level singleton ───────────────────────────────────────────────────

_mcp_server_instance: Optional[MCPServer] = None


def get_mcp_server() -> MCPServer:
    """Get or create the module-level MCP server singleton."""
    global _mcp_server_instance
    if _mcp_server_instance is None:
        _mcp_server_instance = MCPServer()
    return _mcp_server_instance
