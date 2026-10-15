"""Wraps tau-bench's MockAirlineDomainEnv as an PROBE target agent.

The PROBE auditor completely replaces tau-bench's LLM user simulator.
This module provides:

  TauBenchTarget   — target agent backed by tau-bench's airline env;
                     compatible with the TargetAgent.generate() interface.

  TauBenchEpisodeRunner — EpisodeRunner subclass that resets the tau-bench
                          env per episode and synchronises the auditor's
                          scenario with the tau-bench task instruction.

Install tau-bench before importing:
    pip install -e /tmp/tau-bench
    # or:  sys.path.insert(0, "/tmp/tau-bench")
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from ..types import Episode, Role, ToolCall, Turn
from ..trajectory.episode_runner import EpisodeRunner, AuditorProtocol


# ── Null user simulator ────────────────────────────────────────────────

class _NullUserSim:
    """Drop-in replacement for tau-bench's LLMUserSimulationEnv.

    The PROBE auditor generates all user turns, so the user sim must never
    make LLM calls.  reset() returns the task instruction verbatim;
    step() returns the stop marker (never called in normal PROBE flow).
    """

    def reset(self, instruction: str | None = None) -> str:
        return instruction or ""

    def step(self, content: str) -> str:
        return "###STOP###"

    def get_total_cost(self) -> float:
        return 0.0


# ── TauBenchTarget ─────────────────────────────────────────────────────

class TauBenchTarget:
    """Target agent backed by tau-bench's airline environment.

    The target LLM (any InferenceBackend) is given tau-bench's policy wiki
    as its system prompt and has access to all 14 tau-bench airline tools,
    which are executed through tau-bench's stateful in-memory database.

    Usage:
        target = TauBenchTarget(client=vllm_client)
        instruction = target.reset(task_index=0)   # returns task instruction
        turn = target.generate(history)             # called by EpisodeRunner
    """

    def __init__(
        self,
        client: Any,  # InferenceBackend protocol
        task_split: str = "test",
        max_tool_rounds: int = 5,
        tau_bench_path: str | None = None,
    ):
        self.client = client
        self.max_tool_rounds = max_tool_rounds
        self._done: bool = False
        self._current_task: Any = None
        self._task_cursor: int = 0

        self._env = self._make_env(task_split, tau_bench_path)
        self._wiki: str = self._load_wiki(tau_bench_path)
        self._tool_schemas: list[dict] = self._build_tool_schemas(tau_bench_path)

    # ── Public interface ───────────────────────────────────────────────

    @property
    def done(self) -> bool:
        """True once transfer_to_human_agents has been called this episode."""
        return self._done

    def reset(self, task_index: int | None = None) -> str:
        """Reset the tau-bench env for a new task.

        Returns the task instruction, which the caller should use as the
        auditor's scenario description so both sides of the conversation
        share the same goal framing.

        _NullUserSim.reset(instruction) returns the instruction directly,
        so no LLM call is made here.
        """
        self._done = False
        tidx = task_index if task_index is not None else self._next_task_index()
        try:
            reset_resp = self._env.reset(task_index=tidx)
        except TypeError:
            # Some tau-bench versions use task_index as positional arg
            reset_resp = self._env.reset(tidx)

        task = getattr(getattr(reset_resp, "info", None), "task", None)
        self._current_task = task
        return getattr(task, "instruction", None) or getattr(reset_resp, "observation", "")

    def generate(self, history: list[Turn]) -> Turn:
        """Generate target agent response, executing tool calls via env.step()."""
        messages = self._build_messages(history)
        tool_calls_made: list[ToolCall] = []

        for _ in range(self.max_tool_rounds):
            text, new_calls = self.client.generate_with_tools(messages, self._tool_schemas)

            if not new_calls:
                return Turn(
                    role=Role.ASSISTANT,
                    content=text,
                    turn_index=len(history),
                    tool_calls=tuple(tool_calls_made),
                )

            # Execute tool calls through tau-bench env
            oai_tool_calls: list[dict] = []
            for i, tc in enumerate(new_calls):
                result_str, is_done = self._execute_tool(tc.tool_name, tc.arguments)
                if is_done:
                    self._done = True

                try:
                    result_obj: dict = json.loads(result_str)
                    if not isinstance(result_obj, dict):
                        result_obj = {"result": result_obj}
                except (json.JSONDecodeError, TypeError):
                    result_obj = {"error": str(result_str)}

                tool_calls_made.append(ToolCall(
                    tool_name=tc.tool_name,
                    arguments=tc.arguments,
                    result=result_obj,
                ))

                call_id = f"call_{len(tool_calls_made)}_{i}"
                oai_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tc.tool_name,
                        "arguments": json.dumps(tc.arguments),
                    },
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result_str,
                })

            # Insert the assistant message before the tool-result messages
            asst_msg: dict[str, Any] = {
                "role": "assistant",
                "content": text or None,
                "tool_calls": oai_tool_calls,
            }
            messages.insert(len(messages) - len(new_calls), asst_msg)

            if self._done:
                break

        content = text if text else "I've transferred you to a human agent who can assist you further."
        return Turn(
            role=Role.ASSISTANT,
            content=content,
            turn_index=len(history),
            tool_calls=tuple(tool_calls_made),
        )

    def num_tasks(self) -> int:
        return len(getattr(self._env, "tasks", []))

    # ── Private helpers ────────────────────────────────────────────────

    def _next_task_index(self) -> int:
        idx = self._task_cursor
        n = self.num_tasks()
        self._task_cursor = (idx + 1) % max(n, 1)
        return idx

    def _execute_tool(self, tool_name: str, kwargs: dict) -> tuple[str, bool]:
        """Execute one tool call through env.step(). Returns (result_str, is_done)."""
        Action = self._get_action_class()
        if Action is None:
            return json.dumps({"error": f"tau-bench Action type not found"}), False

        try:
            resp = self._env.step(Action(name=tool_name, kwargs=kwargs))
            obs = resp.observation
            obs_str = obs if isinstance(obs, str) else json.dumps(obs)
            return obs_str, bool(resp.done)
        except Exception as e:
            return json.dumps({"error": str(e)}), False

    def _build_messages(self, history: list[Turn]) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = [{"role": "system", "content": self._wiki}]
        for turn in history:
            msgs.append({"role": turn.role.value, "content": turn.content})
        return msgs

    @staticmethod
    def _make_env(task_split: str, tau_bench_path: str | None) -> Any:
        if tau_bench_path is not None:
            import sys
            if tau_bench_path not in sys.path:
                sys.path.insert(0, tau_bench_path)
        try:
            from tau_bench.envs.airline.env import MockAirlineDomainEnv
            from tau_bench.envs.user import UserStrategy
        except ImportError as exc:
            raise ImportError(
                "tau-bench not installed. Run: pip install -e /tmp/tau-bench\n"
                f"Error: {exc}"
            ) from exc

        env = MockAirlineDomainEnv(
            user_strategy=UserStrategy.HUMAN,  # avoids LLM provider requirement at init
            task_split=task_split,
            task_index=0,  # tau-bench bug: randint(0, N) can return N (out of range)
        )
        env.user = _NullUserSim()  # replace after init; reset() returns instruction directly
        return env

    @staticmethod
    def _get_action_class() -> type | None:
        for mod_path in [
            "tau_bench.model_utils.types",
            "tau_bench.envs.base",
            "tau_bench.agents.tool_calling_agent",
        ]:
            try:
                import importlib
                mod = importlib.import_module(mod_path)
                if hasattr(mod, "Action"):
                    return mod.Action
            except ImportError:
                continue
        return None

    @staticmethod
    def _load_wiki(tau_bench_path: str | None) -> str:
        # Try package import first
        for attr_path in [
            ("tau_bench.envs.airline.data.wiki", "WIKI"),
            ("tau_bench.envs.airline", "WIKI"),
        ]:
            try:
                import importlib
                mod = importlib.import_module(attr_path[0])
                wiki = getattr(mod, attr_path[1], None)
                if wiki:
                    return str(wiki)
            except ImportError:
                pass

        # Try file paths
        candidates = [
            Path("/tmp/tau-bench/tau_bench/envs/airline/data/wiki.md"),
            Path("/tmp/tau-bench/tau_bench/envs/airline/wiki.md"),
        ]
        if tau_bench_path:
            candidates.insert(0, Path(tau_bench_path) / "tau_bench/envs/airline/data/wiki.md")
        for p in candidates:
            if p.exists():
                return p.read_text()

        return (
            "You are an airline customer service agent. Follow all airline policies carefully. "
            "Always confirm reservations, cancellations, or modifications with the customer "
            "before executing them. Only provide information grounded in tool results."
        )

    @staticmethod
    def _build_tool_schemas(tau_bench_path: str | None) -> list[dict]:
        if tau_bench_path is not None:
            import sys
            if tau_bench_path not in sys.path:
                sys.path.insert(0, tau_bench_path)
        try:
            from tau_bench.envs.airline.tools import ALL_TOOLS
        except ImportError:
            try:
                import importlib
                mod = importlib.import_module("tau_bench.envs.airline")
                ALL_TOOLS = getattr(mod, "ALL_TOOLS", [])
            except ImportError:
                return []

        schemas = []
        for tool in ALL_TOOLS:
            try:
                info = tool.get_info()
                # get_info() already returns {"type": "function", "function": {...}}
                schemas.append(info)
            except Exception:
                continue
        return schemas


# ── TauBenchEpisodeRunner ──────────────────────────────────────────────

class TauBenchEpisodeRunner(EpisodeRunner):
    """EpisodeRunner that resets the tau-bench env before each episode.

    Before each run() call it:
      1. Picks a scenario from the pool (preferring tau-bench tasks)
      2. Calls target.reset(scenario.task_index) to load the task's DB state
      3. Sets the auditor's scenario to the task instruction
      4. Delegates to the standard EpisodeRunner loop
    """

    def __init__(
        self,
        target: TauBenchTarget,
        scenario_pool: Any,   # ScenarioPool
        logger: Any = None,
        max_turns: int = 14,
    ):
        super().__init__(target=target, logger=logger, max_turns=max_turns)
        self.tb_target = target
        self.scenario_pool = scenario_pool

    def run(
        self,
        auditor: AuditorProtocol,
        initial_context: str | None = None,
        episode_id: str | None = None,
    ) -> Episode:
        # Pick a scenario and synchronise env + auditor before the episode
        scenario = self.scenario_pool.sample()

        # Prefer task_index from scenario; fall back to cursor advance
        task_idx = scenario.task_index if scenario.task_index >= 0 else None
        instruction = self.tb_target.reset(task_index=task_idx)

        # Use task instruction as the auditor scenario (overrides the scenario
        # pool description so the auditor goal matches the DB state)
        if hasattr(auditor, "set_scenario"):
            auditor.set_scenario(instruction or scenario.description)

        return super().run(
            auditor=auditor,
            initial_context=instruction or scenario.description,
            episode_id=episode_id,
        )

    def _should_stop(self, turn: Turn, episode: Episode | None = None) -> bool:
        if self.tb_target.done:
            return True
        return super()._should_stop(turn, episode)
