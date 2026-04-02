from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar
from urllib import error, request

from fastapi import Body, FastAPI
from pydantic import BaseModel, Field

try:
    from openenv.core.client_types import StepResult  # type: ignore
    from openenv.core.env_client import EnvClient  # type: ignore
    from openenv.core.env_server import Action, Environment, Observation, State  # type: ignore
    from openenv.core.env_server.http_server import create_app  # type: ignore
except ImportError:
    A = TypeVar("A")
    O = TypeVar("O")
    S = TypeVar("S")

    class Action(BaseModel):
        """Fallback base action model when OpenEnv is not installed."""

    class Observation(BaseModel):
        reward: Optional[float] = None
        done: bool = False
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        episode_id: Optional[str] = None
        step_count: int = 0

    class Environment(ABC):
        @abstractmethod
        def reset(self, **kwargs: Any) -> Observation:
            raise NotImplementedError

        @abstractmethod
        def step(self, action: Action, **kwargs: Any) -> Observation:
            raise NotImplementedError

        @property
        @abstractmethod
        def state(self) -> State:
            raise NotImplementedError

    @dataclass
    class StepResult(Generic[O]):
        observation: O
        reward: Optional[float]
        done: bool

    class EnvClient(Generic[A, O, S], ABC):
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url.rstrip("/")

        @abstractmethod
        def _step_payload(self, action: A) -> Dict[str, Any]:
            raise NotImplementedError

        @abstractmethod
        def _parse_result(self, payload: Dict[str, Any]) -> StepResult[O]:
            raise NotImplementedError

        @abstractmethod
        def _parse_state(self, payload: Dict[str, Any]) -> S:
            raise NotImplementedError

        def reset(self, **kwargs: Any) -> O:
            payload = self._request("POST", "/reset", kwargs)
            return self._parse_result(
                {
                    "observation": payload,
                    "reward": payload.get("reward"),
                    "done": payload.get("done", False),
                }
            ).observation

        def step(self, action: A) -> StepResult[O]:
            payload = self._request("POST", "/step", {"action": self._step_payload(action)})
            return self._parse_result(payload)

        def state(self) -> S:
            payload = self._request("GET", "/state")
            return self._parse_state(payload)

        def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            url = f"{self.base_url}{path}"
            data = None if payload is None else json.dumps(payload).encode("utf-8")
            req = request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method=method,
            )
            try:
                with request.urlopen(req, timeout=30) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:  # pragma: no cover - thin fallback adapter
                detail = exc.read().decode("utf-8")
                raise RuntimeError(f"{method} {url} failed with {exc.code}: {detail}") from exc

    def create_app(env_class: type[Environment], action_model: type[Action], observation_model: type[Observation], env_name: str) -> FastAPI:
        del observation_model
        env = env_class()
        app = FastAPI(title=env_name)

        @app.get("/health")
        def health() -> Dict[str, str]:
            return {"status": "ok", "env_name": env_name}

        @app.post("/reset")
        def reset(payload: Optional[Dict[str, Any]] = Body(default=None)) -> Dict[str, Any]:
            observation = env.reset(**(payload or {}))
            return observation.model_dump()

        @app.post("/step")
        def step(payload: Optional[Dict[str, Any]] = Body(default=None)) -> Dict[str, Any]:
            action = action_model(**((payload or {}).get("action", {})))
            observation = env.step(action)
            return {
                "observation": observation.model_dump(),
                "reward": observation.reward,
                "done": observation.done,
            }

        @app.get("/state")
        def state() -> Dict[str, Any]:
            return env.state.model_dump()

        return app
