"""Load and validate models.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .suggest import did_you_mean


def _duplicates(values: list[str] | list[int]) -> list[str | int]:
    return sorted({v for v in values if values.count(v) > 1}, key=str)  # type: ignore[arg-type]


class ConfigError(Exception):
    pass


class Service(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    gpus: list[int] = Field(min_length=1)
    runtime: Literal["command", "docker"] = "command"
    # vLLM-style fraction of each GPU's total memory.
    gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    # Real memory the service held per GPU in a previous run (from nvidia-smi).
    measured_mib: dict[int, int] | None = None
    # Extra memory beyond the requested fraction (CUDA graphs etc.), used when not measured.
    overhead_mib: int = Field(default=0, ge=0)
    health: str = "/v1/models"
    # Shell snippet run with `bash -c`; end it with `exec <program>` so signals reach the program.
    command: str | None = None
    # Extra command run first when stopping (e.g. `docker stop my-stt`).
    stop: str | None = None
    start_timeout_s: int = Field(default=900, gt=0)

    @model_validator(mode="after")
    def _measured_gpus_are_used(self) -> Service:
        if self.measured_mib:
            extra = set(self.measured_mib) - set(self.gpus)
            if extra:
                raise ValueError(f"measured_mib lists GPUs {sorted(extra)} that are not in gpus")
        return self


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    services: list[Service] = Field(min_length=1)
    # Free memory that should remain on each GPU after everything has started.
    margin_mib: int = Field(default=1024, ge=0)

    @model_validator(mode="after")
    def _unique_names_and_ports(self) -> Config:
        names = [s.name for s in self.services]
        ports = [s.port for s in self.services]
        for label, dupes in (("name", _duplicates(names)), ("port", _duplicates(ports))):
            if dupes:
                raise ValueError(f"duplicate service {label}: {dupes}")
        return self


def load_config(path: str | Path) -> Config:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as exc:
        hint = ""
        if not path.exists() and path.parent.is_dir():
            hint = did_you_mean(path.name, (p.name for p in path.parent.iterdir() if p.suffix in (".yaml", ".yml")))
        raise ConfigError(f"cannot read {path}: {exc.strerror}.{hint}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping with a 'services' list")
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        lines = [f"  {'.'.join(str(p) for p in e['loc'])}: {e['msg']}{_field_hint(e)}" for e in exc.errors()]
        raise ConfigError(f"invalid config {path}:\n" + "\n".join(lines)) from exc


def _field_hint(error: dict) -> str:
    """For an unknown field, suggest the closest real one (for example gpu_memory_utilisation)."""
    if error.get("type") != "extra_forbidden":
        return ""
    loc = error["loc"]
    model = Service if "services" in loc[:-1] else Config
    hint = did_you_mean(str(loc[-1]), model.model_fields)
    return f". {hint.strip()}" if hint else ""
