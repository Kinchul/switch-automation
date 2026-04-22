from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass(slots=True)
class LoopStatsSnapshot:
    loop_counter: int
    total_elapsed_seconds: float
    loop_elapsed_seconds: float
    status: str
    last_outcome: str | None


@dataclass(slots=True)
class _StatsRecord:
    loop_counter: int = 0
    total_elapsed_seconds: float = 0.0
    current_loop_elapsed_seconds_accum: float = 0.0
    active_loop_started_at: str | None = None
    status: str = "stopped"
    last_outcome: str | None = None
    updated_at: str | None = None
    failed_loop_score_history: dict[str, list[float]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> _StatsRecord:
        failed_history_raw = data.get("failed_loop_score_history", {})
        if not isinstance(failed_history_raw, dict):
            failed_history_raw = {}
        return cls(
            loop_counter=int(data.get("loop_counter", 0)),
            total_elapsed_seconds=float(data.get("total_elapsed_seconds", 0.0)),
            current_loop_elapsed_seconds_accum=float(
                data.get("current_loop_elapsed_seconds_accum", 0.0)
            ),
            active_loop_started_at=data.get("active_loop_started_at"),  # type: ignore[arg-type]
            status=str(data.get("status", "stopped")),
            last_outcome=data.get("last_outcome"),  # type: ignore[arg-type]
            updated_at=data.get("updated_at"),  # type: ignore[arg-type]
            failed_loop_score_history={
                str(state_name): [float(score) for score in state_scores if score is not None]
                for state_name, state_scores in failed_history_raw.items()
                if isinstance(state_scores, list)
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "loop_counter": self.loop_counter,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "current_loop_elapsed_seconds_accum": self.current_loop_elapsed_seconds_accum,
            "active_loop_started_at": self.active_loop_started_at,
            "status": self.status,
            "last_outcome": self.last_outcome,
            "updated_at": self.updated_at,
            "failed_loop_score_history": self.failed_loop_score_history,
        }

    def snapshot(self) -> LoopStatsSnapshot:
        now = _utcnow()
        return LoopStatsSnapshot(
            loop_counter=self.loop_counter,
            total_elapsed_seconds=self.total_elapsed_seconds + self.current_loop_total_seconds(now),
            loop_elapsed_seconds=self.current_loop_total_seconds(now),
            status=self.status,
            last_outcome=self.last_outcome,
        )

    def current_loop_total_seconds(self, now: datetime | None = None) -> float:
        return self.current_loop_elapsed_seconds_accum + self._active_segment_seconds(now)

    def _active_segment_seconds(self, now: datetime | None = None) -> float:
        if self.active_loop_started_at is None:
            return 0.0
        if now is None:
            now = _utcnow()
        started_at = datetime.fromisoformat(self.active_loop_started_at)
        return max(0.0, (now - started_at).total_seconds())


@dataclass(slots=True)
class PersistentLoopStatsStore:
    path: Path
    sequences: dict[str, _StatsRecord] = field(default_factory=dict)
    updated_at: str | None = None

    @classmethod
    def load(cls, path: Path) -> PersistentLoopStatsStore:
        data = _load_json_object(path, label="stats")
        if not isinstance(data, dict):
            return cls(path=path)

        sequences_raw = data.get("sequences", {})
        sequences: dict[str, _StatsRecord] = {}
        if isinstance(sequences_raw, dict):
            for sequence_id, sequence_data in sequences_raw.items():
                if isinstance(sequence_data, dict):
                    sequences[str(sequence_id)] = _StatsRecord.from_dict(sequence_data)

        return cls(
            path=path,
            sequences=sequences,
            updated_at=data.get("updated_at"),  # type: ignore[arg-type]
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _utcnow().isoformat()
        payload = {
            "updated_at": self.updated_at,
            "sequences": {
                sequence_id: record.to_dict()
                for sequence_id, record in sorted(self.sequences.items())
            },
        }
        _write_json_atomically(self.path, payload)
        _chmod_if_possible(self.path, 0o644)

    def snapshot(self, sequence_id: str) -> LoopStatsSnapshot:
        return self._record(sequence_id).snapshot()

    def snapshots(self) -> dict[str, LoopStatsSnapshot]:
        return {
            sequence_id: record.snapshot()
            for sequence_id, record in sorted(self.sequences.items())
        }

    def start_new_loop(self, sequence_id: str) -> LoopStatsSnapshot:
        record = self._record(sequence_id)
        if record.current_loop_total_seconds() > 0:
            record.total_elapsed_seconds += record.current_loop_total_seconds()
            record.current_loop_elapsed_seconds_accum = 0.0
            record.active_loop_started_at = None

        record.current_loop_elapsed_seconds_accum = 0.0
        record.active_loop_started_at = _utcnow().isoformat()
        record.status = "running"
        record.updated_at = _utcnow().isoformat()
        self.save()
        return record.snapshot()

    def record_retry(self, sequence_id: str) -> LoopStatsSnapshot:
        record = self._record(sequence_id)
        record.total_elapsed_seconds += record.current_loop_total_seconds()
        record.current_loop_elapsed_seconds_accum = 0.0
        record.active_loop_started_at = _utcnow().isoformat()
        record.status = "running"
        record.loop_counter += 1
        record.updated_at = _utcnow().isoformat()
        self.save()
        return record.snapshot()

    def checkpoint_running(self, sequence_id: str) -> LoopStatsSnapshot:
        record = self._record(sequence_id)
        if record.active_loop_started_at is not None:
            record.current_loop_elapsed_seconds_accum += record._active_segment_seconds()
            record.active_loop_started_at = _utcnow().isoformat()
            record.status = "running"
            record.updated_at = _utcnow().isoformat()
            self.save()
        return record.snapshot()

    def finish_loop(
        self,
        sequence_id: str,
        outcome: str,
        *,
        status: str = "stopped",
    ) -> LoopStatsSnapshot:
        record = self._record(sequence_id)
        record.total_elapsed_seconds += record.current_loop_total_seconds()
        record.current_loop_elapsed_seconds_accum = 0.0
        record.active_loop_started_at = None
        record.status = status
        record.last_outcome = outcome
        record.updated_at = _utcnow().isoformat()
        self.save()
        return record.snapshot()

    def mark_status(
        self,
        sequence_id: str,
        status: str,
        *,
        last_outcome: str | None = None,
    ) -> LoopStatsSnapshot:
        record = self._record(sequence_id)
        record.status = status
        if last_outcome is not None:
            record.last_outcome = last_outcome
        record.updated_at = _utcnow().isoformat()
        self.save()
        return record.snapshot()

    def normalize_on_startup(self) -> None:
        changed = False
        for record in self.sequences.values():
            if record.active_loop_started_at is not None and record.updated_at is not None:
                updated_at = datetime.fromisoformat(record.updated_at)
                started_at = datetime.fromisoformat(record.active_loop_started_at)
                if updated_at >= started_at:
                    record.current_loop_elapsed_seconds_accum += (updated_at - started_at).total_seconds()
            if record.active_loop_started_at is not None or record.status != "stopped":
                record.active_loop_started_at = None
                record.status = "stopped"
                record.updated_at = _utcnow().isoformat()
                changed = True
        if changed:
            self.save()

    def reset(self) -> None:
        self.sequences = {}
        self.save()

    def failed_loop_score_history(self, sequence_id: str, state_name: str) -> list[float]:
        record = self.sequences.get(sequence_id)
        if record is None:
            return []
        return list(record.failed_loop_score_history.get(state_name, ()))

    def set_failed_loop_score_history(
        self,
        sequence_id: str,
        state_name: str,
        scores: list[float],
    ) -> None:
        record = self._record(sequence_id)
        record.failed_loop_score_history[state_name] = [float(score) for score in scores]
        record.updated_at = _utcnow().isoformat()
        self.save()

    def _record(self, sequence_id: str) -> _StatsRecord:
        record = self.sequences.get(sequence_id)
        if record is None:
            record = _StatsRecord()
            self.sequences[sequence_id] = record
        return record


@dataclass(slots=True)
class PersistentLoopControl:
    path: Path
    command: str = "noop"
    selected_sequence: str | None = None
    updated_at: str | None = None

    @classmethod
    def load(cls, path: Path) -> PersistentLoopControl:
        data = _load_json_object(path, label="control")
        return cls(
            path=path,
            command=str(data.get("command", "noop")) if isinstance(data, dict) else "noop",
            selected_sequence=(data.get("selected_sequence") if isinstance(data, dict) else None),  # type: ignore[arg-type]
            updated_at=(data.get("updated_at") if isinstance(data, dict) else None),  # type: ignore[arg-type]
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _utcnow().isoformat()
        _write_json_atomically(
            self.path,
            {
                "command": self.command,
                "selected_sequence": self.selected_sequence,
                "updated_at": self.updated_at,
            },
        )
        _chmod_if_possible(self.path, 0o666)

    def set_command(self, command: str) -> None:
        self.command = command
        self.save()

    def set_selected_sequence(self, sequence_id: str) -> None:
        self.selected_sequence = sequence_id
        self.save()

    def refresh(self) -> str:
        data = _load_json_object(self.path, label="control")
        if isinstance(data, dict):
            self.command = str(data.get("command", "noop"))
            self.selected_sequence = data.get("selected_sequence")  # type: ignore[arg-type]
            self.updated_at = data.get("updated_at")  # type: ignore[arg-type]
        return self.command


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _load_json_object(path: Path, *, label: str) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        backup_path = _quarantine_invalid_json(path)
        print(
            f"Warning: could not read {label} JSON from {path}: {exc}. "
            f"Moved the invalid file to {backup_path} and starting fresh."
        )
        return None
    if not isinstance(data, dict):
        backup_path = _quarantine_invalid_json(path)
        print(
            f"Warning: {label} JSON in {path} is not an object. "
            f"Moved it to {backup_path} and starting fresh."
        )
        return None
    return data


def _quarantine_invalid_json(path: Path) -> Path:
    timestamp = _utcnow().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(f"{path.stem}.invalid-{timestamp}{path.suffix}")
    counter = 1
    while backup_path.exists():
        backup_path = path.with_name(f"{path.stem}.invalid-{timestamp}-{counter}{path.suffix}")
        counter += 1
    try:
        path.replace(backup_path)
    except OSError:
        pass
    return backup_path


def _write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(path)
    except PermissionError as exc:
        raise PermissionError(
            f"Could not write {path}. This usually means the file or directory is owned by another user "
            f"(for example a previous root-run service). Ensure the camera-loop service runs as your normal user "
            f"and fix ownership of {path.parent}."
        ) from exc


def _chmod_if_possible(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except PermissionError:
        return
    except OSError:
        return
