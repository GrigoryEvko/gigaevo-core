"""Integration tests for the typed CLI entry point in :mod:`run`."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


_EXPERIMENT_BODY = dedent(
    """
    from pathlib import Path
    from gigaevo.config.schemas import (
        BehaviorSpaceConfig,
        ChatOpenAIConfig,
        DataPlaneSettings,
        DefaultPipelineBuilderConfig,
        EnsembleRouterConfig,
        ExperimentConfig,
        FitnessProportionalEliteSelectorConfig,
        IslandConfig,
        PipelineConfig,
        ProblemConfig,
        RedisConfig,
        SingleIslandConfig,
        SteadyStateEngineConfig,
        SumArchiveSelectorConfig,
        TopFitnessMigrantSelectorConfig,
    )


    def build() -> ExperimentConfig:
        redis = RedisConfig()
        return ExperimentConfig(
            name="cli_test",
            seed=99,
            output_dir=Path("{OUTPUT_DIR}"),
            redis=redis,
            dataplane=DataPlaneSettings(redis=redis, key_prefix="gigaevo:cli_test"),
            problem=ProblemConfig(name="cli_test", problem_dir=Path("/srv/x")),
            algorithm=SingleIslandConfig(
                island=IslandConfig(
                    island_id="main",
                    behavior_space=BehaviorSpaceConfig(
                        keys=["fitness"],
                        bounds=[(0.0, 1.0)],
                        resolutions=[100],
                        binning_types=["linear"],
                    ),
                    archive_selector=SumArchiveSelectorConfig(
                        fitness_keys=["fitness"], fitness_key_higher_is_better=[True]
                    ),
                    elite_selector=FitnessProportionalEliteSelectorConfig(
                        fitness_key="fitness"
                    ),
                    migrant_selector=TopFitnessMigrantSelectorConfig(
                        fitness_key="fitness"
                    ),
                )
            ),
            engine=SteadyStateEngineConfig(),
            pipeline=PipelineConfig(builder=DefaultPipelineBuilderConfig()),
            llm=EnsembleRouterConfig(models=[ChatOpenAIConfig(model="gpt-4o-mini")]),
        )
    """
)


def _make_experiment(tmp_path: Path) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    body = _EXPERIMENT_BODY.replace("{OUTPUT_DIR}", str(out))
    exp = tmp_path / "experiment.py"
    exp.write_text(body)
    return exp


class TestCliDryRun:
    def test_dry_run_dumps_config_and_exits_zero(self, tmp_path: Path) -> None:
        from run import main

        exp = _make_experiment(tmp_path)
        exit_code = main([str(exp), "--dry-run"])
        assert exit_code == 0

        out_root = tmp_path / "outputs"
        run_dirs = list(out_root.iterdir())
        assert len(run_dirs) == 1
        config_path = run_dirs[0] / "config.json"
        assert config_path.exists()

        dumped = json.loads(config_path.read_text())
        assert dumped["name"] == "cli_test"
        assert dumped["seed"] == 99

    def test_dry_run_directory_name_is_experiment_id(self, tmp_path: Path) -> None:
        from run import main
        from gigaevo.config.experiment_loader import build_experiment

        exp = _make_experiment(tmp_path)
        cfg = build_experiment(exp)
        expected_id = cfg.experiment_id

        main([str(exp), "--dry-run"])

        run_dir = (tmp_path / "outputs" / expected_id)
        assert run_dir.exists(), (
            f"expected {run_dir}, got {list((tmp_path / 'outputs').iterdir())}"
        )


class TestCliConfigDumpSafety:
    def test_dumped_config_does_not_leak_api_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The on-disk ``config.json`` is the reproducibility record;
        it lives under ``output_dir/experiment_id/`` and is the natural
        artefact for users to share, attach to bug reports, or commit
        to a sweep manifest. The resolved ``OPENAI_API_KEY`` must not
        appear in it."""
        from run import main

        monkeypatch.setenv("OPENAI_API_KEY", "sk-do-not-leak-from-cli")
        exp = _make_experiment(tmp_path)
        exit_code = main([str(exp), "--dry-run"])
        assert exit_code == 0

        out_root = tmp_path / "outputs"
        config_paths = list(out_root.glob("*/config.json"))
        assert len(config_paths) == 1
        text = config_paths[0].read_text()
        assert "sk-do-not-leak-from-cli" not in text
        assert '"api_key"' not in text


class TestCliErrors:
    def test_missing_experiment_file_propagates(self, tmp_path: Path) -> None:
        from run import main

        with pytest.raises(FileNotFoundError):
            main([str(tmp_path / "no_such.py"), "--dry-run"])

    def test_invalid_experiment_raises(self, tmp_path: Path) -> None:
        from run import main
        from gigaevo.config.experiment_loader import ExperimentModuleError

        bad = tmp_path / "bad.py"
        bad.write_text("x = 1\n")  # no build()

        with pytest.raises(ExperimentModuleError):
            main([str(bad), "--dry-run"])


class TestCliTyroOverride:
    def test_seed_override_propagates_to_dumped_config(self, tmp_path: Path) -> None:
        """The tyro path must materialise the override against the
        Pydantic field tree and re-trigger validation; the dumped JSON
        is the ground truth."""
        from run import main

        exp = _make_experiment(tmp_path)
        exit_code = main([str(exp), "--dry-run", "--seed", "7"])
        assert exit_code == 0

        out_root = tmp_path / "outputs"
        run_dirs = list(out_root.iterdir())
        assert len(run_dirs) == 1
        dumped = json.loads((run_dirs[0] / "config.json").read_text())
        assert dumped["seed"] == 7
        # The default seed in the experiment is 99, so the override
        # genuinely changed the resolved value.
        assert dumped["seed"] != 99

    def test_help_with_override_reaches_tyro_with_overrides(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``--help`` must not short-circuit before overrides reshape the
        discriminated-union schema. The combined invocation should print
        the help for the *chosen* variant, not the baseline default."""
        from run import main

        exp = _make_experiment(tmp_path)
        # ``--help`` forwarded to tyro raises SystemExit; ``main``
        # returns its exit code.
        exit_code = main([str(exp), "--help"])
        assert exit_code == 0
        captured = capsys.readouterr()
        # Help output should be on stdout (argparse / tyro convention).
        assert "experiment" in captured.out.lower()

    def test_repeated_flag_emits_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Argparse / tyro silently last-wins on repeated ``--flag``;
        operators should see a warning so they notice the collision."""
        import logging
        from run import main

        # Wire loguru -> caplog so caplog.records includes the warning.
        from loguru import logger as loguru_logger

        class _CaplogSink:
            def write(self, msg: str) -> None:
                caplog.records.append(
                    logging.LogRecord(
                        name="run",
                        level=logging.WARNING,
                        pathname=__file__,
                        lineno=0,
                        msg=msg,
                        args=(),
                        exc_info=None,
                    )
                )

        handler_id = loguru_logger.add(_CaplogSink(), level="WARNING")
        try:
            exp = _make_experiment(tmp_path)
            exit_code = main(
                [str(exp), "--dry-run", "--seed", "5", "--seed", "9"]
            )
        finally:
            loguru_logger.remove(handler_id)
        assert exit_code == 0
        warnings = [
            r.msg
            for r in caplog.records
            if r.levelno == logging.WARNING and "--seed" in str(r.msg)
        ]
        assert warnings, "expected a repeated-flag warning"

    def test_override_triggers_cross_field_validator(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """tyro merges overrides then Pydantic re-validates. An override
        that violates a cross-field invariant must exit non-zero and
        surface a framed error block — not be accepted silently and not
        leak a raw Python stack trace."""
        from run import main

        exp = _make_experiment(tmp_path)
        # The experiment's name is "cli_test", so dataplane.key_prefix
        # must equal "gigaevo:cli_test". Renaming the experiment via
        # CLI without updating key_prefix breaks the invariant.
        exit_code = main([str(exp), "--dry-run", "--name", "renamed"])
        assert exit_code == 2
        captured = capsys.readouterr()
        assert "Configuration validation failed" in captured.err
        # The friendly framed block should not include a Python
        # traceback header.
        assert "Traceback" not in captured.err


class TestCliOutputDirValidation:
    """``_dump_resolved_config`` walks up the requested ``output_dir``
    until it finds an existing ancestor; if that ancestor is not a
    writable directory the run aborts with a typed message instead of
    leaking a low-level ``OSError`` frame from ``mkdir``."""

    def test_output_dir_under_regular_file_rejected(
        self, tmp_path: Path
    ) -> None:
        from run import main

        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        exp = _make_experiment(tmp_path)
        with pytest.raises(ValueError, match="not a directory"):
            main(
                [
                    str(exp),
                    "--dry-run",
                    "--output-dir",
                    str(blocker / "below"),
                ]
            )

    def test_output_dir_under_unwritable_parent_rejected(
        self, tmp_path: Path
    ) -> None:
        from run import main

        readonly = tmp_path / "readonly"
        readonly.mkdir()
        # ``chmod 0o500`` keeps the directory readable + executable so
        # the existence-walk succeeds but blocks new entries underneath.
        readonly.chmod(0o500)
        try:
            exp = _make_experiment(tmp_path)
            with pytest.raises(ValueError, match="not writable"):
                main(
                    [
                        str(exp),
                        "--dry-run",
                        "--output-dir",
                        str(readonly / "below"),
                    ]
                )
        finally:
            readonly.chmod(0o700)
