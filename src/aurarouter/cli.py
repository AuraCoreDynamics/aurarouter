"""AuraRouter command-line interface.

Provides ``main()`` as the ``aurarouter`` console entry-point.  All
subcommands use :class:`~aurarouter.api.AuraRouterAPI` as the unified
backend.  Human-readable output is the default; pass ``--json`` for
machine-readable JSON.
"""

import argparse
import json as _json
import logging
import sys
from typing import Any, Optional

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.CLI")


# ======================================================================
# Helpers
# ======================================================================

def _make_api(args: argparse.Namespace):
    """Construct an AuraRouterAPI from the parsed --config flag."""
    from aurarouter.api import APIConfig, AuraRouterAPI

    return AuraRouterAPI(APIConfig(config_path=getattr(args, "config", None)))


def _print_json(data: Any) -> None:
    """Pretty-print *data* as JSON to stdout."""
    print(_json.dumps(data, indent=2, default=str))


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a simple aligned ASCII table."""
    if not rows:
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        padded = [str(c) for c in row] + [""] * (len(headers) - len(row))
        print(fmt.format(*padded[:len(headers)]))


# ======================================================================
# Subcommand handlers
# ======================================================================

# -- model ---------------------------------------------------------------

def _cmd_model_list(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        models = api.list_models()
        if getattr(args, "filter", None):
            filt = args.filter.lower()
            models = [m for m in models if filt in m.model_id.lower() or filt in m.provider.lower()]
        if args.json:
            _print_json([{"model_id": m.model_id, "provider": m.provider, "config": m.config} for m in models])
        else:
            if not models:
                print("No models configured.")
                return
            rows = [[m.model_id, m.provider, m.config.get("hosting_tier", ""), ", ".join(m.config.get("tags", []))] for m in models]
            _print_table(["MODEL ID", "PROVIDER", "TIER", "TAGS"], rows)
            print(f"\n{len(models)} model(s).")


def _cmd_model_add(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        cfg: dict[str, Any] = {"provider": args.provider}
        if args.endpoint:
            cfg["endpoint"] = args.endpoint
        if args.model_name:
            cfg["model_name"] = args.model_name
        if args.api_key:
            cfg["api_key"] = args.api_key
        if args.tags:
            cfg["tags"] = [t.strip() for t in args.tags.split(",")]
        if args.tier:
            cfg["hosting_tier"] = args.tier
        info = api.add_model(args.model_id, cfg)
        api.save_config()
        print(f"Added model '{info.model_id}' (provider={info.provider}).")


def _cmd_model_edit(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        existing = api.get_model(args.model_id)
        if not existing:
            print(f"Error: Model '{args.model_id}' not found.", file=sys.stderr)
            sys.exit(1)
        cfg = dict(existing.config)
        if args.provider:
            cfg["provider"] = args.provider
        if args.endpoint:
            cfg["endpoint"] = args.endpoint
        if args.model_name:
            cfg["model_name"] = args.model_name
        if args.api_key:
            cfg["api_key"] = args.api_key
        if args.tags:
            cfg["tags"] = [t.strip() for t in args.tags.split(",")]
        if args.tier:
            cfg["hosting_tier"] = args.tier
        api.update_model(args.model_id, cfg)
        api.save_config()
        print(f"Updated model '{args.model_id}'.")


def _cmd_model_remove(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        if not args.force:
            existing = api.get_model(args.model_id)
            if not existing:
                print(f"Error: Model '{args.model_id}' not found.", file=sys.stderr)
                sys.exit(1)
        removed = api.remove_model(args.model_id)
        if removed:
            api.save_config()
            print(f"Removed model '{args.model_id}'.")
        else:
            print(f"Model '{args.model_id}' not found.", file=sys.stderr)
            sys.exit(1)


def _cmd_model_test(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        ok, msg = api.test_model_connection(args.model_id)
        if args.json:
            _print_json({"model_id": args.model_id, "success": ok, "message": msg})
        else:
            status = "OK" if ok else "FAIL"
            print(f"[{status}] {args.model_id}: {msg}")
        if not ok:
            sys.exit(1)


def _cmd_model_auto_tune(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        params = api.auto_tune_model(args.model_id)
        if params:
            api.save_config()
            if args.json:
                _print_json({"model_id": args.model_id, "parameters": params})
            else:
                print(f"Auto-tuned '{args.model_id}':")
                for k, v in params.items():
                    print(f"  {k}: {v}")
        else:
            print(f"Auto-tuning not applicable for '{args.model_id}'.")


# -- route ---------------------------------------------------------------

def _cmd_route_list(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        roles = api.list_roles()
        if args.json:
            _print_json([{"role": r.role, "chain": r.chain} for r in roles])
        else:
            if not roles:
                print("No routing roles configured.")
                return
            rows = [[r.role, " > ".join(r.chain)] for r in roles]
            _print_table(["ROLE", "FALLBACK CHAIN"], rows)


def _cmd_route_set(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        rc = api.set_role_chain(args.role, args.model_ids)
        api.save_config()
        print(f"Set role '{rc.role}': {' > '.join(rc.chain)}")


def _cmd_route_append(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        existing = api.get_role_chain(args.role)
        chain = list(existing.chain) if existing else []
        chain.append(args.model_id)
        rc = api.set_role_chain(args.role, chain)
        api.save_config()
        print(f"Appended to '{rc.role}': {' > '.join(rc.chain)}")


def _cmd_route_remove_model(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        existing = api.get_role_chain(args.role)
        if not existing or args.model_id not in existing.chain:
            print(f"Model '{args.model_id}' not in role '{args.role}'.", file=sys.stderr)
            sys.exit(1)
        chain = [m for m in existing.chain if m != args.model_id]
        api.set_role_chain(args.role, chain)
        api.save_config()
        print(f"Removed '{args.model_id}' from role '{args.role}'.")


def _cmd_route_delete(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        if not args.force:
            existing = api.get_role_chain(args.role)
            if not existing:
                print(f"Role '{args.role}' not found.", file=sys.stderr)
                sys.exit(1)
        removed = api.remove_role(args.role)
        if removed:
            api.save_config()
            print(f"Deleted role '{args.role}'.")
        else:
            print(f"Role '{args.role}' not found.", file=sys.stderr)
            sys.exit(1)


# -- execution -----------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        context = ""
        if args.context:
            from pathlib import Path
            context = Path(args.context).read_text(encoding="utf-8")

        fmt = getattr(args, "format", "text") or "text"

        if getattr(args, "no_review", False):
            # Direct execution without the full IPE loop
            result = api.execute_direct("coding", args.task, json_mode=(fmt == "json"))
            if args.json:
                _print_json({"output": result.text, "model_id": result.model_id})
            else:
                print(result.text)
            return

        result = api.execute_task(
            task=args.task,
            context=context,
            output_format=fmt,
        )
        if args.json:
            _print_json({
                "output": result.output,
                "intent": result.intent,
                "complexity": result.complexity,
                "plan": result.plan,
                "steps_executed": result.steps_executed,
                "review_verdict": result.review_verdict,
                "review_feedback": result.review_feedback,
                "total_elapsed": result.total_elapsed,
            })
        else:
            print(result.output)
            if result.review_verdict:
                print(f"\n[Review: {result.review_verdict}] {result.review_feedback}")


def _cmd_compare(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        model_ids = [m.strip() for m in args.models.split(",")]
        results = api.compare_models(args.prompt, model_ids)
        if args.json:
            _print_json([
                {"model_id": r.model_id, "provider": r.provider, "output": r.text}
                for r in results
            ])
        else:
            for r in results:
                print(f"--- {r.model_id} ({r.provider}) ---")
                print(r.text)
                print()


# -- monitoring ----------------------------------------------------------

def _parse_time_range(range_str: Optional[str]):
    """Parse a time range string like '24h', '7d', '30d' into (start, end)."""
    if not range_str:
        return None
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    unit = range_str[-1].lower()
    try:
        value = int(range_str[:-1])
    except ValueError:
        return None

    if unit == "h":
        start = now - timedelta(hours=value)
    elif unit == "d":
        start = now - timedelta(days=value)
    else:
        return None
    return (start.isoformat(), now.isoformat())


def _cmd_traffic(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        time_range = _parse_time_range(getattr(args, "range", None))
        summary = api.get_traffic(time_range=time_range)
        if args.json:
            _print_json({
                "total_tokens": summary.total_tokens,
                "input_tokens": summary.input_tokens,
                "output_tokens": summary.output_tokens,
                "total_spend": summary.total_spend,
                "spend_by_provider": summary.spend_by_provider,
                "by_model": summary.by_model,
                "projection": summary.projection,
            })
        else:
            print(f"Total tokens:  {summary.total_tokens:,}")
            print(f"  Input:       {summary.input_tokens:,}")
            print(f"  Output:      {summary.output_tokens:,}")
            print(f"Total spend:   ${summary.total_spend:.2f}")
            if summary.spend_by_provider:
                print("Spend by provider:")
                for prov, spend in summary.spend_by_provider.items():
                    print(f"  {prov}: ${spend:.2f}")


def _cmd_privacy(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        time_range = _parse_time_range(getattr(args, "range", None))
        severity = getattr(args, "severity", None)
        summary = api.get_privacy_events(time_range=time_range, severity=severity)
        if args.json:
            _print_json({
                "total_events": summary.total_events,
                "by_severity": summary.by_severity,
                "by_pattern": summary.by_pattern,
            })
        else:
            print(f"Privacy events: {summary.total_events}")
            if summary.by_severity:
                print("By severity:")
                for sev, count in summary.by_severity.items():
                    print(f"  {sev}: {count}")
            if summary.by_pattern:
                print("By pattern:")
                for pat, count in summary.by_pattern.items():
                    print(f"  {pat}: {count}")


def _cmd_health(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        model_id = getattr(args, "model_id", None)
        reports = api.check_health(model_id=model_id)
        if args.json:
            _print_json([
                {"model_id": r.model_id, "healthy": r.healthy, "message": r.message, "latency": r.latency}
                for r in reports
            ])
        else:
            for r in reports:
                status = "OK" if r.healthy else "FAIL"
                lat = f" ({r.latency:.2f}s)" if r.latency > 0 else ""
                print(f"[{status}] {r.model_id}: {r.message}{lat}")


def _cmd_budget(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        status = api.get_budget_status()
        if args.json:
            _print_json(status or {"enabled": False})
        else:
            if status is None:
                print("Budget tracking is disabled.")
                return
            print(f"Daily spend:   ${status['daily_spend']:.2f}" +
                  (f" / ${status['daily_limit']:.2f}" if status.get("daily_limit") else ""))
            print(f"Monthly spend: ${status['monthly_spend']:.2f}" +
                  (f" / ${status['monthly_limit']:.2f}" if status.get("monthly_limit") else ""))
            if not status["allowed"]:
                print(f"BLOCKED: {status['reason']}")


# -- config --------------------------------------------------------------

def _cmd_config_show(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        if args.json:
            import yaml
            data = yaml.safe_load(api.get_config_yaml())
            _print_json(data)
        else:
            print(api.get_config_yaml())


def _cmd_config_set(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        # Navigate dotted key path into config dict
        cfg = api._config.config  # noqa: SLF001
        keys = args.key.split(".")
        target = cfg
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        # Attempt type coercion
        value: Any = args.value
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        target[keys[-1]] = value
        api.save_config()
        print(f"Set {args.key} = {value}")


def _cmd_config_mcp_tool(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        enabled = args.enable
        result = api.set_mcp_tool(args.tool, enabled)
        api.save_config()
        state = "enabled" if result.enabled else "disabled"
        print(f"MCP tool '{result.name}' {state}.")


def _cmd_config_save(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        saved_to = api.save_config()
        print(f"Configuration saved to: {saved_to}")


def _cmd_config_reload(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        api.reload_config()
        print("Configuration reloaded from disk.")


# -- catalog -------------------------------------------------------------

def _cmd_catalog_list(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        entries = api.list_catalog()
        if args.json:
            _print_json([
                {"name": e.name, "type": e.provider_type, "source": e.source,
                 "installed": e.installed, "running": e.running, "version": e.version}
                for e in entries
            ])
        else:
            if not entries:
                print("No providers in catalog.")
                return
            rows = [
                [e.name, e.provider_type, e.source,
                 "Yes" if e.installed else "No",
                 "Running" if e.running else "Stopped"]
                for e in entries
            ]
            _print_table(["NAME", "TYPE", "SOURCE", "INSTALLED", "STATUS"], rows)


def _cmd_catalog_add(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        entry = api.add_catalog_provider(args.name, args.endpoint)
        print(f"Added catalog provider '{entry.name}'.")


def _cmd_catalog_remove(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        removed = api.remove_catalog_provider(args.name)
        if removed:
            print(f"Removed catalog provider '{args.name}'.")
        else:
            print(f"Provider '{args.name}' not found.", file=sys.stderr)
            sys.exit(1)


def _cmd_catalog_start(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        ok = api.start_catalog_provider(args.name)
        if ok:
            print(f"Started '{args.name}'.")
        else:
            print(f"Failed to start '{args.name}'.", file=sys.stderr)
            sys.exit(1)


def _cmd_catalog_stop(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        ok = api.stop_catalog_provider(args.name)
        if ok:
            print(f"Stopped '{args.name}'.")
        else:
            print(f"Failed to stop '{args.name}'.", file=sys.stderr)
            sys.exit(1)


def _cmd_catalog_health(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        if args.name:
            ok, msg = api.check_catalog_provider(args.name)
            if args.json:
                _print_json({"name": args.name, "healthy": ok, "message": msg})
            else:
                status = "OK" if ok else "FAIL"
                print(f"[{status}] {args.name}: {msg}")
        else:
            entries = api.list_catalog()
            results = []
            for e in entries:
                ok, msg = api.check_catalog_provider(e.name)
                results.append({"name": e.name, "healthy": ok, "message": msg})
            if args.json:
                _print_json(results)
            else:
                for r in results:
                    status = "OK" if r["healthy"] else "FAIL"
                    print(f"[{status}] {r['name']}: {r['message']}")


def _cmd_catalog_discover(args: argparse.Namespace) -> None:
    api = _make_api(args)
    with api:
        if args.auto_register:
            count = api.auto_register_catalog_models(args.name)
            print(f"Discovered and registered {count} model(s) from '{args.name}'.")
        else:
            entries = api.list_catalog()
            matched = [e for e in entries if e.name == args.name]
            if matched:
                e = matched[0]
                print(f"Provider: {e.name} ({e.provider_type})")
                print(f"  Source:    {e.source}")
                print(f"  Installed: {e.installed}")
                print(f"  Running:   {e.running}")
                print(f"  Version:   {e.version}")
            else:
                print(f"Provider '{args.name}' not found in catalog.")


# -- catalog artifact commands -------------------------------------------

def _cmd_catalog_artifact_list(args: argparse.Namespace) -> None:
    """List unified catalog artifacts (model/service/analyzer)."""
    api = _make_api(args)
    with api:
        kind = getattr(args, "kind", None)
        artifacts = api.catalog_list(kind=kind)
        if args.json:
            _print_json(artifacts)
        else:
            if not artifacts:
                msg = f"No {kind} artifacts in catalog." if kind else "No artifacts in catalog."
                print(msg)
                return
            rows = [
                [
                    a.get("artifact_id", ""),
                    a.get("kind", "model"),
                    a.get("display_name", ""),
                    a.get("provider", ""),
                    a.get("status", "registered"),
                ]
                for a in artifacts
            ]
            _print_table(["ARTIFACT ID", "KIND", "DISPLAY NAME", "PROVIDER", "STATUS"], rows)
            print(f"\n{len(artifacts)} artifact(s).")


def _cmd_catalog_artifact_get(args: argparse.Namespace) -> None:
    """Get a single catalog artifact by ID."""
    api = _make_api(args)
    with api:
        entry = api.catalog_get(args.artifact_id)
        if entry is None:
            print(f"Artifact '{args.artifact_id}' not found.", file=sys.stderr)
            sys.exit(1)
        if args.json:
            entry["artifact_id"] = args.artifact_id
            _print_json(entry)
        else:
            print(f"Artifact: {args.artifact_id}")
            for k, v in entry.items():
                print(f"  {k}: {v}")


def _cmd_catalog_artifact_register(args: argparse.Namespace) -> None:
    """Register a new artifact in the unified catalog."""
    api = _make_api(args)
    with api:
        data: dict[str, Any] = {
            "kind": args.kind,
            "display_name": args.display_name,
        }
        if args.description:
            data["description"] = args.description
        if args.provider:
            data["provider"] = args.provider
        api.catalog_set(args.artifact_id, data)
        api.save_config()
        if args.json:
            data["artifact_id"] = args.artifact_id
            _print_json(data)
        else:
            print(f"Registered artifact '{args.artifact_id}' (kind={args.kind}).")


def _cmd_catalog_artifact_remove(args: argparse.Namespace) -> None:
    """Remove an artifact from the unified catalog."""
    api = _make_api(args)
    with api:
        if not getattr(args, "force", False):
            existing = api.catalog_get(args.artifact_id)
            if existing is None:
                print(f"Artifact '{args.artifact_id}' not found.", file=sys.stderr)
                sys.exit(1)
        removed = api.catalog_remove(args.artifact_id)
        if removed:
            api.save_config()
            print(f"Removed artifact '{args.artifact_id}'.")
        else:
            print(f"Artifact '{args.artifact_id}' not found.", file=sys.stderr)
            sys.exit(1)


# -- analyzer commands ---------------------------------------------------

def _cmd_analyzer_list(args: argparse.Namespace) -> None:
    """List analyzer artifacts (shortcut for catalog list --kind analyzer)."""
    api = _make_api(args)
    with api:
        artifacts = api.catalog_list(kind="analyzer")
        active_id = api.get_active_analyzer()
        if args.json:
            _print_json({"active": active_id, "analyzers": artifacts})
        else:
            if not artifacts:
                print("No analyzers in catalog.")
                return
            rows = [
                [
                    ("* " if a.get("artifact_id") == active_id else "  ") + a.get("artifact_id", ""),
                    a.get("display_name", ""),
                    a.get("provider", ""),
                    a.get("status", "registered"),
                ]
                for a in artifacts
            ]
            _print_table(["  ANALYZER ID", "DISPLAY NAME", "PROVIDER", "STATUS"], rows)
            if active_id:
                print(f"\nActive analyzer: {active_id}")
            else:
                print("\nNo active analyzer (using built-in default).")


def _cmd_analyzer_active(args: argparse.Namespace) -> None:
    """Show the currently active route analyzer."""
    api = _make_api(args)
    with api:
        active_id = api.get_active_analyzer()
        if args.json:
            _print_json({"active_analyzer": active_id})
        else:
            if active_id:
                print(f"Active analyzer: {active_id}")
                entry = api.catalog_get(active_id)
                if entry:
                    if entry.get("display_name"):
                        print(f"  Display name: {entry['display_name']}")
                    if entry.get("description"):
                        print(f"  Description:  {entry['description']}")
            else:
                print("No active analyzer set (using built-in default).")


def _cmd_analyzer_set(args: argparse.Namespace) -> None:
    """Set the active route analyzer."""
    api = _make_api(args)
    with api:
        # Verify the analyzer exists in catalog
        entry = api.catalog_get(args.analyzer_id)
        if entry is None:
            print(f"Warning: Analyzer '{args.analyzer_id}' not found in catalog.", file=sys.stderr)
        api.set_active_analyzer(args.analyzer_id)
        api.save_config()
        print(f"Active analyzer set to '{args.analyzer_id}'.")


def _cmd_analyzer_clear(args: argparse.Namespace) -> None:
    """Clear the active route analyzer (revert to built-in default)."""
    api = _make_api(args)
    with api:
        api.set_active_analyzer(None)
        api.save_config()
        print("Active analyzer cleared. Using built-in default.")


# ======================================================================
# Backward-compatible legacy subcommands
# ======================================================================

def _cmd_legacy_list_models(args: argparse.Namespace) -> None:
    """Legacy list-models: delegates to local file storage listing."""
    from aurarouter.models.file_storage import FileModelStorage

    storage = FileModelStorage(getattr(args, "dir", None))
    storage.scan()
    models = storage.list_models()

    if not models:
        print("No models found in:", storage.models_dir)
        print("Download one with:  aurarouter download-model --repo <repo> --file <name>")
        return

    print(f"Models in {storage.models_dir}:\n")
    for m in models:
        size_mb = m.get("size_bytes", 0) / (1024 * 1024)
        repo = m.get("repo", "unknown")
        print(f"  {m['filename']}  ({size_mb:.0f} MB)  repo: {repo}")
    print(f"\n{len(models)} model(s) total.")


def _cmd_legacy_remove_model(args: argparse.Namespace) -> None:
    """Legacy remove-model: delegates to local file storage."""
    from aurarouter.models.file_storage import FileModelStorage

    storage = FileModelStorage(getattr(args, "dir", None))
    delete_file = not args.keep_file
    removed = storage.remove(args.file, delete_file=delete_file)

    if removed:
        action = "Removed and deleted" if delete_file else "Unregistered"
        print(f"{action}: {args.file}")
    else:
        print(f"Model not found in registry: {args.file}")
        sys.exit(1)


def _cmd_legacy_download_model(args: argparse.Namespace) -> None:
    """Legacy download-model subcommand."""
    try:
        from aurarouter.models.downloader import download_model
    except ImportError:
        print(
            "huggingface-hub is required for model downloading.\n"
            "Install with:  pip install aurarouter[local]"
        )
        sys.exit(1)

    download_model(repo=args.repo, filename=args.file, dest=args.dest)


# ======================================================================
# Parser construction
# ======================================================================

def _build_parser() -> argparse.ArgumentParser:
    """Build the full argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="aurarouter",
        description="AuraRouter — Multi-model MCP routing fabric.",
    )
    parser.add_argument(
        "--config",
        help="Path to auraconfig.yaml. Falls back to AURACORE_ROUTER_CONFIG env var, "
        "then ~/.auracore/aurarouter/auraconfig.yaml.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output in JSON format.",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Run interactive installer for all supported MCP clients.",
    )
    parser.add_argument(
        "--install-gemini",
        action="store_true",
        help="Register AuraRouter for the Gemini CLI.",
    )

    subparsers = parser.add_subparsers(dest="command")

    # ---- model ----
    model_parser = subparsers.add_parser("model", help="Model management commands.")
    model_sub = model_parser.add_subparsers(dest="model_command")

    # model list
    ml = model_sub.add_parser("list", help="List configured models.")
    ml.add_argument("--json", action="store_true", default=False)
    ml.add_argument("--filter", help="Filter models by ID or provider substring.")

    # model add
    ma = model_sub.add_parser("add", help="Add a new model.")
    ma.add_argument("model_id", help="Unique model identifier.")
    ma.add_argument("--provider", required=True, help="Provider name.")
    ma.add_argument("--endpoint", help="Provider endpoint URL.")
    ma.add_argument("--model-name", help="Provider-specific model name.")
    ma.add_argument("--api-key", help="API key for the provider.")
    ma.add_argument("--tags", help="Comma-separated tags.")
    ma.add_argument("--tier", help="Hosting tier (local, cloud, grid).")
    ma.add_argument("--json", action="store_true", default=False)

    # model edit
    me = model_sub.add_parser("edit", help="Edit an existing model.")
    me.add_argument("model_id", help="Model identifier to edit.")
    me.add_argument("--provider", help="Provider name.")
    me.add_argument("--endpoint", help="Provider endpoint URL.")
    me.add_argument("--model-name", help="Provider-specific model name.")
    me.add_argument("--api-key", help="API key for the provider.")
    me.add_argument("--tags", help="Comma-separated tags.")
    me.add_argument("--tier", help="Hosting tier.")
    me.add_argument("--json", action="store_true", default=False)

    # model remove
    mr = model_sub.add_parser("remove", help="Remove a model.")
    mr.add_argument("model_id", help="Model identifier to remove.")
    mr.add_argument("--force", action="store_true", help="Skip existence check.")
    mr.add_argument("--json", action="store_true", default=False)

    # model test
    mt = model_sub.add_parser("test", help="Test model connectivity.")
    mt.add_argument("model_id", help="Model to test.")
    mt.add_argument("--json", action="store_true", default=False)

    # model auto-tune
    mat = model_sub.add_parser("auto-tune", help="Auto-tune model parameters.")
    mat.add_argument("model_id", help="Model to auto-tune.")
    mat.add_argument("--json", action="store_true", default=False)

    # ---- route ----
    route_parser = subparsers.add_parser("route", help="Routing role management.")
    route_sub = route_parser.add_subparsers(dest="route_command")

    # route list
    rl = route_sub.add_parser("list", help="List all routing roles.")
    rl.add_argument("--json", action="store_true", default=False)

    # route set
    rs = route_sub.add_parser("set", help="Set a role's model chain.")
    rs.add_argument("role", help="Role name.")
    rs.add_argument("model_ids", nargs="+", help="Ordered model IDs.")
    rs.add_argument("--json", action="store_true", default=False)

    # route append
    ra = route_sub.add_parser("append", help="Append a model to a role's chain.")
    ra.add_argument("role", help="Role name.")
    ra.add_argument("model_id", help="Model ID to append.")
    ra.add_argument("--json", action="store_true", default=False)

    # route remove-model
    rrm = route_sub.add_parser("remove-model", help="Remove a model from a role's chain.")
    rrm.add_argument("role", help="Role name.")
    rrm.add_argument("model_id", help="Model ID to remove.")
    rrm.add_argument("--json", action="store_true", default=False)

    # route delete
    rd = route_sub.add_parser("delete", help="Delete an entire role.")
    rd.add_argument("role", help="Role to delete.")
    rd.add_argument("--force", action="store_true", help="Skip existence check.")
    rd.add_argument("--json", action="store_true", default=False)

    # ---- run ----
    run_parser = subparsers.add_parser("run", help="Execute a task through the routing loop.")
    run_parser.add_argument("task", help="Task description.")
    run_parser.add_argument("--context", help="Path to context file.")
    run_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format.")
    run_parser.add_argument("--local-only", action="store_true", help="Use local models only.")
    run_parser.add_argument("--no-review", action="store_true", help="Skip review step.")
    run_parser.add_argument("--json", action="store_true", default=False)

    # ---- compare ----
    cmp_parser = subparsers.add_parser("compare", help="Compare output across models.")
    cmp_parser.add_argument("prompt", help="Prompt to send to each model.")
    cmp_parser.add_argument("--models", required=True, help="Comma-separated model IDs.")
    cmp_parser.add_argument("--json", action="store_true", default=False)

    # ---- traffic ----
    traffic_parser = subparsers.add_parser("traffic", help="Show traffic and usage statistics.")
    traffic_parser.add_argument("--range", help="Time range (e.g. 24h, 7d, 30d).")
    traffic_parser.add_argument("--json", action="store_true", default=False)

    # ---- privacy ----
    privacy_parser = subparsers.add_parser("privacy", help="Show privacy audit events.")
    privacy_parser.add_argument("--range", help="Time range (e.g. 24h, 7d).")
    privacy_parser.add_argument("--severity", choices=["low", "medium", "high"], help="Minimum severity.")
    privacy_parser.add_argument("--json", action="store_true", default=False)

    # ---- health ----
    health_parser = subparsers.add_parser("health", help="Check model health.")
    health_parser.add_argument("model_id", nargs="?", default=None, help="Specific model to check.")
    health_parser.add_argument("--json", action="store_true", default=False)

    # ---- budget ----
    budget_parser = subparsers.add_parser("budget", help="Show budget status.")
    budget_parser.add_argument("--json", action="store_true", default=False)

    # ---- config ----
    config_parser = subparsers.add_parser("config", help="Configuration management.")
    config_sub = config_parser.add_subparsers(dest="config_command")

    cs = config_sub.add_parser("show", help="Show current configuration.")
    cs.add_argument("--json", action="store_true", default=False)

    cset = config_sub.add_parser("set", help="Set a configuration value.")
    cset.add_argument("key", help="Dotted config key (e.g. logging.level).")
    cset.add_argument("value", help="Value to set.")
    cset.add_argument("--json", action="store_true", default=False)

    cmcp = config_sub.add_parser("mcp-tool", help="Enable or disable an MCP tool.")
    cmcp.add_argument("tool", help="Tool name.")
    cmcp_group = cmcp.add_mutually_exclusive_group(required=True)
    cmcp_group.add_argument("--enable", action="store_true", dest="enable")
    cmcp_group.add_argument("--disable", action="store_false", dest="enable")
    cmcp.add_argument("--json", action="store_true", default=False)

    config_sub.add_parser("save", help="Save current configuration to disk.")
    config_sub.add_parser("reload", help="Reload configuration from disk.")

    # ---- catalog ----
    catalog_parser = subparsers.add_parser("catalog", help="Provider catalog management.")
    catalog_sub = catalog_parser.add_subparsers(dest="catalog_command")

    cl = catalog_sub.add_parser("list", help="List catalog providers.")
    cl.add_argument("--json", action="store_true", default=False)

    ca = catalog_sub.add_parser("add", help="Add a manual provider.")
    ca.add_argument("name", help="Provider name.")
    ca.add_argument("--endpoint", required=True, help="Endpoint URL.")
    ca.add_argument("--json", action="store_true", default=False)

    cr = catalog_sub.add_parser("remove", help="Remove a provider.")
    cr.add_argument("name", help="Provider name.")

    cstart = catalog_sub.add_parser("start", help="Start a provider.")
    cstart.add_argument("name", help="Provider name.")

    cstop = catalog_sub.add_parser("stop", help="Stop a provider.")
    cstop.add_argument("name", help="Provider name.")

    ch = catalog_sub.add_parser("health", help="Check provider health.")
    ch.add_argument("name", nargs="?", default=None, help="Specific provider (or all).")
    ch.add_argument("--json", action="store_true", default=False)

    cd = catalog_sub.add_parser("discover", help="Discover models from a provider.")
    cd.add_argument("name", help="Provider name.")
    cd.add_argument("--auto-register", action="store_true",
                    help="Automatically register discovered models.")
    cd.add_argument("--json", action="store_true", default=False)

    # catalog artifact commands (unified catalog)
    cal = catalog_sub.add_parser("artifacts", help="List unified catalog artifacts.")
    cal.add_argument("--kind", choices=["model", "service", "analyzer"],
                     default=None, help="Filter by artifact kind.")
    cal.add_argument("--json", action="store_true", default=False)

    cag = catalog_sub.add_parser("get", help="Get a catalog artifact by ID.")
    cag.add_argument("artifact_id", help="Artifact identifier.")
    cag.add_argument("--json", action="store_true", default=False)

    careg = catalog_sub.add_parser("register", help="Register an artifact in the unified catalog.")
    careg.add_argument("artifact_id", help="Unique artifact identifier.")
    careg.add_argument("--kind", required=True, choices=["model", "service", "analyzer"],
                       help="Artifact kind.")
    careg.add_argument("--display-name", required=True, help="Human-readable name.")
    careg.add_argument("--description", default=None, help="Optional description.")
    careg.add_argument("--provider", default=None, help="Provider or origin identifier.")
    careg.add_argument("--json", action="store_true", default=False)

    cadel = catalog_sub.add_parser("unregister", help="Remove an artifact from the unified catalog.")
    cadel.add_argument("artifact_id", help="Artifact identifier to remove.")
    cadel.add_argument("--force", action="store_true", help="Skip existence check.")
    cadel.add_argument("--json", action="store_true", default=False)

    # ---- analyzer ----
    analyzer_parser = subparsers.add_parser("analyzer", help="Route analyzer management.")
    analyzer_sub = analyzer_parser.add_subparsers(dest="analyzer_command")

    anl = analyzer_sub.add_parser("list", help="List analyzer artifacts.")
    anl.add_argument("--json", action="store_true", default=False)

    ana = analyzer_sub.add_parser("active", help="Show the active route analyzer.")
    ana.add_argument("--json", action="store_true", default=False)

    ans = analyzer_sub.add_parser("set", help="Set the active route analyzer.")
    ans.add_argument("analyzer_id", help="Analyzer artifact ID to activate.")
    ans.add_argument("--json", action="store_true", default=False)

    anc = analyzer_sub.add_parser("clear", help="Clear the active analyzer (use built-in).")
    anc.add_argument("--json", action="store_true", default=False)

    # ---- migrate-config ----
    mig_parser = subparsers.add_parser(
        "migrate-config",
        help="Migrate an old config to the current format (adds catalog section, etc.).",
    )
    mig_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing.",
    )

    # ---- GUI subcommand ----
    gui_parser = subparsers.add_parser("gui", help="Launch the AuraRouter GUI.")
    gui_parser.add_argument(
        "--environment",
        choices=["local", "auragrid"],
        default=None,
        help="Deployment environment to start in (default: local).",
    )

    # ---- Legacy subcommands (backward compat) ----
    dl = subparsers.add_parser(
        "download-model",
        help="Download a GGUF model from HuggingFace Hub.",
    )
    dl.add_argument("--repo", required=True, help="HuggingFace repo ID")
    dl.add_argument("--file", required=True, help="GGUF filename in the repo")
    dl.add_argument("--dest", default=None, help="Destination directory.")

    lm = subparsers.add_parser(
        "list-models",
        help="List locally downloaded GGUF models.",
    )
    lm.add_argument("--dir", default=None, help="Model storage directory.")

    rm = subparsers.add_parser(
        "remove-model",
        help="Remove a downloaded GGUF model.",
    )
    rm.add_argument("--file", required=True, help="GGUF filename to remove")
    rm.add_argument("--keep-file", action="store_true")
    rm.add_argument("--dir", default=None)

    return parser


# ======================================================================
# Dispatch
# ======================================================================

_MODEL_DISPATCH = {
    "list": _cmd_model_list,
    "add": _cmd_model_add,
    "edit": _cmd_model_edit,
    "remove": _cmd_model_remove,
    "test": _cmd_model_test,
    "auto-tune": _cmd_model_auto_tune,
}

_ROUTE_DISPATCH = {
    "list": _cmd_route_list,
    "set": _cmd_route_set,
    "append": _cmd_route_append,
    "remove-model": _cmd_route_remove_model,
    "delete": _cmd_route_delete,
}

_CONFIG_DISPATCH = {
    "show": _cmd_config_show,
    "set": _cmd_config_set,
    "mcp-tool": _cmd_config_mcp_tool,
    "save": _cmd_config_save,
    "reload": _cmd_config_reload,
}

_CATALOG_DISPATCH = {
    "list": _cmd_catalog_list,
    "add": _cmd_catalog_add,
    "remove": _cmd_catalog_remove,
    "start": _cmd_catalog_start,
    "stop": _cmd_catalog_stop,
    "health": _cmd_catalog_health,
    "discover": _cmd_catalog_discover,
    # Unified artifact catalog commands
    "artifacts": _cmd_catalog_artifact_list,
    "get": _cmd_catalog_artifact_get,
    "register": _cmd_catalog_artifact_register,
    "unregister": _cmd_catalog_artifact_remove,
}

_ANALYZER_DISPATCH = {
    "list": _cmd_analyzer_list,
    "active": _cmd_analyzer_active,
    "set": _cmd_analyzer_set,
    "clear": _cmd_analyzer_clear,
}


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args()

    # ---- Install mode (no config needed) ----
    is_install = args.install or args.install_gemini

    if is_install:
        from aurarouter.installers.template import create_config_template

        create_config_template()

        if args.install:
            from aurarouter.installers.registry import install_all

            install_all()
        elif args.install_gemini:
            from aurarouter.installers.gemini import GeminiInstaller

            GeminiInstaller().install()
        return

    # ---- migrate-config ----
    if args.command == "migrate-config":
        from aurarouter.migration import migrate_config_file

        config_path = args.config
        if not config_path:
            from aurarouter.config import _default_config_path
            config_path = str(_default_config_path())
        dry_run = getattr(args, "dry_run", False)
        report = migrate_config_file(config_path, dry_run=dry_run)
        for line in report:
            print(line)
        return

    # ---- Legacy subcommands ----
    if args.command == "download-model":
        _cmd_legacy_download_model(args)
        return
    if args.command == "list-models":
        _cmd_legacy_list_models(args)
        return
    if args.command == "remove-model":
        _cmd_legacy_remove_model(args)
        return

    # ---- GUI subcommand ----
    if args.command == "gui":
        from aurarouter.gui.app import _create_context, launch_gui

        environment = getattr(args, "environment", None)
        context = _create_context(
            environment=environment,
            config_path=args.config,
        )
        launch_gui(context)
        return

    # ---- model subcommand group ----
    if args.command == "model":
        sub = getattr(args, "model_command", None)
        handler = _MODEL_DISPATCH.get(sub)
        if handler:
            handler(args)
        else:
            parser.parse_args(["model", "--help"])
        return

    # ---- route subcommand group ----
    if args.command == "route":
        sub = getattr(args, "route_command", None)
        handler = _ROUTE_DISPATCH.get(sub)
        if handler:
            handler(args)
        else:
            parser.parse_args(["route", "--help"])
        return

    # ---- run ----
    if args.command == "run":
        _cmd_run(args)
        return

    # ---- compare ----
    if args.command == "compare":
        _cmd_compare(args)
        return

    # ---- Monitoring commands ----
    if args.command == "traffic":
        _cmd_traffic(args)
        return
    if args.command == "privacy":
        _cmd_privacy(args)
        return
    if args.command == "health":
        _cmd_health(args)
        return
    if args.command == "budget":
        _cmd_budget(args)
        return

    # ---- config subcommand group ----
    if args.command == "config":
        sub = getattr(args, "config_command", None)
        handler = _CONFIG_DISPATCH.get(sub)
        if handler:
            handler(args)
        else:
            parser.parse_args(["config", "--help"])
        return

    # ---- catalog subcommand group ----
    if args.command == "catalog":
        sub = getattr(args, "catalog_command", None)
        handler = _CATALOG_DISPATCH.get(sub)
        if handler:
            handler(args)
        else:
            parser.parse_args(["catalog", "--help"])
        return

    # ---- analyzer subcommand group ----
    if args.command == "analyzer":
        sub = getattr(args, "analyzer_command", None)
        handler = _ANALYZER_DISPATCH.get(sub)
        if handler:
            handler(args)
        else:
            parser.parse_args(["analyzer", "--help"])
        return

    # ---- Default: run MCP server ----
    from aurarouter.config import ConfigLoader
    from aurarouter.server import create_mcp_server

    try:
        config = ConfigLoader(config_path=args.config)
    except FileNotFoundError:
        print(
            "No configuration file found.\n"
            "Run 'aurarouter --install' to create one, or\n"
            "run 'aurarouter gui' to configure interactively."
        )
        sys.exit(1)

    mcp = create_mcp_server(config)
    mcp.run()
