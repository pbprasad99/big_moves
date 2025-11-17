"""Cross-platform config helpers for Big Moves.

Provides functions to determine candidate locations for a per-user .env file and to load
environment variables from those files. Uses platformdirs when available; otherwise falls
back to sensible platform defaults.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def get_candidate_env_paths(app_name: str = "big-moves") -> List[Path]:
    """Return a list of candidate .env file paths in search order (most specific first).

    Order:
      1. Project working directory: ./ .env (useful for dev)
      2. $XDG_CONFIG_HOME/<app_name>/.env or ~/.config/<app_name>/.env (Unix/mac)
      3. Windows %APPDATA%/LocalAppData/<app_name>/.env
      4. Per-user fallback: ~/.<app_name>.env
    """
    candidates: List[Path] = []

    # 1) project-local .env
    cwd_env = Path.cwd() / ".env"
    candidates.append(cwd_env)

    # Try to use platformdirs if present for niceties
    try:
        from platformdirs import user_config_dir

        cfg_dir = Path(user_config_dir(app_name))
        candidates.append(cfg_dir / ".env")
    except Exception:
        # fallback: XDG or ~/.config
        if os.name == 'nt':
            appdata = os.getenv('LOCALAPPDATA') or os.getenv('APPDATA')
            if appdata:
                candidates.append(Path(appdata) / app_name / '.env')
        else:
            xdg = os.getenv('XDG_CONFIG_HOME')
            if xdg:
                candidates.append(Path(xdg) / app_name / '.env')
            else:
                candidates.append(Path.home() / '.config' / app_name / '.env')

    # 4) per-user hidden file fallback
    candidates.append(Path.home() / f".{app_name}.env")

    return candidates


def _parse_and_set_env(path: Path) -> None:
    """Parse a simple KEY=VALUE file and set os.environ for any missing keys.
    This is a minimal parser only intended for our .env format (no quoting support beyond simple stripping).
    """
    text = path.read_text(encoding='utf8')
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        # do not overwrite existing env vars
        if k not in os.environ or not os.environ.get(k):
            os.environ[k] = v


def load_env_from_paths(paths: List[Path]) -> Optional[Path]:
    """Load the first existing env file from the provided candidate paths.

    Returns the Path loaded, or None if none found.
    """
    # Try python-dotenv first (nicer parsing), fall back to simple parser
    try:
        from dotenv import load_dotenv
        for p in paths:
            try:
                if p.exists():
                    load_dotenv(p)
                    logger.debug('Loaded env from %s via python-dotenv', p)
                    return p
            except Exception:
                logger.exception('Failed loading %s with python-dotenv', p)
    except Exception:
        # python-dotenv not available; continue to fallback parser
        pass

    for p in paths:
        try:
            if p.exists():
                _parse_and_set_env(p)
                logger.debug('Loaded env from %s via simple parser', p)
                return p
        except Exception:
            logger.exception('Failed parsing %s', p)

    return None


def find_and_load_env(app_name: str = "big-moves") -> (List[Path], Optional[Path]):
    """Return (candidate_paths, loaded_path).

    Candidate paths are returned for diagnostics; loaded_path is the actual file that was loaded (or None).
    """
    candidates = get_candidate_env_paths(app_name=app_name)
    loaded = load_env_from_paths(candidates)
    return candidates, loaded
