"""
config.py
---------
Single source of truth for reading MODEL_PATH / DATA_SET.

Why this exists:
- Locally, values come from a `.env` file (via python-dotenv).
- On Streamlit Community Cloud, `.env` files are not deployed; secrets are
  set in the app's dashboard and read via `st.secrets`.
- This helper tries `.env` / real environment variables first (works for
  training.py and prediction.py, which are plain Python, not Streamlit),
  and falls back to `st.secrets` only if running inside Streamlit and the
  env var isn't set. This avoids the crash you hit (StreamlitSecretNotFoundError)
  when no secrets.toml exists locally.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # loads variables from a local .env file, if present


def get_config(key: str, default: str | None = None) -> str:
    # 1. Real environment variable / .env file
    value = os.getenv(key)
    if value:
        return value

    # 2. Streamlit secrets (only available when running under `streamlit run`,
    #    and only if a secrets.toml/dashboard secret actually exists)
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass  # not running under Streamlit, or no secrets configured

    if default is not None:
        return default

    raise KeyError(
        f"'{key}' not found in .env, environment variables, or st.secrets. "
        f"Add it to your .env file, e.g.: {key}=your_value"
    )
