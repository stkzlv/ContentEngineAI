# tools/get_freesound_token.py
import argparse
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# Add project root to path to allow importing from src
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.video.video_config import load_video_config  # noqa: E402


# --- Script ---
def get_new_token():
    """Guide the user through OAuth2 process to get new Freesound refresh token."""
    print("--- Freesound OAuth2 Refresh Token Generator ---")

    # 1. Load config to find the environment variable names
    try:
        env_path = project_root / ".env"
        if not env_path.exists():
            print(f"\nERROR: .env file not found at {env_path}")
            print("Please ensure your .env file exists in the project root.")
            sys.exit(1)

        load_dotenv(dotenv_path=env_path)
        print(f"Loaded environment variables from: {env_path}")

        config_path = project_root / "config" / "video_producer.yaml"
        video_config = load_video_config(config_path)

        client_id_var = video_config.audio_settings.freesound_client_id_env_var
        client_secret_var = video_config.audio_settings.freesound_client_secret_env_var

        client_id = os.getenv(client_id_var)
        client_secret = os.getenv(client_secret_var)

        if not client_id or not client_secret:
            print("\nERROR: Missing Freesound credentials in your .env file.")
            print(f"Ensure '{client_id_var}' and '{client_secret_var}' are set.")
            sys.exit(1)

    except Exception as e:
        print(f"\nERROR: Failed to load configuration: {e}")
        sys.exit(1)

    # 2. Construct the authorization URL
    auth_url = f"https://freesound.org/apiv2/oauth2/authorize/?client_id={client_id}&response_type=code"

    print("\nStep 1: Authorize the Application")
    print(
        "Please visit the following URL in your browser, log in, and "
        "authorize the app:"
    )
    print(f"\n  {auth_url}\n")
    print(
        "After authorizing, you will be redirected. Copy the value of the "
        "'code' parameter from the URL in your browser's address bar."
    )
    print("It will look like: http://localhost/oauth2/callback?code=THIS_IS_THE_CODE")

    # 3. Get the authorization code from the user
    auth_code = input("\nEnter the authorization code here: ").strip()

    if not auth_code:
        print("Authorization code cannot be empty. Exiting.")
        sys.exit(1)

    # 4. Exchange the code for a refresh token
    token_url = "https://freesound.org/apiv2/oauth2/access_token/"  # noqa: S105
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "authorization_code",
        "code": auth_code,
    }

    print("\nStep 2: Exchanging code for refresh token...")
    try:
        response = requests.post(token_url, data=payload, timeout=30)
        response.raise_for_status()
        token_data = response.json()

        refresh_token = token_data.get("refresh_token")

        if not refresh_token:
            print("\nERROR: Could not retrieve refresh token from Freesound.")
            print(f"Response data: {token_data}")
            sys.exit(1)

        print("\n--- SUCCESS! ---")
        print("Your new Freesound refresh token is:")
        print(f"\n  {refresh_token}\n")
        print("Copy this value and paste it into your .env file for the")
        print("'FREESOUND_REFRESH_TOKEN' variable.")

    except requests.exceptions.HTTPError as e:
        print(
            f"\nERROR: HTTP Error {e.response.status_code} when contacting "
            f"Freesound."
        )
        print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A helper script to generate a Freesound OAuth2 refresh token."
    )
    parser.parse_args()  # This is just to provide -h functionality
    get_new_token()
