#!/usr/bin/env python3
r"""Standalone Freesound OAuth2 refresh token generator.

This script helps you obtain a refresh token for Freesound OAuth2 authentication.
It does not require the full application configuration to run.

Usage:
    poetry run python tools/freesound_oauth2_setup.py \
        --client-id YOUR_CLIENT_ID \
        --client-secret YOUR_CLIENT_SECRET

The script will:
1. Print an authorization URL
2. You visit the URL and authorize the app
3. Copy the authorization code from the redirect URL
4. Paste it into the script
5. Script exchanges code for refresh token
6. Automatically update FREESOUND_REFRESH_TOKEN in your .env file
"""

import argparse
import sys
from pathlib import Path

import requests
from dotenv import set_key


def get_freesound_refresh_token(client_id: str, client_secret: str) -> str | None:
    """Guide user through OAuth2 process to get Freesound refresh token.

    Args:
    ----
        client_id: Freesound OAuth2 client ID
        client_secret: Freesound OAuth2 client secret

    Returns:
    -------
        Refresh token string if successful, None otherwise

    """
    print("\n=== Freesound OAuth2 Refresh Token Generator ===\n")

    # Construct authorization URL
    auth_url = (
        f"https://freesound.org/apiv2/oauth2/authorize/"
        f"?client_id={client_id}&response_type=code"
    )

    print("Step 1: Authorize the Application")
    print("-" * 50)
    print(
        "Visit the following URL in your browser, log in to Freesound, "
        "and authorize the app:\n"
    )
    print(f"  {auth_url}\n")
    print(
        "After authorizing, you will be redirected to a URL that looks like:\n"
        "  http://localhost/oauth2/callback?code=THIS_IS_THE_CODE\n"
    )
    print("Copy the value of the 'code' parameter from the redirect URL.")

    # Get authorization code from user
    auth_code = input("\nEnter the authorization code here: ").strip()

    if not auth_code:
        print("ERROR: Authorization code cannot be empty.")
        return None

    # Exchange code for tokens
    print("\nStep 2: Exchanging authorization code for tokens...")
    print("-" * 50)

    token_url = "https://freesound.org/apiv2/oauth2/access_token/"  # noqa: S105
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "authorization_code",
        "code": auth_code,
    }

    try:
        response = requests.post(token_url, data=payload, timeout=30)
        response.raise_for_status()
        token_data = response.json()

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in")

        if not refresh_token:
            print("\nERROR: Could not retrieve refresh token from Freesound.")
            print(f"Response data: {token_data}")
            return None

        print("\n✓ SUCCESS! Tokens obtained from Freesound.\n")
        print(f"Access Token (expires in {expires_in}s):")
        print(f"  {access_token}\n")
        print("Refresh Token:")
        print(f"  {refresh_token}\n")
        print("-" * 50)

        # Auto-update .env file
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            try:
                set_key(
                    str(env_path),
                    "FREESOUND_REFRESH_TOKEN",
                    refresh_token,
                    quote_mode="never",
                )
                print(f"\n✓ Automatically updated {env_path.name}")
                print("  FREESOUND_REFRESH_TOKEN has been saved.")
            except Exception as e:
                print(f"\n⚠ Could not auto-update .env file: {e}")
                print("\nManual update required:")
                print(f"  FREESOUND_REFRESH_TOKEN={refresh_token}")
        else:
            print(f"\n⚠ .env file not found at: {env_path}")
            print("\nManual setup required:")
            print("1. Create a .env file in the project root")
            print(f"2. Add: FREESOUND_REFRESH_TOKEN={refresh_token}")

        print("\nThe system will automatically refresh access tokens as needed.")

        return str(refresh_token)

    except requests.exceptions.HTTPError as e:
        print(f"\nERROR: HTTP {e.response.status_code} from Freesound API")
        print(f"Response: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"\nERROR: Request failed: {e}")
        return None
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}")
        return None


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate Freesound OAuth2 refresh token",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  poetry run python tools/freesound_oauth2_setup.py \\
    --client-id YOUR_CLIENT_ID \\
    --client-secret YOUR_CLIENT_SECRET

For more information on Freesound OAuth2:
  https://freesound.org/docs/api/authentication.html
        """,
    )

    parser.add_argument(
        "--client-id",
        required=True,
        help="Freesound OAuth2 client ID (from app dashboard)",
    )
    parser.add_argument(
        "--client-secret",
        required=True,
        help="Freesound OAuth2 client secret (from app dashboard)",
    )

    args = parser.parse_args()

    # Run OAuth2 flow
    refresh_token = get_freesound_refresh_token(
        client_id=args.client_id, client_secret=args.client_secret
    )

    if refresh_token:
        sys.exit(0)
    else:
        print("\nFailed to obtain refresh token.")
        sys.exit(1)


if __name__ == "__main__":
    main()
