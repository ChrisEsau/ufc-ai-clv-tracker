import requests
import streamlit as st


GITHUB_API_BASE = "https://api.github.com"


def get_github_config():
    owner = st.secrets.get("GITHUB_OWNER", "")
    repo = st.secrets.get("GITHUB_REPO", "")
    token = st.secrets.get("GITHUB_TOKEN", "")
    branch = st.secrets.get("GITHUB_BRANCH", "main")

    return owner, repo, token, branch


def trigger_workflow(workflow_file):
    owner, repo, token, branch = get_github_config()

    if not owner or not repo or not token:
        return False, "Missing GitHub Streamlit secrets."

    url = (
        f"{GITHUB_API_BASE}/repos/"
        f"{owner}/{repo}/actions/workflows/"
        f"{workflow_file}/dispatches"
    )

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    payload = {
        "ref": branch,
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=20,
    )

    if response.status_code in [200, 201, 202, 204]:
        return True, f"Workflow launched: {workflow_file}"

    return False, (
        f"GitHub API error {response.status_code}: "
        f"{response.text}"
    )