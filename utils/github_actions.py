import base64

import requests
import streamlit as st


GITHUB_API_BASE = "https://api.github.com"


def get_github_config():
    owner = st.secrets.get("GITHUB_OWNER", "")
    repo = st.secrets.get("GITHUB_REPO", "")
    token = st.secrets.get("GITHUB_TOKEN", "")
    branch = st.secrets.get("GITHUB_BRANCH", "main")

    return owner, repo, token, branch


def github_headers(token):
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def trigger_workflow(workflow_file, inputs=None):
    owner, repo, token, branch = get_github_config()

    if not owner or not repo or not token:
        return False, "Missing GitHub Streamlit secrets."

    url = (
        f"{GITHUB_API_BASE}/repos/"
        f"{owner}/{repo}/actions/workflows/"
        f"{workflow_file}/dispatches"
    )

    payload = {
        "ref": branch,
        "inputs": inputs or {},
    }

    response = requests.post(
        url,
        headers=github_headers(token),
        json=payload,
        timeout=20,
    )

    if response.status_code in [200, 201, 202, 204]:
        return True, f"Workflow launched: {workflow_file}"

    return False, (
        f"GitHub API error {response.status_code}: "
        f"{response.text}"
    )


def get_workflow_runs(workflow_file, branch=None, per_page=10):
    owner, repo, token, default_branch = get_github_config()

    if not owner or not repo or not token:
        return False, "Missing GitHub Streamlit secrets.", []

    branch = branch or default_branch

    url = (
        f"{GITHUB_API_BASE}/repos/"
        f"{owner}/{repo}/actions/workflows/"
        f"{workflow_file}/runs"
    )

    response = requests.get(
        url,
        headers=github_headers(token),
        params={
            "branch": branch,
            "event": "workflow_dispatch",
            "per_page": per_page,
        },
        timeout=20,
    )

    if response.status_code != 200:
        return False, (
            f"GitHub API error {response.status_code}: "
            f"{response.text}"
        ), []

    return True, "Workflow runs loaded.", response.json().get("workflow_runs", [])


def get_latest_workflow_run(workflow_file, branch=None):
    ok, msg, runs = get_workflow_runs(workflow_file, branch=branch, per_page=10)

    if not ok:
        return False, msg, None

    if not runs:
        return True, "No workflow runs found.", None

    return True, "Latest workflow run loaded.", runs[0]


def read_repo_text_file(path, branch=None):
    owner, repo, token, default_branch = get_github_config()
    if not owner or not repo or not token:
        return False, "Missing GitHub Streamlit secrets.", None, None

    branch = branch or default_branch
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(
        url,
        headers=github_headers(token),
        params={"ref": branch},
        timeout=20,
    )
    if response.status_code != 200:
        return False, f"GitHub API error {response.status_code}: {response.text}", None, None

    payload = response.json()
    encoded = payload.get("content") or ""
    try:
        text = base64.b64decode(encoded).decode("utf-8")
    except Exception as exc:
        return False, f"Could not decode repo file: {exc}", None, None
    return True, "Repo file loaded.", text, payload.get("sha")


def write_repo_text_file(path, content, message, branch=None, sha=None):
    owner, repo, token, default_branch = get_github_config()
    if not owner or not repo or not token:
        return False, "Missing GitHub Streamlit secrets."

    branch = branch or default_branch
    if sha is None:
        ok, msg, _text, fetched_sha = read_repo_text_file(path, branch=branch)
        if ok:
            sha = fetched_sha

    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}"
    payload = {
        "message": message,
        "content": base64.b64encode(str(content).encode("utf-8")).decode("ascii"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    response = requests.put(
        url,
        headers=github_headers(token),
        json=payload,
        timeout=30,
    )
    if response.status_code in [200, 201]:
        return True, f"Repo file saved: {path}"
    return False, f"GitHub API error {response.status_code}: {response.text}"
