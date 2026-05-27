"""Build MoT-Voellmy binary for the current platform.

Usage:
    python _buildMoTVoellmy.py [path/to/source.c]

If a path is provided, that C source file is used. Otherwise, the script tries:
1. MOT_VOELLMY_SOURCE environment variable
2. GitHub API to discover and download the latest source from the upstream repo

On success, the compiled binary is written to this directory alongside the script.
On failure, the script exits with a non-zero code (for pixi task / CI).
When imported from setup.py, it returns a status instead of exiting.
"""

import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request

upstreamRepo = "norwegian-geotechnical-institute/MoT-Voellmy"
upstreamApi = f"https://api.github.com/repos/{upstreamRepo}/contents/"
sourcePattern = re.compile(r"^MoT-Voellmy\..*\.c$")

outputDir = os.path.dirname(os.path.abspath(__file__))


def _findGithubSource():
    """Query GitHub contents API and return the download_url of the .c file.

    Returns None if no matching file is found or the API request fails.
    """
    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = urllib.request.Request(upstreamApi, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            import json

            contents = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Failed to query GitHub API: {e}", file=sys.stderr)
        return None

    for item in contents:
        if item.get("type") != "file":
            continue
        name = item.get("name", "")
        if sourcePattern.match(name):
            url = item.get("download_url")
            if url:
                print(f"Found upstream source: {name}")
                return url, name

    print("No MoT-Voellmy source file found in upstream repo", file=sys.stderr)
    return None


def _downloadSource(url, destPath):
    """Download a file from url to destPath."""
    print(f"Downloading {url} ...")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            with open(destPath, "wb") as f:
                shutil.copyfileobj(resp, f)
        print(f"Downloaded to {destPath}")
        return True
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return False


def _compile(sourcePath):
    """Compile sourcePath for the current platform.

    Returns True on success, False on failure.
    """
    system = platform.system()

    if system == "Linux":
        outName = "MoT-Voellmy_linux.exe"
        cmd = ["gcc", "-Wall", "-pedantic", "-static", "-o", outName, sourcePath, "-lm"]
    elif system == "Windows":
        outName = "MoT-Voellmy_win.exe"
        cmd = ["gcc", "-Wall", "-pedantic", "-o", outName, sourcePath, "-lm"]
    elif system == "Darwin":
        outName = "MoT-Voellmy_mac.exe"
        cmd = ["gcc", "-Wall", "-pedantic", "-o", outName, sourcePath, "-lm"]
    else:
        print(f"Unknown platform: {system}", file=sys.stderr)
        return False

    # Run from outputDir so the binary lands alongside the Python module
    print(f"Compiling: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=outputDir, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Compilation failed:\n{result.stderr}", file=sys.stderr)
            return False
    except FileNotFoundError:
        print("gcc not found. Install gcc or set MOT_VOELLMY_SOURCE to skip download.", file=sys.stderr)
        return False

    # Make executable on Unix
    if system != "Windows":
        outPath = os.path.join(outputDir, outName)
        os.chmod(outPath, 0o755)

    print(f"Compiled {outPath}")
    return True


def buildMoTVoellmy(sourcePath=None):
    """Compile MoT-Voellmy binary.

    Args:
        sourcePath: Optional path to a local .c file. If None, tries
                    MOT_VOELLMY_SOURCE env var, then GitHub download.

    Returns:
        True if compilation succeeded, False otherwise.
    """
    # 1. CLI argument
    if sourcePath is not None:
        if not os.path.isfile(sourcePath):
            print(f"Source file not found: {sourcePath}", file=sys.stderr)
            return False
        return _compile(sourcePath)

    # 2. Environment variable
    envSource = os.environ.get("MOT_VOELLMY_SOURCE")
    if envSource:
        if not os.path.isfile(envSource):
            print(f"MOT_VOELLMY_SOURCE file not found: {envSource}", file=sys.stderr)
            return False
        return _compile(envSource)

    # 3. GitHub download
    result = _findGithubSource()
    if result is None:
        return False

    url, filename = result
    dest = os.path.join(outputDir, filename)

    # Use cached copy if available and download is a fallback
    if not os.path.isfile(dest):
        if not _downloadSource(url, dest):
            return False

    return _compile(dest)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else None
    success = buildMoTVoellmy(source)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
