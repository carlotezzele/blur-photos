"""Send files to the Recycle Bin / Trash. Never deletes permanently.

- WSL with files on a Windows drive (/mnt/...): Windows Recycle Bin, via PowerShell.
- Otherwise: the system trash via the optional `send2trash` package.
If neither is available, nothing is deleted and an error is raised.
"""
import base64, json, os, shutil, subprocess


class TrashUnavailable(RuntimeError):
    pass


def recycle(paths):
    paths = [p for p in dict.fromkeys(paths) if os.path.lexists(p)]
    if not paths:
        return []
    if shutil.which("powershell.exe") and shutil.which("wslpath") and all(p.startswith("/mnt/") for p in paths):
        win = [subprocess.check_output(["wslpath", "-w", p], text=True).strip() for p in paths]
        b64 = base64.b64encode(json.dumps(win).encode()).decode()
        script = ("Add-Type -AssemblyName Microsoft.VisualBasic; "
                  f"$p = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{b64}')) | ConvertFrom-Json; "
                  "foreach ($f in @($p)) { [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile("
                  "$f, 'OnlyErrorDialogs', 'SendToRecycleBin') }")
        subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                       cwd="/mnt/c", capture_output=True, timeout=120)
    else:
        try:
            from send2trash import send2trash
        except ImportError:
            raise TrashUnavailable("no Recycle Bin available: pip install send2trash")
        for p in paths:
            send2trash(p)
    left = [p for p in paths if os.path.lexists(p)]
    if left:
        raise RuntimeError("could not move to the Recycle Bin: " + ", ".join(os.path.basename(p) for p in left))
    return paths
