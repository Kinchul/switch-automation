[CmdletBinding()]
param(
    [string]$PiHost,
    [string]$RemoteDir,
    [switch]$DryRun,
    [double]$DebounceSeconds = 1.0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Edit these once for your usual Pi target. Command-line arguments still override them.
$DefaultHost = "vkindschi@192.168.1.181"
$DefaultRemoteDir = "~/switch-automation"

if (-not $PiHost) {
    $PiHost = $DefaultHost
}

if (-not $RemoteDir) {
    $RemoteDir = $DefaultRemoteDir
}

function Quote-Sh {
    param([Parameter(Mandatory = $true)][string]$Value)

    if ($Value.Length -eq 0) {
        return "''"
    }

    return "'" + ($Value -replace "'", "'""'""'") + "'"
}

function Test-ExcludedRelativePath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $normalized = $RelativePath -replace "\\", "/"
    $parts = $normalized.Split("/", [System.StringSplitOptions]::RemoveEmptyEntries)
    foreach ($part in $parts) {
        if ($part -in @(".git", ".venv", ".pytest_cache", ".mypy_cache", "__pycache__", ".codex", ".vscode", "debug")) {
            return $true
        }
    }

    $leaf = Split-Path $normalized -Leaf
    if ($leaf -like "*.pyc" -or $leaf -like "*.pyo" -or $leaf -like "*.pyd") {
        return $true
    }

    return $false
}

function Get-RemoteParentAndLeaf {
    param([Parameter(Mandatory = $true)][string]$Path)

    $trimmed = $Path.Trim().TrimEnd("/")
    if ($trimmed.Length -eq 0) {
        throw "RemoteDir must not be empty."
    }

    $slash = $trimmed.LastIndexOf("/")
    if ($slash -lt 0) {
        return @{
            Parent = "."
            Leaf = $trimmed
        }
    }

    if ($slash -eq 0) {
        return @{
            Parent = "/"
            Leaf = $trimmed.Substring(1)
        }
    }

    return @{
        Parent = $trimmed.Substring(0, $slash)
        Leaf = $trimmed.Substring($slash + 1)
    }
}

function Invoke-Sync {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [string]$CurrentPiHost,
        [string]$CurrentRemoteDir,
        [switch]$CurrentDryRun,
        [string[]]$ChangedPaths = @()
    )

    $startedAt = Get-Date
    Write-Host ""
    Write-Host "[$($startedAt.ToString('HH:mm:ss'))] Sync starting..."
    if ($ChangedPaths.Count -gt 0) {
        Write-Host "[$($startedAt.ToString('HH:mm:ss'))] Triggered by:"
        foreach ($path in ($ChangedPaths | Sort-Object -Unique)) {
            Write-Host "  - $path"
        }
    }

    $remote = Get-RemoteParentAndLeaf -Path $CurrentRemoteDir
    if (-not $remote.Leaf) {
        throw "RemoteDir must end with a directory name."
    }

    $stagingRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("switch-automation-sync-" + [System.Guid]::NewGuid().ToString("N"))
    $stagingRepo = Join-Path $stagingRoot $remote.Leaf
    $copiedRelativePaths = New-Object System.Collections.Generic.List[string]

    $exitCode = 0
    New-Item -ItemType Directory -Path $stagingRepo -Force | Out-Null

    try {
        $items = Get-ChildItem -LiteralPath $RepoRoot -Force -Recurse
        foreach ($item in $items) {
            $relative = $item.FullName.Substring($RepoRoot.Length).TrimStart("\")
            if ([string]::IsNullOrWhiteSpace($relative)) {
                continue
            }
            if (Test-ExcludedRelativePath -RelativePath $relative) {
                continue
            }

            $destination = Join-Path $stagingRepo $relative
            if ($item.PSIsContainer) {
                New-Item -ItemType Directory -Path $destination -Force | Out-Null
                continue
            }

            $destinationParent = Split-Path -Parent $destination
            if ($destinationParent) {
                New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
            }
            Copy-Item -LiteralPath $item.FullName -Destination $destination -Force
            $normalizedRelative = $relative -replace "\\", "/"
            [void]$copiedRelativePaths.Add($normalizedRelative)
        }

        $stagedFiles = Get-ChildItem -LiteralPath $stagingRepo -Recurse -File | Sort-Object FullName
        Write-Host "Prepared $($stagedFiles.Count) files for upload to ${CurrentPiHost}:$CurrentRemoteDir"

        if ($CurrentDryRun) {
            $stagedFiles | ForEach-Object {
                $_.FullName.Substring($stagingRepo.Length).TrimStart("\") -replace "\\", "/"
            }
        }
        else {
            $ssh = (Get-Command ssh.exe -ErrorAction Stop).Source
            $scp = (Get-Command scp.exe -ErrorAction Stop).Source

            $remoteParentQuoted = Quote-Sh $remote.Parent
            & $ssh $CurrentPiHost "mkdir -p $remoteParentQuoted"
            $exitCode = $LASTEXITCODE
            if ($exitCode -eq 0) {
                Write-Host "Uploading $stagingRepo -> ${CurrentPiHost}:$($remote.Parent)/"
                & $scp -r $stagingRepo "${CurrentPiHost}:$($remote.Parent)/"
                $exitCode = $LASTEXITCODE
                if ($exitCode -eq 0) {
                    Write-Host "SYNC_SUMMARY files=$($copiedRelativePaths.Count) target=${CurrentPiHost}:$CurrentRemoteDir"
                }
            }
        }
    }
    finally {
        if (Test-Path $stagingRoot) {
            Remove-Item -LiteralPath $stagingRoot -Recurse -Force
        }
    }

    $endedAt = Get-Date

    if ($exitCode -eq 0) {
        Write-Host "[$($endedAt.ToString('HH:mm:ss'))] Sync complete."
    }
    else {
        Write-Host "[$($endedAt.ToString('HH:mm:ss'))] Sync failed with exit code $exitCode."
    }

    return $exitCode
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $repoRoot
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true
$watcher.NotifyFilter = [System.IO.NotifyFilters]'FileName, DirectoryName, LastWrite, Size, CreationTime'

$pendingChange = $false
$lastRelevantChangeAt = [DateTime]::MinValue
$pendingPaths = New-Object System.Collections.Generic.List[string]

$onChange = {
    $repoRootPath = $Event.MessageData.RepoRoot
    $fullPath = [string]$Event.SourceEventArgs.FullPath
    if (-not $fullPath.StartsWith($repoRootPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        return
    }

    $relativePath = $fullPath.Substring($repoRootPath.Length).TrimStart('\')
    if ([string]::IsNullOrWhiteSpace($relativePath)) {
        return
    }
    if (Test-ExcludedRelativePath -RelativePath $relativePath) {
        return
    }

    $script:pendingChange = $true
    $script:lastRelevantChangeAt = Get-Date
    $normalized = $relativePath -replace "\\", "/"
    $script:pendingPaths.Add($normalized)
    Write-Host "[$($script:lastRelevantChangeAt.ToString('HH:mm:ss'))] Change detected: $normalized"
}

$subscriptions = @()
$subscriptions += Register-ObjectEvent -InputObject $watcher -EventName Changed -Action $onChange -MessageData @{ RepoRoot = $repoRoot }
$subscriptions += Register-ObjectEvent -InputObject $watcher -EventName Created -Action $onChange -MessageData @{ RepoRoot = $repoRoot }
$subscriptions += Register-ObjectEvent -InputObject $watcher -EventName Deleted -Action $onChange -MessageData @{ RepoRoot = $repoRoot }
$subscriptions += Register-ObjectEvent -InputObject $watcher -EventName Renamed -Action $onChange -MessageData @{ RepoRoot = $repoRoot }

try {
    Write-Host "Watching $repoRoot for changes. Press Ctrl+C to stop."
    Invoke-Sync -RepoRoot $repoRoot -CurrentPiHost $PiHost -CurrentRemoteDir $RemoteDir -CurrentDryRun:$DryRun | Out-Null

    while ($true) {
        Wait-Event -Timeout 0.5 | Out-Null

        if (-not $pendingChange) {
            continue
        }

        $quietFor = ((Get-Date) - $lastRelevantChangeAt).TotalSeconds
        if ($quietFor -lt $DebounceSeconds) {
            continue
        }

        $changedPaths = @($pendingPaths | Select-Object -Unique)
        $pendingChange = $false
        $pendingPaths.Clear()
        Invoke-Sync `
            -RepoRoot $repoRoot `
            -CurrentPiHost $PiHost `
            -CurrentRemoteDir $RemoteDir `
            -CurrentDryRun:$DryRun `
            -ChangedPaths $changedPaths | Out-Null
    }
}
finally {
    foreach ($subscription in $subscriptions) {
        try {
            Unregister-Event -SubscriptionId $subscription.Id
        }
        catch {
        }
        try {
            $subscription | Remove-Job -Force
        }
        catch {
        }
    }

    $watcher.EnableRaisingEvents = $false
    $watcher.Dispose()
}
