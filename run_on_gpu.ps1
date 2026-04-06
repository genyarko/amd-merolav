# run_on_gpu.ps1 — Copy migrated file to MI300X and run it to prove AMD compatibility
# Usage: .\run_on_gpu.ps1 rocm_output\demo_input.py

param(
    [Parameter(Mandatory=$true)]
    [string]$MigratedFile
)

$envFile = "$PSScriptRoot\.env"
function Get-EnvVar($name) {
    $line = Get-Content $envFile | Select-String "^${name}="
    if ($line) { return $line.ToString().Split("=", 2)[1].Trim() }
    return $null
}

$IP = Get-EnvVar "DROPLET_IP"
if (-not $IP) { Write-Error "DROPLET_IP not set in .env"; exit 1 }

$fileName = Split-Path $MigratedFile -Leaf

Write-Host "`n[1/3] Copying $fileName to MI300X ($IP)..." -ForegroundColor Cyan
scp -o StrictHostKeyChecking=no $MigratedFile "root@${IP}:~/run_${fileName}"

Write-Host "`n[2/3] Running on MI300X GPU..." -ForegroundColor Cyan
ssh -o StrictHostKeyChecking=no "root@${IP}" @"
source /root/venv/bin/activate
echo '--- GPU Status ---'
rocm-smi --showuse
echo ''
echo '--- Running $fileName ---'
python3 ~/run_${fileName} 2>&1
echo ''
echo '--- GPU After Run ---'
rocm-smi --showuse
"@

Write-Host "`n[3/3] Done." -ForegroundColor Green
