# =============================================================================
# launch.ps1 — Full end-to-end: create droplet → setup → download → serve
# =============================================================================

$envFile = "$PSScriptRoot\.env"

function Get-EnvVar($name) {
    $line = Get-Content $envFile | Select-String "^${name}="
    if ($line) { return $line.ToString().Split("=", 2)[1].Trim() }
    return $null
}

$DO_TOKEN  = Get-EnvVar "DIGITAL_OCEAN"
$HF_TOKEN  = Get-EnvVar "HUGGINGFACE_TOKEN"
$SSH_KEY   = "55426474"
$REGION    = "atl1"
$SIZE      = "gpu-mi300x1-192gb"
$IMAGE     = 220895104
$NAME      = "rocm-mi300x-$(Get-Date -Format 'yyyyMMdd-HHmm')"

if (-not $DO_TOKEN) { Write-Error "DIGITAL_OCEAN not found in .env"; exit 1 }
if (-not $HF_TOKEN) { Write-Error "HUGGINGFACE_TOKEN not found in .env"; exit 1 }

$headers = @{ Authorization = "Bearer $DO_TOKEN"; "Content-Type" = "application/json" }

# =============================================================================
# STEP 1: Poll until droplet is created
# =============================================================================
Write-Host "`n[1/4] Waiting for MI300X GPU availability..." -ForegroundColor Cyan

$body = @{
    name     = $NAME
    region   = $REGION
    size     = $SIZE
    image    = $IMAGE
    ssh_keys = @([int]$SSH_KEY)
    backups  = $false
    ipv6     = $false
    monitoring = $false
} | ConvertTo-Json

$dropletId = $null
while (-not $dropletId) {
    try {
        $response = Invoke-RestMethod -Uri "https://api.digitalocean.com/v2/droplets" `
            -Method POST -Headers $headers -Body $body
        $dropletId = $response.droplet.id
        Write-Host "  Droplet created! ID: $dropletId" -ForegroundColor Green
    } catch {
        $msg = ($_.ErrorDetails.Message | ConvertFrom-Json -ErrorAction SilentlyContinue).message
        Write-Host "  $(Get-Date -Format 'HH:mm:ss') Not available yet ($msg). Retrying in 5 min..."
        Start-Sleep -Seconds 300
    }
}

# =============================================================================
# STEP 2: Wait for droplet to boot and get IP
# =============================================================================
Write-Host "`n[2/4] Waiting for droplet to boot..." -ForegroundColor Cyan

$ip = $null
$attempts = 0
while (-not $ip -and $attempts -lt 12) {
    Start-Sleep -Seconds 15
    $droplet = Invoke-RestMethod -Uri "https://api.digitalocean.com/v2/droplets/$dropletId" -Headers $headers
    $ip = $droplet.droplet.networks.v4 | Where-Object { $_.type -eq "public" } | Select-Object -ExpandProperty ip_address -ErrorAction SilentlyContinue
    $attempts++
}

if (-not $ip) { Write-Error "Could not get droplet IP after 3 minutes."; exit 1 }

Write-Host "  IP: $ip" -ForegroundColor Green
Write-Host "  SSH: ssh root@$ip"

# Wait a bit more for SSH daemon to be ready
Start-Sleep -Seconds 30

# =============================================================================
# STEP 3: Copy scripts to droplet
# =============================================================================
Write-Host "`n[3/4] Copying scripts to droplet..." -ForegroundColor Cyan

scp -o StrictHostKeyChecking=no `
    "$PSScriptRoot\setup_gpu.sh" `
    "$PSScriptRoot\serve.sh" `
    "root@${ip}:~/"

Write-Host "  Scripts copied." -ForegroundColor Green

# =============================================================================
# STEP 4: Run setup (install deps + download models) then start servers
# =============================================================================
Write-Host "`n[4/4] Running setup and starting servers on droplet..." -ForegroundColor Cyan
Write-Host "  This will take a while (model downloads ~128GB total)."
Write-Host "  Logs: ssh root@$ip 'tail -f /models/logs/*.log'"

ssh -o StrictHostKeyChecking=no "root@${ip}" @"
set -e
bash ~/setup_gpu.sh '$HF_TOKEN'
nohup bash ~/serve.sh > ~/serve.log 2>&1 &
echo "Servers starting in background. Check: tail -f ~/serve.log"
"@

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  DONE! Droplet is live." -ForegroundColor Green
Write-Host "  IP:      $ip" -ForegroundColor Green
Write-Host "  Planner: http://${ip}:8000/v1  (DeepSeek-R1)" -ForegroundColor Green
Write-Host "  Executor: http://${ip}:8001/v1  (Qwen2.5-Coder)" -ForegroundColor Green
Write-Host "  SSH:     ssh root@$ip" -ForegroundColor Green
Write-Host "  Logs:    ssh root@$ip 'tail -f ~/serve.log'" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Save IP to .env for the agent
$envContent = Get-Content $envFile
if ($envContent -match "^DROPLET_IP=") {
    $envContent = $envContent -replace "^DROPLET_IP=.*", "DROPLET_IP=$ip"
} else {
    $envContent += "DROPLET_IP=$ip"
}
$envContent | Set-Content $envFile
Write-Host "`n  DROPLET_IP=$ip saved to .env" -ForegroundColor Yellow
