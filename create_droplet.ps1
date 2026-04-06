# Load token from .env
$token = (Get-Content "$PSScriptRoot\.env" | Select-String "^DIGITAL_OCEAN=").ToString().Split("=",2)[1]
$headers = @{ Authorization = "Bearer $token"; "Content-Type" = "application/json" }

# Step 1: Find the single MI300X size slug
Write-Host "Fetching all available sizes..."
$sizes = Invoke-RestMethod -Uri "https://api.digitalocean.com/v2/sizes?per_page=200" -Headers $headers

Write-Host "`nAll size slugs containing 'gpu' or 'mi300':"
$sizes.sizes | Where-Object { $_.slug -like "*gpu*" -or $_.slug -like "*mi300*" } | ForEach-Object { Write-Host "  $($_.slug)" }

Write-Host "`nTotal sizes returned: $($sizes.sizes.Count)"

$sizeSlug = "gpu-mi300x1-192gb"
Write-Host "`nUsing size: $sizeSlug"

# Step 2: Poll until droplet is created
$body = @{
    name     = "rocm-7-2-software-gpu-mi300x-devcloud-atl1"
    region   = "atl1"
    size     = $sizeSlug
    image    = 220895104
    ssh_keys = @(55426474)
    backups  = $false
    ipv6     = $false
    monitoring = $false
} | ConvertTo-Json

Write-Host "`nPolling every 5 minutes until GPU becomes available..."

while ($true) {
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Attempting to create droplet..."
    try {
        $response = Invoke-RestMethod -Uri "https://api.digitalocean.com/v2/droplets" `
            -Method POST -Headers $headers -Body $body
        $dropletId = $response.droplet.id
        Write-Host "`nSUCCESS! Droplet created."
        Write-Host "  ID:   $dropletId"
        Write-Host "  Name: $($response.droplet.name)"
        Write-Host "`nWaiting 2 minutes for droplet to boot..."
        Start-Sleep -Seconds 120

        # Fetch IP
        $droplet = Invoke-RestMethod -Uri "https://api.digitalocean.com/v2/droplets/$dropletId" -Headers $headers
        $ip = $droplet.droplet.networks.v4 | Where-Object { $_.type -eq "public" } | Select-Object -ExpandProperty ip_address
        Write-Host "`nDroplet IP: $ip"
        Write-Host "Connect with: ssh root@$ip"
        break
    } catch {
        $msg = $_.ErrorDetails.Message | ConvertFrom-Json -ErrorAction SilentlyContinue
        Write-Host "  Not available yet ($($msg.message)). Retrying in 5 minutes..."
        Start-Sleep -Seconds 300
    }
}
