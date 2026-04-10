$ip = "134.199.195.3"
$url = "http://${ip}:8000/v1/chat/completions"

$body = @{
    model    = "/models/DeepSeek-R1-Distill-Qwen-32B"
    messages = @(
        @{ role = "user"; content = "Say hello in one word" }
    )
    max_tokens = 10
} | ConvertTo-Json

Write-Host "Testing planner at $url ..."

try {
    $response = Invoke-RestMethod -Uri $url -Method POST `
        -Headers @{ "Content-Type" = "application/json"; "Authorization" = "Bearer EMPTY" } `
        -Body $body -TimeoutSec 60

    Write-Host "SUCCESS!"
    Write-Host "Response: $($response.choices[0].message.content)"
} catch {
    Write-Host "FAILED: $_"
}
