# Force UTF-8 so Rich can render → and other unicode characters
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$result = Measure-Command {
    python -m cli.main demo\demo_complex.py --verbose --force-agents
}

Write-Host ""
Write-Host "Total time: $([math]::Round($result.TotalSeconds, 1)) seconds"
