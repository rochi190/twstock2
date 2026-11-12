# 續跑 FinMind 下載（可中斷續跑；已存在的檔案會跳過；遇流量限制自動等一小時）
# 用法（在 repo 根目錄）：  powershell -ExecutionPolicy Bypass -File research\resume.ps1
Set-Location (Split-Path $PSScriptRoot -Parent)
foreach ($ds in @('rev','inst','per','margin','hold')) {
    for ($try = 0; $try -lt 12; $try++) {
        Write-Host "[$(Get-Date -Format s)] start $ds try $try"
        python -m backend.altdata.finmind --datasets $ds --universe 300
        $code = $LASTEXITCODE
        Write-Host "[$(Get-Date -Format s)] $ds exit=$code"
        if ($code -eq 0) { break }
        if ($code -eq 2) { Write-Host "流量限制，等 1 小時"; Start-Sleep -Seconds 3700 } else { Start-Sleep -Seconds 90 }
    }
}
Write-Host "ALL_DONE"
