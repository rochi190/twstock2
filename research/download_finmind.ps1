Set-Location E:\twstock2
$log = 'C:\Users\neill\.claude\jobs\bffcee47\tmp\finmind.log'
foreach ($ds in @('rev','inst','per','margin','hold')) {
    for ($try = 0; $try -lt 12; $try++) {
        "[$(Get-Date -Format s)] start $ds try $try" | Out-File -Append -Encoding utf8 $log
        python -m backend.altdata.finmind --datasets $ds --universe 300 *>> $log
        $code = $LASTEXITCODE
        "[$(Get-Date -Format s)] $ds exit=$code" | Out-File -Append -Encoding utf8 $log
        if ($code -eq 0) { break }
        if ($code -eq 2) { Start-Sleep -Seconds 3700 } else { Start-Sleep -Seconds 90 }
    }
}
"[$(Get-Date -Format s)] ALL_DONE" | Out-File -Append -Encoding utf8 $log
