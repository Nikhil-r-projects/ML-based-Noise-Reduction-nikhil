Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

$outDir = Join-Path $PSScriptRoot "..\data\sample_voices"
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

$phrases = @(
    @{ Name = "voice_alpha.wav"; Text = "Alpha team moving towards checkpoint Bravo." },
    @{ Name = "voice_charlie.wav"; Text = "Charlie team reached sector four." },
    @{ Name = "voice_bravo.wav"; Text = "Bravo team waiting at checkpoint two." },
    @{ Name = "voice_delta.wav"; Text = "Delta squad under fire at objective Iron." },
    @{ Name = "voice_echo.wav"; Text = "Echo element holding position at sector seven." },
    @{ Name = "voice_viper.wav"; Text = "Viper recon requesting evac at landing zone Alpha." }
)

foreach ($p in $phrases) {
    $filePath = Join-Path $outDir $p.Name
    Write-Host "Generating: $($p.Name) -> $($p.Text)"
    $synth.SetOutputToWaveFile($filePath)
    $synth.Speak($p.Text)
}

$synth.Dispose()
Write-Host "Voice synthesis complete!"
