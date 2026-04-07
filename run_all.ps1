$ErrorActionPreference = "Stop"

$datasets = @("CICIDS2017", "ToN-IoT", "NbAIoT")
$seeds = @(42)  # Reduced to 1 seed for speed

# 1. Main Federation Experiment
Write-Host "`n[1/4] Running main federation experiments..."
foreach ($seed in $seeds) {
    foreach ($ds in $datasets) {
        Write-Host "  Dataset=$ds Seed=$seed"
        python -u experiments/main_experiment.py --dataset $ds --seed $seed --rounds 50
    }
}

# 2. Byzantine robustness experiment
Write-Host "`n[2/4] Running Byzantine experiments..."
foreach ($seed in $seeds) {
    foreach ($ds in $datasets) {
        python -u experiments/byzantine_experiment.py --dataset $ds --seed $seed --rounds 50
    }
}

# 3. Zero-day detection experiment
Write-Host "`n[3/4] Running zero-day experiments..."
foreach ($seed in $seeds) {
    foreach ($ds in $datasets) {
        python -u experiments/zeroday_experiment.py --dataset $ds --seed $seed --rounds 50 --holdout 2
    }
}

# 4. Ablation study
Write-Host "`n[4/4] Running ablation study..."
foreach ($seed in $seeds) {
    python -u experiments/ablation_study.py --dataset CICIDS2017 --seed $seed --rounds 50
}

# 5. Statistical tests & plots
Write-Host "`nGenerating statistical analysis..."
python -u experiments/statistical_tests.py

Write-Host "`n=============================================="
Write-Host "  All experiments complete!"
Write-Host "  Results -> ./results/"
Write-Host "  LaTeX tables -> ./results/tables/"
Write-Host "  Plots -> ./results/plots/"
Write-Host "=============================================="
