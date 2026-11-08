"""
NTL Digital Twin Lab: Experiment Runner
Run benchmarks and compare configurations with statistical bootstrap verification.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab.engine import SimulationEngine
from lab.metrics import compute_confusion_analysis, bootstrap_confidence_interval


def main():
    print("=" * 70)
    print("      NTL DIGITAL TWIN SIMULATION LAB - EXPERIMENT BENCHMARK")
    print("=" * 70)
    
    engine = SimulationEngine()
    engine.build_or_load_base_oof()
    
    # Configuration 1: Raw Baseline (No Post-Processing)
    print("\n[1] Running Baseline (Raw Predictions, No Sinkhorn)...")
    res_raw = engine.simulate_pipeline(sinkhorn_reg=0.0, clip_val=0.0, use_dispersion=False, use_length_bias=False)
    boot_raw = bootstrap_confidence_interval(engine.y_true, res_raw['y_pred'])
    print(f"  -> Score: {res_raw['balanced_accuracy']:.4f} | 95% CI: [{boot_raw['ci_low']:.4f}, {boot_raw['ci_high']:.4f}] (+/- {boot_raw['noise_floor_pm']:.4f})")
    
    # Configuration 2: Champion V59 Anchor (Sinkhorn 0.45 + Length Bias + Dispersion)
    print("\n[2] Running Champion V59 Setup (Sinkhorn 0.45 + Mix 0.35 + Length Bias + Dispersion)...")
    res_v59 = engine.simulate_pipeline(sinkhorn_reg=0.45, target_mix=0.35, clip_val=1.5, use_dispersion=True, use_length_bias=True)
    boot_v59 = bootstrap_confidence_interval(engine.y_true, res_v59['y_pred'])
    print(f"  -> Score: {res_v59['balanced_accuracy']:.4f} | 95% CI: [{boot_v59['ci_low']:.4f}, {boot_v59['ci_high']:.4f}]")
    
    # Configuration 3: Champion V61 Judith Setup (Sinkhorn 0.60 + Evidence Clip 1.2)
    print("\n[3] Running Champion V61 Setup (Judith007 Sinkhorn 0.60 + Evidence Clip 1.2)...")
    res_v61 = engine.simulate_pipeline(sinkhorn_reg=0.60, target_mix=0.35, clip_val=1.2, use_dispersion=True, use_length_bias=True)
    boot_v61 = bootstrap_confidence_interval(engine.y_true, res_v61['y_pred'])
    print(f"  -> Score: {res_v61['balanced_accuracy']:.4f} | 95% CI: [{boot_v61['ci_low']:.4f}, {boot_v61['ci_high']:.4f}]")
    diff = res_v61['balanced_accuracy'] - res_v59['balanced_accuracy']
    print(f"  -> V61 vs V59 Delta: {diff:+.4f} points!")
    
    # Configuration 4: Grid Search for Optimal Sinkhorn
    print("\n[4] Running Sinkhorn 2D Grid Search...")
    best_params, best_score, df_grid = engine.grid_search_sinkhorn()
    
    # Configuration 5: Diagnostic Breakdown of V61
    print("\n[5] Detailed Diagnostic Analysis of V61:")
    diag = compute_confusion_analysis(engine.y_true, res_v61['y_pred'])
    print(f"  -> Inter-Family Accuracy (Vendor separation): {diag['inter_family_accuracy']*100:.2f}%")
    print("  -> Intra-Family Accuracy (Sister model separation):")
    for fam, acc in diag['intra_family_accuracy'].items():
        print(f"     * {fam.capitalize():8s}: {acc*100:.2f}%")

    print("\n" + "=" * 70)
    print("               EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == '__main__':
    main()

