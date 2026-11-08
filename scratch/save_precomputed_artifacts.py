import numpy as np, pandas as pd
import test_cross_72_milestone as m
import test_cross_74_verifiers as v

print("Saving precomputed artifacts into scratch/cache_cv_data.npz...", flush=True)

np.savez_compressed(
    'scratch/cache_cv_data.npz',
    base_logits=m.base_logits,
    gemini_cols=m.cluster_oof['gemini_flash'][0],
    gemini_probs=m.cluster_oof['gemini_flash'][1],
    gemini_rel=m.cluster_oof['gemini_flash'][2],
    gpt56_cols=m.cluster_oof['gpt56'][0],
    gpt56_probs=m.cluster_oof['gpt56'][1],
    gpt56_rel=m.cluster_oof['gpt56'][2],
    glm_cols=m.cluster_oof['glm'][0],
    glm_probs=m.cluster_oof['glm'][1],
    glm_rel=m.cluster_oof['glm'][2],
    oof_fbl_glm_diff=m.oof_fbl_glm_diff,
    oof_gem_pair_diff=m.oof_gem_pair_diff,
    oof_so_diff=v.oof_so_diff,
    oof_kg_diff=v.oof_kg_diff,
    oof_st_diff=v.oof_st_diff,
    y_true=m.y_true,
    all_pids=m.all_pids,
    unique_pids=m.unique_pids,
    labels=m.LABELS
)

print("Artifacts saved successfully!")

