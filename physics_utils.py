import torch
import torch.nn as nn
def compute_physics_loss(model, t_physics, drugs, k_params, scalers):
    required_keys = [
        'hill_coeff', 'IC50_vem', 'IC50_tram', 'IC50_pi3k', 'IC50_ras',
        'k_paradox', 'k_egfr', 'k_egfr_deg', 'k_her2', 'k_her2_deg',
        'k_her3', 'k_her3_deg', 'k_igf', 'k_igf_deg',
        'k_erk_rtk', 'Km_rtk', 'k_up', 'k_erk_sos', 'Km_sos',
        'k_akt_rtk', 'Km_artk', 'k_craf', 'k_craf_deg', 'k_mek', 'k_mek_deg', 'k_mek_braf',
        'k_erk', 'k_erk_deg', 'k_dusp_synth', 'k_dusp_deg', 'k_dusp_cat',
        'Km_dusp', 'Km_dusp_s', 'n_dusp', 'k_raf_pi3k', 'Km_raf_pi3k',
        'k_erk_pi3k', 'Km_erk_pi3k', 'k_akt', 'k_akt_deg', 
        'k_4ebp1', 'k_4ebp1_deg', 'k_4ebp1_comp', 'Km_4ebp1', 'k_akt_raf', 'Km_akt_raf',
        'k_her2_tx', 'k_her3_tx', 'k_ras_pi3k_frac',
        'K_sat_egfr', 'K_sat_her2', 'K_sat_her3', 'K_sat_igfr',
        'K_sat_craf', 'K_sat_mek', 'K_sat_erk', 'K_sat_akt', 'K_sat_4ebp1',
        'w_egfr', 'w_her2', 'w_her3', 'w_igf1r', 'w_craf',
        'w_mek', 'w_erk', 'w_dusp6', 'w_akt', 'w_4ebp1'
    ]
    missing = [k for k in required_keys if k not in k_params]
    if missing:
        raise KeyError(f"compute_physics_loss: missing k_params keys: {missing}")

    t_physics.requires_grad_(True)
    y_pred_norm = model(t_physics, drugs)
    dy_dt_norm = torch.autograd.grad(
        outputs=y_pred_norm,
        inputs=t_physics,
        grad_outputs=torch.ones_like(y_pred_norm),
        create_graph=True,
        retain_graph=True
    )[0]
    y = y_pred_norm * scalers['y_range'] + scalers['y_min']
    dy_dt = (dy_dt_norm * scalers['y_range']) / scalers['t_range']
    pEGFR  = y[:, 0]
    HER2   = y[:, 1]
    HER3   = y[:, 2]
    IGF1R  = y[:, 3]
    pCRAF  = y[:, 4]
    pMEK   = y[:, 5]
    pERK   = y[:, 6]
    DUSP6  = y[:, 7]
    pAKT   = y[:, 8]
    p4EBP1 = y[:, 9]
    Vem    = drugs[:, 0]
    Tram   = drugs[:, 1]
    PI3Ki  = drugs[:, 2]
    RasInh = drugs[:, 3]
    n = torch.clamp(k_params['hill_coeff'], 1.0, 4.0)
    eps = 1e-7
    Vem_inhibition = ((Vem+eps)**n) / (torch.abs(k_params['IC50_vem'])**n + (Vem+eps)**n + 1e-8)
    Tram_effect    = ((Tram+eps)**n) / (torch.abs(k_params['IC50_tram'])**n + (Tram+eps)**n + 1e-8)
    PI3Ki_effect   = ((PI3Ki+eps)**n) / (torch.abs(k_params['IC50_pi3k'])**n + (PI3Ki+eps)**n + 1e-8)
    Ras_effect     = ((RasInh+eps)**n) / (torch.abs(k_params['IC50_ras'])**n + (RasInh+eps)**n + 1e-8)
    K_sat_egfr  = torch.abs(k_params['K_sat_egfr'])
    K_sat_her2  = torch.abs(k_params['K_sat_her2'])
    K_sat_her3  = torch.abs(k_params['K_sat_her3'])
    K_sat_igfr  = torch.abs(k_params['K_sat_igfr'])
    K_sat_craf  = torch.abs(k_params['K_sat_craf'])
    K_sat_mek   = torch.abs(k_params['K_sat_mek'])
    K_sat_erk   = torch.abs(k_params['K_sat_erk'])
    K_sat_akt   = torch.abs(k_params['K_sat_akt'])
    K_sat_4ebp1 = torch.abs(k_params['K_sat_4ebp1'])
    k_paradox = torch.abs(k_params['k_paradox'])
    pCRAF_floored = pCRAF.clamp(min=0.05)  
    pi3ki_attenuation = 1.0 - 0.7 * PI3Ki_effect
    Vem_paradox = (k_paradox * Vem * K_sat_craf / (K_sat_craf + pCRAF_floored + 1e-8)
                   * pi3ki_attenuation)

    k_erk_rtk = torch.abs(k_params['k_erk_rtk'])
    Km_rtk = torch.abs(k_params['Km_rtk'])
    ERK_feedback = k_erk_rtk * pERK / (Km_rtk + pERK + 1e-8)
    k_up = torch.abs(k_params['k_up'])
    drug_relief = k_up * (Vem_inhibition + Tram_effect + PI3Ki_effect)
    k_erk_sos = torch.abs(k_params['k_erk_sos'])
    Km_sos = torch.abs(k_params['Km_sos'])
    k_akt_rtk = torch.abs(k_params['k_akt_rtk'])
    Km_artk = torch.abs(k_params['Km_artk'])
    ERK_to_SOS = k_erk_sos * pERK / (Km_sos + pERK + 1e-8)
    AKT_to_RTK = k_akt_rtk * pAKT / (Km_artk + pAKT + 1e-8)
    RTK_total  = pEGFR + HER2 + 1.5 * HER3 + IGF1R
    RAS_GTP    = RTK_total * (1.0 - ERK_to_SOS) * (1.0 - AKT_to_RTK) * (1.0 - Ras_effect)
    k_raf_pi3k = torch.abs(k_params['k_raf_pi3k'])
    Km_raf_pi3k = torch.abs(k_params['Km_raf_pi3k'])
    k_erk_pi3k = torch.abs(k_params['k_erk_pi3k'])
    Km_erk_pi3k = torch.abs(k_params['Km_erk_pi3k'])
    RAF_to_PI3K = k_raf_pi3k * pCRAF / (Km_raf_pi3k + pCRAF + 1e-8)
    ERK_to_PI3K = k_erk_pi3k * pERK / (Km_erk_pi3k + pERK + 1e-8)
    k_ras_pi3k_frac = torch.abs(k_params['k_ras_pi3k_frac'])
    PI3K_input = RTK_total * (1.0 - ERK_to_PI3K) * (1.0 - k_ras_pi3k_frac * Ras_effect) + RAF_to_PI3K
    k_akt_raf = torch.abs(k_params['k_akt_raf'])
    Km_akt_raf = torch.abs(k_params['Km_akt_raf'])
    AKT_RAF_inhib  = k_akt_raf * pAKT / (Km_akt_raf + pAKT + 1e-8)
    k_egfr = torch.abs(k_params['k_egfr'])
    k_egfr_deg = torch.abs(k_params['k_egfr_deg'])
    res_pEGFR = dy_dt[:, 0] - (k_egfr * (1.0 + drug_relief) * K_sat_egfr / (K_sat_egfr + pEGFR + 1e-8) - (k_egfr_deg + ERK_feedback) * pEGFR)
    k_her2 = torch.abs(k_params['k_her2'])
    k_her2_deg = torch.abs(k_params['k_her2_deg'])
    k_her2_tx = torch.abs(k_params['k_her2_tx'])
    
    res_HER2 = dy_dt[:, 1] - (
        k_her2 * K_sat_her2 / (K_sat_her2 + HER2 + 1e-8)
        + k_her2_tx * (1.0 - pERK / (K_sat_erk + pERK + 1e-8))
        - k_her2_deg * HER2
    )
    k_her3 = torch.abs(k_params['k_her3'])
    k_her3_deg = torch.abs(k_params['k_her3_deg'])
    k_her3_tx = torch.abs(k_params['k_her3_tx'])
    
    res_HER3 = dy_dt[:, 2] - (
        k_her3 * K_sat_her3 / (K_sat_her3 + HER3 + 1e-8)
        + k_her3_tx * (1.0 - pERK / (K_sat_erk + pERK + 1e-8))
        - k_her3_deg * HER3
    )
    k_igf = torch.abs(k_params['k_igf'])
    k_igf_deg = torch.abs(k_params['k_igf_deg'])
    
    res_IGF1R = dy_dt[:, 3] - (k_igf * K_sat_igfr / (K_sat_igfr + IGF1R + 1e-8) - (k_igf_deg + ERK_feedback + AKT_to_RTK) * IGF1R)
    k_craf = torch.abs(k_params['k_craf'])
    k_craf_deg = torch.abs(k_params['k_craf_deg'])
    
    res_pCRAF = dy_dt[:, 4] - (
        k_craf * RAS_GTP * K_sat_craf / (K_sat_craf + pCRAF + 1e-8)
        + Vem_paradox
        - (k_craf_deg + AKT_RAF_inhib) * pCRAF
    )
    k_mek = torch.abs(k_params['k_mek'])
    k_mek_deg = torch.abs(k_params['k_mek_deg'])
    k_mek_braf = torch.abs(k_params['k_mek_braf'])
    
    _braf_ic50 = 0.04  # fixed constant uM — A375 vemurafenib IC50 for BRAF V600E
    Vem_braf_inhibition = ((Vem+eps)**n) / (_braf_ic50**n + (Vem+eps)**n + 1e-8)
    res_pMEK = dy_dt[:, 5] - (
        (k_mek * pCRAF + k_mek_braf * (1.0 - Vem_braf_inhibition))
        * (1.0 - Tram_effect)
        * K_sat_mek / (K_sat_mek + pMEK + 1e-8)
        - k_mek_deg * pMEK
    )
    k_dusp_cat = torch.abs(k_params['k_dusp_cat'])
    Km_dusp = torch.abs(k_params['Km_dusp'])
    DUSP6_activity = k_dusp_cat * DUSP6 / (Km_dusp + DUSP6 + 1e-8)
    k_erk = torch.abs(k_params['k_erk'])
    k_erk_deg = torch.abs(k_params['k_erk_deg'])
    res_pERK  = dy_dt[:, 6] - (k_erk * pMEK * K_sat_erk / (K_sat_erk + pERK + 1e-8) - (k_erk_deg + DUSP6_activity) * pERK)
    n_dusp = torch.clamp(k_params['n_dusp'], 1.5, 3.5)
    k_dusp_synth = torch.abs(k_params['k_dusp_synth'])
    Km_dusp_s = torch.abs(k_params['Km_dusp_s'])
    DUSP6_induction = k_dusp_synth * ((pERK+eps)**n_dusp) / (Km_dusp_s**n_dusp + (pERK+eps)**n_dusp + 1e-8)
    k_dusp_deg = torch.abs(k_params['k_dusp_deg'])
    res_DUSP6 = dy_dt[:, 7] - (DUSP6_induction - k_dusp_deg * DUSP6)
    k_akt = torch.abs(k_params['k_akt'])
    k_akt_deg = torch.abs(k_params['k_akt_deg'])
    res_pAKT  = dy_dt[:, 8] - (k_akt * PI3K_input * (1.0 - PI3Ki_effect) * K_sat_akt / (K_sat_akt + pAKT + 1e-8) - k_akt_deg * pAKT)

    k_4ebp1 = torch.abs(k_params['k_4ebp1'])
    k_4ebp1_deg = torch.abs(k_params['k_4ebp1_deg'])
    k_4ebp1_basal = torch.abs(k_params['k_4ebp1_comp'])   # reuse existing param
    res_p4EBP1 = dy_dt[:, 9] - (
        k_4ebp1 * pAKT * K_sat_4ebp1 / (K_sat_4ebp1 + p4EBP1 + 1e-8)
        + k_4ebp1_basal
        - k_4ebp1_deg * p4EBP1
    )
    physics_loss = (
        torch.abs(k_params['w_egfr']) * torch.mean(res_pEGFR**2) +
        torch.abs(k_params['w_her2']) * torch.mean(res_HER2**2) +
        torch.abs(k_params['w_her3']) * torch.mean(res_HER3**2) +
        torch.abs(k_params['w_igf1r']) * torch.mean(res_IGF1R**2) +
        torch.abs(k_params['w_craf']) * torch.mean(res_pCRAF**2) +
        torch.abs(k_params['w_mek']) * torch.mean(res_pMEK**2) +
        torch.abs(k_params['w_erk']) * torch.mean(res_pERK**2) +
        torch.abs(k_params['w_dusp6']) * torch.mean(res_DUSP6**2) +
        torch.abs(k_params['w_akt']) * torch.mean(res_pAKT**2) +
        torch.abs(k_params['w_4ebp1']) * torch.mean(res_p4EBP1**2)
    )
    return physics_loss
    
def compute_conservation_loss(y_pred_norm, scalers):
    """
    Computes conservation and biological constraint losses for pathway consistency.
    """
    y = y_pred_norm * scalers['y_range'] + scalers['y_min']
    pCRAF  = y[:, 4]
    pMEK   = y[:, 5]
    pERK   = y[:, 6]
    DUSP6  = y[:, 7]
    pAKT   = y[:, 8]
    p4EBP1 = y[:, 9]
    neg_penalty = torch.mean(torch.relu(-y))
    mapk_order = torch.mean(torch.relu(pMEK - 2.5 * pCRAF)**2) +   torch.mean(torch.relu(pERK - 2.5 * pMEK)**2)
    dusp6_corr = torch.mean((DUSP6 - 0.8 * pERK)**2)
    conservation_loss = neg_penalty + 0.02 * mapk_order +  0.015 * dusp6_corr
    return conservation_loss

def compute_steady_state_loss(model, t_physics, drugs, scalers):
    """
    Computes the steady-state loss: penalizes dX/dt ≠ 0 at collocation
    points where ALL drugs = 0.
    Biological rationale:
        In BRAF V600E melanoma without drug treatment, the constitutive
        MAPK/PI3K pathway is at steady state. All species should have
        dX/dt ≈ 0 because there is no perturbation driving the system
        away from equilibrium.
    This loss is ONLY applied to no-drug collocation points (drugs = [0,0,0,0]).
    Args:
        model: The PINN model.
        t_physics: Collocation time points (normalized), shape (N, 1).
        drugs: Drug concentrations at collocation points, shape (N, 4).
        scalers: Dict with 'y_range', 'y_min', 't_range' for un-normalization.
    Returns:
        Scalar loss tensor (mean squared temporal derivative at no-drug points).
    """
    drug_sum = drugs.sum(dim=1)
    no_drug_mask = (drug_sum < 1e-6)
    n_no_drug = no_drug_mask.sum().item()
    if n_no_drug == 0:
        return torch.tensor(0.0, device=t_physics.device, requires_grad=True)
    t_nd = t_physics[no_drug_mask].clone().detach().requires_grad_(True)
    drugs_nd = drugs[no_drug_mask]
    y_pred_norm = model(t_nd, drugs_nd)
    dy_dt_norm = torch.autograd.grad(
        outputs=y_pred_norm,
        inputs=t_nd,
        grad_outputs=torch.ones_like(y_pred_norm),
        create_graph=True,
        retain_graph=True, # Ensure graph is retained for shared model autograd tracing
    )[0]
    dy_dt = (dy_dt_norm * scalers['y_range']) / scalers['t_range']
    y_scale = scalers['y_range'] + 1e-8
    dy_dt_scaled = dy_dt / y_scale.unsqueeze(0)
    steady_state_loss = torch.mean(dy_dt_scaled ** 2)
    return steady_state_loss
