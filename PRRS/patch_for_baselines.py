import re
import sys
import os

os.chdir('/home/h2n/l2r/CP-PRE/PRRS')

def patch_file(fname, seeds_list_str):
    with open(fname, 'r') as f:
        content = f.read()

    # 1. Patch FNOBlock2d
    content = re.sub(
        r'class FNOBlock2d\(nn\.Module\):\n    def __init__\(self, modes1, modes2, n_vars, width\):',
        r'class FNOBlock2d(nn.Module):\n    def __init__(self, modes1, modes2, n_vars, width, dropout=0.0):',
        content
    )
    content = re.sub(
        r'self\.b    = nn\.Conv3d\(2, width, 1\)\s*# grid has 2 channels \(x,y\)',
        r'self.b    = nn.Conv3d(2, width, 1)   # grid has 2 channels (x,y)\n        self.drop = nn.Dropout3d(p=dropout) if dropout > 0.0 else nn.Identity()',
        content
    )
    content = re.sub(
        r'return F\.gelu\(self\.mlp2\(F\.gelu\(self\.mlp1\(self\.spec\(x\)\)\)\) \+ self\.w\(x\) \+ self\.b\(grid\)\)',
        r'return self.drop(F.gelu(self.mlp2(F.gelu(self.mlp1(self.spec(x)))) + self.w(x) + self.b(grid)))',
        content
    )

    # 2. Patch FNO2D
    content = re.sub(
        r'def __init__\(self, T_in, step, modes1, modes2, n_vars, width\):',
        r'def __init__(self, T_in, step, modes1, modes2, n_vars, width, dropout=0.0):',
        content
    )
    content = re.sub(
        r'FNOBlock2d\(modes1, modes2, n_vars, width\)',
        r'FNOBlock2d(modes1, modes2, n_vars, width, dropout)',
        content
    )

    # 3. Patch predict_ar
    predict_ar_new = """@torch.no_grad()
def predict_ar(model, a_enc, out_norm, cfg, num_passes=1):
    if num_passes > 1:
        model.train() # MC Dropout active
    else:
        model.eval()
    T_out, step = cfg["T_out"], cfg["step"]
    all_preds_decoded = []
    import torch
    device = next(model.parameters()).device
    for k in range(num_passes):
        inp   = a_enc.to(device)
        preds = []
        for t in range(0, T_out, step):
            out = model(inp)
            preds.append(out)
            inp = torch.cat([inp[..., step:], out], dim=-1)
        pred_enc = torch.cat(preds, dim=-1).cpu()
        all_preds_decoded.append(out_norm.decode(pred_enc))
    if num_passes == 1:
        return all_preds_decoded[0]
    return torch.stack(all_preds_decoded, dim=0)

@torch.no_grad()
def ensemble_predict_ar(models, a_enc, out_norm, cfg):
    all_preds = []
    for model in models:
        all_preds.append(predict_ar(model, a_enc, out_norm, cfg, num_passes=1))
    import torch
    return torch.stack(all_preds, dim=0)
"""
    content = re.sub(
        r'@torch\.no_grad\(\)\ndef predict_ar\(model, a_enc, out_norm, cfg\):.*?return out_norm\.decode\(pred_enc\)\n',
        predict_ar_new,
        content,
        flags=re.DOTALL
    )

    # 4. main patch
    content = re.sub(r'def main\(suffix=""\):', r'def main(suffix="", method="prrs", dropout_p=0.0):', content)
    
    content = re.sub(
        r'model = FNO2D\(cfg\["T_in"\], cfg\["step"\], cfg\["modes"\], cfg\["modes"\],\s*cfg\["num_vars"\], cfg\["width"\]\)\.to\(device\)',
        r'model = FNO2D(cfg["T_in"], cfg["step"], cfg["modes"], cfg["modes"], cfg["num_vars"], cfg["width"], dropout=dropout_p).to(device)',
        content
    )

    # train logic for ensemble:
    train_replacement = """    if method == "ensemble":
        print("  [Skipping training for ensemble inference]")
        pass
    else:
        opt   = torch.optim.Adam"""
    content = re.sub(r'opt\s*=\s*torch\.optim\.Adam', train_replacement, content, count=1)
    
    # 5. Inference
    eval_orig = r'    pred_cal = predict_ar\(model, a_cal_enc, out_norm, cfg\)\n    pred_val = predict_ar\(model, a_val_enc, out_norm, cfg\)\n    pred_ood = predict_ar\(model, a_ood_enc, out_norm, cfg\)'
    
    prefix_name = fname.replace('baseline_', '').replace('.py', '')
    
    eval_new = f"""    if method == "ensemble":
        models = []
        for s in {seeds_list_str}:
            m = FNO2D(cfg["T_in"], cfg["step"], cfg["modes"], cfg["modes"], cfg["num_vars"], cfg["width"], dropout=0.0).to(device)
            chk_path = os.path.join(RESULTS, '{prefix_name}_fno_seed'+str(s)+'.pt')
            m.load_state_dict(torch.load(chk_path, map_location=device))
            m.eval()
            models.append(m)
        pred_cal_k = ensemble_predict_ar(models, a_cal_enc, out_norm, cfg)
        pred_val_k = ensemble_predict_ar(models, a_val_enc, out_norm, cfg)
        pred_ood_k = ensemble_predict_ar(models, a_ood_enc, out_norm, cfg)
    elif method == "mc_dropout":
        pred_cal_k = predict_ar(model, a_cal_enc, out_norm, cfg, num_passes=20)
        pred_val_k = predict_ar(model, a_val_enc, out_norm, cfg, num_passes=20)
        pred_ood_k = predict_ar(model, a_ood_enc, out_norm, cfg, num_passes=20)
    else:
        pred_cal_k = predict_ar(model, a_cal_enc, out_norm, cfg, num_passes=1).unsqueeze(0)
        pred_val_k = predict_ar(model, a_val_enc, out_norm, cfg, num_passes=1).unsqueeze(0)
        pred_ood_k = predict_ar(model, a_ood_enc, out_norm, cfg, num_passes=1).unsqueeze(0)
        
    pred_cal = pred_cal_k.mean(dim=0)
    pred_val = pred_val_k.mean(dim=0)
    pred_ood = pred_ood_k.mean(dim=0)"""
    
    content = re.sub(eval_orig, eval_new, content)

    # 6. Score 
    score_orig = r'    scores_cal = pre_score_wave\(pred_cal, D_wave\)\n    scores_val = pre_score_wave\(pred_val, D_wave\)\n    scores_ood = pre_score_wave\(pred_ood, D_wave\)'
    score_new = """    if method in ["mc_dropout", "ensemble"]:
        scores_cal = pred_cal_k.std(dim=0).mean(dim=(1,2,3,4)).numpy()
        scores_val = pred_val_k.std(dim=0).mean(dim=(1,2,3,4)).numpy()
        scores_ood = pred_ood_k.std(dim=0).mean(dim=(1,2,3,4)).numpy()
    else:
        scores_cal = pre_score_wave(pred_cal, D_wave)
        scores_val = pre_score_wave(pred_val, D_wave)
        scores_ood = pre_score_wave(pred_ood, D_wave)"""
    content = re.sub(score_orig, score_new, content)
    
    # Also for NS2D which uses pre_score_ns2d
    score_orig2 = r'    scores_cal = pre_score_ns2d\(pred_cal, D_ns, norm_pred=True\)\n    scores_val = pre_score_ns2d\(pred_val, D_ns, norm_pred=True\)\n    scores_ood = pre_score_ns2d\(pred_ood, D_ns, norm_pred=True\)'
    score_new2 = """    if method in ["mc_dropout", "ensemble"]:
        scores_cal = pred_cal_k.std(dim=0).mean(dim=(1,2,3,4)).numpy()
        scores_val = pred_val_k.std(dim=0).mean(dim=(1,2,3,4)).numpy()
        scores_ood = pred_ood_k.std(dim=0).mean(dim=(1,2,3,4)).numpy()
    else:
        scores_cal = pre_score_ns2d(pred_cal, D_ns, norm_pred=True)
        scores_val = pre_score_ns2d(pred_val, D_ns, norm_pred=True)
        scores_ood = pre_score_ns2d(pred_ood, D_ns, norm_pred=True)"""
    content = re.sub(score_orig2, score_new2, content)
    
    # 7. argparse
    arg_orig = """    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Re-seed with specified seed
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    suffix = f"_seed{SEED}\""""
    arg_new = """    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', type=str, default='prrs')
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    suffix = f"_{args.method}_seed{SEED}\""""
    content = content.replace(arg_orig, arg_new)
    
    content = content.replace("results = main(suffix=suffix)", "results = main(suffix=suffix, method=args.method, dropout_p=args.dropout)")
    
    with open(fname, 'w') as f:
        f.write(content)

patch_file('baseline_wave2d.py', '[0, 42, 1, 2, 3]')
print("Done Wave2D")
patch_file('baseline_ns2d.py', '[0, 1, 2, 3, 4]')
print("Done NS2D")
