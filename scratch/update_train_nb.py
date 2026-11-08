import json

# Read current notebook
with open('train_sovereign_modernbert_31k/train-sovereign-modernbert-31k.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 5: Add temperature calibration and evaluation
c5_code = """# Step 4: Part B - Fine-Tuning ModernBERT-Large on 31k Pool (1 Epoch)
print('=== Step 4: Fine-Tuning ModernBERT-Large on 31k Pool ===')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ModernBERTClassifier(enc_dir, num_classes=18).to(device)

save_dir = '/kaggle/working/sovereign_modernbert'
os.makedirs(os.path.join(save_dir, 'encoder'), exist_ok=True)

def save_modernbert(b_acc=None, tag='best', temp=1.0):
    print(f'--> Saving ModernBERT checkpoint ({tag}, T={temp:.3f})...', flush=True)
    model.encoder.save_pretrained(os.path.join(save_dir, 'encoder'))
    tokenizer.save_pretrained(os.path.join(save_dir, 'encoder'))
    calibrated_w = (model.classifier.weight.detach().cpu() / temp).float()
    calibrated_b = (model.classifier.bias.detach().cpu() / temp).float()
    save_file({
        'weight': calibrated_w,
        'bias': calibrated_b
    }, os.path.join(save_dir, 'head.safetensors'))
    with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
        json.dump({
            'labels': LABELS,
            'max_len': 384,
            'hidden_size': model.config.hidden_size,
            'val_acc': float(b_acc) if b_acc else None,
            'temperature': float(temp),
            'timestamp': time.time()
        }, f, indent=2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW([
    {'params': model.encoder.parameters(), 'lr': 2e-5, 'weight_decay': 0.01},
    {'params': model.classifier.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}
])

EPOCHS = 1
TOTAL_STEPS = len(train_loader) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * TOTAL_STEPS), num_training_steps=TOTAL_STEPS)
scaler = torch.cuda.amp.GradScaler()

start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for step, (input_ids, attention_mask, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        running_loss += loss.item()
        if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
            elapsed = (time.time() - start_time) / 60.0
            avg_loss = running_loss / (step + 1)
            print(f'Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {avg_loss:.4f} | Elapsed: {elapsed:.1f}m', flush=True)
            
    # Evaluate on Validation & Compute Temperature Calibration
    model.eval()
    val_preds, val_targets = [], []
    val_logits_list = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits, _ = model(input_ids, attention_mask)
            val_logits_list.append(logits.float().cpu().numpy())
            val_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            val_targets.extend(labels.numpy())
            
    val_bacc = balanced_accuracy_score(val_targets, val_preds)
    all_val_logits = np.concatenate(val_logits_list, axis=0)
    val_targets = np.array(val_targets)
    
    # Guo et al. Temperature Calibration
    from scipy.optimize import minimize_scalar
    def nll_fn(t):
        t = max(t, 0.1)
        scaled = all_val_logits / t
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        correct_p = probs[np.arange(len(val_targets)), val_targets]
        return -np.mean(np.log(np.maximum(correct_p, 1e-12)))

    res_t = minimize_scalar(nll_fn, bounds=(0.5, 3.0), method='bounded')
    opt_temp = float(res_t.x) if res_t.success else 1.25
    print(f'=== Epoch {epoch+1} Done | Val Macro-Recall: {val_bacc:.4f} | Calibrated T: {opt_temp:.3f} ===', flush=True)
    save_modernbert(b_acc=val_bacc, tag='final_calibrated', temp=opt_temp)

print(f'*** Training Finished in {(time.time() - start_time)/60.0:.1f} minutes! ***')
"""

nb['cells'][5]['source'] = [c5_code]

with open('train_sovereign_modernbert_31k/train-sovereign-modernbert-31k.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated train-sovereign-modernbert-31k.ipynb successfully!')

