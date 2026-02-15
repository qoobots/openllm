import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb


class FinancialComplianceTrainer:
    """金融合规审查微调训练器"""
    
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model.model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4
        )
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        num_training_steps = len(self.train_loader) * config['num_epochs']
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=num_training_steps
        )
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['use_fp16'])
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=self.config['use_fp16']):
                outputs = self.model(**batch)
                loss = outputs.loss
            
            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            if step % self.config['logging_steps'] == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'train/epoch': epoch
                })
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self, epoch):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=self.config['use_fp16']):
                outputs = self.model(**batch)
                loss = outputs.loss
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        wandb.log({'val/loss': avg_loss, 'val/epoch': epoch})
        
        return avg_loss
    
    def train(self):
        """完整训练流程"""
        wandb.init(project="financial-compliance-finetune", config=self.config)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate(epoch)
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save_pretrained(self.config['output_dir'])
                print(f"Saved best model with val_loss: {val_loss:.4f}")
        
        wandb.finish()
