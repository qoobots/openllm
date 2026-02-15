import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb
import flwr as fl  # 联邦学习框架


class MedicalDiagnosisTrainer:
    """医疗AI诊断微调训练器
    
    支持本地训练和联邦学习训练
    """
    
    def __init__(self, model, train_dataset, val_dataset, config, use_federated=False):
        self.model = model.model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.use_federated = use_federated
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            collate_fn=self._collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4,
            collate_fn=self._collate_fn
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
    
    def _collate_fn(self, batch):
        """自定义collate函数"""
        # 处理多模态数据
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
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
        correct = 0
        total = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=self.config['use_fp16']):
                outputs = self.model(**batch)
                loss = outputs.loss
            
            total_loss += loss.item()
            
            # 计算准确率
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == batch['labels']).sum().item()
            total += batch['labels'].numel()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        wandb.log({
            'val/loss': avg_loss,
            'val/accuracy': accuracy,
            'val/epoch': epoch
        })
        
        return avg_loss, accuracy
    
    def train(self):
        """本地训练"""
        wandb.init(project="medical-diagnosis-finetune", config=self.config)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss, val_acc = self.evaluate(epoch)
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save_pretrained(self.config['output_dir'])
                print(f"Saved best model with val_loss: {val_loss:.4f}")
        
        wandb.finish()


class FederatedLearningClient(fl.client.NumPyClient):
    """联邦学习客户端"""
    
    def __init__(self, model, train_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.device = device
    
    def get_parameters(self, config):
        """获取模型参数"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """设置模型参数"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """本地训练一轮"""
        self.set_parameters(parameters)
        
        # 执行本地训练
        self.model.train()
        for epoch in range(config["local_epochs"]):
            for batch in self.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                loss.backward()
                
                # 更新参数
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.data -= config["learning_rate"] * param.grad
                        param.grad.zero_()
        
        # 返回更新后的参数
        return self.get_parameters(config), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        """评估模型"""
        self.set_parameters(parameters)
        # 实现评估逻辑
        return 0.5, len(self.train_loader.dataset), {}
