import clip
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from enum import Enum
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

class FinetuneMode(Enum):
    """Different finetuning strategies"""
    FULL = "full"                    # Train all parameters
    PARTIAL_HEAD = "partial_head"    # Train only final projection layers
    PARTIAL_TOP = "partial_top"      # Train top N layers + projections
    PARTIAL_LORA = "partial_lora"    # Use LoRA adapters (Low-Rank Adaptation)
    CUSTOM = "custom"                # Custom parameter selection

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""
    def __init__(self, in_features, out_features, rank=16, alpha=16, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.dropout(self.lora_A(x))) * self.scaling

class CLIPFineTuner:
    def __init__(self, model_name="ViT-B/32", device="cuda"):
        self.device = device
        self.model_name = model_name

        # Load CLIP model and preprocessing
        self.model, self.preprocess = clip.load(model_name, device=device)

        # Store original parameters for partial training
        self.original_params = {}
        for name, param in self.model.named_parameters():
            self.original_params[name] = param.clone().detach()

        print(f"Loaded CLIP model: {model_name}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def setup_finetuning(self, mode=FinetuneMode.FULL, **kwargs):
        """
        Setup which parameters to train based on the finetuning mode

        Args:
            mode: FinetuneMode enum
            **kwargs: Additional parameters for specific modes
                - top_layers: Number of top layers to train (for PARTIAL_TOP)
                - lora_rank: Rank for LoRA adaptation (for PARTIAL_LORA)
                - custom_patterns: List of parameter name patterns (for CUSTOM)
        """
        # First, freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_params = 0

        if mode == FinetuneMode.FULL:
            # Train all parameters
            for param in self.model.parameters():
                param.requires_grad = True
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        elif mode == FinetuneMode.PARTIAL_HEAD:
            # Train only final projection layers
            patterns = [
                'visual.proj',      # Vision projection
                'text_projection',  # Text projection
                'logit_scale'      # Temperature parameter
            ]

            for name, param in self.model.named_parameters():
                if any(pattern in name for pattern in patterns):
                    param.requires_grad = True
                    trainable_params += param.numel()

        elif mode == FinetuneMode.PARTIAL_TOP:
            # Train top N layers + projections
            top_layers = kwargs.get('top_layers', 2)

            # Always train projections
            projection_patterns = ['visual.proj', 'text_projection', 'logit_scale']

            for name, param in self.model.named_parameters():
                # Train projection layers
                if any(pattern in name for pattern in projection_patterns):
                    param.requires_grad = True
                    trainable_params += param.numel()

                # Train top layers of vision transformer
                if 'visual.transformer.resblocks' in name:
                    # Extract layer number
                    layer_num = int(name.split('.resblocks.')[1].split('.')[0])
                    total_layers = self.model.visual.transformer.layers
                    if layer_num >= total_layers - top_layers:
                        param.requires_grad = True
                        trainable_params += param.numel()

                # Train top layers of text transformer
                if 'transformer.resblocks' in name and 'visual' not in name:
                    layer_num = int(name.split('.resblocks.')[1].split('.')[0])
                    total_layers = self.model.transformer.layers
                    if layer_num >= total_layers - top_layers:
                        param.requires_grad = True
                        trainable_params += param.numel()

        elif mode == FinetuneMode.PARTIAL_LORA:
            # Add LoRA adapters to attention layers
            lora_rank = kwargs.get('lora_rank', 16)
            lora_alpha = kwargs.get('lora_alpha', 16)

            self._add_lora_adapters(lora_rank, lora_alpha)
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        elif mode == FinetuneMode.CUSTOM:
            # Custom parameter selection
            custom_patterns = kwargs.get('custom_patterns', [])

            for name, param in self.model.named_parameters():
                if any(pattern in name for pattern in custom_patterns):
                    param.requires_grad = True
                    trainable_params += param.numel()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_percentage = (trainable_params / total_params) * 100

        print(f"\nFinetuning Mode: {mode.value}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_percentage:.2f}%)")
        print(f"Frozen parameters: {total_params - trainable_params:,}")

        # Print trainable parameter names for verification
        print("\nTrainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.numel():,} params")

    # def _add_lora_adapters(self, rank=16, alpha=16):
    #     """Add LoRA adapters to attention layers"""
    #     # This is a simplified LoRA implementation
    #     # In practice, you'd need to modify the attention modules more carefully

    #     def add_lora_to_linear(module, name):
    #         if isinstance(module, nn.Linear):
    #             # Create LoRA adapter
    #             lora = LoRALayer(
    #                 module.in_features,
    #                 module.out_features,
    #                 rank=rank,
    #                 alpha=alpha
    #             )

    #             # Add as a new module
    #             setattr(module, 'lora_adapter', lora)

    #             # Enable gradients for LoRA parameters
    #             for param in lora.parameters():
    #                 param.requires_grad = True

    #             # Modify forward pass to include LoRA
    #             original_forward = module.forward
    #             def new_forward(x):
    #                 base_output = original_forward(x)
    #                 if hasattr(module, 'lora_adapter'):
    #                     lora_output = module.lora_adapter(x)
    #                     return base_output + lora_output
    #                 return base_output

    #             module.forward = new_forward

    #     # Apply LoRA to attention layers
    #     for name, module in self.model.named_modules():
    #         if 'attn' in name and isinstance(module, nn.Linear):
    #             if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']):
    #                 add_lora_to_linear(module, name)

    def _add_lora_adapters(self, rank=16, alpha=16):
        """Add LoRA adapters to attention layers"""

        def add_lora_to_linear(module, name):
            if isinstance(module, nn.Linear):
                # Freeze the original linear layer
                for param in module.parameters():
                    param.requires_grad = False

                # Create LoRA adapter
                lora = LoRALayer(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    alpha=alpha
                )
                # Add as a new module
                setattr(module, 'lora_adapter', lora)

                # Enable gradients for LoRA parameters only
                for param in lora.parameters():
                    param.requires_grad = True

                # Store original forward
                original_forward = module.forward

                def new_forward(x):
                    # Ensure x requires gradients for LoRA computation
                    if not x.requires_grad and any(p.requires_grad for p in lora.parameters()):
                        x = x.requires_grad_(True)

                    base_output = original_forward(x)
                    if hasattr(module, 'lora_adapter'):
                        lora_output = module.lora_adapter(x)
                        # Handle gradient flow properly
                        if base_output.requires_grad or lora_output.requires_grad:
                            return base_output + lora_output
                        else:
                            return base_output.detach() + lora_output.detach()
                    return base_output

                module.forward = new_forward

        # FIXED: Collect modules first, then modify
        modules_to_modify = []
        for name, module in self.model.named_modules():
            if 'attn' in name and isinstance(module, nn.Linear):
                if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']):
                    modules_to_modify.append((module, name))

        # Now modify the collected modules
        for module, name in modules_to_modify:
            add_lora_to_linear(module, name)

    def get_trainable_parameters(self):
        """Get only trainable parameters for optimizer"""
        return [p for p in self.model.parameters() if p.requires_grad]

    def contrastive_loss(self, logits_per_image, logits_per_text, temperature=0.07):
        """Compute CLIP's contrastive loss"""
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=self.device, dtype=torch.long)

        logits_per_image = logits_per_image / temperature
        logits_per_text = logits_per_text / temperature

        loss_i2t = nn.CrossEntropyLoss()(logits_per_image, labels)
        loss_t2i = nn.CrossEntropyLoss()(logits_per_text, labels)

        return (loss_i2t + loss_t2i) / 2.0

    def train_epoch(self, dataloader, optimizer, scheduler, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        # Set frozen layers to eval mode to prevent BatchNorm updates
        for name, module in self.model.named_modules():
            if not any(param.requires_grad for param in module.parameters()):
                module.eval()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            texts = batch['text'].to(self.device)

            # Forward pass
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(texts)

            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # Compute logits
            logit_scale = self.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            # Compute loss
            loss = self.contrastive_loss(logits_per_image, logits_per_text)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.get_trainable_parameters(), max_norm=1.0)

            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                images = batch['image'].to(self.device)
                texts = batch['text'].to(self.device)

                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(texts)

                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                logit_scale = self.model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                loss = self.contrastive_loss(logits_per_image, logits_per_text)
                total_loss += loss.item()

                # Compute accuracy
                predictions = logits_per_image.argmax(dim=1)
                targets = torch.arange(len(images), device=self.device)
                correct_predictions += (predictions == targets).sum().item()
                total_predictions += len(images)

        return total_loss / len(dataloader), correct_predictions / total_predictions

    def finetune(self, train_dataset, val_dataset=None,
                 finetune_mode=FinetuneMode.FULL, epochs=10, lr=1e-5,
                 batch_size=32, weight_decay=0.1,
                 experiment_id=0,
                 **mode_kwargs):
        """
        Finetune the CLIP model with specified strategy

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            finetune_mode: FinetuneMode enum
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size
            weight_decay: Weight decay
            **mode_kwargs: Additional arguments for finetuning mode
        """
        # Setup finetuning strategy
        self.setup_finetuning(finetune_mode, **mode_kwargs)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )

        # Setup optimizer - only for trainable parameters
        trainable_params = self.get_trainable_parameters()

        # Adjust learning rate based on finetuning mode
        if finetune_mode == FinetuneMode.PARTIAL_HEAD:
            effective_lr = lr * 10  # Higher LR for head-only training
        elif finetune_mode == FinetuneMode.PARTIAL_LORA:
            effective_lr = lr * 5   # Moderate increase for LoRA
        else:
            effective_lr = lr

        optimizer = optim.AdamW(
            trainable_params,
            lr=effective_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6
        )

        # Setup scheduler
        total_steps = len(train_loader) * epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=effective_lr * 0.01
        )

        print(f"\nTraining Configuration:")
        print(f"Mode: {finetune_mode.value}")
        print(f"Learning rate: {effective_lr:.2e}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Total steps: {total_steps}")

        best_val_loss = float('inf')

        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_params({
                "mode": finetune_mode.value,
                "lr": effective_lr,
                "epochs": epochs,
                "batch_size": batch_size
            })

            for epoch in range(1, epochs + 1):
                print(f"\nEpoch {epoch}/{epochs}")

                train_loss = self.train_epoch(train_loader, optimizer, scheduler, epoch)
                print(f"Train Loss: {train_loss:.4f}")
                mlflow.log_metrics({
                    "train_loss": train_loss
                }, step=epoch)

                if val_loader:
                    val_loss, val_acc = self.validate(val_loader)
                    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    mlflow.log_metrics({
                        "val_loss": val_loss,
                        "val_accuracy": val_acc
                    }, step=epoch)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_checkpoint(
                            f"best_{finetune_mode.value}_epoch_{epoch}.pt",
                            epoch, optimizer, scheduler, finetune_mode
                        )
                        print(f"New best model saved!")

                # if epoch % 5 == 0:
                #     self.save_checkpoint(
                #         f"checkpoint_{finetune_mode.value}_epoch_{epoch}.pt",
                #         epoch, optimizer, scheduler, finetune_mode
                #     )

    def save_checkpoint(self, filename, epoch, optimizer, scheduler, finetune_mode):
        """Save model checkpoint with finetuning metadata"""
        # checkpoint = {
        #     'epoch': epoch,
        #     'model_state_dict': self.model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'scheduler_state_dict': scheduler.state_dict(),
        #     'finetune_mode': finetune_mode.value,
        #     'model_name': self.model_name,
        #     'trainable_params': [name for name, param in self.model.named_parameters() if param.requires_grad]
        # }
        # torch.save(checkpoint, filename)
        # print(f"Checkpoint saved: {filename}")
        mlflow.pytorch.log_model(self.model, "model")
