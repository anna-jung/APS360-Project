import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ttt_utils import ttt_mlp


class TTTBase(nn.Module):
    """Base TTT layer adapted from the reference implementation."""
    
    def __init__(self, model_dim: int, proj_dim: int, mini_batch_size: int = 3, 
                 ttt_base_lr: float = 1.0, num_heads: Optional[int] = 8):
        super().__init__()
        
        # Main dimensions
        self.width = model_dim  # model dimension
        self.proj_dim = proj_dim  # num_heads * head_dim from reference
        
        # If num_heads is provided, compute head_dim; otherwise assume single head
        if num_heads is not None:
            assert proj_dim % num_heads == 0, f"proj_dim {proj_dim} must be divisible by num_heads {num_heads}"
            self.num_heads = num_heads
            self.head_dim = proj_dim // num_heads
        else:
            self.num_heads = 1
            self.head_dim = proj_dim
            
        self.mini_batch_size = mini_batch_size
        self.ttt_base_lr = ttt_base_lr
        
        self._init_qkvo_proj()
        self._init_ttt_lr_gate()
        self._init_ttt_ln()
        
        self.post_norm = nn.LayerNorm(self.proj_dim, eps=1e-6)
    
    def init_weights(self):
        """Initialize weights following the reference implementation."""
        for linear in (self.wq, self.wk, self.wv):
            nn.init.normal_(linear.weight, mean=0.0, std=0.02)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        nn.init.normal_(self.wo.weight, mean=0.0, std=0.02)
        if self.wo.bias is not None:
            nn.init.zeros_(self.wo.bias)
        
        self.post_norm.reset_parameters()
        nn.init.ones_(self.ttt_norm_weight.data)
        nn.init.zeros_(self.ttt_norm_bias)
        nn.init.normal_(self.learnable_ttt_lr_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.learnable_ttt_lr_bias)
    
    def _init_qkvo_proj(self):
        """Initialize Q, K, V, and output projections."""
        self.wq = nn.Linear(self.width, self.proj_dim, bias=True)
        self.wk = nn.Linear(self.width, self.proj_dim, bias=True)
        self.wv = nn.Linear(self.width, self.proj_dim, bias=True)
        self.wo = nn.Linear(self.proj_dim, self.width, bias=True)
    
    def _init_ttt_lr_gate(self):
        """Initialize learnable learning rate gates."""
        linear_weight_data = nn.Linear(self.width, 1, bias=True).weight.data
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.stack(
                [torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)],
                dim=0,
            )
        )
        
        linear_bias_data = nn.Linear(self.width, 1, bias=True).bias.data
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.stack(
                [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
                dim=0,
            )
        )
    
    def _init_ttt_ln(self):
        """Initialize per-head layer normalization parameters."""
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))
    
    def get_qkv_projections(self, hidden_states):
        """Get Q, K, V projections from hidden states."""
        XQ = self.wq(hidden_states)
        XK = self.wk(hidden_states)
        XV = self.wv(hidden_states)
        return XQ, XK, XV
    
    def get_eta(self, X):
        """Compute learning rate modulation (eta) for TTT."""
        # X shape: [B, num_mini_batch, mini_batch_size, width]
        ttt_lr = torch.einsum("bnkc,hdc->bhnkd", X, self.learnable_ttt_lr_weight) + self.learnable_ttt_lr_bias.reshape(1, -1, 1, 1, 1)
        ttt_lr = F.sigmoid(ttt_lr)  # [B, H, num_mini_batch, mini_batch_size, 1]
        ttt_lr = ttt_lr.permute(0, 1, 2, 4, 3)  # [B, H, num_mini_batch, 1, mini_batch_size]
        return self.ttt_base_lr * ttt_lr / self.head_dim
    
    def ln_reconstruction_target(self, XV, XK):
        """Apply layer normalization to reconstruction targets."""
        XV = XV - XK
        eps = 1e-8
        
        # Compute mean and std over the head dimension (last dimension)
        mean = XV.mean(dim=-1, keepdim=True)
        std = XV.std(dim=-1, keepdim=True)
        
        # Normalize
        XV = (XV - mean) / (std + eps)
        
        # Apply per-head weight and bias
        XV = self.ttt_norm_weight.unsqueeze(0).unsqueeze(0) * XV + self.ttt_norm_bias.unsqueeze(0).unsqueeze(0)
        
        return XV + XK
    
    def reshape_to_mini_batch(self, X, XQ, XK, XV):
        """Reshape inputs to mini-batch format for TTT processing."""
        B, L = X.shape[:2]
        num_mini_batch = L // self.mini_batch_size
        
        # Reshape input
        X = X.reshape(B, num_mini_batch, self.mini_batch_size, self.width)
        
        # Reshape Q, K, V to [B, num_heads, num_mini_batch, mini_batch_size, head_dim]
        XQ = XQ.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XK = XK.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XV = XV.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        XQ = XQ.reshape(B, self.num_heads, num_mini_batch, self.mini_batch_size, self.head_dim)
        XK = XK.reshape(B, self.num_heads, num_mini_batch, self.mini_batch_size, self.head_dim)
        XV = XV.reshape(B, self.num_heads, num_mini_batch, self.mini_batch_size, self.head_dim)
        
        return X, XQ, XK, XV
    
    def process_input(self, hidden_states: torch.Tensor):
        """Process input hidden states into TTT format."""
        B, L = hidden_states.shape[:2]
        # Get Q, K, V projections
        XQ, XK, XV = self.get_qkv_projections(hidden_states)
        
        # L2 normalize Q and K
        XQ = F.normalize(XQ.view(B, L, self.num_heads, self.head_dim), p=2, dim=-1)
        XK = F.normalize(XK.view(B, L, self.num_heads, self.head_dim), p=2, dim=-1)
        XV = XV.view(B, L, self.num_heads, self.head_dim)
        
        # Prepare reconstruction target
        XV = self.ln_reconstruction_target(XV, XK).contiguous()

        # Reshape to mini-batch format
        hidden_states_mb, XQ, XK, XV = self.reshape_to_mini_batch(
            hidden_states.view(B // self.mini_batch_size, L * self.mini_batch_size, -1), 
            XQ.view(B // self.mini_batch_size, L * self.mini_batch_size, -1), 
            XK.view(B // self.mini_batch_size, L * self.mini_batch_size, -1), 
            XV.view(B // self.mini_batch_size, L * self.mini_batch_size, -1)
        )

        # Get learning rate modulation
        eta = self.get_eta(hidden_states_mb)
        eta = 1 / self.mini_batch_size * eta.repeat(1, 1, 1, self.mini_batch_size, 1)
        
        inputs = {
            "XQ": XQ,
            "XK": XK,
            "XV": XV,
            "eta": eta,
        }
        
        return inputs
    
    def ttt(self, inputs):
        """TTT computation - to be implemented by subclasses."""
        raise NotImplementedError("ttt method must be implemented in TTTBase subclasses.")
    
    def forward(self, hidden_states: torch.Tensor):
        """Forward pass through TTT layer."""
        B, L = hidden_states.shape[:2]
        
        # Process input and apply TTT
        hidden_states_out = self.ttt(self.process_input(hidden_states))
        hidden_states_out = hidden_states_out.reshape(B, L, -1)
        
        # Apply post-norm and output projection
        hidden_states_out = self.post_norm(hidden_states_out)
        hidden_states_out = self.wo(hidden_states_out)
        
        return hidden_states_out

class TTTMLPLayer(TTTBase):
    """TTT layer with MLP inner model (using TTTMLP)."""
    
    def __init__(self, model_dim: int, proj_dim: int, mini_batch_size: int = 3, 
                 ttt_base_lr: float = 1.0, num_heads: Optional[int] = None):
        super().__init__(model_dim, proj_dim, mini_batch_size, ttt_base_lr, num_heads)
        
        # MLP TTT parameters (2-layer MLP: head_dim -> 4*head_dim -> head_dim)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
    
    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.W1, mean=0.0, std=0.02)
        nn.init.zeros_(self.b1)
        nn.init.normal_(self.W2, mean=0.0, std=0.02)
        nn.init.zeros_(self.b2)
    
    def ttt(self, inputs):
        """Apply TTT with MLP inner model using reference implementation."""
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        
        # Tile parameters for batch processing
        W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
        b1_states = torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1))
        W2_states = torch.tile(self.W2.unsqueeze(0), dims=(B, 1, 1, 1))
        b2_states = torch.tile(self.b2.unsqueeze(0), dims=(B, 1, 1, 1))
        
        # Use reference ttt_mlp implementation with checkpoint_group_size=0 for simplicity
        XQW_batch = ttt_mlp(
            inputs["XK"], inputs["XQ"], inputs["XV"], inputs["eta"],
            self.ttt_norm_weight, self.ttt_norm_bias,
            W1_states, b1_states, W2_states, b2_states,
            checkpoint_group_size=min(num_mini_batch, 16)
        )
        
        # XQW_batch is [B, num_mini_batch, mini_batch_size, H, head_dim]
        # Reshape to [B, L, proj_dim] for TTTBase to handle final projection
        XQW_batch = XQW_batch.reshape(B, L, self.proj_dim)
        
        return XQW_batch



class TTT(nn.Module):
    """TTT model with multiple TTTMLPLayer instances."""
    
    def __init__(self, num_layers: int, model_dim: int, proj_dim: int, 
                 mini_batch_size: int = 3, ttt_base_lr: float = 0.1, 
                 num_heads: Optional[int] = None):
        super().__init__()
        
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.proj_dim = proj_dim
        self.mini_batch_size = mini_batch_size
        self.ttt_base_lr = ttt_base_lr
        self.num_heads = num_heads
        
        # Create TTT layers
        self.layers = nn.ModuleList([
            TTTMLPLayer(
                model_dim=model_dim,
                proj_dim=proj_dim,
                mini_batch_size=mini_batch_size,
                ttt_base_lr=ttt_base_lr,
                num_heads=num_heads
            )
            for _ in range(num_layers)
        ])
        
        # Initialize all layers
        for layer in self.layers:
            layer.init_weights()
    
    def forward(self, layer_idx: int, old_hidden_states: torch.Tensor, hidden_states: torch.Tensor = None):
        """Apply TTT layer at the specified index.
        
        Args:
            layer_idx: Layer index
            old_hidden_states: Previous hidden states (before transformer layer)
            hidden_states: Current hidden states (after transformer layer), optional
        
        Returns:
            Updated hidden states
        """
        if layer_idx >= len(self.layers) or layer_idx < 0:
            return old_hidden_states if hidden_states is None else hidden_states
        
        # Use current hidden states if provided, otherwise use old hidden states
        input_states = hidden_states if hidden_states is not None else old_hidden_states
        
        return input_states + self.layers[layer_idx](input_states)
    
    def forward_all_layers(self, hidden_states_list):
        """Apply TTT to a list of hidden states from different layers."""
        outputs = []
        for i, hidden_states in enumerate(hidden_states_list):
            if i < len(self.layers):
                output = self.layers[i](hidden_states)
                outputs.append(output)
            else:
                outputs.append(hidden_states)
        return outputs
    
    def get_layer(self, layer_idx: int):
        """Get a specific TTT layer."""
        if 0 <= layer_idx < len(self.layers):
            return self.layers[layer_idx]
        return None