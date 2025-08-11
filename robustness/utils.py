from typing import Optional, TypeVar

import torch
from torch.nn.utils.spectral_norm import SpectralNorm
from torch.nn.modules import Module
import torch.nn.functional as F

T_module = TypeVar("T_module", bound=Module)

class ConvSpectralNorm(SpectralNorm):
    def reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        # Return weight as-is to preserve spatial structure
        return weight
  
    def compute_weight(self, module: Module, do_power_iteration: bool) -> torch.Tensor:
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
      
        is_conv = isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d))
        is_conv_transpose = isinstance(module, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d))
      
        if not (is_conv or is_conv_transpose):
            # Fall back to original matrix-based method for linear layers
            return super().compute_weight(module, do_power_iteration)
      
        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    if is_conv:
                        # Forward: v = conv(u, weight)
                        v_new = F.conv2d(u.unsqueeze(0), weight, padding="same")
                        v = F.normalize(v_new.squeeze(0).flatten(), dim=0, eps=self.eps, out=v)
                        # Backward: u = conv_transpose(v, weight) 
                        u_new = F.conv_transpose2d(v.view_as(u).unsqueeze(0), weight, padding="same")
                        u = F.normalize(u_new.squeeze(0).flatten(), dim=0, eps=self.eps, out=u)
                    else:  # conv_transpose
                        # Forward: v = conv_transpose(u, weight)
                        v_new = F.conv_transpose2d(u.unsqueeze(0), weight, padding="same")
                        v = F.normalize(v_new.squeeze(0).flatten(), dim=0, eps=self.eps, out=v)
                        # Backward: u = conv(v, weight)
                        u_new = F.conv2d(v.view_as(u).unsqueeze(0), weight, padding="same")
                        u = F.normalize(u_new.squeeze(0).flatten(), dim=0, eps=self.eps, out=u)
              
                if self.n_power_iterations > 0:
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)
      
        if is_conv:
            Wv = F.conv2d(v.view_as(u).unsqueeze(0), weight, padding="same")
        else:
            Wv = F.conv_transpose2d(v.view_as(u).unsqueeze(0), weight, padding="same")
      
        sigma = torch.dot(u.flatten(), Wv.squeeze(0).flatten())
        weight = weight / sigma
        return weight


def conv_spectral_norm(
    module: T_module,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: Optional[int] = None,
) -> T_module:
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    ConvSpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module
    
def project_weights_after_step(model):
    for module in model.modules():
        for hook in module._forward_pre_hooks.values():
            if hasattr(hook, "compute_weight") and hasattr(hook, "name"):
                weight_name = hook.name + "_orig"
                if hasattr(module, weight_name):
                    with torch.no_grad():
                        hook.compute_weight(module, do_power_iteration=True)
