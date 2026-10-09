"""Official pretrained DNS3 GTCRN; no training or per-test alignment fitting."""
import importlib.util
import hashlib
from pathlib import Path
import numpy as np
import torch


class GTCRNEnhancer:
    def __init__(self, root):
        root=Path(root)
        spec=importlib.util.spec_from_file_location('official_gtcrn',root/'gtcrn.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        torch.set_num_threads(1)
        self.model=module.GTCRN().eval().cpu()
        self.checkpoint=root/'checkpoints/model_trained_on_dns3.tar'
        checkpoint=torch.load(self.checkpoint,map_location='cpu',weights_only=True)
        self.model.load_state_dict(checkpoint['model'])
        self.window=torch.hann_window(512).sqrt()
        self.metadata=dict(checkpoint_sha256=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
                           parameters=sum(p.numel() for p in self.model.parameters()),
                           trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                           parameter_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters()),
                           checkpoint_bytes=self.checkpoint.stat().st_size,
                           torch_version=torch.__version__,threads=1)

    @torch.inference_mode()
    def __call__(self, x):
        t=torch.tensor(x,dtype=torch.float32)
        s=torch.stft(t,512,256,512,self.window,return_complex=True)
        # Official model consumes [..., frequency, time, real/imaginary].
        y=self.model(torch.view_as_real(s)[None])[0]
        if not torch.is_complex(y): y=torch.view_as_complex(y.contiguous())
        out=torch.istft(y,512,256,512,self.window,length=len(x))
        return out.numpy().astype(np.float64)
