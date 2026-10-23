"""Synthetic tensors only: select a CPU thread count, never scientific parameters."""
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
import torch
from trace.physionet2019.internal import make_model
from trace.physionet2019.internal_training import step

if __name__=="__main__":
    print("CUDA available:",torch.cuda.is_available(),flush=True)
    x=torch.randn(512,4,99)
    y=torch.randn(512,4,33)
    mask=torch.ones_like(y,dtype=torch.bool)
    for threads in (1,2,4):
        torch.set_num_threads(threads)
        model=make_model("gru")
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        for _ in range(3):
            step(model,"gru",x,y,mask,optimizer)
        start=time.perf_counter()
        for _ in range(30):
            step(model,"gru",x,y,mask,optimizer)
        print({"threads":threads,"seconds_per_batch":(time.perf_counter()-start)/30},flush=True)
