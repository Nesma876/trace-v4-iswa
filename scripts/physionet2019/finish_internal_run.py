"""One-shot continuation of an already running internal training process, not a scheduler."""
import argparse
import ctypes
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from trace.physionet2019.internal import OUT, CHECKPOINTS, REPO, config, verify
from trace.physionet2019.pipeline import write_json
from trace.physionet2019.internal_reporting import finisher_jbi_preflight


def complete_training():
    return all((CHECKPOINTS/name/str(seed)/"metadata.json").exists()
               for name in config()["models"] for seed in config()["seeds"])


def main(pid):
    verify()
    finisher_jbi_preflight()
    status_path=OUT/"analysis/run_status.json"
    write_json(status_path,{"phase":"training","training_pid":pid,"Set_B_predictive_evaluation_performed":False,"Set_B_locked":True})
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x100000,False,pid) # SYNCHRONIZE access only
    if not handle and not complete_training():
        raise RuntimeError("Training process unavailable and training is incomplete")
    try:
        while not complete_training():
            if kernel.WaitForSingleObject(handle,0)==0:
                raise RuntimeError("Training process exited before all twenty fits completed")
            time.sleep(15)
    finally:
        if handle:kernel.CloseHandle(handle)
    for phase,script in (("internal_evaluation","evaluate_internal.py"),("statistical_reporting","report_internal.py")):
        write_json(status_path,{"phase":phase,"Set_B_predictive_evaluation_performed":False,"Set_B_locked":True})
        subprocess.run([sys.executable,str(REPO/"scripts/physionet2019"/script)],cwd=REPO,check=True)
    result=subprocess.run([sys.executable,"-m","pytest","-q"],cwd=REPO,capture_output=True,text=True)
    write_json(OUT/"analysis/final_test_run.json",{"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr})
    if result.returncode:
        raise RuntimeError("Final regression suite failed")
    verify()
    write_json(status_path,{"phase":"computed_pending_visual_QA_and_final_review","Set_B_predictive_evaluation_performed":False,"Set_B_locked":True,
                           "readiness":"NOT READY until final visual QA, scientific review and results commit"})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--training-pid",type=int,required=True)
    args=parser.parse_args()
    try:
        main(args.training_pid)
    except Exception as exc:
        write_json(OUT/"analysis/run_status.json",{"phase":"FAILED","error":str(exc),"traceback":traceback.format_exc(),"Set_B_predictive_evaluation_performed":False,"Set_B_locked":True})
        raise
