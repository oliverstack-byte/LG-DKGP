from lgdkgp.common.io import file_hash, digest, write_json

def validate_resume(path, identity):
    import torch
    state=torch.load(path,map_location="cpu",weights_only=False)
    if state.get("lewm_run_identity",{}).get("identity_sha256") != identity:
        raise ValueError("Resume identity differs")
    if not state.get("optimizer_states"):
        raise ValueError("Full-state optimizer checkpoint required")
    return {"epoch":state["epoch"],"global_step":state["global_step"]}
