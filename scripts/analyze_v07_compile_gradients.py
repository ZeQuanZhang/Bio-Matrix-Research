"""Offline raw-gradient geometry and global-clip analysis of saved compile probes.

Preserves original coordinate failures and tolerances. This script neither changes
acceptance criteria nor executes models, optimizers, CUDA, or training.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots',type=Path,required=True)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--clip-norm',type=float,default=1.)
    parser.add_argument('--atol',type=float)
    parser.add_argument('--rtol',type=float)
    args=parser.parse_args()
    if args.clip_norm<=0 or any(value is not None and value<0 for value in (args.atol,args.rtol)):
        parser.error('Positive clip norm and nonnegative tolerances required')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    snapshots=torch.load(args.snapshots,map_location='cpu',weights_only=True)
    probe_root=args.snapshots.parent
    protocol_path=probe_root/'protocol.json'
    protocol=json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    atol=protocol.get('atol',3e-5) if args.atol is None else args.atol
    rtol=protocol.get('rtol',3e-3) if args.rtol is None else args.rtol
    result_path=probe_root/'result.json'
    original=json.loads(result_path.read_text()) if result_path.exists() else {}
    parity_path=probe_root/'parity.json'
    parity=json.loads(parity_path.read_text()) if parity_path.exists() else original.get('parity',{})
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root=args.root or probe_root/f'offline_gradient_analysis_{stamp}'
    root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):parser.error('Output root must be empty; original evidence is preserved')

    def geometry(reference,candidate):
        reference=reference.double().flatten();candidate=candidate.double().flatten()
        delta=candidate-reference
        left_norm=float(reference.norm());right_norm=float(candidate.norm());delta_norm=float(delta.norm())
        dot=float((reference*candidate).sum())
        cosine=max(-1.,min(1.,dot/(left_norm*right_norm))) if left_norm and right_norm else (1. if not left_norm and not right_norm else None)
        alpha=dot/(left_norm*left_norm) if left_norm else None
        orthogonal=float((candidate-alpha*reference).norm())/left_norm if left_norm else None
        return dict(elements=reference.numel(),reference_l2=left_norm,candidate_l2=right_norm,
                    difference_l2=delta_norm,relative_l2=delta_norm/left_norm if left_norm else None,
                    cosine=cosine,max_abs=float(delta.abs().max()) if delta.numel() else 0.,
                    rms_difference=float(delta.square().mean().sqrt()) if delta.numel() else 0.,
                    candidate_to_reference_norm_ratio=right_norm/left_norm if left_norm else None,
                    candidate_projection_scale=alpha,orthogonal_relative_l2=orthogonal)

    def index_tuple(index,shape):
        result=[]
        for size in reversed(shape):
            result.append(index%size);index//=size
        return list(reversed(result))

    pairs={}
    for pair,left_name,right_name in [('core_vs_tensor','core_eager','tensor_eager'),
                                      ('tensor_vs_compiled','tensor_eager','compiled'),
                                      ('core_vs_compiled','core_eager','compiled')]:
        if left_name not in snapshots or right_name not in snapshots:
            pairs[pair]=dict(status='not_available');continue
        left=snapshots[left_name]['gradients'];right=snapshots[right_name]['gradients']
        assert left.keys()==right.keys(),'Parameter names differ between snapshots'
        valid=[];none=[]
        for name in left:
            if left[name] is None or right[name] is None:
                none.append(dict(parameter=name,reference_none=left[name] is None,candidate_none=right[name] is None))
            else:
                assert left[name].shape==right[name].shape,(name,'gradient shape mismatch')
                assert bool(left[name].isfinite().all()&right[name].isfinite().all()),(name,'nonfinite raw gradient')
                valid.append(name)
        ref=torch.cat([left[name].double().flatten() for name in valid])
        cand=torch.cat([right[name].double().flatten() for name in valid])
        norm_ref=float(ref.norm());norm_cand=float(cand.norm())
        scale_ref=min(1.,args.clip_norm/(norm_ref+1e-6));scale_cand=min(1.,args.clip_norm/(norm_cand+1e-6))
        layers={};parameters={};coordinate_failures=0
        for name in valid:
            a,b=left[name],right[name]
            # Same saved dtype and reference-sided rule used by the original probe.
            difference=(a-b).abs();allowed=atol+rtol*a.abs();bad=difference>allowed
            failures=int(bad.sum());coordinate_failures+=failures
            examples=[]
            if failures:
                candidates=bad.flatten().nonzero().flatten()
                ratios=(difference/allowed.clamp_min(1e-30)).flatten()[candidates]
                chosen=candidates[ratios.topk(min(8,len(candidates))).indices]
                for flat_index in chosen.tolist():
                    av=float(a.flatten()[flat_index]);bv=float(b.flatten()[flat_index])
                    examples.append(dict(index=index_tuple(flat_index,a.shape),reference=av,candidate=bv,
                        abs_difference=abs(av-bv),original_allowed=float(allowed.flatten()[flat_index]),
                        original_tolerance_ratio=float((difference/allowed.clamp_min(1e-30)).flatten()[flat_index]),
                        reference_abs_fraction_of_global_l2=abs(av)/norm_ref if norm_ref else None,
                        reference_after_global_clip=av*scale_ref,candidate_after_global_clip=bv*scale_cand,
                        abs_difference_after_global_clip=abs(av*scale_ref-bv*scale_cand)))
            parameters[name]=dict(raw=geometry(a,b),global_clip=geometry(a.double()*scale_ref,b.double()*scale_cand),
                                  original_coordinate_failures=failures,worst_failed_coordinates=examples)
            layer=name.rsplit('.',1)[0] if '.' in name else name
            layers.setdefault(layer,[]).append(name)
        layer_results={}
        for layer,names in layers.items():
            a=torch.cat([left[name].double().flatten() for name in names])
            b=torch.cat([right[name].double().flatten() for name in names])
            layer_results[layer]=dict(parameters=names,raw=geometry(a,b),global_clip=geometry(a*scale_ref,b*scale_cand),
                                     original_coordinate_failures=sum(parameters[name]['original_coordinate_failures'] for name in names))
        pairs[pair]=dict(status='analyzed',reference=left_name,candidate=right_name,
            original_probe_pair_close=parity.get(pair,{}).get('close'),
            original_coordinate_failures=coordinate_failures,none_gradients=none,
            raw_global=geometry(ref,cand),global_clip=geometry(ref*scale_ref,cand*scale_cand),
            global_clip_parameters=dict(max_norm=args.clip_norm,epsilon=1e-6,reference_scale=scale_ref,candidate_scale=scale_cand,
              arithmetic='Deterministic CPU float64 norms/scales applied to saved float32 raw gradients; this isolates geometry, not CUDA foreach reduction roundoff'),
            layers=layer_results,parameters=parameters)
    result=dict(analysis_only=True,changes_acceptance_criteria=False,model_executed=False,cuda_used=False,
                snapshots=str(args.snapshots.resolve()),snapshots_sha256=sha256(args.snapshots),
                script_sha256=sha256(Path(__file__)),torch=str(torch.__version__),
                atol=atol,rtol=rtol,original_status=original.get('status'),
                original_numerically_accepted=original.get('numerically_accepted'),
                original_optimizer_trajectory_accepted=original.get('optimizer_trajectory_accepted'),
                source_hashes=original.get('source_hashes',protocol.get('source_hashes')),
                original_script_sha256=original.get('script_sha256',protocol.get('script_sha256')),
                interpretation='Raw coordinate failures remain failures. L2/cosine/clipped diagnostics describe scale and cancellation sensitivity; eager-vs-eager observations do not independently establish repeat nondeterminism.',
                pairs=pairs)
    (root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    summary={name:{key:row.get(key) for key in ('status','original_probe_pair_close','original_coordinate_failures','raw_global','global_clip')}
             for name,row in pairs.items()}
    print(json.dumps(dict(output=str(root/'result.json'),pairs=summary),ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
