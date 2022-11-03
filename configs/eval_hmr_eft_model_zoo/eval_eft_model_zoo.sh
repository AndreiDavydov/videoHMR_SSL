#!/bin/sh
runfile=run/run_proc.py
configdir=configs/eval_hmr_eft_model_zoo

echo "mpii_eft evaluation..."
bento console --local --kernel body_tracking --file $runfile -- -- --cfg $configdir/eval_hmr_mpii.yaml
echo "h36m_eft evaluation..."
bento console --local --kernel body_tracking --file $runfile -- -- --cfg $configdir/eval_hmr_h36m.yaml
echo "coco_part_eft evaluation..."
bento console --local --kernel body_tracking --file $runfile -- -- --cfg $configdir/eval_hmr_cocopart.yaml
echo "coco_all_eft evaluation..."
bento console --local --kernel body_tracking --file $runfile -- -- --cfg $configdir/eval_hmr_cocoall.yaml
echo "coco_all+h36m+mpii_eft evaluation..."
bento console --local --kernel body_tracking --file $runfile -- -- --cfg $configdir/eval_hmr_cocoall_h36m_mpii.yaml
echo "coco_all+h36m+mpii+posetrack+lsp+ochuman_eft evaluation..."
bento console --local --kernel body_tracking --file $runfile -- -- --cfg $configdir/eval_hmr_cocoall_h36m_mpii_posetrack_lsptrain_ochuman.yaml
echo "coco_all+h36m+mpii+posetrack+lsp+ochuman_eft_3DPW-T evaluation..."
bento console --local --kernel body_tracking --file $runfile -- -- --cfg $configdir/eval_hmr_cocoall_h36m_mpii_posetrack_lsptrain_ochuman_3dpwtrain.yaml
