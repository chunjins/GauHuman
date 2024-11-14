SEQUENCES=("313")
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="../data/zju_mocap/$SEQUENCE"
    python train.py -s $dataset --eval --exp_name zju_mocap/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 1200
done