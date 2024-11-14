SEQUENCES=("313")
for SEQUENCE in ${SEQUENCES[@]}; do
    python render.py -m output/zju_mocap/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 1200
done