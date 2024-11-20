SEQUENCES=('100846' '100990' '102107' '102145' '103708' '200173' '204112' '204129')
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="data/mvhuman/$SEQUENCE"
    python render.py -m output/mvhuman/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 2000 --white_background
done
