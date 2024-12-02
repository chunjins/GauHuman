SEQUENCES=('377' '386' '387' '390' '392' '393' '394')
SEQUENCES=('390')
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="data/zju_mocap/$SEQUENCE"
    python render.py -m output/zju_mocap/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 2000 --skip_novel_view
done
