SEQUENCES=('313' '377' '386' '387' '390' '392' '393' '394')

for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="../data/zju_mocap/$SEQUENCE"
    python train.py -s $dataset --eval --exp_name zju_mocap/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 2000
done