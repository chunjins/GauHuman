DATASET='dna_rendering'
SEQUENCES=('0018_05')
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="../data/$DATASET/$SEQUENCE"
    python render.py -m output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 2000 --skip_novel_pose
done
