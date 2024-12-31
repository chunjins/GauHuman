DATASET='dna_rendering'
SEQUENCES=('0018_05')
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="../data/$DATASET/$SEQUENCE"
    python train.py -s $dataset --eval --exp_name ${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 2000 --white_background
done