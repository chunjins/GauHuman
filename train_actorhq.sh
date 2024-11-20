SEQUENCES=('actor0101' 'actor0301' 'actor0601' 'actor0701')
SEQUENCES=('actor0301')
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="../data/actorhq/$SEQUENCE"
    python train.py -s $dataset --eval --exp_name actorhq/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 2000
done