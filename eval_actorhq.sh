conda activate gart
SEQUENCES=('actor0101' 'actor0301' 'actor0601' 'actor0701')
SEQUENCES=('actor0301')
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="../data/actorhq/$SEQUENCE"
    python render.py -m ../output/actorhq/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 1200 --white_background  --split novel_view --img_scale 1.0
done
