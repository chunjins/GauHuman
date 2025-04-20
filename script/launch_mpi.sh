conda activate gauhuman

#DATASET='mpi'
#SEQUENCES=('Antonia' 'Magdalena')
#for SEQUENCE in ${SEQUENCES[@]}; do
#    dataset="../data/$DATASET/$SEQUENCE"
#    python train.py -s $dataset --eval --exp_name ${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 2000 --white_background --split train --img_scale 1.0
#    python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 2000 --white_background  --split novel_pose --img_scale 1.0
#done

#DATASET='mpi'
#SEQUENCES=('0056' 'FranziRed')
#for SEQUENCE in ${SEQUENCES[@]}; do
#    dataset="../data/$DATASET/$SEQUENCE"
##    python train.py -s $dataset --eval --exp_name ${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 2000 --white_background --split train --img_scale 1.0
##    python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 2000 --white_background  --split novel_pose --img_scale 1.0
#    python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 2000 --white_background  --split novel_view --img_scale 1.0
#done


DATASET='mpi'
#SEQUENCES=('0056' 'FranziRed')
#SEQUENCES=('0056')
#STARTS=(0 500) nv

SEQUENCES=('FranziRed')
STARTS=(0 500 1000 1500 2000 2500) #nv
STARTS=(0 500) #nv

for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="../data/$DATASET/$SEQUENCE"
#    python train.py -s $dataset --eval --exp_name ${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 2000 --white_background --split train --img_scale 1.0
#    python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 2000 --white_background  --split novel_pose --img_scale 1.0
    for ((i = 0; i < ${#STARTS[@]}; i++)); do
        CURRENT=${STARTS[i]}
        # Check if there's a next element
        if ((i + 1 < ${#STARTS[@]})); then
            NEXT=${STARTS[i + 1]}
        else
            NEXT=-1  # Or handle the last element differently
        fi
        python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl \
        --actor_gender neutral --iteration 2000 --white_background  --split novel_pose --img_scale 1.0 \
        --eval_start ${CURRENT} --eval_end ${NEXT}
    done
done