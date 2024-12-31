conda activate gauhuman

#DATASET='actorhq'
#SEQUENCES=('actor0101' 'actor0301' 'actor0601' 'actor0701' 'actor0801')
#for SEQUENCE in ${SEQUENCES[@]}; do
#    dataset="../data/$DATASET/$SEQUENCE"
#    python train.py -s $dataset --eval --exp_name ${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 2000 --white_background --split train --img_scale 0.5
##    python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 2000 --white_background  --split novel_pose --img_scale 1.0
##    python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 2000 --white_background  --split novel_view --img_scale 1.0
#done

DATASET='mvhuman'
SEQUENCES=('100846' '100990' '102107' '102145' '103708' '200173' '204112' '204129')
STARTS=(0 500 1000)

for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="../data/$DATASET/$SEQUENCE"
#    python train.py -s $dataset --eval --exp_name ${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 2000 --white_background --split train --img_scale 1.0

    for ((i = 0; i < ${#STARTS[@]}; i++)); do
        CURRENT=${STARTS[i]}
        # Check if there's a next element
        if ((i + 1 < ${#STARTS[@]})); then
            NEXT=${STARTS[i + 1]}
        else
            NEXT=-1  # Or handle the last element differently
        fi
        python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl \
        --actor_gender neutral --iteration 2000 --white_background  --split novel_view --img_scale 1.0 \
        --eval_start ${CURRENT} --eval_end ${NEXT}
    done
done

#DATASET='mpi'
#SEQUENCES=('Antonia' 'Magdalena')
#for SEQUENCE in ${SEQUENCES[@]}; do
#    dataset="../data/$DATASET/$SEQUENCE"
#    python train.py -s $dataset --eval --exp_name ${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 2000 --white_background --split train --img_scale 0.5
##    python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 2000 --white_background  --split novel_pose --img_scale 1.0
#done
#
#DATASET='mpi'
#SEQUENCES=('0056' 'FranziRed')
#for SEQUENCE in ${SEQUENCES[@]}; do
#    dataset="../data/$DATASET/$SEQUENCE"
#    python train.py -s $dataset --eval --exp_name ${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 2000 --white_background --split train --img_scale 0.5
##    python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 2000 --white_background  --split novel_pose --img_scale 1.0
#done

