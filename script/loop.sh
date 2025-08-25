#!/bin/bash
conda activate gauhuman

#DATASET='actorhq'
#SEQUENCES=('actor0101' 'actor0301' 'actor0601' 'actor0701' 'actor0801')
#STARTS=(0)
#for SEQUENCE in ${SEQUENCES[@]}; do
#    dataset="../data/$DATASET/$SEQUENCE"
#    for ((i = 0; i < ${#STARTS[@]}; i++)); do
#        CURRENT=${STARTS[i]}exit
#        if ((i + 1 < ${#STARTS[@]})); then
#            NEXT=${STARTS[i + 1]}
#        else
#            NEXT=-1  # Or handle the last element differently
#        fi
##        python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl \
##        --actor_gender neutral --iteration 2000 --white_background  --split novel_view --img_scale 1.0 \
##        --eval_start ${CURRENT} --eval_end ${NEXT} --skip 15
##
##        python render.py -s $dataset -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl \
##        --actor_gender neutral --iteration 2000 --white_background  --split novel_pose --img_scale 1.0 \
##        --eval_start ${CURRENT} --eval_end ${NEXT} --skip 5
#
#        python render.py -s $dataset -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl \
#        --actor_gender neutral --iteration 2000 --white_background  --split mesh_novel_view --img_scale 1.0 \
#        --eval_start ${CURRENT} --eval_end ${NEXT} --skip 1
#
#        python render.py -s $dataset -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl \
#        --actor_gender neutral --iteration 2000 --white_background  --split mesh_novel_pose --img_scale 1.0 \
#        --eval_start ${CURRENT} --eval_end ${NEXT} --skip 1
#    done
#done

DATASET='mvhuman'
#SEQUENCES=('100846' '100990' '102107' '102145' '103708' '200173' '204112' '204129')
SEQUENCES=('100846')
STARTS=(0)
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="../data/$DATASET/$SEQUENCE"
    for ((i = 0; i < ${#STARTS[@]}; i++)); do
        CURRENT=${STARTS[i]}
        # Check if there's a next element
        if ((i + 1 < ${#STARTS[@]})); then
            NEXT=${STARTS[i + 1]}
        else
            NEXT=-1  # Or handle the last element differently
        fi
#        python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl \
#        --actor_gender neutral --iteration 2000 --white_background  --split novel_view --img_scale 1.0 \
#        --eval_start ${CURRENT} --eval_end ${NEXT} --skip 10

        python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl \
        --actor_gender neutral --iteration 2000 --white_background  --split mesh_novel_view --img_scale 1.0 \
        --eval_start ${CURRENT} --eval_end ${NEXT} --skip 10
    done
done

#DATASET='mpi'
#SEQUENCES=('Antonia' 'Magdalena')
#STARTS=(0)
#for SEQUENCE in ${SEQUENCES[@]}; do
#    dataset="../data/$DATASET/$SEQUENCE"
#    for ((i = 0; i < ${#STARTS[@]}; i++)); do
#        CURRENT=${STARTS[i]}
#        # Check if there's a next element
#        if ((i + 1 < ${#STARTS[@]})); then
#            NEXT=${STARTS[i + 1]}
#        else
#            NEXT=-1  # Or handle the last element differently
#        fi
#        python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl \
#        --actor_gender neutral --iteration 2000 --white_background  --split novel_pose --img_scale 1.0 \
#        --eval_start ${CURRENT} --eval_end ${NEXT} --skip 2
#
#        python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl \
#            --actor_gender neutral --iteration 2000 --white_background  --split mesh_novel_pose --img_scale 1.0 \
#            --eval_start ${CURRENT} --eval_end ${NEXT} --skip 1
#
#        python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl \
#            --actor_gender neutral --iteration 2000 --white_background  --split mesh_training --img_scale 1.0 \
#            --eval_start ${CURRENT} --eval_end ${NEXT} --skip 1
#    done
#done


#DATASET='mpi'
#SEQUENCES=('0056' 'FranziRed')
#STARTS=(0)
#for SEQUENCE in ${SEQUENCES[@]}; do
#    dataset="../data/$DATASET/$SEQUENCE"
#    for ((i = 0; i < ${#STARTS[@]}; i++)); do
#        CURRENT=${STARTS[i]}
#        # Check if there's a next element
#        if ((i + 1 < ${#STARTS[@]})); then
#            NEXT=${STARTS[i + 1]}
#        else
#            NEXT=-1  # Or handle the last element differently
#        fi
#        python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl \
#        --actor_gender neutral --iteration 2000 --white_background  --split novel_view --img_scale 1.0 \
#        --eval_start ${CURRENT} --eval_end ${NEXT} --skip 5
#
#        python render.py -s $dataset -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl \
#        --actor_gender neutral --iteration 2000 --white_background  --split novel_pose --img_scale 1.0 \
#        --eval_start ${CURRENT} --eval_end ${NEXT} --skip 2
#
#        python render.py -s $dataset -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl \
#        --actor_gender neutral --iteration 2000 --white_background  --split mesh_novel_view --img_scale 1.0 \
#        --eval_start ${CURRENT} --eval_end ${NEXT} --skip 1
#
#        python render.py -s $dataset -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl \
#        --actor_gender neutral --iteration 2000 --white_background  --split mesh_novel_pose --img_scale 1.0 \
#        --eval_start ${CURRENT} --eval_end ${NEXT} --skip 1
#    done
#done

#DATASET='synwild'
#SEQUENCES=('00000_random' '00020_Dance' '00027_Phonecall' '00069_Dance' '00070_Dance')
#STARTS=(0)
#for SEQUENCE in ${SEQUENCES[@]}; do
#    dataset="../data/$DATASET/$SEQUENCE"
#    for ((i = 0; i < ${#STARTS[@]}; i++)); do
#        CURRENT=${STARTS[i]}
#        # Check if there's a next element
#        if ((i + 1 < ${#STARTS[@]})); then
#            NEXT=${STARTS[i + 1]}
#        else
#            NEXT=-1  # Or handle the last element differently
#        fi
#       python render.py -s $dataset -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl \
#        --actor_gender neutral --iteration 2000 --white_background  --split mesh_training --img_scale 1.0 \
#        --eval_start ${CURRENT} --eval_end ${NEXT} --skip 1
#    done
#done