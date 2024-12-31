conda activate gauhuman

DATASET='actorhq'
SEQUENCES=('actor0101' 'actor0301' 'actor0601' 'actor0701' 'actor0801')
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="../data/$DATASET/$SEQUENCE"
    python train.py -s $dataset --eval --exp_name ${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 2000 --white_background --split train --img_scale 1.0
    python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 2000 --white_background  --split novel_pose --img_scale 1.0
    python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 2000 --white_background  --split novel_view --img_scale 1.0
done

DATASET='mvhuman'
subjects=('100846' '100990' '102107' '102145' '103708' '200173' '204112' '204129')
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="../data/$DATASET/$SEQUENCE"
    python train.py -s $dataset --eval --exp_name ${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 2000 --white_background --split train --img_scale 1.0
    python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 2000 --white_background  --split novel_view --img_scale 1.0
done

DATASET='mpi'
subjects=('Antonia' 'Magdalena')
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="../data/$DATASET/$SEQUENCE"
    python train.py -s $dataset --eval --exp_name ${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 2000 --white_background --split train --img_scale 1.0
    python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 2000 --white_background  --split novel_pose --img_scale 1.0
done

DATASET='mpi'
subjects=('0056' 'FranziRed')
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="../data/$DATASET/$SEQUENCE"
    python train.py -s $dataset --eval --exp_name ${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 2000 --white_background --split train --img_scale 1.0
    python render.py -m ../output/${DATASET}/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 2000 --white_background  --split novel_pose --img_scale 1.0
done

