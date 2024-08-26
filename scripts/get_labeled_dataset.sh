# download green dataset
sh ./scripts/clean-preprocess.sh temp_part_a 2021-10-15 2021-10-22
sh ./scripts/clean-preprocess.sh temp_part_b 2021-12-05 2021-12-13
sh ./scripts/clean-preprocess.sh temp_part_c 2022-03-19 2022-03-28
sh ./scripts/clean-preprocess.sh temp_part_d 2022-04-19 2022-04-27
sh ./scripts/clean-preprocess.sh temp_part_e 2022-08-06 2022-08-12
sh ./scripts/clean-preprocess.sh temp_part_f 2022-10-07 2022-10-16
sh ./scripts/clean-preprocess.sh temp_part_g 2023-01-01 2023-01-07
sh ./scripts/clean-preprocess.sh temp_part_h 2023-08-01 2023-08-14
sh ./scripts/clean-preprocess.sh temp_part_i 2023-08-14 2023-08-18
sh ./scripts/clean-preprocess.sh temp_part_j 2023-08-22 2023-08-25
sh ./scripts/clean-preprocess.sh temp_part_k 2023-12-17 2023-12-22
sh ./scripts/clean-preprocess.sh temp_part_l 2024-01-02 2024-01-06
sh ./scripts/clean-preprocess.sh temp_part_m 2024-01-15 2024-01-19
sh ./scripts/clean-preprocess.sh temp_part_n 2024-01-31 2024-02-06
sh ./scripts/clean-preprocess.sh temp_part_o 2024-02-09 2024-02-13
sh ./scripts/clean-preprocess.sh temp_part_p 2024-02-22 2024-03-01
sh ./scripts/clean-preprocess.sh temp_part_q 2024-03-04 2024-03-07
sh ./scripts/clean-preprocess.sh temp_part_r 2024-03-20 2024-03-25
sh ./scripts/clean-preprocess.sh temp_part_s 2024-03-28 2024-03-31
sh ./scripts/clean-preprocess.sh temp_part_t 2024-04-12 2024-04-16

flux-merge --dataset temp_part_a temp_part_b temp_part_c temp_part_d temp_part_e temp_part_f \
    temp_part_g temp_part_h temp_part_i temp_part_j temp_part_k temp_part_l temp_part_m \
    temp_part_n temp_part_o temp_part_p temp_part_q temp_part_r temp_part_s temp_part_t \
    --export dataset-cat-green-raw

# download orange dataset
sh ./scripts/clean-preprocess.sh temp_part_x 2023-04-01 2023-04-27
sh ./scripts/clean-preprocess.sh temp_part_y 2023-07-01 2023-07-17

flux-merge --dataset temp_part_x temp_part_y --export dataset-cat-orange-raw

rm -rf temp_part_*

sh ./scripts/clean-postprocess.sh dataset-cat-green
sh ./scripts/clean-postprocess.sh dataset-cat-orange