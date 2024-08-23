# How to set dataset

1. Download dataset at [this link](https://itsacid-my.sharepoint.com/:f:/g/personal/5025201216_student_its_ac_id/Ekl4y-VZ9fxMtMmDsRKF0kABKfD8EQGZDMQhCdIt3LxZzg?e=Pl1Yro)
2. Make `/dataset/` folder at root directory
3. Copy all dataset folders to the `/dataset/` directory

## Install environment
```bash
pip install -r requirements.txt
```

## Explanation
- Ada 4 dataset yang dapat dilakukan eksperimen, yaitu National Illness (ILI), Exchange, Weather, dan ETT
- Setiap runner script dapat diakses di [./rlscripts/M/](./rlscripts/M/)
- Setiap dataset memiliki runner scriptnya di folder yang sesuai dengan nama datasetnya
- Tiap script melakukan 3x run untuk 1 skenario eksperimen. Misalnya, untuk eksperimen ILI_36_24, dilakukan 3x eksperimen dengan deskripsi `(des)` normal_0, normal_1, normal_2 (**Dapat melihat [./rlscirpts/M/ILI/ILI_all_normal.sh](./rlscripts/M/ILI/ILI_all_normal.sh) untuk lebih jelasnya**)
- Tiap 1x run eksperimen, terdapat 2 file yang di run, yaitu `run_open_net.py` dan `run_rlmc.py`
- File `run_open_net.py` digunakan untuk melakukan *training* jaringan OpenNet, sedangkan file `run_rlmc.py` digunakan untuk melakukan *training* jaringan *reinforcement learning*-nya.

## Dataset Information
| Dataset Name      | Dataset Code  | Dataset Runner Folder                 |
| ------------      | :----------:  | ---------------------                 |
| National Illness  | ILI           | [ILI scripts](./rlscripts/M/ILI/)     |
| Exchange          | Exchange      | [Exchange scripts](./rlscripts/M/Exchange/)     |
| Weather           | Weather       | [Weather scripts](./rlscripts/M/Weather/)      |
| Electricity       | ETTm2         | [ETT scripts](./rlscripts/M/ETT/)     |

## Run python
```bash
bash ./rlscripts/M/{dataset_name}/{dataset_name}_all_normal.sh
```

## Results
- Setelah melakukan running eksperimen, hasil dari eksperimen dapat dilihat pada [`./checkpoints`](./checkpoints/). untuk tiap nama eksperimennya.
- Tiap folder akan berisi 4 subfolder dan 1 file model.
- Sub-sub folder tersebut, yaitu: *dataset, rl_bm, testing_results, train_results*
- 1 file model memiliki nama `checkpoint.pth`

### Keterangan Subfolder
- Subfolder `dataset` akan berisi data-data yang nantinya dipakai pada saat *training* jaringan *reinforcement learning*
- Subfolder `rl_bm` merupakan data base model dari tiap *learner* jaringan OpenNet
- Subfolder `testing_results` berisi data hasil *test* untuk OpenNet dan *reinforcement learning* yang dipisah berdasarkan nama folder
- Subfolder `training_results` berisi data hasil *train* untuk OpenNet dan *reinforcement learning* yang dipisah berdasarkan nama folder