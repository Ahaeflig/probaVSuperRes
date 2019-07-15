# PROBA-V Super Resolution
## Multi-image Super-resolution

In this repo I tackle the PROBA-V Super Resolution competition.

Report [here](https://neuralburst.com/probav-super-resolution/ "Report")

## Running the code

### Generate TFRecords

```bash
python generate_tfrecords.py --data_dir $path_to_folder$
```

### Run the training scripts

```bash
python train_srcnn.py -v --epoch 1000 --learning_rate 0.0001 --batch_size 4 
```
Resume training with ```--model_path $path$```

### Run predictions on the test set

```bash
python predict.py $path_to_model$ --output_dir $path$
```


### Sample Result
(models still training)

#### SRCNN
![resolved sample](https://neuralburst.com/content/images/2019/07/399-1.png)

#### SRGAN
