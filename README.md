# PROBA-V Super Resolution
## Multi-image Super-resolution

My attempt at the the PROBA-V Super Resolution competition.

Report [here](https://neuralburst.com/probav-super-resolution/ "Report")

## Requirements
Developed on tensorflow docker image: 2.0.0b1-gpu-py3-jupyter

## Running the code

### Generate TFRecords
Expects a folder with the [data](https://kelvins.esa.int/proba-v-super-resolution/data/ "Data") already downloaded.
```bash
python generate_tfrecords.py --data_dir $path_to_folder$
```

### Train with the training scripts

```bash
python train_srcnn.py -v --epoch 1000 --learning_rate 0.0001 --batch_size 4 
```
Resume training with ```--model_path $path$``` (for train_gan.py, resuming is not yet supported)

### Train with notebooks

SRCNN_notebook.ipynb and SRGAN_notebook.ipynb shows how to train each model

### Run predictions on the test set

```bash
python predict.py $path_to_model$ --output_dir $path$
```

### Scores
Lower is better:
<table class="tg">
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-0pky">Epoch</th>
    <th class="tg-0pky">Strategy (LRs)</th>
    <th class="tg-0pky">Losses</th>
    <th class="tg-0pky">Train Score</th>
    <th class="tg-0pky">Test Score</th>
  </tr>
  <tr>
    <td class="tg-0pky">Baseline</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">None</td>
    <td class="tg-0pky">None</td>
    <td class="tg-0pky">1.0062784200046624</td>
    <td class="tg-0pky">1.00000007339574</td>
  </tr>
  <tr>
    <td class="tg-0pky">SRCNN</td>
    <td class="tg-0pky">30</td>
    <td class="tg-0pky">Top-4 + Median</td>
    <td class="tg-0pky">SSIM</td>
    <td class="tg-0pky">1.0306669966571653</td>
    <td class="tg-0pky">###</td>
  </tr>
  <tr>
    <td class="tg-0pky">SRCNN</td>
    <td class="tg-0pky">60</td>
    <td class="tg-0pky">Top-4 + Median</td>
    <td class="tg-0pky">SSIM</td>
    <td class="tg-0pky">1.012628179551261</td>
    <td class="tg-0pky">###</td>
  </tr>
  <tr>
    <td class="tg-0pky">SRCNN</td>
    <td class="tg-0pky">80</td>
    <td class="tg-0pky">Top-35 + Median + normalize</td>
    <td class="tg-0pky">cMSE + cMAE</td>
    <td class="tg-0pky">1.2344703595833206</td>
    <td class="tg-0pky">###</td>
  </tr>
  <tr>
    <td class="tg-0pky">SRGAN</td>
    <td class="tg-0pky">10</td>
    <td class="tg-0pky">Top-4 + Median</td>
    <td class="tg-0pky">SSIM + GAN</td>
    <td class="tg-0pky">1.0041622868681903</td>
    <td class="tg-0pky">###</td>
  </tr>
  <tr>
    <td class="tg-0lax">SRGAN</td>
    <td class="tg-0lax">30</td>
    <td class="tg-0lax">Top-4 + Median</td>
    <td class="tg-0lax">SSIM + GAN</td>
    <td class="tg-0lax">1.00079132784449</td>
    <td class="tg-0lax">###</td>
  </tr>
</table>

### Sample Result

#### SRCNN
![resolved sample](https://neuralburst.com/content/images/2019/07/image-3.png)

#### SRGAN
![resolved sample](https://neuralburst.com/content/images/2019/07/image-6.png)
