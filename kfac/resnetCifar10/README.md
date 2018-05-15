## Runs ResNet experiments
Example Usage:

`python cifar10_main.py --optimizer Adam --data_dir $HOME/datasets/cifar-10-batches-bin  --model_dir ./logs/ResNet32	--train_epochs 250	--epochs_per_eval 1  --batch_size 128 --lr 0.0005 --noiseLambda 0.1 --useNoiseType DiagF --eps 1e-8`

Values for each command options:

| Option        | Description / Values                  | 
|---------------|-------------------------| 
| `optimizer`     | `Adam`, `AdamNoisy` (noise to weights), `SGDMom` | 
| `useNoiseType`  | `None`, `DiagF`, `Gauss`(isotropic)  | 
| `data_dir`      | `/path/to/cifar-10-batches-bin` (Has to be that folder) |
| `model_dir`    | Base path where the model checkpoints will be stored. Sub folders will be created for different hyperparameter configurations |
| `train_epochs` | Number of epochs to train |
| `epochs_per_eval` | The number of epochs before evaluating train/test acc/loss |
| `batch_size` | The batch size to use |
| `lr` | Learning rate for the optimizer |
| `noiseLambda` | A scalar value to multiply the noise by before adding to the gradients or weights |
| `eps` | Epsilon value for Adam |

Also included are some bash scripts (`create_experiments_40.sh`) to generate/run sbatch experiment scripts. These probably will need to be modified a bit to work for your specific SLURM environment. 

The bash script simply varies over the batch sizes (and corresponding `train_epochs` and learning rate)
