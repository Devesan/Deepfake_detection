import sys
import jax
sys.path.insert(1, '/home/gdevesan_gmail_com/resh_trans/vision_transformer')
print(jax.device_count())
from vit_jax import train
from vit_jax.configs import models as models_config
from vit_jax import checkpoint
from vit_jax import models
from vit_jax import input_pipeline
from vit_jax.configs import common as common_config
# model_config = models_config.MODEL_CONFIGS['ViT-B_16']
# model = models.VisionTransformer(num_classes=2, **model_config)
# print(model)

# config = common_config.with_dataset(common_config.get_config(),'home/gdevesan_gmail_com/datasets/processed_images')
# ds_train,ds_test = input_pipeline.get_datasets(config=config)

# del config

# batch = next(iter(ds_test.as_numpy_iterator()))
# # Note the shape : [num_local_devices, local_batch_size, h, w, c]
# batch['image'].shape


# variables = jax.jit(lambda: model.init(
#     jax.random.PRNGKey(0),
#     # Discard the "num_local_devices" dimension of the batch for initialization.
#     batch['image'][0, :1],
#     train=False,
# ), backend='cpu')()
params = checkpoint.load(
    "/home/gdevesan_gmail_com/models_checkpoint/vit-1639849430/checkpoint_10000"
)