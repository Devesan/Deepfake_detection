import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/gdevesan_gmail_com/resh_trans/vision_transformer')

# from vit_jax import input_pipeline
# from vit_jax import models
from vit_jax import checkpoint

checkpoint.load('/home/gdevesan_gmail_com/resh_trans/vit-1639785689/checkpoint_10000')
# vis_model = models.VisionTransformer
# checkpoint_path = '/home/common/vit-1639785689'

# vis_model.load(checkpoint_path)

# print(vis_model)