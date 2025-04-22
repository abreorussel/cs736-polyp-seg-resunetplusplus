import torch
import cv2
import numpy as np
from pytorch_grad_cam import ScoreCAM, GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms import v2


from core.res_unet_plus import ResUnetPlusPlus
from core.res_unet import ResUnet
from dataset.polyps_dataloader import *


# Resunet++
model = ResUnetPlusPlus(channel=3)
checkpoint_path = 'checkpoints/resunet++/default_checkpoint_32000.pt'


#Resunet
# model = ResUnet(3)
# checkpoint_path = 'checkpoints/resunet/default_checkpoint_50000.pt'

checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

transform = v2.Compose([
    GrayscaleNormalizationSI(mean=0.5, std=0.5),
    ToTensorSI(),
])


target_layer = model.aspp_out.output  # resunet++
# target_layer = model.up_residual_conv3 # resunet

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:3' if use_cuda else 'cpu')
model.to(device)
cam = ScoreCAM(model=model, target_layers=[target_layer])

# img_path = 'new_data/Kvasir-SEG/test/images/cju1bm8063nmh07996rsjjemq.jpg'  #perfect
# img_path = 'new_data/Kvasir-SEG/test/images/cju1c4fcu40hl07992b8gj0c8.jpg'   ok
# img_path = 'new_data/Kvasir-SEG/test/images/cju1cbokpuiw70988j4lq1fpi.jpg'   #perfect
# img_path = 'new_data/Kvasir-SEG/test/images/cju1cdxvz48hw0801i0fjwcnk.jpg'
# img_path = 'new_data/Kvasir-SEG/test/images/cju1cnnziug1l0835yh4ropyg.jpg'  #perfect
img_path = 'new_data/Kvasir-SEG/test/images/cju1d31sp4d4k0878r3fr02ul.jpg'


orig_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
transformed_img = transform(orig_bgr)

orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB) / 255.0

print(transformed_img.shape)
_, H, W = transformed_img.shape
mask = np.ones((H, W), dtype=np.float32)

targets = [SemanticSegmentationTarget(0, mask)]

grayscale_cam = cam(input_tensor=torch.from_numpy(transformed_img).unsqueeze(0).float().to(device), targets=targets)[0]  # HxW numpy array
visualization = show_cam_on_image(orig_rgb, grayscale_cam, use_rgb=True)

cv2.imwrite('scorecam_overlay.png', cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
print('ScoreCAM overlay saved to scorecam_overlay.png')
