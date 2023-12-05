import torch
pretrained_weights=torch.load('weights/detr-r50-e632da11.pth')
num_classes=1
pretrained_weights["model"]["class_embed.weight"].resize_(num_classes+1,256)
pretrained_weights["model"]["class_embed.bias"].resize_(num_classes+1)
torch.save(pretrained_weights,"detr_r50_%d.pth"%num_classes)
