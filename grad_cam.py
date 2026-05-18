
import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = []
        self.featuremaps = []

        # Register hooks
        target_layer.register_forward_hook(self._save_featuremaps)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_featuremaps(self, module, input, output):
        self.featuremaps.append(output.clone().detach())

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].clone().detach())

    def __call__(self, image, label=None):
        self.featuremaps.clear()
        self.gradients.clear()

        output = self.model(image)

        if label is None:
            label = output.argmax(dim=1).item()

        # Compute the loss with respect to the correct class
        loss = output[0, label]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        # Get gradients and featuremaps
        gradients = self.gradients[-1]
        featuremaps = self.featuremaps[-1]

        # Compute Grad-CAM
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * featuremaps).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[2:], mode='bilinear', align_corners=False)

        # Normalize cam to [0, 1]
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return output, cam.squeeze().cpu().numpy()
