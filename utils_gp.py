import torch
import torch.nn as nn


def gp_gradient_penalty(critic, real, fake, device="cpu"):
    batch_size, c, h, w = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, c, h, w).to(device)
    interpolated_image = (real * epsilon) + (fake * (1 - epsilon))

    # calculate critic scores
    score = critic(interpolated_image)

    # first we take the gradient of score with respect to the inputs/interpolated_image
    gradient = torch.autograd.grad(
        inputs=interpolated_image,
        outputs=score,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Compute the gradient penalty with respect to the gradient
    # # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1) # l2 normalization
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty
