import torch
import clip
from PIL import Image
import requests
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToPILImage
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visual_attacker import Attacker
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import kornia.filters as kf

# Create directory to save the results
output_dir = "defense_results"
os.makedirs(output_dir, exist_ok=True)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
num_iter = 11

# Download the CIFAR-100 dataset and pick consistent images
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
n_samples = 10  # Fixed number of images
fixed_indices = list(range(n_samples))  # Use first 50 images (fixed)
images_and_labels = [(cifar100[i][0], cifar100[i][1]) for i in fixed_indices]
target_class_id = 0
batch_size = 2
eps = 8/255

# Range of sigma values
sigmas = [i / 5 for i in range(1, 16)]
specific_attack_success_rates = []
unspecific_attack_success_rates = []
denoised_specific_attack_success_rates = []
denoised_unspecific_attack_success_rates = []
consistency_success_rates = []
consistency_with_clip = []

def get_one_hot_from_class(class_name):
    return torch.tensor([cifar100.classes.index(class_name)])
# Function to save batch results and return top 5 probabilities
def save_batch_denoised_figures(images_and_labels, batch_index, model, attacker, sig):
    batch_ground_truth_preds = []
    batch__denoised_ground_truth_preds = []
    batch_specific_attack_preds = []
    batch__denoised_specific_attack_preds = []
    batch_unspecific_attack_preds = []
    batch__denoised_unspecific_attack_preds = []
    

    for i, (image, labels) in enumerate(images_and_labels):
        image_input = preprocess(image).unsqueeze(0).to(device)
        target = torch.tensor([target_class_id]).to(device)  # Set target class for specific attack
        model_output = get_one_hot_from_class(attacker.generate_prompt(image_input)).to(device)
        # Perform specific and unspecific attacks
        adv_img_specific, _ = attacker.attack_specific(image_input, target, num_iter=num_iter)
        adv_img_unspecific, _ = attacker.attack_unspecific(image_input, model_output, num_iter=num_iter)

        #denoise the images
        denoised_image_input = kf.bilateral_blur(image_input, kernel_size=(5, 5), sigma_color=sig, sigma_space=(sig,sig))
        denoised_adv_img_specific = kf.bilateral_blur(adv_img_specific, kernel_size=(5, 5), sigma_color=sig, sigma_space=(sig,sig))
        denoised_adv_img_unspecific = kf.bilateral_blur(adv_img_unspecific, kernel_size=(5, 5), sigma_color=sig, sigma_space=(sig,sig))

        # Get predictions
        ground_truth_prediction = attacker.generate_prompt(image_input)
        denoised_ground_truth_prediction = attacker.generate_prompt(denoised_image_input)
        specific_attack_prediction = attacker.generate_prompt(adv_img_specific.to(device))
        denoised_specific_attack_prediction = attacker.generate_prompt(denoised_adv_img_specific.to(device))
        unspecific_attack_prediction = attacker.generate_prompt(adv_img_unspecific.to(device))
        denoised_unspecific_attack_prediction = attacker.generate_prompt(denoised_adv_img_unspecific.to(device))

        # Collect predictions for success calculation later
        batch_ground_truth_preds.append(ground_truth_prediction)
        batch__denoised_ground_truth_preds.append(denoised_ground_truth_prediction)
        batch_specific_attack_preds.append(specific_attack_prediction)
        batch__denoised_specific_attack_preds.append(denoised_specific_attack_prediction)
        batch_unspecific_attack_preds.append(unspecific_attack_prediction)
        batch__denoised_unspecific_attack_preds.append(denoised_unspecific_attack_prediction)

    return batch_ground_truth_preds, batch__denoised_ground_truth_preds, batch_specific_attack_preds, batch__denoised_specific_attack_preds, batch_unspecific_attack_preds, batch__denoised_unspecific_attack_preds

# Create attacker instance
attacker = Attacker(model, cifar100.classes, device=device, eps=eps)

# Loop over different sigma values and calculate success rates
for sig in sigmas:
    print(f"Processing for sigma = {sig}")
    
    # Reset success counters for this epsilon
    specific_attack_success = 0
    denoised_specific_attack_success = 0
    denoised_right_label = 0
    unspecific_attack_success = 0
    denoised_unspecific_attack_success = 0
    normal_predict_success = 0
    same_as_clip = 0

    # Process images in batches
    for batch_index in range(0, n_samples, batch_size):
        batch_images_and_labels = images_and_labels[batch_index:batch_index + batch_size]

        # Save the current batch results as a figure with top 5 predictions
        ground_truth_preds, denoised_ground_truth_preds, specific_preds, denoised_specific_preds, unspecific_preds, denoised_unspecific_preds = save_batch_denoised_figures(
            batch_images_and_labels, batch_index // batch_size, model, attacker, sig
        )

        _, labels = zip(*batch_images_and_labels)

        # Perform success rate calculations using the returned predictions
        for gt_pred, dns_gt_pred, spec_pred, dns_spec_pred, unspec_pred, dns_unspec_pred, label in zip(ground_truth_preds, denoised_ground_truth_preds, specific_preds, denoised_specific_preds, unspecific_preds, denoised_unspecific_preds, labels):
            if spec_pred == cifar100.classes[target_class_id]:  # Success if it predicts the target
                specific_attack_success += 1
            if dns_spec_pred == cifar100.classes[target_class_id]:  # Success if it predicts the target
                denoised_specific_attack_success += 1
            if dns_spec_pred == label:  # Failure if it gets the right label
                denoised_right_label += 1
            if unspec_pred != gt_pred:  # Success if it predicts anything other than the original prediction
                unspecific_attack_success += 1
            if dns_unspec_pred != gt_pred:  # Success if it predicts anything other than the original prediction
                denoised_unspecific_attack_success += 1
            if dns_gt_pred == label:  # Success if it predicts the same as the original label
                normal_predict_success += 1
            if dns_gt_pred == gt_pred: # Success if it predicts the same as the original prediction
                same_as_clip += 1
            

    # Calculate success rates for this sigma
    specific_attack_success_rate = specific_attack_success / n_samples * 100
    unspecific_attack_success_rate = unspecific_attack_success / n_samples * 100
    denoised_specific_attack_success_rate = denoised_specific_attack_success / n_samples * 100
    denoised_unspecific_attack_success_rate = denoised_unspecific_attack_success / n_samples * 100
    consistency_success_rate = normal_predict_success / n_samples *100
    denoised_right_label_rate = denoised_right_label / n_samples *100
    same_as_clip_rate = same_as_clip / n_samples *100

    specific_attack_success_rates.append(specific_attack_success_rate)
    unspecific_attack_success_rates.append(unspecific_attack_success_rate)
    denoised_specific_attack_success_rates.append(denoised_specific_attack_success_rate)
    denoised_unspecific_attack_success_rates.append(denoised_unspecific_attack_success_rate)
    consistency_success_rates.append(consistency_success_rate)
    consistency_with_clip.append(same_as_clip_rate)

    print(f"Sigma: {sig}, Specific Attack Success Rate: {specific_attack_success_rate:.2f}%")
    print(f"Sigma: {sig}, Unspecific Attack Success Rate: {unspecific_attack_success_rate:.2f}%")
    print(f"Sigma: {sig}, Denoised Specific Attack Success Rate: {denoised_specific_attack_success_rate:.2f}%")
    print(f"Sigma: {sig}, Denoised Unspecific Attack Success Rate: {denoised_unspecific_attack_success_rate:.2f}%")
    print(f"Sigma: {sig}, Consistency Success Rate: {consistency_success_rate:.2f}%")
    print(f"Sigma: {sig}, Consistency Success Rate: {consistency_success_rate:.2f}%")

# Plot success rate vs sigma
plt.figure(figsize=(8, 6))
plt.plot(sigmas, specific_attack_success_rates, label="Specific Attack", marker='o', color='blue')
plt.plot(sigmas, unspecific_attack_success_rates, label="Unspecific Attack", marker='o', color='red')
plt.plot(sigmas, denoised_specific_attack_success_rates, label="Denoised Specific Attack", marker='o', color='green')
plt.plot(sigmas, denoised_unspecific_attack_success_rates, label="Denoised Unspecific Attack", marker='o', color='purple')
plt.plot(sigmas, consistency_success_rates, label='Consistency without attack', marker='o', color='yellow')
plt.title("Adversarial Attack Success Rate vs Sigma")
plt.xlabel("Sigma")
plt.ylabel("Success Rate (%)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "attack_success_vs_sigma.png"))
plt.close()