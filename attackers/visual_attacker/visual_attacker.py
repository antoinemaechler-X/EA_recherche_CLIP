import torch
from tqdm import tqdm
import clip
import random
import kornia.filters as kf
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import seaborn as sns

def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images
    
        

class Attacker:
    def __init__(self,model,classes, device='cuda:0', eps = 1/255):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eps = eps
        self.classes = classes
        self.text = clip.tokenize(self.classes).to(self.device)

        self.model.eval()
        self.model.requires_grad_(False)
    def loss(self, img, target):
        x = denormalize(img).clone().to(self.device)
        x_adv = normalize(x)
        logits_per_image, logits_per_text = self.model(x_adv, self.text)
        # Calculate loss and append it to the loss list
        target_loss = torch.nn.functional.cross_entropy(logits_per_image, target)
        return(target_loss)
    
    def generate_prompt(self,image):
        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(image, self.text)
            probs = torch.nn.functional.softmax(logits_per_image, dim=-1)
        return self.classes[probs.argmax().item()]
    
    def generate_prompt_v2(self,image):
    # for finetuned model
        with torch.no_grad():
            logits_per_image = self.model(image, self.text)[0]
            probs = torch.nn.functional.softmax(logits_per_image, dim=-1)
        return self.classes[probs.argmax().item()]

    def get_top5_probs(self, image):
        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(image, self.text)
            probs = torch.nn.functional.softmax(logits_per_image, dim=-1)  
            
            top5_probs, top5_indices = probs.topk(5, dim=-1)
            
        return [(top5_probs[0, i].item(), self.classes[top5_indices[0, i].item()]) for i in range(5)]
    
    def predict(self, img):
        return(self.generate_prompt(img))

    def proba_vect(self, img):
        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(img, self.text)
        return logits_per_image

    def attack_specific(self, img, target, num_iter = 2000, alpha = 0.01):
        adv_noise = torch.randn_like(img).to(self.device) * 2 * self.eps - self.eps
        x = denormalize(img).clone().to(self.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data 
        adv_noise.requires_grad = True
        adv_noise.retain_grad()

        loss_values = []

        for t in tqdm(range(num_iter)):
            x_adv = normalize(x + adv_noise)
            logits_per_image, logits_per_text = self.model(x_adv, self.text)            
            target_loss = torch.nn.functional.cross_entropy(logits_per_image,target.to(self.device))
            loss_values.append(target_loss.item())
            
            target_loss.backward()
            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-self.eps, self.eps)
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data            
            adv_noise.grad.zero_()
            self.model.zero_grad()
            
            if t % 10 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv = x + adv_noise
                x_adv = normalize(x_adv)

                with torch.no_grad():
                    print('>>> Sample Outputs')
                    print(self.generate_prompt(x_adv))
                adv_img_prompt = denormalize(x_adv).detach().cpu()
        return adv_img_prompt, loss_values
    
    def attack_specific_vfinetuned(self, img, target, num_iter = 2000, alpha = 0.01):
        # to have only one input, cf model finetuned
        adv_noise = torch.randn_like(img).to(self.device) * 2 * self.eps - self.eps
        x = denormalize(img).clone().to(self.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data 
        adv_noise.requires_grad = True
        adv_noise.retain_grad()

        loss_values = []

        for t in tqdm(range(num_iter)):
            x_adv = normalize(x + adv_noise)
            logits_per_image = self.model(x_adv, self.text)[0]         
            target_loss = torch.nn.functional.cross_entropy(logits_per_image,target.to(self.device))
            loss_values.append(target_loss.item())
            
            target_loss.backward()
            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-self.eps, self.eps)
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data            
            adv_noise.grad.zero_()
            self.model.zero_grad()
            
            if t % 10 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv = x + adv_noise
                x_adv = normalize(x_adv)

                with torch.no_grad():
                    print('>>> Sample Outputs')
                    # print(self.generate_prompt_v2(x_adv))
                adv_img_prompt = denormalize(x_adv).detach().cpu()
        return adv_img_prompt, loss_values
    
    def attack_unspecific(self, img, model_output, num_iter=2000, alpha = 0.01):
        adv_noise = torch.randn_like(img).to(self.device) * 2 * self.eps - self.eps
        x = denormalize(img).clone().to(self.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data 
        adv_noise.requires_grad = True
        adv_noise.retain_grad()

        loss_values = []

        for t in tqdm(range(num_iter)):
            x_adv = normalize(x + adv_noise)
            logits_per_image, logits_per_text = self.model(x_adv, self.text)            
            target_loss = -torch.nn.functional.cross_entropy(logits_per_image,model_output.to(self.device)) # moins la loss pour que le modèle s'éloigne de l'output original
            loss_values.append(target_loss.item())
            
            target_loss.backward()
            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-self.eps, self.eps)
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data            
            adv_noise.grad.zero_()
            self.model.zero_grad()
            
            if t % 10 == 0:
                print('######### Output - Iter = %d ##########' % t)
                x_adv = x + adv_noise
                x_adv = normalize(x_adv)

                with torch.no_grad():
                    print('>>> Sample Outputs')
                    print(self.generate_prompt(x_adv))
                adv_img_prompt = denormalize(x_adv).detach().cpu()
        return adv_img_prompt, loss_values
    
    def eot_attack_specific(self, img, target, num_iter=2000, alpha=0.01, sigma_values=None, n_monte_carlo = 3):
        adv_noise = torch.randn_like(img).to(self.device) * 2 * self.eps - self.eps
        x = denormalize(img).clone().to(self.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
        adv_noise.requires_grad = True
        adv_noise.retain_grad()

        if sigma_values is None:
            sigma_values = [x / 20 for x in range(1, 100)]  # Default sigma range for Gaussian blur

        loss_values = []

        for t in tqdm(range(num_iter)):
            eot_loss = 0
            for _ in range(n_monte_carlo): 
                sigma = random.choice(sigma_values)
                blurred_img = kf.bilateral_blur(x + adv_noise, kernel_size=(5, 5), sigma_color=sigma, sigma_space=(sigma,sigma)) # TODO : Check the impact of kernel_size, allow free choice of sigma_space and sigma_color
                x_adv = normalize(blurred_img)
                logits_per_image, logits_per_text = self.model(x_adv, self.text)
                target_loss = torch.nn.functional.cross_entropy(logits_per_image, target.to(self.device))
                eot_loss += target_loss
            eot_loss /= 10 
            loss_values.append(eot_loss.item())

            eot_loss.backward()
            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-self.eps, self.eps)
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
            adv_noise.grad.zero_()
            self.model.zero_grad()

            if t % 10 == 0:
                print(f'######### Output - Iter = {t} ##########')
                x_adv = x + adv_noise
                x_adv = normalize(x_adv)

                with torch.no_grad():
                    print('>>> Sample Outputs')
                    print(self.generate_prompt(x_adv))
        x_adv = (x + adv_noise).detach().cpu()
        return x_adv, loss_values
    
    def eot_attack_unspecific(self, img, model_output, num_iter=2000, alpha=0.01, sigma_values=None, n_monte_carlo = 3):
        adv_noise = torch.randn_like(img).to(self.device) * 2 * self.eps - self.eps
        x = denormalize(img).clone().to(self.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
        adv_noise.requires_grad = True
        adv_noise.retain_grad()

        if sigma_values is None:
            sigma_values = [x / 20 for x in range(1, 100)]  # Default sigma range for Gaussian blur

        loss_values = []

        for t in tqdm(range(num_iter)):
            eot_loss = 0
            for _ in range(n_monte_carlo): 
                sigma = random.choice(sigma_values)
                blurred_img = kf.bilateral_blur(x + adv_noise, kernel_size=(5, 5), sigma_color=sigma, sigma_space=(sigma,sigma)) # TODO : Check the impact of kernel_size, allow free choice of sigma_space and sigma_color
                x_adv = normalize(blurred_img)
                logits_per_image, logits_per_text = self.model(x_adv, self.text)
                target_loss = -torch.nn.functional.cross_entropy(logits_per_image, model_output.to(self.device))
                eot_loss += target_loss
            eot_loss /= 10 
            loss_values.append(eot_loss.item())

            eot_loss.backward()
            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-self.eps, self.eps)
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
            adv_noise.grad.zero_()
            self.model.zero_grad()

            if t % 10 == 0:
                print(f'######### Output - Iter = {t} ##########')
                x_adv = x + adv_noise
                x_adv = normalize(x_adv)

                with torch.no_grad():
                    print('>>> Sample Outputs')
                    print(self.generate_prompt(x_adv))
        x_adv = (x + adv_noise).detach().cpu()
        return x_adv, loss_values