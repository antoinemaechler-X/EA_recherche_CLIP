# EA recherche CLIP robustness

The goal of this project is to enhance the robustness of the VLM CLIP against adversarial attacks.
Adversarial attacks are small modifications of images (not perceivable by human eye) but that lead the VLM to answer falsely (unspecific attack) or another specific output chosen by attacker (specific attack).

We first implement attacks and then see how we can defend against them and enhance the robustness of CLIP.

**Attacks**

We use three methods: Projected Gradient Descent (specific), local maximization of loss function (unspecific attack) and Expectation Over Transformation (EOT).
We get such results, that is classified by the VLM as "an image of a car".

<img width="497" alt="Capture d’écran 2024-10-28 à 16 57 01" src="https://github.com/user-attachments/assets/fad062ae-bfe3-4bf1-9f73-e5acc4202a0d">

All further tests are done on cifar-100 dataset.

**Defense before inference**

We use 2 mains techniques in defense before inference:

- Gaussian noise method: we add some random Gaussian noise to the image and use Monte-Carlo method to get the most likely output.
- Denoising method: we use and optimize denoising methods (Blur + edge conservation) to smooth the perturbation of the attack, thus defend while keeping the characteristics of the image (edge preserving)

We get the following results for the denoising method against specific PGD depending on sigma (the spacial parameter of the blur):

<img width="800" alt="image" src="https://github.com/user-attachments/assets/fa0aafb3-cf83-4d90-aa48-92536cf5ff26" />

**Defense through finetuning**

We use here Unsupervised Adversarial Finetuning to make the model more robust. We get the following results: 

<img width="587" alt="image" src="https://github.com/user-attachments/assets/1d8fb75c-4351-41f7-bcee-ee716a752187" />

Because of the drop in performance against non-attacked images, we implement **Orthogonal finetuning** to defend while preserving the initial performance of the model. We get the follonwing results:

<img width="585" alt="image" src="https://github.com/user-attachments/assets/b8444740-a5f2-470e-872d-697a97d34e89" />

The pdf report is available in the github.
