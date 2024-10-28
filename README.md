# EA recherche CLIP robustness

**[Ongoing project]**

The goal of this project is to enhance the robustness of the VLM CLIP against adversarial attacks.
Adversarial attacks are small modifications of images (not perceivable by human eye) but that lead the VLM to answer falsely (unspecific attack) or another specific outputchosen by attacker (specific attack).

We first implement attacks and then see how we can defend against them and enhance the robustness of CLIP.

**Attacks**

We use two methods: gradient descent (specific attack) and local maximization of loss function (unspecific attack).
We get such results, that is classified by the VLM as "an image of a car".

<img width="497" alt="Capture d’écran 2024-10-28 à 16 57 01" src="https://github.com/user-attachments/assets/fad062ae-bfe3-4bf1-9f73-e5acc4202a0d">

All further tests are done on cifar-100 dataset.

**Defense**

We use 2 mains techniques in defense:

- Gaussian noise method: we add some random Gaussian noise to the image and use Monte-Carlo method to get the most likely output.
- Denoising method: we use and optimize denoising methods (Blur + edge conservation, using in particular kornia.filters.bilateral_blur) to smooth the perturbation of the attack.

We get the following results for denoising method:

![attack_specific_success_vs_sigma](https://github.com/user-attachments/assets/829fbc5f-9691-4c57-ac51-1d7e916ae04a)

**Next steps**

This is an ongoing project.
The next steps are the implementation of unsupervized adversarial fine-tuning, leveraging an orthogonal training approach.

