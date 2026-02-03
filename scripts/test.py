import torch
# from sae_lens import SAE

# # Load your SAE and extract the decoder row
# sae = SAE.from_pretrained(
#     release="goodfire-llama-3.1-8b-instruct",
#     sae_id="layer_19",
#     device="cuda",
#     dtype=torch.bfloat16,
# )
# feature_idx = 18506  # Your suspect feature index
# v_dec = sae.W_dec[feature_idx]

# # Save for EasySteer to load
# torch.save(v_dec, "vectors/feature_18506_vector.pt")

a = 5.0
b = torch.tensor([1.0, 7.0, 3.0])
print(torch.clamp(a - b, min=0).unsqueeze(-1))
