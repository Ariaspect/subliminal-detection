import torch
from vllm.steer_vectors.algorithms.template import AlgorithmTemplate
from vllm.steer_vectors.algorithms.factory import register_algorithm


@register_algorithm("sae_min_clamp")
class SAEMinClamp(AlgorithmTemplate):
    """
    Algorithm to ensure a latent feature is active at a minimum strength.
    h' = h + max(0, alpha - current_projection) * decoder_vector
    """

    def _transform(self, hidden_states: torch.Tensor, params) -> torch.Tensor:
        # Get feature direction and target strength
        alpha = params.get("alpha", 1.0)
        # params['vector'] is the SAE decoder vector for your target latent
        v_dec = params.get("vector").to(hidden_states.device)

        # 1. Project current hidden state onto the feature direction
        # Conceptually: how active is this feature naturally?
        current_act = torch.einsum("blh,h->bl", hidden_states, v_dec)

        # 2. Calculate necessary 'nudge' to hit the alpha threshold
        # If natural act is < alpha, we add the difference
        nudge = torch.clamp(alpha - current_act, min=0).unsqueeze(-1)

        # 3. Apply the minimal addition needed to reach the clamp value
        return hidden_states + (nudge * v_dec)
