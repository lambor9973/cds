import torch
import torch.nn.functional as F
import numpy as np


class LobachevskyPrototypeCorrector:
    def __init__(self, device, gamma_weight=10., cov=1.0):
        self.gamma_weight = gamma_weight
        self.cov = cov
        self.lambda_base_strength = 1.
        self.epsilon_cov_reg = 1e-6
        self.epsilon_lambda_denom = 1e-9
        self.device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data
        return torch.from_numpy(data).float().to(self.device)

    def correct_prototypes(self,
                           old_task_means_np: np.ndarray,
                           old_task_covariances_np: np.ndarray,
                           new_task_means_before_np: np.ndarray,
                           new_task_covariances_before_np: np.ndarray,
                           new_task_means_after_np: np.ndarray,
                           new_task_covariances_after_np: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
        """
        Correct class prototype means and covariances using simplified structure and heuristic Lambda.

        Args:
            old_task_means_np (np.ndarray): Old task means (K_old, feature_dim).
            old_task_covariances_np (np.ndarray): Old task covariances (K_old, feature_dim, feature_dim).
            new_task_means_before_np (np.ndarray): New means (old model) (K_new, feature_dim).
            new_task_covariances_before_np (np.ndarray): New covariances (old model) (K_new, feature_dim, feature_dim).
            new_task_means_after_np (np.ndarray): New means (new model) (K_new, feature_dim).
            new_task_covariances_after_np (np.ndarray): New covariances (new model) (K_new, feature_dim, feature_dim).

        Returns:
            tuple[np.ndarray, np.ndarray]: Corrected old means and old covariances (numpy arrays).
        """

        num_old_classes = old_task_means_np.shape[0]
        num_new_classes = new_task_means_before_np.shape[0]

        if num_old_classes == 0:
            feature_dim_fallback = new_task_means_before_np.shape[1] if num_new_classes > 0 else 1
            return np.array([]).reshape(0, feature_dim_fallback), np.array([]).reshape(0, feature_dim_fallback,
                                                                                       feature_dim_fallback)

        feature_dim = old_task_means_np.shape[1]

        if num_new_classes == 0:
            print("Warning: No new task prototypes provided. Returning original old prototypes.")
            return old_task_means_np, old_task_covariances_np

        mu_old_k_orig = self._to_tensor(old_task_means_np).to(self.device)
        Sigma_old_k_orig = self._to_tensor(old_task_covariances_np).to(self.device)
        mu_new_j_before = self._to_tensor(new_task_means_before_np).to(self.device)
        Sigma_new_j_before = self._to_tensor(new_task_covariances_before_np).to(self.device)
        mu_new_j_after = self._to_tensor(new_task_means_after_np).to(self.device)
        Sigma_new_j_after = self._to_tensor(new_task_covariances_after_np).to(self.device)

        # --- Mean Correction ---
        delta_mu_j_global = mu_new_j_after - mu_new_j_before

        corrected_mu_old_k_list = []
        all_weights_wk_list = []

        for k in range(num_old_classes):
            mu_old_k_current = mu_old_k_orig[k]

            mu_old_k_norm = F.normalize(mu_old_k_current.unsqueeze(0), p=2, dim=1)
            mu_new_j_before_norm = F.normalize(mu_new_j_before, p=2, dim=1)

            s_kj = torch.matmul(mu_old_k_norm, mu_new_j_before_norm.t()).squeeze(0)

            w_kj = F.softmax(self.gamma_weight * s_kj, dim=0)
            all_weights_wk_list.append(w_kj)

            Delta_mu_old_k = torch.sum(w_kj.unsqueeze(1) * delta_mu_j_global, dim=0)

            corrected_mu_old_k = mu_old_k_current + Delta_mu_old_k
            corrected_mu_old_k_list.append(corrected_mu_old_k)

        final_corrected_mu_old_k = torch.stack(corrected_mu_old_k_list, dim=0)

        # --- Covariance Correction ---
        corrected_Sigma_old_k_list = []
        for k in range(num_old_classes):
            Sigma_old_k_current = Sigma_old_k_orig[k]
            weights_for_current_k = all_weights_wk_list[k]

            total_directional_expansion_term_for_k = torch.zeros_like(Sigma_old_k_current)

            for j in range(num_new_classes):
                delta_mu_j_component = delta_mu_j_global[j]
                w_kj_scalar = weights_for_current_k[j]

                lambda_kj_heuristic = 0.0

                delta_mu_j_norm_sq = torch.sum(delta_mu_j_component ** 2)

                if delta_mu_j_norm_sq > self.epsilon_lambda_denom:
                    var_j_before_numerator = torch.matmul(
                        delta_mu_j_component.unsqueeze(0),
                        torch.matmul(Sigma_new_j_before[j], delta_mu_j_component.unsqueeze(1))
                    ).squeeze()
                    v_j_before = var_j_before_numerator / delta_mu_j_norm_sq

                    var_j_after_numerator = torch.matmul(
                        delta_mu_j_component.unsqueeze(0),
                        torch.matmul(Sigma_new_j_after[j], delta_mu_j_component.unsqueeze(1))
                    ).squeeze()
                    v_j_after = var_j_after_numerator / delta_mu_j_norm_sq

                    relative_stretch_factor_j = v_j_before / (v_j_after + self.epsilon_lambda_denom)

                    lambda_kj_heuristic = self.cov

                v_kj_component = w_kj_scalar * delta_mu_j_component

                directional_expansion_for_j = lambda_kj_heuristic * torch.matmul(
                    v_kj_component.unsqueeze(1),
                    v_kj_component.unsqueeze(0)
                )

                total_directional_expansion_term_for_k += directional_expansion_for_j

            corrected_Sigma_old_k = (Sigma_old_k_current +
                                     total_directional_expansion_term_for_k +
                                     self.epsilon_cov_reg * torch.eye(feature_dim, device=self.device))
            corrected_Sigma_old_k_list.append(corrected_Sigma_old_k)

        final_corrected_Sigma_old_k = torch.stack(corrected_Sigma_old_k_list, dim=0)

        return final_corrected_mu_old_k.cpu().numpy(), final_corrected_Sigma_old_k.cpu()
