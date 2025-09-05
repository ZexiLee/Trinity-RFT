import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch

from trinity.buffer import BufferWriter
from trinity.common.experience import Experience
from trinity.utils.monitor import gather_metrics
from trinity.utils.registry import Registry
from trinity.utils.timer import Timer

ADD_STRATEGY = Registry("add_strategy")


class AddStrategy(ABC):
    def __init__(self, writer: BufferWriter, **kwargs) -> None:
        self.writer = writer

    @abstractmethod
    async def add(self, experiences: List[Experience], step: int) -> Tuple[int, Dict]:
        """Add experiences to the buffer.

        Args:
            experiences (`Experience`): The experiences to be added.
            step (`int`): The current step number.

        Returns:
            `int`: The number of experiences added to the buffer.
            `Dict`: Metrics for logging.
        """

    @classmethod
    @abstractmethod
    def default_args(cls) -> dict:
        """Get the default arguments of the add strategy.

        Returns:
            `dict`: The default arguments.
        """


class GroupAdvantageStrategy(AddStrategy):
    """An example AddStrategy that calculates group advantages."""

    @abstractmethod
    def group_experiences(self, exps: List[Experience]) -> Dict[str, List[Experience]]:
        """Group experiences by a certain criterion.

        Args:
            exps (List[Experience]): List of experiences to be grouped.

        Returns:
            Dict[str, List[Experience]]: A dictionary where keys are group identifiers and values are lists of experiences.
        """

    @abstractmethod
    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        """Calculate advantages for a group of experiences.

        Args:
            group_id (str): The identifier for the group of experiences.
            exps (List[Experience]): List of experiences in the group.

        Returns:
            Tuple[List[Experience], Dict]: A tuple containing the modified list of experiences and a dictionary of metrics.
        """

    async def add(self, exps: List[Experience], step: int) -> Tuple[int, Dict]:
        if len(exps) == 0:
            return 0, {}
        exp_groups = self.group_experiences(exps)
        cnt = 0
        metric_list = []
        tasks = []
        for group_id, group_exps in exp_groups.items():
            group_exps, group_metrics = self.calculate_group_advantage(group_id, group_exps)
            metric_list.append(group_metrics)
            cnt += len(group_exps)
            if len(group_exps) > 0:
                tasks.append(self.writer.write_async(group_exps))
        if tasks:
            await asyncio.gather(*tasks)
        try:
            metrics = gather_metrics(metric_list, "group_advantages")
        except ValueError:
            metrics = {}  # empty metric list causes ValueError, ignore it
        return cnt, metrics



@ADD_STRATEGY.register_module("grpo")
class GRPOAddStrategy(GroupAdvantageStrategy):
    """An example AddStrategy that calculates GRPO advantages."""

    def __init__(self, writer: BufferWriter, epsilon: float = 1e-6, **kwargs) -> None:
        super().__init__(writer)
        self.epsilon = epsilon

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            if len(exps) == 1:
                group_reward_mean = torch.tensor(0.0)
                group_reward_std = torch.tensor(1.0)
            else:
                rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)
                group_reward_mean = torch.mean(rewards)
                group_reward_std = torch.std(rewards)
            for exp in exps:
                score = (exp.reward - group_reward_mean) / (group_reward_std + self.epsilon)
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()

            metrics = {
                "reward_mean": group_reward_mean.item(),
                "reward_std": group_reward_std.item(),
            }

        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"epsilon": 1e-6}

@ADD_STRATEGY.register_module("opo")
class OPOAddStrategy(GroupAdvantageStrategy):
    """An example AddStrategy that calculates OPO advantages."""

    def __init__(self, writer: BufferWriter, epsilon: float = 1e-6, **kwargs) -> None:
        super().__init__(writer)
        self.epsilon = epsilon

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)

            # If there's only one sample, its reward is the mean, and we use 1.0 for std to avoid division by zero.
            if len(exps) == 1:
                weighted_reward_mean = rewards[0]
                group_reward_std = torch.tensor(1.0)

            else:
                # Calculate the length of each response.
                response_lengths = torch.tensor(
                    [len(exp.tokens) - exp.prompt_length for exp in exps], dtype=torch.float32
                )

                # Calculate the response-length-weighted reward mean as the new baseline.
                # baseline = sum(reward * length) / sum(length)
                numerator = torch.sum(rewards * response_lengths)
                denominator = torch.sum(response_lengths)
                weighted_reward_mean = numerator / (denominator + self.epsilon)

            # Calculate advantage using the new weighted mean as the baseline.
            for exp in exps:
                score = (exp.reward - weighted_reward_mean)
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()

            # Update metrics to reflect the new baseline.
            metrics = {
                "reward_mean": float(weighted_reward_mean),
                "reward_std": float(group_reward_std)
            }


        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"epsilon": 1e-6}



@ADD_STRATEGY.register_module("grpo-opo")
class GRPOOPOAddStrategy(GroupAdvantageStrategy):
    """An example AddStrategy that calculates GRPO + OPO advantages."""

    def __init__(self, writer: BufferWriter, epsilon: float = 1e-6, alpha: float = 1.0, **kwargs) -> None:
        super().__init__(writer)
        self.epsilon = epsilon

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)

            # If there's only one sample, its reward is the mean, and we use 1.0 for std to avoid division by zero.
            if len(exps) == 1:
                weighted_reward_mean = rewards[0]
                group_reward_std = torch.tensor(1.0)
                baseline_delta = torch.tensor(0.0)
                reward_response_length_slope = 0.0

            else:
                # Calculate the length of each response.
                response_lengths = torch.tensor(
                    [len(exp.tokens) - exp.prompt_length for exp in exps], dtype=torch.float32
                )

                # Calculate the response-length-weighted reward mean as the new baseline.
                # baseline = sum(reward * length) / sum(length)
                numerator = torch.sum(rewards * response_lengths)
                denominator = torch.sum(response_lengths)
                weighted_reward_mean = numerator / (denominator + self.epsilon)

                # Standard deviation is still calculated on the original, unweighted rewards as per the requirement.
                group_reward_std = torch.std(rewards)

                ### response reward regression ###
                median_l = torch.median(response_lengths)
                # 防御：median>=1 保证安全；仍加eps
                tilde_l = response_lengths / (median_l + self.epsilon)
                log_tilde_l = torch.log(tilde_l)

                # 线性回归 slope（仅用于指标观测）
                def get_linear_slope(x: torch.Tensor, y: torch.Tensor) -> float:
                    if x.dim() != 1 or x.shape != y.shape:
                        raise ValueError("Inputs must be 1D tensors of the same shape.")
                    x_mean = x.mean()
                    y_mean = y.mean()
                    covariance = torch.sum((x - x_mean) * (y - y_mean))
                    variance = torch.sum((x - x_mean) ** 2)
                    if variance == 0:
                        return float("inf")
                    slope = covariance / variance
                    return slope.item()
                
                reward_response_length_slope = get_linear_slope(log_tilde_l, rewards)

                group_reward_mean = torch.mean(rewards)
                baseline_delta = weighted_reward_mean - group_reward_mean
                ### response reward regression ###

            # Calculate advantage using the new weighted mean as the baseline.
            for exp in exps:
                score = (exp.reward - weighted_reward_mean) / (group_reward_std + self.epsilon)
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()

            # Update metrics to reflect the new baseline.
            metrics = {
                "reward_mean": float(weighted_reward_mean),
                "reward_std": float(group_reward_std),
                "baseline_delta": float(baseline_delta),
                "reward_response_length_slope": float(reward_response_length_slope),
            }


        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"epsilon": 1e-6}


@ADD_STRATEGY.register_module("grpo-opo-new")
class GRPOOPONewStrategy(GroupAdvantageStrategy):
    """
    GRPO + Adaptive OPO (Power-OPO with LOO baseline).
    - Assumption proxy: ||∇ log π||^2 ∝ (length / median(length))^alpha
    - Advantages use a leave-one-out (LOO) baseline:
        b_{-i} = (S1 - w_i r_i) / (S0 - w_i)
        A_i = r_i - b_{-i}
      Then standardize by unweighted std(rewards) (keeps behavior consistent).
    Notes:
      * Signatures and returned fields unchanged.
      * Extra metrics include 'alpha' and 'alpha_objective'.
    """

    def __init__(self, writer: BufferWriter, epsilon: float = 1e-6,  alpha: float = 1.0, **kwargs) -> None:
        super().__init__(writer)
        self.epsilon = epsilon
        self.alpha = alpha

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)

            if len(exps) == 1:
                weighted_reward_mean = rewards[0]
                group_reward_std = torch.tensor(1.0)
                group_reward_mean = rewards[0]
                baseline_delta = torch.tensor(0.0)
                reward_response_length_slope = 0.0

                # 优势：0（除以std后仍为0）
                score = torch.tensor(0.0)
                exps[0].advantages = score * exps[0].action_mask
                exps[0].returns = exps[0].advantages.clone()

            else:
                ### response reward regression ###
                response_lengths = torch.tensor(
                    [max(1, len(exp.tokens) - exp.prompt_length) for exp in exps],
                    dtype=torch.float32,
                )
                median_l = torch.median(response_lengths)
                # 防御：median>=1 保证安全；仍加eps
                tilde_l = response_lengths / (median_l + self.epsilon)
                log_tilde_l = torch.log(tilde_l)

                # 线性回归 slope（仅用于指标观测）
                def get_linear_slope(x: torch.Tensor, y: torch.Tensor) -> float:
                    if x.dim() != 1 or x.shape != y.shape:
                        raise ValueError("Inputs must be 1D tensors of the same shape.")
                    x_mean = x.mean()
                    y_mean = y.mean()
                    covariance = torch.sum((x - x_mean) * (y - y_mean))
                    variance = torch.sum((x - x_mean) ** 2)
                    if variance == 0:
                        return float("inf")
                    slope = covariance / variance
                    return slope.item()
                
                reward_response_length_slope = get_linear_slope(log_tilde_l, rewards)
                ### response reward regression ###


                # 计算 S0/S1/S2 与 J(alpha)
                def stats_at(alpha: torch.Tensor):
                    # w = (l_tilde) ** alpha = exp(alpha * log l_tilde)
                    w = torch.exp(alpha * log_tilde_l)
                    S0 = w.sum()
                    S1 = (w * rewards).sum()
                    S2 = (w * rewards * rewards).sum()
                    return w, S0, S1, S2

                w, S0, S1, _ = stats_at(self.alpha)

                group_reward_mean = torch.mean(rewards)
                weighted_reward_mean = S1 / (S0 + self.epsilon)  # 汇总基线（用于记录）

                baseline_delta = weighted_reward_mean - group_reward_mean

                # 标准差保持未加权版本（与参考实现一致）
                group_reward_std = torch.std(rewards)

                # —— LOO 基线与优势 —— #
                # b_{-i} = (S1 - w_i r_i) / (S0 - w_i)
                S0_minus = S0 - w
                S1_minus = S1 - w * rewards
                b_minus = S1_minus / (S0_minus + self.epsilon)
                advantages = rewards - b_minus  # A_i（未标准化）

                # 写回
                std_denom = group_reward_std + self.epsilon
                for i, exp in enumerate(exps):
                    score = advantages[i] / std_denom
                    exp.advantages = score * exp.action_mask
                    exp.returns = exp.advantages.clone()

            metrics = {
                "reward_mean": float(weighted_reward_mean),
                "reward_std": float(group_reward_std),
                "alpha": float(self.alpha),
                "baseline_delta": float(baseline_delta),
                "reward_response_length_slope": float(reward_response_length_slope),
            }

        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"epsilon": 1e-6}


@ADD_STRATEGY.register_module("grpo-aopo")
class GRPOAOPOAddStrategy(GroupAdvantageStrategy):
    """
    GRPO + OPO with adaptive length^alpha baseline.
    - Assumption: ||∇ log π||^2 ∝ length^alpha
    - For each group, choose alpha by minimizing
        J(alpha) = sum_i l_i^alpha (r_i - b_alpha)^2
                 = S2 - S1^2 / S0,
      where b_alpha = S1 / S0, S0=∑v_i, S1=∑v_i r_i, S2=∑v_i r_i^2, v_i=l_i^alpha.
    - Then compute advantages with this baseline.
    Notes:
      * Std remains the unweighted std(rewards), matching your original behavior.
      * Signatures and returned fields unchanged. Extra metrics include 'alpha' and 'alpha_objective'.
    """

    def __init__(self, writer: BufferWriter, epsilon: float = 1e-6, **kwargs) -> None:
        super().__init__(writer)
        self.epsilon = epsilon
        # 可通过 kwargs 覆盖；不提供也有默认网格
        self.alpha_grid = kwargs.get(
            "alpha_grid", [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        )

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)

            # 单样本组：退化处理
            if len(exps) == 1:
                weighted_reward_mean = rewards[0]
                group_reward_std = torch.tensor(1.0)
                chosen_alpha = torch.tensor(0.0)
                J_star = torch.tensor(0.0)
                # —— 修正：补齐用于 metrics 的字段 —— 
                group_reward_mean = rewards[0]
                baseline_delta = torch.tensor(0.0)
                reward_response_length_slope = 0.0
            else:
                # 计算每条响应的长度（至少为 1，防止 log(0)）
                response_lengths = torch.tensor(
                    [max(1, len(exp.tokens) - exp.prompt_length) for exp in exps],
                    dtype=torch.float32,
                )
                log_l = torch.log(response_lengths)

                def stats_at(alpha: torch.Tensor):
                    # v = l ** alpha = exp(alpha * log l)
                    v = torch.exp(alpha * log_l)
                    S0 = v.sum()
                    S1 = (v * rewards).sum()
                    S2 = (v * rewards * rewards).sum()
                    return S0, S1, S2

                def J(alpha: torch.Tensor):
                    S0, S1, S2 = stats_at(alpha)
                    return S2 - (S1 * S1) / (S0 + self.epsilon)

                def get_linear_slope(x: torch.Tensor, y: torch.Tensor) -> float:
                    """
                    Calculates the slope 'a' of a linear regression (y = a*x + b).
                    """
                    if x.dim() != 1 or x.shape != y.shape:
                        raise ValueError("Inputs must be 1D tensors of the same shape.")
                    x_mean = x.mean()
                    y_mean = y.mean()
                    covariance = torch.sum((x - x_mean) * (y - y_mean))
                    variance = torch.sum((x - x_mean) ** 2)
                    if variance == 0:
                        return float('inf')
                    slope = covariance / variance
                    return slope.item()

                # —— 修正：语义一致，x 为 response_lengths，y 为 rewards —— 
                reward_response_length_slope = get_linear_slope(response_lengths, rewards)

                # —— 1D 最小化：在小网格上选取最优 alpha ——
                alpha_candidates = torch.tensor(self.alpha_grid, dtype=torch.float32)
                J_vals = torch.stack([J(a) for a in alpha_candidates])
                best_idx = torch.argmin(J_vals)
                chosen_alpha = alpha_candidates[best_idx]
                J_star = J_vals[best_idx]

                group_reward_mean = torch.mean(rewards)

                # —— 用最优 alpha 计算加权均值基线 ——
                S0, S1, _ = stats_at(chosen_alpha)
                weighted_reward_mean = S1 / (S0 + self.epsilon)

                baseline_delta = weighted_reward_mean - group_reward_mean

                # 标准差保持未加权版本（与参考实现一致）
                group_reward_std = torch.std(rewards)

            # 计算优势并写回（与参考实现一致）
            for exp in exps:
                score = (exp.reward - weighted_reward_mean) / (group_reward_std + self.epsilon)
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()

            metrics = {
                "reward_mean": float(weighted_reward_mean),
                "reward_std": float(group_reward_std),
                "alpha": float(chosen_alpha),
                "alpha_objective": float(J_star),
                "baseline_delta": float(baseline_delta),
                "reward_response_length_slope": float(reward_response_length_slope),
            }

        return exps, metrics


    @classmethod
    def default_args(cls) -> dict:
        return {"epsilon": 1e-6}


@ADD_STRATEGY.register_module("grpo-aopo-new")
class GRPOAOPONewStrategy(GroupAdvantageStrategy):
    """
    GRPO + Adaptive OPO (Power-OPO with LOO baseline).
    - Assumption proxy: ||∇ log π||^2 ∝ (length / median(length))^alpha
    - For each group, choose alpha by minimizing
        J(alpha) = sum_i w_i (r_i - b_alpha)^2
                 = S2 - S1^2 / S0,
      where w_i = (l_i / median(l))^alpha, b_alpha = S1 / S0,
            S0=∑w_i, S1=∑w_i r_i, S2=∑w_i r_i^2.
    - Advantages use a leave-one-out (LOO) baseline:
        b_{-i} = (S1 - w_i r_i) / (S0 - w_i)
        A_i = r_i - b_{-i}
      Then standardize by unweighted std(rewards) (keeps behavior consistent).
    Notes:
      * Signatures and returned fields unchanged.
      * Extra metrics include 'alpha' and 'alpha_objective'.
    """

    def __init__(self, writer: BufferWriter, epsilon: float = 1e-6, **kwargs) -> None:
        super().__init__(writer)
        self.epsilon = epsilon
        # 默认网格（可通过 kwargs 覆盖）
        self.alpha_grid = kwargs.get("alpha_grid", [0.0, 0.5, 1.0, 1.5, 2.0])

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)

            # 单样本组：退化处理（A=0）
            if len(exps) == 1:
                weighted_reward_mean = rewards[0]
                group_reward_std = torch.tensor(1.0)
                chosen_alpha = torch.tensor(0.0)
                J_star = torch.tensor(0.0)
                group_reward_mean = rewards[0]
                baseline_delta = torch.tensor(0.0)
                reward_response_length_slope = 0.0

                # 优势：0（除以std后仍为0）
                score = torch.tensor(0.0)
                exps[0].advantages = score * exps[0].action_mask
                exps[0].returns = exps[0].advantages.clone()

            else:
                # 响应长度（至少为1），并做中位数归一化：tilde_l = l / median(l)

                # print("old_logprob!!!", exp.logprobs)
                # print("old_logprob.shape", exp.old_logprob.shape)


                ### response reward regression ###
                response_lengths = torch.tensor(
                    [max(1, len(exp.tokens) - exp.prompt_length) for exp in exps],
                    dtype=torch.float32,
                )
                median_l = torch.median(response_lengths)
                # 防御：median>=1 保证安全；仍加eps
                tilde_l = response_lengths / (median_l + self.epsilon)
                log_tilde_l = torch.log(tilde_l)

                # 线性回归 slope（仅用于指标观测）
                def get_linear_slope(x: torch.Tensor, y: torch.Tensor) -> float:
                    if x.dim() != 1 or x.shape != y.shape:
                        raise ValueError("Inputs must be 1D tensors of the same shape.")
                    x_mean = x.mean()
                    y_mean = y.mean()
                    covariance = torch.sum((x - x_mean) * (y - y_mean))
                    variance = torch.sum((x - x_mean) ** 2)
                    if variance == 0:
                        return float("inf")
                    slope = covariance / variance
                    return slope.item()
                
                reward_response_length_slope = get_linear_slope(log_tilde_l, rewards)
                ### response reward regression ###


                # 计算 S0/S1/S2 与 J(alpha)
                def stats_at(alpha: torch.Tensor):
                    # w = (l_tilde) ** alpha = exp(alpha * log l_tilde)
                    w = torch.exp(alpha * log_tilde_l)
                    S0 = w.sum()
                    S1 = (w * rewards).sum()
                    S2 = (w * rewards * rewards).sum()
                    return w, S0, S1, S2

                def J(alpha: torch.Tensor):
                    _, S0, S1, S2 = stats_at(alpha)
                    return S2 - (S1 * S1) / (S0 + self.epsilon)

                # —— 1D 网格最小化选择 alpha ——
                alpha_candidates = torch.tensor(self.alpha_grid, dtype=torch.float32)
                J_vals = torch.stack([J(a) for a in alpha_candidates])
                best_idx = torch.argmin(J_vals)
                chosen_alpha = alpha_candidates[best_idx]
                J_star = J_vals[best_idx]

                # 用最优 alpha 的权重与统计量
                w, S0, S1, _ = stats_at(chosen_alpha)

                group_reward_mean = torch.mean(rewards)
                weighted_reward_mean = S1 / (S0 + self.epsilon)  # 汇总基线（用于记录）

                baseline_delta = weighted_reward_mean - group_reward_mean

                # 标准差保持未加权版本（与参考实现一致）
                group_reward_std = torch.std(rewards)

                # —— LOO 基线与优势 —— #
                # b_{-i} = (S1 - w_i r_i) / (S0 - w_i)
                S0_minus = S0 - w
                S1_minus = S1 - w * rewards
                b_minus = S1_minus / (S0_minus + self.epsilon)
                advantages = rewards - b_minus  # A_i（未标准化）

                # 写回
                std_denom = group_reward_std + self.epsilon
                for i, exp in enumerate(exps):
                    score = advantages[i] / std_denom
                    exp.advantages = score * exp.action_mask
                    exp.returns = exp.advantages.clone()

            metrics = {
                "reward_mean": float(weighted_reward_mean),
                "reward_std": float(group_reward_std),
                "alpha": float(chosen_alpha),
                "alpha_objective": float(J_star),
                "baseline_delta": float(baseline_delta),
                "reward_response_length_slope": float(reward_response_length_slope),
            }

        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"epsilon": 1e-6}


@ADD_STRATEGY.register_module("grpo-rs")
class GRPORSAddStrategy(GroupAdvantageStrategy):
    """
    An AddStrategy that calculates GRPO-RS (Robust Scaling) advantages.
    This version uses median for centering and Interquartile Range (IQR) for scaling,
    making it more robust to outlier rewards.
    """

    def __init__(self, writer: BufferWriter, epsilon: float = 1e-6, **kwargs) -> None:
        super().__init__(writer)
        self.epsilon = epsilon

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            # For a single sample or fewer, normalization is meaningless.
            # Default to a median of 0 and IQR of 1 for a stable advantage of 0.
            if len(exps) <= 1:
                group_reward_median = torch.tensor(0.0)
                group_reward_iqr = torch.tensor(1.0)
            else:
                rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)
                
                # --- Core Logic Change: From Mean/Std to Median/IQR ---
                # 1. Use Median for centering
                group_reward_median = torch.median(rewards)
                
                # 2. Use Interquartile Range (IQR) for scaling
                q1 = torch.quantile(rewards, 0.25)
                q3 = torch.quantile(rewards, 0.75)
                group_reward_iqr = q3 - q1
                # --- End of Core Logic Change ---

            for exp in exps:
                # The advantage is now calculated using robust statistics
                score = (exp.reward - group_reward_median) / (group_reward_iqr + self.epsilon)
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()

            # Update metrics to reflect the new statistics
            metrics = {
                "reward_median": group_reward_median.item(),
                "reward_iqr": group_reward_iqr.item(),
            }

        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"epsilon": 1e-6}

@ADD_STRATEGY.register_module("grpo-rss")
class GRPORSSAddStrategy(GroupAdvantageStrategy):
    """
    An AddStrategy that calculates GRPO-RSS advantages.
    This version uses median for centering and Standard Deviation for scaling.
    """

    def __init__(self, writer: BufferWriter, epsilon: float = 1e-6, **kwargs) -> None:
        super().__init__(writer)
        self.epsilon = epsilon

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            if len(exps) <= 1:
                # Default for single samples
                group_reward_median = torch.tensor(0.0)
                group_reward_std = torch.tensor(1.0)
            else:
                rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)

                # --- Core Logic Change: Median for centering, Std for scaling ---
                # 1. Use Median for centering (as before)
                group_reward_median = torch.median(rewards)

                # 2. Use Standard Deviation for scaling (changed from IQR)
                group_reward_std = torch.std(rewards)
                # --- End of Core Logic Change ---

            for exp in exps:
                # The advantage is calculated using median and standard deviation
                score = (exp.reward - group_reward_median) / (group_reward_std + self.epsilon)
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()

            # Update metrics to reflect the new statistics
            metrics = {
                "reward_median": group_reward_median.item(),
                "reward_std": group_reward_std.item(),
            }

        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"epsilon": 1e-6}

@ADD_STRATEGY.register_module("opmd")
class OPMDAddStrategy(GroupAdvantageStrategy):
    """An example AddStrategy that calculates OPMD advantages."""

    def __init__(
        self, writer: BufferWriter, opmd_baseline: str = "mean", tau: float = 1.0, **kwargs
    ) -> None:
        super().__init__(writer)
        assert opmd_baseline in [
            "mean",
            "logavgexp",
        ], f"opmd_baseline must be 'mean' or 'logavgexp', got {opmd_baseline}"
        self.opmd_baseline = opmd_baseline
        self.tau = tau

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            if len(exps) == 1:
                group_baseline = torch.tensor(0.0)
            else:
                group_rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)
                if self.opmd_baseline == "mean":
                    group_baseline = torch.mean(group_rewards)
                else:
                    group_baseline = self.tau * (
                        torch.logsumexp(group_rewards / self.tau, dim=-1)
                        - torch.log(torch.tensor(len(exps)))
                    )
            for exp in exps:
                score = exp.reward - group_baseline
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()
            metrics = {
                "group_baseline": group_baseline,
            }
        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"opmd_baseline": "mean", "tau": 1.0}


@ADD_STRATEGY.register_module("reward_variance")
class RewardVarianceAddStrategy(AddStrategy):
    """An example AddStrategy that filters experiences based on a reward variance threshold."""

    def __init__(self, writer: BufferWriter, variance_threshold: float = 0.0, **kwargs) -> None:
        super().__init__(writer)
        self.variance_threshold = variance_threshold

    async def add(self, experiences: List[Experience], step: int) -> Tuple[int, Dict]:
        cnt = 0
        metrics = {}
        tasks = []
        with Timer(metrics, "add_strategy_time"):
            grouped_experiences = group_by(experiences, id_type="task")
            for _, group_exps in grouped_experiences.items():
                if len(group_exps) < 2:
                    continue
                rewards = [exp.reward for exp in group_exps]
                variance = np.var(rewards)
                if variance <= self.variance_threshold:
                    continue
                cnt += len(group_exps)
                tasks.append(self.writer.write_async(group_exps))
            if tasks:
                await asyncio.gather(*tasks)
        return cnt, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"variance_threshold": 0.0}


def group_by(
    experiences: List[Experience], id_type: Literal["task", "run", "step"]
) -> Dict[str, List[Experience]]:
    """Group experiences by ID."""
    if id_type == "task":
        id_type = "tid"
    elif id_type == "run":
        id_type = "rid"
    elif id_type == "step":
        id_type = "sid"
    else:
        raise ValueError(f"Unknown id_type: {id_type}")
    grouped = {}
    for exp in experiences:
        group_id = getattr(exp.eid, id_type)
        if group_id not in grouped:
            grouped[group_id] = []
        grouped[group_id].append(exp)
    return grouped
