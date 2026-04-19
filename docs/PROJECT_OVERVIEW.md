# TetraRL: Project Overview

> **Target Venue**: IEEE Transactions on Computers (TC) / RTSS / RTAS
> **Theme**: 4-D Preference-Conditioned Multi-Objective RL for Embedded Systems

## 1. High-Level Vision

TetraRL is a systems-centric framework for deploying Deep Reinforcement Learning (DRL) on resource-constrained embedded platforms (e.g., Jetson Orin AGX, Orin Nano). It extends prior conference work (R³ + DuoJoule) by addressing **4-D optimization** simultaneously:

*   **R**eal-time (Tail Latency, p99)
*   **R**eward (Task Performance)
*   **R**AM (Memory Footprint)
*   **R**echarge (Energy Consumption)

Unlike pure ML benchmarking, TetraRL focuses on the **systems engineering** required to make RL robust, adaptable, and safe on physical edge hardware.

## 2. Core Narrative: Why TetraRL?

### The Problem with Standard RL
1.  **Single-Objective**: Standard RL maximizes reward. If a platform overheats or runs out of memory, standard RL cannot adapt unless retrained with new penalty weights.
2.  **Failure Handling**: Neural networks offer statistical guarantees, not hard guarantees. In safety-critical embedded systems, relying *only* on the neural network to learn hard constraints leads to catastrophic failure rates.
3.  **Systems Overhead**: RL inference and telemetry polling must be extremely cheap to avoid stealing cycles from the real-time application it is supposed to be managing.

### The TetraRL Solution
TetraRL acknowledges that **multi-objective optimization on physical hardware is fundamentally harder than single-objective optimization**. It accepts trade-offs (e.g., potentially lower absolute reward) in exchange for adaptability and safety.

*   **Adaptability**: A single trained policy accepts a preference vector ($\omega$) at runtime, allowing the system to instantly pivot priorities (e.g., "Low battery detected $\rightarrow$ shift preference to energy conservation") without retraining.
*   **Safety (Constraint Respect)**: A deterministic override layer catches RL violations and enforces hardware-level constraints, reducing failure rates drastically compared to pure RL.

## 3. Four-Layer System Architecture

TetraRL handles the systems concerns (failure, generalization, overhead) through four distinct layers:

| Component | Primary Concern Addressed | Mechanism |
| :--- | :--- | :--- |
| **Preference Plane** | **Generalization** | Exposes a preference vector ($\omega$) interface. The underlying agent (e.g., Preference-Conditioned PPO) learns a continuous Pareto front, allowing instant runtime adaptation to new user priorities. |
| **RL Arbiter** | **Algorithm Abstraction** | A unified `ResourcePrimitives` interface abstracts away the differences between on-policy (PPO) and off-policy (SAC/DQN) algorithms, handling observation/action spaces uniformly. |
| **Resource Manager** | **Hardware Interfacing & Overhead** | Low-overhead interfaces to Jetson hardware. Uses an asynchronous `tegrastats` daemon to poll metrics without blocking the main control loop, and interfaces with `sysfs` for DVFS control. |
| **Override Layer** | **Failure Handling (Hard Constraints)** | A deterministic safety net. If the RL arbiter's action violates critical thresholds (e.g., OOM risk, thermal limits), this layer overrides the action with hard kernel-level interventions (e.g., `freq_down`, `mem_evict`), independent of RL learning. |

## 4. Evaluation Strategy & Metrics

TetraRL evaluates systems integration on real hardware, not just algorithmic convergence in simulation.

### Environments
*   **DAG-scheduler-MO**: A custom multi-objective environment simulating real-time DAG task scheduling (stressing latency and energy).
*   **PyBullet-HalfCheetah** (or FFmpeg co-runner): Continuous control/compute-heavy tasks to stress memory and DVFS interactions.

### Key Metrics
1.  **Hypervolume (HV)**: Measures the volume of the 4-D Pareto front captured by the agent. *Note: Single-objective baselines (like MAX-A) may occasionally dominate aggregate HV by excelling on one axis. TetraRL's goal is competitive HV with vastly superior adaptability.*
2.  **Lagrangian Violation Rate**: Proves the efficacy of the Override Layer (target: $< 5\%$ violation vs. $\sim 25\%$ for Lagrangian-only RL).
3.  **Framework Overhead**: Measures the ms/step cost of the TetraRL framework itself (target: $< 30\%$ of bare-RL step time on Nano).
4.  **Dynamic Preference Switching**: Demonstrates smooth adaptation when $\omega$ changes mid-episode.

### Experimental Matrix
*   **Hardware**: Jetson Orin AGX (high compute) vs. Jetson Orin Nano (constrained).
*   **Algorithms**: TetraRL (Preference PPO) vs. PD-MORL, FOCOPS, PPO-Lagrangian, Envelope MORL, DuoJoule, PCN.

## 5. Development Roadmap (W7-W12)

This project follows an aggressive, agent-assisted implementation roadmap:

*   **W7-W8**: Core framework implementation, environments (4-D DAG), baseline integration, Nano/Orin platform profiles, and component ablation studies.
*   **W9**: Nano deep-validation, DVFS-DRL-Multitask baseline implementation, multi-env scaling tests.
*   **W10**: Full empirical evaluation matrix (HV comparison, override validation, dynamic switching).
*   **W11**: Final figure generation (CDF panels, Pareto fronts) and Nano-GRPO micro-benchmark appendix.
*   **W12**: Full paper drafting (LaTeX) and internal review.

---
*Generated: 2026-04-19. Refer to `MEMORY.md` and `docs/action-plan-weekly.md` for real-time task status.*
